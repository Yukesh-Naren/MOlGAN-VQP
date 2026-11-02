import os
import math
import gzip
import pickle
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress noisy RDKit C++ warnings/deprecation logs
RDLogger.DisableLog('rdApp.*')

# Allowed atom types for QM9:
ATOM_TYPES = ['PAD', 'C', 'N', 'O', 'F']
ATOMIC_NUMBERS = [0, 6, 7, 8, 9]

# 5 Bond types matching QM9 dataset:
BOND_TYPES = [
    None,                           # Index 0: No bond
    Chem.rdchem.BondType.SINGLE,    # Index 1: Single bond
    Chem.rdchem.BondType.DOUBLE,    # Index 2: Double bond
    Chem.rdchem.BondType.TRIPLE,    # Index 3: Triple bond
    Chem.rdchem.BondType.AROMATIC   # Index 4: Aromatic bond
]

# Load SA and NP scoring models if present
def _load_pickle_gz(filepath):
    if os.path.exists(filepath):
        try:
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    return None

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sa_raw = _load_pickle_gz(os.path.join(_BASE_DIR, 'data', 'SA_score.pkl.gz'))
SA_MODEL = {i[j]: float(i[0]) for i in _sa_raw for j in range(1, len(i))} if _sa_raw else {}

_np_raw = _load_pickle_gz(os.path.join(_BASE_DIR, 'data', 'NP_score.pkl.gz'))
NP_MODEL = _np_raw if _np_raw else {}

# SMARTS patterns for physically unstable, explosive, or strained structural motifs:
UNSTABLE_SMARTS_PATTERNS = [
    Chem.MolFromSmarts('[#7]~[#7]~[#7]'),        # Contiguous 3+ nitrogens (triazenes, azides, etc.)
    Chem.MolFromSmarts('[#8]~[#8]'),            # Peroxides & poly-oxides (O-O)
    Chem.MolFromSmarts('[#7,#8]1~[#7,#8]~*1'),  # Strained 3-membered rings with multiple heteroatoms
    Chem.MolFromSmarts('[#9]~[#7,#8,#9]'),      # Fluorine bonded to heteroatom (N-F, O-F, F-F)
]

# REINVENT Structural Alert & Strain Filters:
BREDT_VIOLATION = Chem.MolFromSmarts('[$([#6;R1;!$(*=,#[!#6])]=,#[#6;R1])]')
STRAINED_RINGS = [
    Chem.MolFromSmarts('C1=CC1'),  # Cyclopropene
    Chem.MolFromSmarts('C1=CO1'),  # Oxirene
    Chem.MolFromSmarts('C1=CN1'),  # 2H-Azirine
]

def check_reinvent_structural_alerts(mol):
    """Returns True if molecule violates Bredt's rule or contains strained 3-membered rings."""
    if mol is None:
        return False
    if BREDT_VIOLATION is not None and mol.HasSubstructMatch(BREDT_VIOLATION):
        return True
    for pat in STRAINED_RINGS:
        if pat is not None and mol.HasSubstructMatch(pat):
            return True
    return False


class ScaffoldMemoryFilter:
    """
    REINVENT Scaffold Diversity Memory.
    Tracks Bemis-Murcko scaffolds in memory buckets to mathematically prevent mode collapse.
    When a scaffold exceeds bucket_size (e.g. 25 occurrences), the reward is zeroed out.
    """
    def __init__(self, bucket_size=25, penalty=0.0):
        self.bucket_size = bucket_size
        self.penalty = penalty
        self.memory = {}

    def score_and_update(self, mol):
        if mol is None:
            return 1.0
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if not scaffold:
                scaffold = Chem.MolToSmiles(mol)
            count = self.memory.get(scaffold, 0)
            if count >= self.bucket_size:
                return self.penalty
            self.memory[scaffold] = count + 1
            return 1.0
        except Exception:
            return 1.0

    def reset(self):
        self.memory.clear()


def matrices_to_mols(node_indices, edge_indices):
    """
    Converts node indices and edge indices into a list of RDKit Mol objects.
    """
    if isinstance(node_indices, torch.Tensor):
        node_indices = node_indices.detach().cpu().numpy()
    if isinstance(edge_indices, torch.Tensor):
        edge_indices = edge_indices.detach().cpu().numpy()

    batch_size, num_nodes = node_indices.shape
    mols = []

    for b in range(batch_size):
        rw_mol = Chem.RWMol()
        node_to_atom_idx = {}
        valid_nodes = 0

        # 1. Add atoms
        for i in range(num_nodes):
            atom_type = int(node_indices[b, i])
            if 0 < atom_type < len(ATOMIC_NUMBERS):  # Skip PAD (0)
                atomic_num = ATOMIC_NUMBERS[atom_type]
                atom = Chem.Atom(atomic_num)
                atom_idx = rw_mol.AddAtom(atom)
                node_to_atom_idx[i] = atom_idx
                valid_nodes += 1

        if valid_nodes == 0:
            mols.append(None)
            continue

        # 2. Add bonds
        for i in range(num_nodes):
            if i not in node_to_atom_idx:
                continue
            for j in range(i + 1, num_nodes):
                if j not in node_to_atom_idx:
                    continue
                bond_type_idx = int(edge_indices[b, i, j])
                if 0 < bond_type_idx < len(BOND_TYPES):
                    rw_mol.AddBond(node_to_atom_idx[i], node_to_atom_idx[j], BOND_TYPES[bond_type_idx])

        # 3. Sanitize and validate
        mol = rw_mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
            # Ensure connected molecule of sufficient size (>= 3 heavy atoms)
            frags = Chem.GetMolFrags(mol, asMols=True)
            if len(frags) > 1:
                largest = max(frags, key=lambda m: m.GetNumAtoms())
                if largest.GetNumHeavyAtoms() < 3:
                    mols.append(None)
                    continue
                mol = largest

            if Chem.MolToSmiles(mol) == '':
                mols.append(None)
            else:
                mols.append(mol)
        except Exception:
            mols.append(None)

    return mols


def calculate_qed(mol):
    if mol is None:
        return 0.0
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def calculate_logp(mol, norm=True):
    if mol is None:
        return 0.0
    try:
        logp = Crippen.MolLogP(mol)
        if norm:
            norm_logp = (logp + 2.12178879609) / (6.0429063424 + 2.12178879609)
            return float(np.clip(norm_logp, 0.0, 1.0))
        return float(logp)
    except Exception:
        return 0.0


def calculate_sas(mol, norm=True):
    if mol is None or not SA_MODEL:
        return 0.0
    try:
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.0
        nf = 0
        for bitId, v in fps.items():
            nf += v
            score1 += SA_MODEL.get(bitId, -4.0) * v
        score1 /= max(nf, 1)

        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = sum(1 for x in ri.AtomRings() if len(x) > 8)

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = math.log10(2) if nMacrocycles > 0 else 0.0

        score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5 if nAtoms > len(fps) else 0.0

        sascore = score1 + score2 + score3
        sascore = 11.0 - (sascore - (-4.0) + 1.0) / (2.5 - (-4.0)) * 9.0
        if sascore > 8.0:
            sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
        sascore = np.clip(sascore, 1.0, 10.0)

        if norm:
            norm_sas = (5.0 - sascore) / (5.0 - 1.5)
            return float(np.clip(norm_sas, 0.0, 1.0))
        return float(sascore)
    except Exception:
        return 0.0


def calculate_np(mol, norm=True):
    if mol is None or not NP_MODEL:
        return 0.0
    try:
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        bits = fp.GetNonzeroElements()
        score = sum(NP_MODEL.get(bit, 0.0) for bit in bits) / float(max(mol.GetNumAtoms(), 1))

        if score > 4.0:
            score = 4.0 + math.log10(score - 4.0 + 1.0)
        elif score < -4.0:
            score = -4.0 - math.log10(-4.0 - score + 1.0)

        if norm:
            norm_np = (score - (-3.0)) / (1.0 - (-3.0))
            return float(np.clip(norm_np, 0.0, 1.0))
        return float(score)
    except Exception:
        return 0.0


def calculate_dc(mol, norm_logp=None, sas_norm=None, is_novel=1.0):
    if mol is None:
        return 0.0
    try:
        if norm_logp is None:
            norm_logp = calculate_logp(mol, norm=True)
        if sas_norm is None:
            sas_norm = calculate_sas(mol, norm=True)

        decay = 0.025
        if norm_logp <= 0.210:
            bump = math.exp(- (norm_logp - 0.210) ** 2 / decay)
        elif norm_logp >= 0.945:
            bump = math.exp(- (norm_logp - 0.945) ** 2 / decay)
        else:
            bump = 1.0

        novelty_term = is_novel + (1.0 - is_novel) * 0.3
        dc = (bump + sas_norm + novelty_term) / 3.0
        return float(np.clip(dc, 0.0, 1.0))
    except Exception:
        return 0.0


def calculate_batch_tanimoto_diversity(mols):
    """
    Computes pairwise Morgan fingerprint (ECFP4) Tanimoto diversity multipliers for a batch.
    Penalizes duplicate or near-identical scaffolds: max(0.0, 1.0 - max_sim^2).
    """
    B = len(mols)
    fps = []
    for mol in mols:
        if mol is not None:
            try:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.append(fp)
            except Exception:
                fps.append(None)
        else:
            fps.append(None)

    div_factors = []
    for i in range(B):
        if fps[i] is None:
            div_factors.append(1.0)
            continue

        max_sim = 0.0
        for j in range(B):
            if i != j and fps[j] is not None:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim > max_sim:
                    max_sim = sim

        factor = max(0.0, 1.0 - (max_sim ** 2))
        div_factors.append(factor)

    return div_factors


class CompositeRewardCalculator(nn.Module):
    """
    Composite Reward Function with REINVENT Mechanisms:
      1. Quantum Policy / Evaluator reward R_Q
      2. Discriminator realness probability D(G)
      3. Property objective R_prop (QED, etc.)
      4. REINVENT Scaffold Memory Filter (Bemis-Murcko memory bucket penalty)
      5. REINVENT Structural Alert & Strain Filters (Bredt rule, cyclopropenes, oxirenes)
      6. In-Batch Morgan fingerprint diversity multiplier
    """
    def __init__(self, quantum_evaluator=None, alpha=0.4, beta=0.3, gamma=0.3,
                 prop_metric='qed', scaffold_bucket_size=25, use_filter=True,
                 use_scaffold_memory=True, use_tanimoto=True):
        super(CompositeRewardCalculator, self).__init__()
        self.quantum_evaluator = quantum_evaluator
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prop_metric = prop_metric.lower()
        self.use_filter = use_filter
        self.use_scaffold_memory = use_scaffold_memory
        self.use_tanimoto = use_tanimoto

        # REINVENT Scaffold Memory Filter to eliminate mode collapse
        self.scaffold_filter = ScaffoldMemoryFilter(bucket_size=scaffold_bucket_size, penalty=0.0)

    def get_property_scores(self, mols):
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(0.0)
            elif self.prop_metric == 'qed':
                scores.append(calculate_qed(mol))
            elif self.prop_metric == 'sas':
                scores.append(calculate_sas(mol, norm=True))
            elif self.prop_metric == 'np':
                scores.append(calculate_np(mol, norm=True))
            elif self.prop_metric == 'dc':
                scores.append(calculate_dc(mol))
            elif self.prop_metric == 'logp':
                scores.append(calculate_logp(mol, norm=True))
            else:
                scores.append(calculate_qed(mol))
        return torch.tensor(scores, dtype=torch.float32).view(-1, 1)

    def forward(self, mols, discriminator_scores=None):
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        batch_size = len(mols)

        # 1. Quantum Reward R_Q in [0, 1]
        if self.quantum_evaluator is not None:
            r_quantum = self.quantum_evaluator.evaluate_mols(mols).to(device)
        else:
            r_quantum = torch.zeros(batch_size, 1, device=device)

        # 2. Discriminator Realness in [0.0, 1.0]
        if discriminator_scores is not None:
            if discriminator_scores.dim() == 1:
                discriminator_scores = discriminator_scores.unsqueeze(-1)
            prob_real = torch.sigmoid(discriminator_scores).to(device)
        else:
            prob_real = torch.zeros(batch_size, 1, device=device)

        # 3. Target Property Score R_prop in [0, 1]
        r_prop = self.get_property_scores(mols).to(device)

        # 4. In-batch Tanimoto diversity multipliers
        if self.use_tanimoto:
            tanimoto_factors = calculate_batch_tanimoto_diversity(mols)
        else:
            tanimoto_factors = [1.0] * batch_size

        # 5. Physical Stability, REINVENT Structural Alerts, and Scaffold Memory
        rewards = []
        stable_flags = []
        for i, mol in enumerate(mols):
            if mol is None:
                rewards.append(-0.5)  # Invalid graph penalty
                stable_flags.append(0.0)
                continue

            # When filters are disabled, reward pure generation based on quantum + D + prop
            if not self.use_filter:
                stable_flags.append(1.0)
                r_pos = (
                    0.05 +
                    self.alpha * r_quantum[i].item() +
                    self.beta * prob_real[i].item() +
                    self.gamma * r_prop[i].item()
                )
                rewards.append(min(1.0, max(0.0, r_pos)))
                continue

            n_heavy = mol.GetNumHeavyAtoms()
            n_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)

            # Must have at least 40% Carbons (>= 2 Carbons for small 4-5 atom molecules, >= 4 for 9 atoms)
            min_c = max(2, int(math.ceil(0.40 * n_heavy)))
            if n_heavy < 3 or n_c < min_c:
                rewards.append(-0.3)  # Low-carbon artifact penalty
                stable_flags.append(0.0)
                continue

            # Check standard unstable motifs
            has_unstable_motif = False
            for pat in UNSTABLE_SMARTS_PATTERNS:
                if pat is not None and mol.HasSubstructMatch(pat):
                    has_unstable_motif = True
                    break

            if has_unstable_motif:
                rewards.append(-0.3)  # Chemically unstable motif penalty
                stable_flags.append(0.0)
                continue

            # REINVENT Structural Alert & Bond Strain Check (Bredt's rule, strained 3-rings)
            if check_reinvent_structural_alerts(mol):
                rewards.append(-0.25)  # Strained cage/ring penalty
                stable_flags.append(0.0)
                continue

            # REINVENT Scaffold Memory Filter: checks if scaffold bucket is saturated (if enabled)
            scaffold_factor = 1.0
            if self.use_scaffold_memory:
                scaffold_factor = self.scaffold_filter.score_and_update(mol)
                if scaffold_factor <= 0.0:
                    rewards.append(0.0)  # Zero reward for over-sampled scaffold (forces exploration)
                    stable_flags.append(1.0)
                    continue

            # Molecule is genuine, stable organic compound
            stable_flags.append(1.0)
            c_frac = n_c / float(n_heavy)
            c_factor = min(1.0, c_frac / 0.5)

            # Base positive reward
            r_pos = (
                0.05 +
                self.alpha * r_quantum[i].item() +
                self.beta * prob_real[i].item() +
                self.gamma * r_prop[i].item()
            ) * c_factor

            # In-Batch Tanimoto Diversity Multiplier & Scaffold Factor
            r_pos = r_pos * tanimoto_factors[i] * scaffold_factor
            rewards.append(min(1.0, r_pos))

        total_reward = torch.tensor(rewards, dtype=torch.float32, device=device).view(-1, 1)
        stable_tensor = torch.tensor(stable_flags, dtype=torch.float32, device=device).view(-1, 1)

        breakdown = {
            'r_quantum': r_quantum,
            'r_disc': prob_real,
            'r_prop': r_prop,
            'stable_flags': stable_tensor,
            'stability_rate': stable_tensor.mean().item(),
            'validity_rate': sum(1 for m in mols if m is not None) / float(batch_size)
        }

        return total_reward, breakdown
