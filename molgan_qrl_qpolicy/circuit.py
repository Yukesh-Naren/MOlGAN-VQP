import torch
import torch.nn as nn
import pennylane as qml
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


class QuantumPhysChemEvaluator(nn.Module):
    """
    Quantum Physicochemical Reward Circuit for MolGAN-QRL.
    Encodes 4 key Lipinski drug-likeness descriptors into 4 qubits calibrated to the QM9 chemical space:
      - Qubit 0: Molecular Weight (MW)
      - Qubit 1: LogP (Lipophilicity)
      - Qubit 2: Hydrogen Bond Donors (HBD)
      - Qubit 3: Hydrogen Bond Acceptors (HBA)

    Architecture:
      - Feature Encoding: RY(x_i)
      - Data Re-Uploading: Features re-encoded at each layer
      - Ansatz Depth: L = 3 layers of parameterized Rot(alpha, beta, gamma)
      - Entanglement: Circular CNOT ring
      - Readout: Pauli-Z expectation value on Qubit 0 normalized to [0, 1]
    """
    def __init__(self, n_qubits=4, n_layers=3):
        super(QuantumPhysChemEvaluator, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Variational weights: [n_layers, n_qubits, 3] for Rot(alpha, beta, gamma)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        def _circuit(features, weights):
            for l in range(self.n_layers):
                # 1. Feature encoding with RY rotations
                for i in range(self.n_qubits):
                    qml.RY(features[i], wires=i)

                # 2. Parameterized Euler rotations Rot(alpha, beta, gamma)
                for i in range(self.n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

                # 3. Circular CNOT entangling ring
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            return qml.expval(qml.PauliZ(0))

        self.qnode = qml.QNode(_circuit, self.dev, interface="torch", diff_method="backprop")

    def extract_physchem_features(self, mol):
        """
        Extracts and scales 4 Lipinski descriptors to rotation angles in [0, pi]
        calibrated specifically to the QM9 drug-like chemical domain.
        """
        if mol is None:
            return torch.zeros(self.n_qubits, device=self.weights.device)

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        pi = 3.141592653589793
        # Scaled to QM9 chemical distributions (MW: 40-150, LogP: -2.0 to 3.0)
        theta_mw = pi * max(0.0, min((mw - 40.0) / 110.0, 1.0))
        theta_logp = pi * max(0.0, min((logp + 2.0) / 5.0, 1.0))
        theta_hbd = pi * max(0.0, min(hbd / 4.0, 1.0))
        theta_hba = pi * max(0.0, min(hba / 5.0, 1.0))

        return torch.tensor([theta_mw, theta_logp, theta_hbd, theta_hba],
                            dtype=torch.float32,
                            device=self.weights.device)

    def forward_single(self, features):
        """
        Executes quantum circuit for a single 4-dimensional feature vector.
        Normalizes Pauli-Z expectation value from [-1, 1] to [0, 1].
        """
        raw_val = self.qnode(features, self.weights)
        norm_val = (raw_val + 1.0) / 2.0
        return torch.clamp(norm_val, 0.0, 1.0)

    def evaluate_mols(self, mols):
        """
        Batch evaluation of a list of RDKit molecules.
        Molecules that are None, trivial fragments (< 3 heavy atoms),
        or lack any Carbon atoms receive a score of 0.0.
        """
        scores = []
        for mol in mols:
            if mol is None:
                scores.append(torch.tensor([0.0], device=self.weights.device))
                continue

            num_heavy = mol.GetNumHeavyAtoms()
            num_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)

            # Require at least 3 heavy atoms and at least 1 Carbon to qualify for drug rewards
            if num_heavy < 3 or num_c == 0:
                scores.append(torch.tensor([0.0], device=self.weights.device))
                continue

            feats = self.extract_physchem_features(mol)
            r_q = self.forward_single(feats)
            scores.append(r_q.view(1))

        return torch.stack(scores).view(-1, 1)
