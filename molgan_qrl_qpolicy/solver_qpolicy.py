import os
import time
import math
import pickle
import csv
import logging
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Draw

from molgan_qrl_qpolicy.circuit import QuantumPhysChemEvaluator
from molgan_qrl_qpolicy.reward import (
    matrices_to_mols,
    calculate_qed,
    calculate_sas,
    calculate_np,
    calculate_dc,
    calculate_logp,
    CompositeRewardCalculator
)
from molgan_qrl_qpolicy.models_qpolicy import ActorRGCN_Quantum, CriticRGCN, DiscriminatorRGCN
from molgan_qrl_qpolicy.ppo_buffer import PPOBuffer


class QM9Dataset(Dataset):
    """
    Lightweight PyTorch Dataset loader for QM9 sparse dataset.
    Converts integer node/edge labels into one-hot tensors:
      - Nodes: [N, 5] (PAD:0, C:1, N:2, O:3, F:4)
      - Adjacency: [N, N, 5] (None:0, Single:1, Double:2, Triple:3, Aromatic:4)
    """
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        self.data_A = data['data_A']  # [M, 9, 9] integer indices
        self.data_X = data['data_X']  # [M, 9] integer indices
        self.labels = data.get('labels', None)
        if self.labels is None:
            self.labels = np.ones(len(self.data_X), dtype=np.float32)

        self.smiles = set(data['smiles']) if 'smiles' in data else set()
        self.num_samples = len(self.data_X)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data_X[idx]
        a = self.data_A[idx]
        lbl = self.labels[idx]

        x_t = torch.tensor(x, dtype=torch.long)
        a_t = torch.tensor(a, dtype=torch.long)
        lbl_t = torch.tensor(lbl, dtype=torch.float32)

        x_one_hot = F.one_hot(x_t, num_classes=5).float()
        a_one_hot = F.one_hot(a_t, num_classes=5).float()

        return x_one_hot, a_one_hot, lbl_t


class SolverQPolicy:
    """
    Training Orchestrator for MolGAN-QRL with Dedicated 1-Qubit = 1-Atom Policy Head.
    Saves timestamped run directories:
      Results/qpolicy_quantum_GAN/<timestamp>/
        ├── log_dir/    (train.log, metrics.csv)
        ├── model_dir/  (checkpoints)
        └── img_dir/    (epoch_*.png)
    """
    def __init__(self,
                 dataset_path='data/qm9_with_negatives.sparsedataset',
                 save_dir='Results/qpolicy_quantum_GAN',
                 batch_size=16,
                 epochs=50,
                 ppo_epochs=4,
                 clip_eps=0.2,
                 entropy_coef=0.10,
                 temperature=1.2,
                 alpha=0.4,
                 beta=0.3,
                 gamma=0.3,
                 prop_metric='qed',
                 scaffold_bucket_size=25,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 lr_disc=1e-4,
                 atom_policy_type='dedicated_quantum',
                 edge_policy_type='quantum',
                 use_valence_mask=True,
                 use_filter=True,
                 use_scaffold_memory=True,
                 use_tanimoto=True,
                 enforce_connectivity=True,
                 use_atom_filters=True,
                 min_atoms=4,
                 max_atoms=9,
                 device=None):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.atom_policy_type = atom_policy_type
        self.edge_policy_type = edge_policy_type
        self.use_valence_mask = use_valence_mask
        self.use_filter = use_filter
        self.use_scaffold_memory = use_scaffold_memory
        self.use_tanimoto = use_tanimoto
        self.enforce_connectivity = enforce_connectivity
        self.use_atom_filters = use_atom_filters
        self.batch_size = batch_size
        self.epochs = epochs
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        self.save_dir = save_dir

        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = os.path.join(save_dir, timestamp)
        self.log_dir = os.path.join(self.run_dir, 'log_dir')
        self.model_dir = os.path.join(self.run_dir, 'model_dir')
        self.img_dir = os.path.join(self.run_dir, 'img_dir')

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        self.log_file = os.path.join(self.log_dir, 'train.log')
        self.csv_file = os.path.join(self.log_dir, 'metrics.csv')
        self._init_logger()
        self._init_csv()

        self.logger.info(f"Initialized Dedicated Q-Policy run at: {self.run_dir}")

        # 1. Dataset & DataLoader
        self.logger.info(f"Loading QM9 dataset from {dataset_path}...")
        self.dataset = QM9Dataset(dataset_path)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.dataset_smiles = self.dataset.smiles
        self.logger.info(f"Dataset loaded: {len(self.dataset)} molecules.")

        # 2. Quantum Evaluator & Composite Reward
        self.logger.info("Initializing 4-Qubit Variational Quantum Property Evaluator...")
        self.quantum_evaluator = QuantumPhysChemEvaluator(n_qubits=4, n_layers=3).to(self.device)
        self.reward_calculator = CompositeRewardCalculator(
            quantum_evaluator=self.quantum_evaluator,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            prop_metric=prop_metric,
            scaffold_bucket_size=scaffold_bucket_size,
            use_filter=self.use_filter,
            use_scaffold_memory=self.use_scaffold_memory,
            use_tanimoto=self.use_tanimoto
        ).to(self.device)

        # 3. Networks: Actor with Q-Policy Head (Atom + Edge), Critic, Discriminator
        self.logger.info(f"Initializing Actor (Atom: {self.atom_policy_type}, Edge: {self.edge_policy_type}, Masking: {self.use_valence_mask}, Filters: {self.use_filter}), Critic, Discriminator...")
        self.actor = ActorRGCN_Quantum(
            max_atoms=self.max_atoms,
            min_atoms=self.min_atoms,
            num_atom_types=4,
            num_edge_types=5,
            atom_policy_type=self.atom_policy_type,
            edge_policy_type=self.edge_policy_type,
            use_valence_mask=self.use_valence_mask,
            enforce_connectivity=self.enforce_connectivity,
            use_atom_filters=self.use_atom_filters
        ).to(self.device)
        self.critic = CriticRGCN(node_dim=5, edge_types=5).to(self.device)
        self.discriminator = DiscriminatorRGCN(node_dim=5, edge_types=5).to(self.device)

        # 4. Rollout Buffer
        self.buffer = PPOBuffer(clip_eps=clip_eps)

        # 5. Optimizers
        self.optimizer_A = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_C = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)

        self.logger.info(f"Dedicated Q-Policy Solver initialized on device: {self.device}")

    def _init_logger(self):
        self.logger = logging.getLogger(f"MolGAN_QPolicy_{time.time()}")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.log_file)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _init_csv(self):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'time_sec', 'loss_d', 'loss_pi', 'loss_v', 'r_avg',
                'valid_rate', 'stable_rate', 'unique_rate', 'novel_rate', 'qed', 'logp', 'sas', 'np', 'dc'
            ])

    def train_epoch(self, epoch):
        self.actor.eval()
        self.critic.eval()
        self.discriminator.train()

        total_loss_d = 0.0
        total_loss_pi = 0.0
        total_loss_v = 0.0
        total_rewards = []
        all_generated_mols = []

        data_iter = iter(self.dataloader)

        for step in range(len(self.dataloader)):
            try:
                batch_x, batch_adj, batch_labels = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch_x, batch_adj, batch_labels = next(data_iter)

            batch_x = batch_x.to(self.device)
            batch_adj = batch_adj.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Phase 1: Environment Rollout with Dedicated 1-Qubit = 1-Atom Policy
            with torch.no_grad():
                nodes, edges, node_feat, adj_t, trajs = self.actor.generate_mols(
                    batch_size=self.batch_size, device=self.device, temperature=self.temperature,
                    min_atoms=self.min_atoms, max_atoms=self.max_atoms
                )
                old_log_probs, _ = self.actor.evaluate_trajectory_log_probs(trajs, temperature=self.temperature)
                values = self.critic(node_feat, adj_t)

                mols = matrices_to_mols(nodes, edges)
                all_generated_mols.extend(mols)

                disc_scores = self.discriminator(node_feat, adj_t)
                rewards, breakdown = self.reward_calculator(mols, disc_scores)

            total_rewards.append(rewards.mean().item())
            self.buffer.store(trajs, old_log_probs, rewards, values)

            # Phase 2: Discriminator Update
            self.optimizer_D.zero_grad()
            dataset_logits = self.discriminator(batch_x, batch_adj).view(-1)
            fake_logits = self.discriminator(node_feat.detach(), adj_t.detach()).view(-1)

            target_dataset = torch.where(batch_labels > 0.5, torch.full_like(dataset_logits, 0.9), torch.zeros_like(dataset_logits))
            loss_d_dataset = F.binary_cross_entropy_with_logits(dataset_logits, target_dataset)
            loss_d_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))

            loss_d = loss_d_dataset + loss_d_fake
            loss_d.backward()
            self.optimizer_D.step()
            total_loss_d += loss_d.item()

            # Phase 3: PPO Optimization for Quantum Actor & Classical Critic
            self.actor.train()
            self.critic.train()
            buf_data = self.buffer.get_data()

            for _ in range(self.ppo_epochs):
                self.optimizer_A.zero_grad()
                new_log_probs, entropy = self.actor.evaluate_trajectory_log_probs(buf_data['trajectories'], temperature=self.temperature)
                log_ratio = torch.clamp(new_log_probs - buf_data['old_log_probs'], -5.0, 5.0)
                ratio = torch.exp(log_ratio)
                advantages = buf_data['advantages']

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                loss_pi = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                loss_pi.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.optimizer_A.step()

                self.optimizer_C.zero_grad()
                curr_values = self.critic(node_feat.detach(), adj_t.detach())
                loss_v = F.mse_loss(curr_values, buf_data['returns'])

                loss_v.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.optimizer_C.step()

            total_loss_pi += loss_pi.item()
            total_loss_v += loss_v.item()
            self.buffer.clear()

            if (step + 1) % 10 == 0 or step == len(self.dataloader) - 1:
                self.logger.info(
                    f"  [Epoch {epoch:02d} | Step {step+1:04d}/{len(self.dataloader)}] "
                    f"Loss_D: {loss_d.item():.4f} | Loss_Pi: {loss_pi.item():.4f} | "
                    f"Loss_V: {loss_v.item():.4f} | Batch_R: {rewards.mean().item():.4f}"
                )

        n_steps = len(self.dataloader)
        avg_loss_d = total_loss_d / n_steps
        avg_loss_pi = total_loss_pi / n_steps
        avg_loss_v = total_loss_v / n_steps
        avg_reward = np.mean(total_rewards)

        metrics = self.evaluate_molecules(all_generated_mols)
        metrics.update({
            'loss_d': avg_loss_d,
            'loss_pi': avg_loss_pi,
            'loss_v': avg_loss_v,
            'avg_reward': avg_reward
        })

        valid_sample = [m for m in all_generated_mols if m is not None][:16]
        if valid_sample:
            try:
                legends = [f"{Chem.MolToSmiles(m)}\nQED: {calculate_qed(m):.2f}" for m in valid_sample]
                img = Draw.MolsToGridImage(valid_sample, molsPerRow=4, subImgSize=(250, 250), legends=legends)
                img.save(os.path.join(self.img_dir, f"epoch_{epoch}.png"))
            except Exception:
                pass

        return metrics

    def evaluate_molecules(self, mols):
        valid_mols = [m for m in mols if m is not None]
        valid_count = len(valid_mols)
        total_count = len(mols)

        valid_rate = (valid_count / total_count * 100.0) if total_count > 0 else 0.0

        if valid_count > 0:
            smiles_list = [Chem.MolToSmiles(m) for m in valid_mols]
            unique_smiles = set(smiles_list)
            unique_rate = len(unique_smiles) / valid_count * 100.0

            novel_count = sum(1 for s in unique_smiles if s not in self.dataset_smiles)
            novel_rate = novel_count / len(unique_smiles) * 100.0

            from molgan_qrl_qpolicy.reward import UNSTABLE_SMARTS_PATTERNS
            stable_count = sum(
                1 for m in valid_mols
                if sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6) >= max(2, int(math.ceil(0.40 * m.GetNumHeavyAtoms())))
                and not any(pat and m.HasSubstructMatch(pat) for pat in UNSTABLE_SMARTS_PATTERNS)
            )
            stable_rate = (stable_count / total_count * 100.0) if total_count > 0 else 0.0

            qeds = [calculate_qed(m) for m in valid_mols]
            logps = [calculate_logp(m, norm=True) for m in valid_mols]
            sass = [calculate_sas(m, norm=True) for m in valid_mols]
            nps = [calculate_np(m, norm=True) for m in valid_mols]
            dcs = [calculate_dc(m) for m in valid_mols]

            mean_qed = float(np.mean(qeds))
            mean_logp = float(np.mean(logps))
            mean_sas = float(np.mean(sass))
            mean_np = float(np.mean(nps))
            mean_dc = float(np.mean(dcs))
        else:
            unique_rate = 0.0
            novel_rate = 0.0
            stable_rate = 0.0
            mean_qed = 0.0
            mean_logp = 0.0
            mean_sas = 0.0
            mean_np = 0.0
            mean_dc = 0.0

        return {
            'valid_rate': valid_rate,
            'stable_rate': stable_rate,
            'unique_rate': unique_rate,
            'novel_rate': novel_rate,
            'qed': mean_qed,
            'logp': mean_logp,
            'sas': mean_sas,
            'np': mean_np,
            'dc': mean_dc
        }

    def train(self):
        self.logger.info(f"Starting Dedicated 1-Qubit=1-Atom Q-Policy Training for {self.epochs} epochs...")
        start_time = time.time()
        best_valid_rate = -1.0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(epoch)
            elapsed = time.time() - t0

            # Write metrics row to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, round(elapsed, 2),
                    round(metrics['loss_d'], 4),
                    round(metrics['loss_pi'], 4),
                    round(metrics['loss_v'], 4),
                    round(metrics['avg_reward'], 4),
                    round(metrics['valid_rate'], 2),
                    round(metrics['stable_rate'], 2),
                    round(metrics['unique_rate'], 2),
                    round(metrics['novel_rate'], 2),
                    round(metrics['qed'], 3),
                    round(metrics['logp'], 3),
                    round(metrics['sas'], 3),
                    round(metrics['np'], 3),
                    round(metrics['dc'], 3)
                ])

            self.logger.info(
                f"[Epoch {epoch:03d}/{self.epochs:03d}] ({elapsed:.1f}s) | "
                f"Loss_D: {metrics['loss_d']:.4f} | Loss_Pi: {metrics['loss_pi']:.4f} | Loss_V: {metrics['loss_v']:.4f} | "
                f"R_avg: {metrics['avg_reward']:.4f} | Valid: {metrics['valid_rate']:.1f}% | Stable: {metrics['stable_rate']:.1f}% | "
                f"Unique: {metrics['unique_rate']:.1f}% | Novel: {metrics['novel_rate']:.1f}% | "
                f"QED: {metrics['qed']:.3f} | SAS: {metrics['sas']:.3f} | DC: {metrics['dc']:.3f}"
            )

            # Save best checkpoint
            if metrics['valid_rate'] > best_valid_rate:
                best_valid_rate = metrics['valid_rate']
                torch.save(self.actor.state_dict(), os.path.join(self.model_dir, 'best_actor.pt'))
                torch.save(self.critic.state_dict(), os.path.join(self.model_dir, 'best_critic.pt'))
                torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, 'best_discriminator.pt'))
                self.logger.info(f"Saved checkpoint (best) at epoch {epoch} to {self.model_dir}")

            if epoch % 5 == 0 or epoch == self.epochs:
                torch.save(self.actor.state_dict(), os.path.join(self.model_dir, f'epoch_{epoch}.pt'))
                self.logger.info(f"Saved checkpoint (epoch_{epoch}) at epoch {epoch} to {self.model_dir}")

        total_min = (time.time() - start_time) / 60.0
        self.logger.info(f"Dedicated Q-Policy Training completed in {total_min:.2f} minutes. Best Validity: {best_valid_rate:.1f}%.")
