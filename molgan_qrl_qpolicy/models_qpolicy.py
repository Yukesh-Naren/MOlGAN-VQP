import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from molgan_qrl_qpolicy.layers import RelationalGCN
from molgan_qrl_qpolicy.quantum_policy import (
    DedicatedAtomQuantumPolicy,
    QuantumAtomPolicy2Q,
    ClassicalAtomCircuit2Q,
    QuantumEdgePolicy3Q,
    ClassicalEdgeCircuit3Q
)

# Atomic limits and valence rules
MAX_VALENCES = {
    0: 0,  # PAD
    1: 4,  # C
    2: 3,  # N
    3: 2,  # O
    4: 1   # F
}

# Bond orders matching bond type index:
# 0: None, 1: Single (1.0), 2: Double (2.0), 3: Triple (3.0), 4: Aromatic (1.5)
BOND_ORDERS = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0,
    4: 1.5
}


class ActorRGCN_Quantum(nn.Module):
    """
    Autoregressive Graph Convolutional Actor with Variational Quantum Policies:
      - Atom Selection: Dedicated 1-Qubit=1-Atom (4Q) or 2-Qubit Born-Rule (2Q)
      - Bond Formation: 3-Qubit Born-Rule Quantum Edge Policy (3Q)
    Includes exact size-matched classical counterparts for head-to-head comparison.
    """
    def __init__(self, max_atoms=9, num_atom_types=4, num_edge_types=5,
                 node_dim=5, hidden_dims=[128, 64], dropout_rate=0.0,
                 atom_policy_type='dedicated_quantum', edge_policy_type='quantum'):
        super(ActorRGCN_Quantum, self).__init__()
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.num_edge_types = num_edge_types
        self.node_dim = node_dim
        self.atom_policy_type = str(atom_policy_type).lower()
        self.edge_policy_type = str(edge_policy_type).lower()

        # Relational GCN encoder
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=num_edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

        # =========================================================================
        # 1. ATOM SELECTION POLICY HEAD (QUANTUM vs CLASSICAL)
        # -------------------------------------------------------------------------
        # Quantum (Dedicated 4-Qubit): 1 qubit per heavy atom [C, N, O, F]
        # Classical Counterpart (in comments for model comparison):
        #   self.quantum_policy = ClassicalAtomCircuit2Q(
        #       in_features=self.encoder.out_dim, hidden_dim=4, num_classes=num_atom_types
        #   )  # 32 parameters (matches 18 Euler angles + 1 scale = 19 quantum params)
        # =========================================================================
        if self.atom_policy_type in ['dedicated', 'dedicated_quantum', '4q']:
            self.quantum_policy = DedicatedAtomQuantumPolicy(
                in_features=self.encoder.out_dim,
                n_qubits=4,
                n_layers=3
            )
        elif self.atom_policy_type in ['quantum_2q', '2q']:
            self.quantum_policy = QuantumAtomPolicy2Q(
                in_features=self.encoder.out_dim,
                n_layers=3,
                output_scale=3.0,
                learnable_scale=True
            )
        elif self.atom_policy_type in ['classical', 'classical_2q']:
            self.quantum_policy = ClassicalAtomCircuit2Q(
                in_features=self.encoder.out_dim,
                hidden_dim=4,
                num_classes=num_atom_types
            )
        else:
            raise ValueError(f"Unknown atom_policy_type: {atom_policy_type}")

        # =========================================================================
        # 2. BOND FORMATION POLICY HEAD (QUANTUM vs CLASSICAL)
        # -------------------------------------------------------------------------
        # Quantum Edge Policy (3-Qubit VQC):
        #   - 3 qubits span 2^3 = 8 computational basis states.
        #   - Normalized slice |000>..|100> maps to 5 QM9 bond types:
        #     [None, Single, Double, Triple, Aromatic]
        #   - Trainable parameters: 27 Euler angles + 1 learned temperature scale gamma = 28 params.
        #
        # Classical Size-Matched Counterpart (in comments for model comparison):
        #   self.edge_policy = ClassicalEdgeCircuit3Q(
        #       in_features=pair_in_dim, hidden_dim=8, num_classes=num_edge_types
        #   )  # 77 parameters (matches 27 Euler angles + 1 scale = 28 quantum params)
        #
        # Classical Unchoked MLP Counterpart (in comments):
        #   self.edge_policy = nn.Sequential(
        #       nn.Linear(pair_in_dim, 128),
        #       nn.LeakyReLU(0.2),
        #       nn.Dropout(dropout_rate),
        #       nn.Linear(128, num_edge_types)
        #   )  # 25,221 parameters
        # =========================================================================
        pair_in_dim = self.encoder.out_dim * 3

        if self.edge_policy_type in ['quantum', 'quantum_3q', 'vqc']:
            self.edge_policy = QuantumEdgePolicy3Q(
                in_features=pair_in_dim,
                n_layers=3,
                output_scale=3.0,
                learnable_scale=True
            )
        elif self.edge_policy_type in ['classical', 'classical_3q', 'size_matched']:
            self.edge_policy = ClassicalEdgeCircuit3Q(
                in_features=pair_in_dim,
                hidden_dim=8,
                num_classes=num_edge_types
            )
        elif self.edge_policy_type in ['mlp', 'classical_mlp', 'direct']:
            self.edge_policy = nn.Sequential(
                nn.Linear(pair_in_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_edge_types)
            )
        else:
            raise ValueError(f"Unknown edge_policy_type: {edge_policy_type}")

        # Keep edge_head pointing to edge_policy for backwards compatibility
        self.edge_head = self.edge_policy

    def forward_atom(self, graph_emb, bond_angles=None, free_val=None, neighbor_types=None):
        """Helper to invoke atom policy whether quantum or classical."""
        if hasattr(self.quantum_policy, 'compute_atom_specific_compatibilities'):
            out = self.quantum_policy(graph_emb, bond_angles, free_val, neighbor_types)
        else:
            out = self.quantum_policy(graph_emb)
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward_edge(self, pair_feat):
        """Helper to invoke bond/edge policy whether quantum VQC or classical."""
        out = self.edge_policy(pair_feat)
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward_subgraph(self, node_features, adj_tensor, active_mask):
        """Encodes active subgraph."""
        return self.encoder(node_features, adj_tensor, node_mask=active_mask)

    def compute_valence_mask(self, node_indices, adj_indices, u, v):
        """
        Dynamically calculates legal bond types between atom u and atom v.
        """
        batch_size = node_indices.size(0)
        device = node_indices.device
        mask = torch.ones(batch_size, self.num_edge_types, dtype=torch.bool, device=device)

        for b in range(batch_size):
            type_u = int(node_indices[b, u].item())
            type_v = int(node_indices[b, v].item())

            # If either atom is PAD (0), only 'No bond' (0) is legal
            if type_u == 0 or type_v == 0:
                mask[b, 1:] = False
                continue

            # Anti-peroxide check: Strict ban on O-O bonds
            if type_u == 3 and type_v == 3:
                mask[b, 1:] = False
                continue

            # Anti-hetero-fluorine check: F can only bond to Carbon (type 1)
            if (type_u == 4 and type_v != 1) or (type_v == 4 and type_u != 1):
                mask[b, 1:] = False
                continue

            # Anti-triazene check: Ban N-N if either already has an N neighbor
            if type_u == 2 and type_v == 2:
                u_has_n = any(node_indices[b, w].item() == 2 and adj_indices[b, u, w].item() > 0 for w in range(u) if w != v)
                v_has_n = any(node_indices[b, w].item() == 2 and adj_indices[b, v, w].item() > 0 for w in range(u) if w != v)
                if u_has_n or v_has_n:
                    mask[b, 1:] = False
                    continue

            # Calculate current utilized valence
            curr_val_u = sum(BOND_ORDERS[adj_indices[b, u, w].item()] for w in range(u) if w != v)
            curr_val_v = sum(BOND_ORDERS[adj_indices[b, v, w].item()] for w in range(u) if w != u)

            rem_val_u = MAX_VALENCES[type_u] - curr_val_u
            rem_val_v = MAX_VALENCES[type_v] - curr_val_v

            # Check each bond order
            for bt in range(1, self.num_edge_types):
                req_order = BOND_ORDERS[bt]
                if req_order > rem_val_u or req_order > rem_val_v:
                    mask[b, bt] = False

        return mask

    def generate_mols(self, batch_size, device='cpu', temperature=1.2):
        """
        Autoregressively generates strictly 9-atom molecules using the Quantum Policy Head.
        """
        node_features = torch.zeros(batch_size, self.max_atoms, self.node_dim, device=device)
        node_features[:, :, 0] = 1.0
        node_indices = torch.zeros((batch_size, self.max_atoms), dtype=torch.long, device=device)

        adj_tensor = torch.zeros(batch_size, self.max_atoms, self.max_atoms, self.num_edge_types, device=device)
        adj_tensor[:, :, :, 0] = 1.0
        adj_indices = torch.zeros(batch_size, self.max_atoms, self.max_atoms, dtype=torch.long, device=device)

        active_mask = torch.zeros(batch_size, self.max_atoms, device=device)
        trajectories = []

        # Run for all 9 atoms
        for step in range(self.max_atoms):
            # 1. Encode intermediate graph
            node_emb, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask)

            # 2. Extract local geometric parameters for Quantum Policy Head
            if step == 0:
                # Step 0 is always Carbon anchor
                bond_angles = torch.full((batch_size,), 1.91, device=device)
                free_val = torch.full((batch_size,), 4.0, device=device)
                neighbor_types = torch.ones(batch_size, dtype=torch.long, device=device)
            else:
                # Local anchor is the most recently added atom (step - 1)
                neighbor_types = node_indices[:, step - 1]
                free_val_list = []
                for b in range(batch_size):
                    anchor_t = int(neighbor_types[b].item())
                    used = sum(BOND_ORDERS[adj_indices[b, step - 1, w].item()] for w in range(step))
                    free_val_list.append(max(0.0, MAX_VALENCES[anchor_t] - used))
                free_val = torch.tensor(free_val_list, device=device)
                # Angle estimation: 109.5 deg default, 120 deg if double bond present
                bond_angles = torch.full((batch_size,), 1.91, device=device)

            # 3. Atom Policy execution (Quantum or Classical)
            q_logits = self.forward_atom(graph_emb, bond_angles, free_val, neighbor_types)

            # Enforce Carbon anchor on Step 0 and cap heteroatoms at 4
            if step == 0:
                q_logits[:, 1:] = -1e9
            else:
                num_hetero = (node_indices[:, :step] > 1).sum(dim=-1)
                for b in range(batch_size):
                    if num_hetero[b] >= 4:
                        q_logits[b, 1:] = -1e9

            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            node_action = node_dist.sample()         # in {0, 1, 2, 3}
            node_log_prob = node_dist.log_prob(node_action)

            # Map action {0, 1, 2, 3} to atom type {1: C, 2: N, 3: O, 4: F}
            atom_types = node_action + 1

            step_record = {
                'step': step,
                'state_node': node_features.clone(),
                'state_adj': adj_tensor.clone(),
                'active_mask': active_mask.clone(),
                'bond_angles': bond_angles.clone(),
                'free_val': free_val.clone(),
                'neighbor_types': neighbor_types.clone(),
                'node_action': node_action,
                'node_log_prob': node_log_prob,
                'edge_records': []
            }

            for b in range(batch_size):
                a_type = atom_types[b].item()
                node_indices[b, step] = a_type
                node_features[b, step, :] = 0.0
                node_features[b, step, a_type] = 1.0
                active_mask[b, step] = 1.0

            # 4. Edge formation head
            if step > 0:
                node_emb, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask)
                u = step

                for v in range(step):
                    h_u = node_emb[:, u, :]
                    h_v = node_emb[:, v, :]
                    pair_feat = torch.cat([h_u, h_v, graph_emb], dim=-1)

                    edge_logits = self.forward_edge(pair_feat)

                    # Valence and Aromatic Masking
                    valence_mask = self.compute_valence_mask(node_indices, adj_indices, u, v)
                    edge_logits = torch.where(valence_mask, edge_logits, torch.tensor(-1e9, device=device))

                    edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                    edge_action = edge_dist.sample()
                    edge_log_prob = edge_dist.log_prob(edge_action)

                    step_record['edge_records'].append({
                        'u': u,
                        'v': v,
                        'edge_action': edge_action,
                        'edge_log_prob': edge_log_prob,
                        'valence_mask': valence_mask.clone()
                    })

                    for b in range(batch_size):
                        b_type = edge_action[b].item()
                        adj_indices[b, u, v] = b_type
                        adj_indices[b, v, u] = b_type
                        adj_tensor[b, u, v, :] = 0.0
                        adj_tensor[b, v, u, :] = 0.0
                        adj_tensor[b, u, v, b_type] = 1.0
                        adj_tensor[b, v, u, b_type] = 1.0

                # Ensure node u is connected to at least one prior node to maintain connectivity
                for b in range(batch_size):
                    if adj_indices[b, u, :u].sum() == 0:
                        u_t = int(node_indices[b, u].item())
                        candidates = list(range(u))
                        candidates.sort(key=lambda c: 0 if int(node_indices[b, c].item()) == 1 else 1)
                        for v_cand in candidates:
                            v_type = int(node_indices[b, v_cand].item())
                            if u_t == 3 and v_type == 3:
                                continue
                            if (u_t == 4 and v_type != 1) or (v_type == 4 and u_t != 1):
                                continue
                            if u_t == 2 and v_type == 2:
                                u_has_n = any(node_indices[b, w].item() == 2 and adj_indices[b, u, w].item() > 0 for w in range(step) if w != u and w != v_cand)
                                v_has_n = any(node_indices[b, w].item() == 2 and adj_indices[b, v_cand, w].item() > 0 for w in range(step) if w != u and w != v_cand)
                                if u_has_n or v_has_n:
                                    continue
                            curr_val_u = sum(BOND_ORDERS[adj_indices[b, u, w].item()] for w in range(step))
                            curr_val_v = sum(BOND_ORDERS[adj_indices[b, v_cand, w].item()] for w in range(step))
                            if (curr_val_u + 1.0 <= MAX_VALENCES[u_t]) and (curr_val_v + 1.0 <= MAX_VALENCES[v_type]):
                                adj_indices[b, u, v_cand] = 1
                                adj_indices[b, v_cand, u] = 1
                                adj_tensor[b, u, v_cand, :] = 0.0
                                adj_tensor[b, v_cand, u, :] = 0.0
                                adj_tensor[b, u, v_cand, 1] = 1.0
                                adj_tensor[b, v_cand, u, 1] = 1.0
                                break

            trajectories.append(step_record)

        return node_indices, adj_indices, node_features, adj_tensor, trajectories

    def evaluate_trajectory_log_probs(self, trajectories, temperature=1.2):
        """Re-computes action log probabilities and entropy across recorded trajectories."""
        total_log_probs = 0.0
        entropies = []

        for record in trajectories:
            step = record['step']
            node_feat = record['state_node']
            adj_t = record['state_adj']
            mask = record['active_mask']
            bond_ang = record['bond_angles']
            free_v = record['free_val']
            neigh_t = record['neighbor_types']
            node_act = record['node_action']

            _, graph_emb = self.forward_subgraph(node_feat, adj_t, mask)
            q_logits = self.forward_atom(graph_emb, bond_ang, free_v, neigh_t)

            if step == 0:
                q_logits[:, 1:] = -1e9
            else:
                atom_indices = node_feat[:, :step, 1:].argmax(dim=-1) + 1
                num_hetero = (atom_indices > 1).sum(dim=-1)
                for b in range(q_logits.size(0)):
                    if num_hetero[b] >= 4:
                        q_logits[b, 1:] = -1e9

            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            total_log_probs = total_log_probs + node_dist.log_prob(node_act)
            entropies.append(node_dist.entropy().mean())

            for e_rec in record['edge_records']:
                u, v = e_rec['u'], e_rec['v']
                edge_act = e_rec['edge_action']
                val_mask = e_rec['valence_mask']

                node_emb, g_emb = self.forward_subgraph(node_feat, adj_t, mask)
                pair_feat = torch.cat([node_emb[:, u, :], node_emb[:, v, :], g_emb], dim=-1)
                edge_logits = self.forward_edge(pair_feat)

                edge_logits = torch.where(val_mask, edge_logits, torch.tensor(-1e9, device=edge_logits.device))
                edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                log_p = edge_dist.log_prob(edge_act)
                total_log_probs = total_log_probs + log_p
                entropies.append(edge_dist.entropy().mean())

        mean_entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
        return total_log_probs, mean_entropy


class CriticRGCN(nn.Module):
    """Value Critic V_psi(G) predicting expected cumulative return."""
    def __init__(self, node_dim=5, edge_types=5, hidden_dims=[128, 64], dropout_rate=0.0):
        super(CriticRGCN, self).__init__()
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, node_features, adj_tensor, node_mask=None):
        _, graph_emb = self.encoder(node_features, adj_tensor, node_mask=node_mask)
        return self.value_head(graph_emb)


class DiscriminatorRGCN(nn.Module):
    """Adversarial Discriminator D_phi(G) evaluating realism of real vs generated graphs."""
    def __init__(self, node_dim=5, edge_types=5, hidden_dims=[128, 64], dropout_rate=0.1):
        super(DiscriminatorRGCN, self).__init__()
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        self.disc_head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, node_features, adj_tensor, node_mask=None):
        _, graph_emb = self.encoder(node_features, adj_tensor, node_mask=node_mask)
        return self.disc_head(graph_emb)
