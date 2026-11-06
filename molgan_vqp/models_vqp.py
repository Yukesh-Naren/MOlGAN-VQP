import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from molgan_vqp.layers import RelationalGCN
from molgan_vqp.policy_qc import JerbiAtomPolicy, JerbiEdgePolicy

# Atomic limits and valence rules
MAX_VALENCES = {
    0: 0,  # PAD
    1: 4,  # C
    2: 3,  # N
    3: 2,  # O
    4: 1   # F
}

# Bond orders matching bond type index (Kekule representation - no aromatic bonds):
# 0: None, 1: Single (1.0), 2: Double (2.0), 3: Triple (3.0)
BOND_ORDERS = {
    0: 0.0,
    1: 1.0,
    2: 2.0,
    3: 3.0
}


class ActorRGCN_Jerbi(nn.Module):
    """
    Autoregressive Graph Convolutional Actor with Jerbi et al. (NeurIPS 2021)
    Variational Quantum Policies (Softmax-PQC with Multi-Layer Data Re-Uploading):

      - Atom Policy: 4-Qubit Jerbi Softmax-PQC over heavy atoms {C, N, O, F}
        Data re-uploading layers + CNOT entanglement + Pauli-Z expectation values + Softmax
      - Bond Policy: 4-Qubit Jerbi Softmax-PQC over Kekule bonds {None, Single, Double, Triple}
        Data re-uploading layers + CNOT entanglement + Pauli-Z expectation values + Softmax
      - Valence and Chemical Masking ensuring 100% chemically valid molecules
    """
    def __init__(self, max_atoms=9, min_atoms=4, num_atom_types=4, num_edge_types=4,
                 node_dim=5, hidden_dims=[128, 64], dropout_rate=0.0,
                 n_layers_atom=3, n_layers_edge=3, output_scale=3.0,
                 use_valence_mask=True, enforce_connectivity=True, use_atom_filters=True):
        super(ActorRGCN_Jerbi, self).__init__()
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.num_atom_types = num_atom_types
        self.num_edge_types = num_edge_types
        self.node_dim = node_dim
        self.use_valence_mask = use_valence_mask
        self.enforce_connectivity = enforce_connectivity
        self.use_atom_filters = use_atom_filters

        # 1. Relational Graph Convolutional Encoder for Subgraphs
        self.encoder = RelationalGCN(
            node_dim=node_dim,
            edge_types=num_edge_types,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )

        # 2. Jerbi Data Re-Uploading Softmax-PQC for Atom Selection (4 qubits -> C, N, O, F)
        self.atom_policy = JerbiAtomPolicy(
            in_features=self.encoder.out_dim,
            n_layers=n_layers_atom,
            output_scale=output_scale,
            learnable_scale=True
        )

        # 3. Jerbi Data Re-Uploading Softmax-PQC for Edge Selection (4 qubits -> None, Single, Double, Triple)
        pair_in_dim = self.encoder.out_dim * 3
        self.edge_policy = JerbiEdgePolicy(
            in_features=pair_in_dim,
            n_layers=n_layers_edge,
            output_scale=output_scale,
            learnable_scale=True
        )

    def forward_atom(self, graph_emb):
        """Invokes Jerbi Atom Policy."""
        logits, _ = self.atom_policy(graph_emb)
        return logits

    def forward_edge(self, pair_feat):
        """Invokes Jerbi Bond Policy."""
        logits, _ = self.edge_policy(pair_feat)
        return logits

    def forward_subgraph(self, node_features, adj_tensor, active_mask):
        """Encodes active subgraph."""
        return self.encoder(node_features, adj_tensor, node_mask=active_mask)

    def compute_valence_mask(self, node_indices, adj_indices, u, v):
        """
        Dynamically calculates legal bond types between atom u and atom v.
        """
        batch_size = node_indices.size(0)
        mask = torch.ones(batch_size, self.num_edge_types, dtype=torch.bool, device=node_indices.device)

        for b in range(batch_size):
            type_u = int(node_indices[b, u].item())
            type_v = int(node_indices[b, v].item())

            if type_u == 0 or type_v == 0:
                mask[b, 1:] = False
                continue

            # Anti-peroxide check: Ban O-O single/double/triple bonds
            if type_u == 3 and type_v == 3:
                mask[b, 1:] = False
                continue

            # Fluorine termination check: F can only form 1 single bond to C
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

    def generate_mols(self, batch_size, device=None, temperature=1.2, min_atoms=None, max_atoms=None, target_sizes=None):
        """
        Autoregressively generates molecules using Jerbi Softmax-PQCs with dynamic atom sizes (e.g. 4 to 9).
        """
        if device is None:
            device = next(self.parameters()).device

        if min_atoms is None:
            min_atoms = getattr(self, 'min_atoms', 4)
        if max_atoms is None:
            max_atoms = self.max_atoms

        if target_sizes is None:
            if min_atoms == max_atoms:
                target_sizes = torch.full((batch_size,), max_atoms, dtype=torch.long, device=device)
            else:
                target_sizes = torch.randint(min_atoms, max_atoms + 1, (batch_size,), dtype=torch.long, device=device)

        # Buffers for generated graphs
        node_indices = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=device)
        adj_indices = torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.long, device=device)

        node_features = torch.zeros(batch_size, max_atoms, self.node_dim, device=device)
        node_features[:, :, 0] = 1.0  # Default PAD one-hot

        adj_tensor = torch.zeros(batch_size, max_atoms, max_atoms, self.num_edge_types, device=device)
        adj_tensor[:, :, :, 0] = 1.0  # Default None edge one-hot

        active_mask = torch.zeros(batch_size, max_atoms, device=device)
        trajectories = []

        for step in range(max_atoms):
            active_step_mask = (step < target_sizes)

            if not active_step_mask.any():
                break

            # 1. Graph representation of current active subgraph
            if step == 0:
                graph_emb = torch.zeros(batch_size, self.encoder.out_dim, device=device)
            else:
                _, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask)

            # 2. Jerbi Atom Policy execution
            q_logits = self.forward_atom(graph_emb)

            # Enforce Carbon anchor on Step 0 and cap heteroatoms at 4 (if atom filters enabled)
            if self.use_atom_filters:
                if step == 0:
                    q_logits[:, 1:] = -1e9
                else:
                    num_hetero = (node_indices[:, :step] > 1).sum(dim=-1)
                    for b in range(batch_size):
                        if active_step_mask[b] and num_hetero[b] >= 4:
                            q_logits[b, 1:] = -1e9

            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            node_action = node_dist.sample()         # in {0, 1, 2, 3}
            node_log_prob = node_dist.log_prob(node_action)
            atom_types = node_action + 1

            step_record = {
                'step': step,
                'state_node': node_features.clone(),
                'state_adj': adj_tensor.clone(),
                'active_mask': active_mask.clone(),
                'node_action': node_action,
                'node_log_prob': node_log_prob,
                'active_mask_step': active_step_mask.clone(),
                'edge_records': []
            }

            for b in range(batch_size):
                if active_step_mask[b]:
                    a_type = atom_types[b].item()
                    node_indices[b, step] = a_type
                    node_features[b, step, :] = 0.0
                    node_features[b, step, a_type] = 1.0
                    active_mask[b, step] = 1.0
                else:
                    node_indices[b, step] = 0
                    node_features[b, step, :] = 0.0
                    node_features[b, step, 0] = 1.0
                    active_mask[b, step] = 0.0

            # 3. Jerbi Edge formation head
            if step > 0:
                node_emb, graph_emb = self.forward_subgraph(node_features, adj_tensor, active_mask)
                u = step

                for v in range(step):
                    h_u = node_emb[:, u, :]
                    h_v = node_emb[:, v, :]
                    pair_feat = torch.cat([h_u, h_v, graph_emb], dim=-1)

                    edge_logits = self.forward_edge(pair_feat)

                    # Valence and Chemical Masking (if enabled)
                    if self.use_valence_mask:
                        valence_mask = self.compute_valence_mask(node_indices, adj_indices, u, v)
                        edge_logits = torch.where(valence_mask, edge_logits, torch.tensor(-1e9, device=device))
                    else:
                        valence_mask = None

                    edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                    edge_action = edge_dist.sample()
                    edge_log_prob = edge_dist.log_prob(edge_action)

                    step_record['edge_records'].append({
                        'u': u,
                        'v': v,
                        'edge_action': edge_action,
                        'edge_log_prob': edge_log_prob,
                        'valence_mask': valence_mask.clone() if valence_mask is not None else None,
                        'active_mask_step': active_step_mask.clone()
                    })

                    for b in range(batch_size):
                        if active_step_mask[b]:
                            b_type = edge_action[b].item()
                            adj_indices[b, u, v] = b_type
                            adj_indices[b, v, u] = b_type
                            adj_tensor[b, u, v, :] = 0.0
                            adj_tensor[b, v, u, :] = 0.0
                            adj_tensor[b, u, v, b_type] = 1.0
                            adj_tensor[b, v, u, b_type] = 1.0
                        else:
                            adj_indices[b, u, v] = 0
                            adj_indices[b, v, u] = 0
                            adj_tensor[b, u, v, :] = 0.0
                            adj_tensor[b, v, u, :] = 0.0
                            adj_tensor[b, u, v, 0] = 1.0
                            adj_tensor[b, v, u, 0] = 1.0

                # Ensure node u is connected to maintain molecular connectivity (if enabled)
                if self.enforce_connectivity:
                    for b in range(batch_size):
                        if active_step_mask[b] and adj_indices[b, u, :u].sum() == 0:
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
        B = trajectories[0]['node_action'].size(0)
        device = trajectories[0]['node_action'].device
        total_log_probs = torch.zeros(B, device=device)
        entropies = []

        for record in trajectories:
            step = record['step']
            node_feat = record['state_node']
            adj_t = record['state_adj']
            mask = record['active_mask']
            node_act = record['node_action']
            active_step_mask = record.get('active_mask_step', None)

            node_emb, graph_emb = self.forward_subgraph(node_feat, adj_t, mask)
            q_logits = self.forward_atom(graph_emb)

            if self.use_atom_filters:
                if step == 0:
                    q_logits[:, 1:] = -1e9
                else:
                    atom_indices = node_feat[:, :step, 1:].argmax(dim=-1) + 1
                    num_hetero = (atom_indices > 1).sum(dim=-1)
                    for b in range(q_logits.size(0)):
                        if num_hetero[b] >= 4:
                            q_logits[b, 1:] = -1e9

            node_dist = Categorical(logits=q_logits / max(temperature, 1e-4))
            node_lp = node_dist.log_prob(node_act)
            if active_step_mask is not None:
                node_lp = torch.where(active_step_mask, node_lp, torch.zeros_like(node_lp))
                if active_step_mask.any():
                    entropies.append(node_dist.entropy()[active_step_mask].mean())
            else:
                entropies.append(node_dist.entropy().mean())
            total_log_probs = total_log_probs + node_lp

            for e_rec in record['edge_records']:
                u, v = e_rec['u'], e_rec['v']
                edge_act = e_rec['edge_action']
                val_mask = e_rec['valence_mask']
                edge_active = e_rec.get('active_mask_step', active_step_mask)

                pair_feat = torch.cat([node_emb[:, u, :], node_emb[:, v, :], graph_emb], dim=-1)
                edge_logits = self.forward_edge(pair_feat)

                if self.use_valence_mask and val_mask is not None:
                    edge_logits = torch.where(val_mask, edge_logits, torch.tensor(-1e9, device=edge_logits.device))
                edge_dist = Categorical(logits=edge_logits / max(temperature, 1e-4))
                edge_lp = edge_dist.log_prob(edge_act)
                if edge_active is not None:
                    edge_lp = torch.where(edge_active, edge_lp, torch.zeros_like(edge_lp))
                    if edge_active.any():
                        entropies.append(edge_dist.entropy()[edge_active].mean())
                else:
                    entropies.append(edge_dist.entropy().mean())
                total_log_probs = total_log_probs + edge_lp

        mean_entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=device)
        return total_log_probs, mean_entropy


class CriticRGCN(nn.Module):
    """Value Critic V_psi(G) predicting expected cumulative return."""
    def __init__(self, node_dim=5, edge_types=4, hidden_dims=[128, 64], dropout_rate=0.0):
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
    def __init__(self, node_dim=5, edge_types=4, hidden_dims=[128, 64], dropout_rate=0.1):
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
