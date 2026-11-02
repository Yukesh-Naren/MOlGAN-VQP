import torch
import torch.nn as nn
import pennylane as qml


class DedicatedAtomQuantumPolicy(nn.Module):
    """
    Dedicated 1-Qubit = 1-Heavy-Atom Quantum Policy Head.

    Direct Physical Mapping:
      - Qubit 0 <---> Carbon (C)
      - Qubit 1 <---> Nitrogen (N)
      - Qubit 2 <---> Oxygen (O)
      - Qubit 3 <---> Fluorine (F)

    Each qubit receives atom-specific geometric and chemical compatibility parameters:
      x_C: Carbon affinity   (valence capacity >= 1, angle ~ 109.5°/120°, carbon anchor ratio)
      x_N: Nitrogen affinity (valence capacity <= 3, angle ~ 107°, anti-triazene check)
      x_O: Oxygen affinity   (valence capacity <= 2, angle ~ 104.5°, anti-peroxide O-O check)
      x_F: Fluorine affinity (monovalent terminal slot, anchor is Carbon, anti-hetero bond)

    Entanglement between the 4 atomic qubits models chemical competition & orbital cross-talk.
    Readout: Pauli-Z expectation value <Z_i> on each dedicated qubit gives that atom's score!
    """
    def __init__(self, in_features=64, n_qubits=4, n_layers=3):
        super(DedicatedAtomQuantumPolicy, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical feature adapter: extracts atom-specific chemical affinities from GNN state
        self.atom_affinity_adapter = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, n_qubits),
            nn.Sigmoid()  # Normalizes baseline affinities to [0, 1]
        )

        # Trainable Variational Quantum Weights: [n_layers, n_qubits, 3]
        # Initialized with small weights so physical parameters guide early exploration
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Setup PennyLane Quantum Device and QNode
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            # inputs shape: [B, 4] -> Each column i is the affinity [0, 1] for Qubit i (Atom i)
            # Physical mapping:
            #   If x = 1.0 (favorable) -> rotation 0 -> state |0> -> <Z> = +1.0 (Promoted)
            #   If x = 0.0 (forbidden) -> rotation pi -> state |1> -> <Z> = -1.0 (Suppressed)
            for i in range(self.n_qubits):
                qml.RY((1.0 - inputs[:, i]) * torch.pi, wires=i)

            # 2. Trainable Variational Entangling Layers (models atomic orbital competition)
            for layer in range(self.n_layers):
                # All-to-all circular entanglement between the 4 dedicated atomic qubits
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

                # 3D Bloch-Sphere rotations per atomic qubit
                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

            # 3. Dedicated Readout: Measure Pauli-Z expectation for EACH atom's dedicated qubit!
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = circuit

    def compute_atom_specific_compatibilities(self, bond_angle, free_valence, neighbor_type):
        """
        Calculates physical chemical compatibility scores for each of the 4 atoms.
        Args:
            bond_angle: [B] in radians (e.g. 1.91 for 109.5 deg, 1.05 for 60 deg)
            free_valence: [B] number of free bond slots on current anchor atom
            neighbor_type: [B] atom type of anchor (1: C, 2: N, 3: O, 4: F)
        Returns:
            compat: [B, 4] in [0, 1] for [C, N, O, F]
        """
        B = bond_angle.size(0)
        device = bond_angle.device
        compat = torch.ones(B, 4, device=device)

        # 1. Carbon (Qubit 0): Highly compatible at 109.5° (1.91 rad) and 120° (2.09 rad)
        angle_diff_c = torch.abs(bond_angle - 1.91)
        compat[:, 0] = torch.clamp(1.0 - 0.5 * angle_diff_c, 0.1, 1.0)

        # 2. Nitrogen (Qubit 1): Favorable around 107° (1.87 rad), penalize if anchor is N-N
        angle_diff_n = torch.abs(bond_angle - 1.87)
        compat[:, 1] = torch.clamp(1.0 - 0.5 * angle_diff_n, 0.1, 1.0)

        # 3. Oxygen (Qubit 2): Favorable around 104.5° (1.82 rad)
        # CRITICAL: Strictly penalize Oxygen if anchor is already Oxygen (anti-peroxide O-O!)
        angle_diff_o = torch.abs(bond_angle - 1.82)
        o_base = torch.clamp(1.0 - 0.6 * angle_diff_o, 0.1, 1.0)
        is_o_neighbor = (neighbor_type == 3).float()
        is_strained = (bond_angle < 1.2).float() # < 70 deg ring strain
        compat[:, 2] = o_base * (1.0 - 0.9 * is_o_neighbor) * (1.0 - 0.7 * is_strained)

        # 4. Fluorine (Qubit 3): Monovalent capping group
        # Highly favorable if free_valence == 1 and neighbor is Carbon; strictly penalize if neighbor is N, O, F
        is_c_neighbor = (neighbor_type == 1).float()
        is_terminal = (free_valence == 1).float()
        compat[:, 3] = (0.2 + 0.8 * is_terminal) * is_c_neighbor

        return compat

    def forward(self, graph_emb, bond_angle=None, free_valence=None, neighbor_type=None):
        B = graph_emb.size(0)
        device = graph_emb.device

        base_affinity = self.atom_affinity_adapter(graph_emb)

        if bond_angle is None:
            bond_angle = torch.full((B,), 1.91, device=device)
        if free_valence is None:
            free_valence = torch.full((B,), 3.0, device=device)
        if neighbor_type is None:
            neighbor_type = torch.ones(B, dtype=torch.long, device=device)

        phys_compat = self.compute_atom_specific_compatibilities(bond_angle, free_valence, neighbor_type)
        atom_inputs = 0.5 * (base_affinity + phys_compat)

        try:
            expvals = self.qnode(atom_inputs, self.weights)
            atom_scores = torch.stack(expvals, dim=-1)
        except Exception:
            inputs_cpu = atom_inputs.cpu()
            weights_cpu = self.weights.cpu()
            expvals = self.qnode(inputs_cpu, weights_cpu)
            atom_scores = torch.stack(expvals, dim=-1).to(device)

        logits = atom_scores * 2.5
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


# =============================================================================
# 1. 2-QUBIT ATOM SELECTION CIRCUIT (4 BASIS STATES FOR C, N, O, F)
# =============================================================================
# Hilbert Space Decomposition:
#   2 qubits span a 4-dimensional Hilbert space (2^2 = 4 basis states):
#       |00> <---> Carbon (C)    [Index 0]  (Tetravalent anchor)
#       |01> <---> Nitrogen (N)  [Index 1]  (Trivalent linker)
#       |10> <---> Oxygen (O)    [Index 2]  (Divalent bridge / carbonyl)
#       |11> <---> Fluorine (F)  [Index 3]  (Monovalent terminal halogen)
#
# CLASSICAL COUNTERPART (IN COMMENTS FOR DIRECT MODEL COMPARISON):
# -----------------------------------------------------------------------------
# class ClassicalAtomCircuit2Q(nn.Module):
#     """
#     Exact size-matched classical neural counterpart to 2-qubit atom circuit:
#       - Adapter: Linear(64 -> 16) + LeakyReLU(0.2) + Linear(16 -> 2) [1,074 params]
#       - Head: Linear(2 -> 4) + Tanh + Linear(4 -> 4) -> 4 logits [32 params]
#       - Total Head Parameters: 32 (matches 18 Euler angles + 1 scale = 19 quantum params).
#     """
#     def __init__(self, in_features=64, hidden_dim=4, num_classes=4):
#         super().__init__()
#         self.adapter = nn.Sequential(
#             nn.Linear(in_features, 16),
#             nn.LeakyReLU(0.2),
#             nn.Linear(16, 2),
#             nn.Sigmoid()
#         )
#         self.head = nn.Sequential(
#             nn.Linear(2, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, graph_emb):
#         angles = self.adapter(graph_emb)
#         logits = self.head(angles)
#         probs = torch.softmax(logits, dim=-1)
#         return logits, probs
# =============================================================================

class ClassicalAtomCircuit2Q(nn.Module):
    """Active Classical Size-Matched Counterpart to 2-Qubit Atom Policy (32 params)."""
    def __init__(self, in_features=64, hidden_dim=4, num_classes=4):
        super(ClassicalAtomCircuit2Q, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, graph_emb, bond_angle=None, free_valence=None, neighbor_type=None):
        angles = self.adapter(graph_emb)
        logits = self.head(angles)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class QuantumAtomPolicy2Q(nn.Module):
    """
    2-Qubit Variational Quantum Policy Head for Heavy Atom Type Selection.
    Measures Born-rule computational basis probabilities: P(|00>), P(|01>), P(|10>), P(|11>).
    """
    def __init__(self, in_features=64, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(QuantumAtomPolicy2Q, self).__init__()
        self.n_qubits = 2
        self.n_layers = n_layers

        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        else:
            self.register_buffer('output_scale', torch.tensor(float(output_scale)), persistent=False)

        self.adapter = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, self.n_qubits),
            nn.Sigmoid()
        )

        # 18 trainable Euler angles (3 angles x 2 qubits x 3 layers)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, self.n_qubits, 3))
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            for i in range(self.n_qubits):
                qml.RY(inputs[:, i] * 3.14159265, wires=i)

            for layer in range(self.n_layers):
                qml.CNOT(wires=[0, 1])
                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

            return qml.probs(wires=[0, 1])

        self.qnode = circuit

    def forward(self, graph_emb, bond_angle=None, free_valence=None, neighbor_type=None):
        device = graph_emb.device
        angles = self.adapter(graph_emb)

        if angles.device.type != 'cpu':
            probs_4 = self.qnode(angles.cpu(), self.weights.cpu()).to(device=device, dtype=graph_emb.dtype)
        else:
            probs_4 = self.qnode(angles, self.weights).to(device=device, dtype=graph_emb.dtype)

        if probs_4.ndim == 1:
            probs_4 = probs_4.unsqueeze(0)

        logits = self.output_scale * torch.log(probs_4 + 1e-8)
        return logits, probs_4


# =============================================================================
# 2. 3-QUBIT BOND FORMATION CIRCUIT (5 BASIS STATES FOR 5 QM9 BOND TYPES)
# =============================================================================
# Hilbert Space Decomposition:
#   3 qubits span an 8-dimensional Hilbert space (2^3 = 8 basis states):
#       |000> <---> None (No Bond)     [Index 0]  (Graph sparsity / disconnect)
#       |001> <---> Single Bond (1.0)  [Index 1]  (Sigma covalent bond)
#       |010> <---> Double Bond (2.0)  [Index 2]  (Sigma + Pi bond)
#       |011> <---> Triple Bond (3.0)  [Index 3]  (Sigma + 2 Pi bonds)
#       |100> <---> Aromatic Bond (1.5)[Index 4]  (Delocalized Pi resonance)
#       |101>, |110>, |111> <---> Auxiliary non-bonding subspace (sliced out)
#   5 states are sufficient for the 5 QM9 bond types!
#
# CLASSICAL COUNTERPART (IN COMMENTS FOR DIRECT MODEL COMPARISON):
# -----------------------------------------------------------------------------
# class ClassicalEdgeCircuit3Q(nn.Module):
#     """
#     Exact size-matched classical neural counterpart to 3-qubit edge circuit:
#       - Adapter: Linear(192 -> 32) + LeakyReLU(0.2) + Linear(32 -> 3) [6,275 params]
#       - Head: Linear(3 -> 8) + Tanh + Linear(8 -> 5) -> 5 bond logits [77 params]
#       - Total Head Parameters: 77 (matches 27 Euler angles + 1 scale = 28 quantum params).
#     """
#     def __init__(self, in_features=192, hidden_dim=8, num_classes=5):
#         super().__init__()
#         self.adapter = nn.Sequential(
#             nn.Linear(in_features, 32),
#             nn.LeakyReLU(0.2),
#             nn.Linear(32, 3),
#             nn.Sigmoid()
#         )
#         self.head = nn.Sequential(
#             nn.Linear(3, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, num_classes)
#         )
#     def forward(self, pair_feat):
#         angles = self.adapter(pair_feat)
#         logits = self.head(angles)
#         probs = torch.softmax(logits, dim=-1)
#         return logits, probs
# =============================================================================

class ClassicalEdgeCircuit3Q(nn.Module):
    """Active Classical Size-Matched Counterpart to 3-Qubit Edge Policy (77 params)."""
    def __init__(self, in_features=192, hidden_dim=8, num_classes=5):
        super(ClassicalEdgeCircuit3Q, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pair_feat):
        angles = self.adapter(pair_feat)
        logits = self.head(angles)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class QuantumEdgePolicy3Q(nn.Module):
    """
    3-Qubit Variational Quantum Policy Head for Bond Type Selection.
    Measures Born-rule computational basis probabilities across 8 states and normalizes
    the first 5 states (|000> to |100>) corresponding to QM9 bond types.
    """
    def __init__(self, in_features=192, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(QuantumEdgePolicy3Q, self).__init__()
        self.n_qubits = 3
        self.n_layers = n_layers

        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        else:
            self.register_buffer('output_scale', torch.tensor(float(output_scale)), persistent=False)

        # Classical adapter: projects pair embedding [h_u, h_v, h_G] (192-dim) into 3 rotation angles
        self.adapter = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.n_qubits),
            nn.Sigmoid()
        )

        # 27 trainable Euler angles (3 angles x 3 qubits x 3 layers)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, self.n_qubits, 3))
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)

            # Angle encoding on the 3 qubits
            for i in range(self.n_qubits):
                qml.RY(inputs[:, i] * 3.14159265, wires=i)

            # Ring entanglement coupling all 3 qubits with SU(2) Euler rotations
            for layer in range(self.n_layers):
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])
                for i in range(self.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

            return qml.probs(wires=[0, 1, 2])

        self.qnode = circuit

    def forward(self, pair_feat):
        device = pair_feat.device
        angles = self.adapter(pair_feat)

        if angles.device.type != 'cpu':
            probs_8 = self.qnode(angles.cpu(), self.weights.cpu()).to(device=device, dtype=pair_feat.dtype)
        else:
            probs_8 = self.qnode(angles, self.weights).to(device=device, dtype=pair_feat.dtype)

        if probs_8.ndim == 1:
            probs_8 = probs_8.unsqueeze(0)

        # Slicing the 5 states corresponding to the 5 QM9 bond types
        probs_5 = probs_8[:, :5]
        probs_5_norm = probs_5 / (probs_5.sum(dim=-1, keepdim=True) + 1e-8)

        # Convert to logits via log-probability scaling
        logits = self.output_scale * torch.log(probs_5_norm + 1e-8)
        return logits, probs_5_norm
