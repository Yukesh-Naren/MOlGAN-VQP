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
            # <Z_0>: Carbon score
            # <Z_1>: Nitrogen score
            # <Z_2>: Oxygen score
            # <Z_3>: Fluorine score
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
        """
        Args:
            graph_emb: [B, 64] GNN graph representation
            bond_angle: [B] local bond angle in radians
            free_valence: [B] free bond capacity
            neighbor_type: [B] anchor atom type
        Returns:
            logits: [B, 4] for PPO Categorical action distribution
            probs:  [B, 4] normalized selection probabilities
        """
        B = graph_emb.size(0)
        device = graph_emb.device

        # Classical baseline affinity from GNN embedding: [B, 4]
        base_affinity = self.atom_affinity_adapter(graph_emb)

        # Physical geometric compatibility: [B, 4]
        if bond_angle is None:
            bond_angle = torch.full((B,), 1.91, device=device) # 109.5 deg default
        if free_valence is None:
            free_valence = torch.full((B,), 3.0, device=device)
        if neighbor_type is None:
            neighbor_type = torch.ones(B, dtype=torch.long, device=device) # Carbon default

        phys_compat = self.compute_atom_specific_compatibilities(bond_angle, free_valence, neighbor_type)

        # Combined atomic input: each qubit receives its tailored physical-chemical parameter!
        atom_inputs = 0.5 * (base_affinity + phys_compat)

        # Execute dedicated 1-qubit = 1-atom quantum circuit
        inputs_cpu = atom_inputs.cpu()
        weights_cpu = self.weights.cpu()

        expvals = self.qnode(inputs_cpu, weights_cpu)
        # expvals is a list of 4 tensors [B], stack to [B, 4]
        atom_scores = torch.stack(expvals, dim=-1).to(device)

        # Softmax over the 4 dedicated qubit expectation values
        logits = atom_scores * 2.5
        probs = torch.softmax(logits, dim=-1)

        return logits, probs
