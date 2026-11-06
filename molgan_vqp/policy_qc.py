import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class JerbiDataReuploadingPQC(nn.Module):
    """
    Parametrized Quantum Policy Head based on Jerbi et al. (NeurIPS 2021):
    'Parametrized Quantum Policies for Reinforcement Learning' (arXiv:2103.05577).

    Key Characteristics:
      1. Multi-Layer Data Re-Uploading:
         The classical state representation s is repeatedly re-injected across all L layers
         via single-qubit rotations, expanding the accessible Fourier frequency spectrum.
      2. Entangling Layer:
         Circular / Ladder CNOT entanglement connecting adjacent qubits.
      3. Variational Ansatz:
         General SU(2) Euler rotations Rot(alpha, beta, gamma) = Rz(alpha) Ry(beta) Rz(gamma)
         on every qubit at each layer.
      4. Measurement of Pauli Observables:
         Measures expectation values <Z_i> = <psi| sigma_z^(i) |psi> in [-1, +1] for each action.
      5. Softmax Policy Readout:
         Maps expectation values to action logits via learnable inverse temperature beta:
           logits = beta * <Z>
           pi(a|s) = Softmax(logits)
    """
    def __init__(self, in_features, n_qubits=4, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(JerbiDataReuploadingPQC, self).__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Data Re-Uploading Adapters:
        # Maps input features s in R^{in_features} to rotation angles in [-pi, pi] for each layer and qubit.
        # Shape: (L, in_features, n_qubits)
        self.input_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, n_qubits),
                nn.Tanh()
            )
            for _ in range(n_layers)
        ])

        # Trainable Variational Weights:
        # 3 Euler angles (alpha, beta, gamma) per qubit per layer: (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Trainable Softmax scaling factor beta (inverse temperature)
        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(float(output_scale)))
        else:
            self.register_buffer('output_scale', torch.tensor(float(output_scale)), persistent=False)

        # Quantum Device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(encoded_angles, weights):
            """
            encoded_angles: (B, L, n_qubits)
            weights: (L, n_qubits, 3)
            """
            if encoded_angles.ndim == 2:
                encoded_angles = encoded_angles.unsqueeze(0)

            for l in range(self.n_layers):
                # 1. Data Re-Uploading: Inject classical state angles via Ry rotations
                for i in range(self.n_qubits):
                    qml.RY(encoded_angles[:, l, i], wires=i)

                # 2. Entanglement Layer: Circular CNOT ladder
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

                # 3. Variational SU(2) Rotations
                for i in range(self.n_qubits):
                    qml.Rot(
                        weights[l, i, 0],
                        weights[l, i, 1],
                        weights[l, i, 2],
                        wires=i
                    )

            # Measure Pauli-Z expectation values for each qubit/action
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = circuit

    def forward(self, state_features):
        """
        Args:
            state_features: Tensor of shape (B, in_features)
        Returns:
            logits: Tensor of shape (B, n_qubits)
            expvals: Tensor of shape (B, n_qubits) in [-1, 1]
        """
        device = state_features.device
        batch_size = state_features.size(0)

        # Compute data re-uploading angles for each layer: (B, L, n_qubits)
        angles_list = []
        for l in range(self.n_layers):
            angles_l = math.pi * self.input_adapters[l](state_features)  # scaled to [-pi, pi]
            angles_list.append(angles_l)
        angles = torch.stack(angles_list, dim=1)  # (B, L, n_qubits)

        # Execute QNode (with CPU/CUDA fallback)
        if angles.device.type != 'cpu':
            res = self.qnode(angles.cpu(), self.weights.cpu())
            expvals = torch.stack(list(res), dim=-1).to(device=device, dtype=state_features.dtype)
        else:
            res = self.qnode(angles, self.weights)
            expvals = torch.stack(list(res), dim=-1).to(device=device, dtype=state_features.dtype)

        if expvals.ndim == 1:
            expvals = expvals.unsqueeze(0)

        # Jerbi Softmax Readout: logits = beta * <Z>
        logits = self.output_scale * expvals
        return logits, expvals


class JerbiAtomPolicy(nn.Module):
    """
    Jerbi Softmax-PQC for Heavy Atom Selection:
      - 4 qubits correspond to 4 atom types:
          Qubit 0: Carbon (C)
          Qubit 1: Nitrogen (N)
          Qubit 2: Oxygen (O)
          Qubit 3: Fluorine (F)
      - Measures <Z_0>, <Z_1>, <Z_2>, <Z_3>.
      - Softmax action selection over atoms.
    """
    def __init__(self, in_features=64, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(JerbiAtomPolicy, self).__init__()
        self.pqc = JerbiDataReuploadingPQC(
            in_features=in_features,
            n_qubits=4,
            n_layers=n_layers,
            output_scale=output_scale,
            learnable_scale=learnable_scale
        )

    def forward(self, graph_emb):
        logits, expvals = self.pqc(graph_emb)
        return logits, expvals


class JerbiEdgePolicy(nn.Module):
    """
    Jerbi Softmax-PQC for Bond Formation:
      - 4 qubits correspond to 4 Kekule bond types:
          Qubit 0: None (No bond)
          Qubit 1: Single bond
          Qubit 2: Double bond
          Qubit 3: Triple bond
      - Measures <Z_0>, <Z_1>, <Z_2>, <Z_3>.
      - Softmax action selection over candidate edge types.
    """
    def __init__(self, in_features=192, n_layers=3, output_scale=3.0, learnable_scale=True):
        super(JerbiEdgePolicy, self).__init__()
        self.pqc = JerbiDataReuploadingPQC(
            in_features=in_features,
            n_qubits=4,
            n_layers=n_layers,
            output_scale=output_scale,
            learnable_scale=learnable_scale
        )

    def forward(self, pair_feat):
        logits, expvals = self.pqc(pair_feat)
        return logits, expvals
