# MolGAN-Jerbi-VQP: Variational Quantum Policies for Molecular Generation

This package implements an autoregressive molecular graph generative model grounded directly on the foundational theoretical and architectural framework of:

> **"Parametrized Quantum Policies for Reinforcement Learning"**  
> Sofiene Jerbi, Casper Gyurik, Simon C. Marshall, Hans J. Briegel, Vedran Dunjko  
> *Advances in Neural Information Processing Systems (NeurIPS 2021)*  
> [arXiv:2103.05577](https://arxiv.org/abs/2103.05577)

---

## Architectural Principles from Jerbi et al. (2021)

### 1. Multi-Layer Data Re-Uploading
In near-term quantum devices, encoding continuous observations only once at the circuit input restricts the accessible Fourier frequencies to $\pm 1$.  
Following Jerbi et al., the molecular state embeddings (graph and pair representations from an RGCN) are **re-encoded across every layer $l \in \{1, \dots, L\}$** via single-qubit Pauli-$Y$ rotations:
$$\phi_{l, i} = \pi \cdot \tanh(\mathbf{w}_{l, i} \cdot \mathbf{s} + b_{l, i})$$
This turns the circuit into a degree-$L$ multivariate Fourier series with universal function approximation capacity.

### 2. Entanglement Ladder & Variational Ansatz
Each layer couples adjacent qubits via a circular $\text{CNOT}$ ladder followed by arbitrary $SU(2)$ Euler rotations:
$$U(\theta_l) = \bigotimes_{i} \left[ R_z(\theta_{l, i, 0}) \, R_y(\theta_{l, i, 1}) \, R_z(\theta_{l, i, 2}) \right]$$

### 3. Pauli Observable Measurement
For each candidate discrete action (atom type or bond type), the quantum circuit measures the expectation value of a single-qubit Pauli-$Z$ observable:
$$\langle \hat{Z}_i \rangle = \langle \psi(\mathbf{s}, \theta) | \sigma_z^{(i)} | \psi(\mathbf{s}, \theta) \rangle \in [-1, +1]$$

### 4. Softmax Policy Scaling
The bounded continuous expectation values are scaled by a learnable inverse temperature $\beta$ and converted into action probabilities via Softmax:
$$\pi_\theta(a \mid \mathbf{s}) = \frac{\exp\left(\beta \cdot \langle \hat{Z}_a \rangle\right)}{\sum_{a'} \exp\left(\beta \cdot \langle \hat{Z}_{a'} \rangle\right)}$$

---

## Dual Quantum Policy Heads for Chemistry

1. **Jerbi Atom Policy (`JerbiAtomPolicy`)**:
   - $4$ Qubits correspond to the 4 heavy atom types in QM9:
     - Qubit 0: Carbon ($\text{C}$)
     - Qubit 1: Nitrogen ($\text{N}$)
     - Qubit 2: Oxygen ($\text{O}$)
     - Qubit 3: Fluorine ($\text{F}$)
   - Input: Pooled molecular subgraph embedding $\mathbf{g} \in \mathbb{R}^{64}$.
   - Output: Atom action probability distribution $\pi_{\text{atom}}(a \mid \mathbf{g})$.

2. **Jerbi Bond Policy (`JerbiEdgePolicy`)**:
   - $4$ Qubits correspond to the 4 Kekulé bond types:
     - Qubit 0: None (No bond)
     - Qubit 1: Single bond ($1.0$)
     - Qubit 2: Double bond ($2.0$)
     - Qubit 3: Triple bond ($3.0$)
   - Input: Concatenated pair and graph features $[\mathbf{h}_u, \mathbf{h}_v, \mathbf{g}] \in \mathbb{R}^{192}$.
   - Valence Masking: Dynamic octet rule masking enforces physical valences before Softmax.
   - Output: Bond action probability distribution $\pi_{\text{edge}}(b \mid \mathbf{h}_u, \mathbf{h}_v, \mathbf{g})$.

---

## Usage

### Training

To train MolGAN with the Jerbi Softmax-PQC policy on QM9:

```bash
python -u molgan_vqp/main_vqp.py \
    --dataset_path data/qm9_5k_kekulized.sparsedataset \
    --save_dir Results/molgan_jerbi_softmax_pqc \
    --prop_metric sas \
    --n_layers_atom 3 \
    --n_layers_edge 3 \
    --epochs 200 \
    --batch_size 16 \
    --device cuda
```
