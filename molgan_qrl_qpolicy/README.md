# MolGAN-QRL with Dedicated 1-Qubit = 1-Heavy-Atom Policy Head (Q-Actor)

This package implements an autoregressive **Quantum Policy Gradient** architecture for molecular graph generation. Instead of using classical linear projection layers for atom type selection, the decision-making engine is a **Dedicated 1-Qubit = 1-Heavy-Atom Parameterized Quantum Circuit (PQC)**.

---

## 1. Physical Qubit Allocation

In standard QML, qubits often represent abstract features. Here, we establish a **strict 1-to-1 physical correspondence** between each qubit on the quantum register and each heavy atom type in the QM9 chemical space:

| Qubit Register | Assigned Atom | Physical Valence | Native Role in Molecule |
| :---: | :---: | :---: | :--- |
| **Qubit 0** | **Carbon (C)** | 4 | Primary structural backbone, ring, and bridge former |
| **Qubit 1** | **Nitrogen (N)** | 3 | Polar functional groups, amines, amides, aromatic heterocycles |
| **Qubit 2** | **Oxygen (O)** | 2 | Alcohols, ethers, carbonyls ($C=O$), and esters |
| **Qubit 3** | **Fluorine (F)** | 1 | Terminal monovalent halogen capping group |

---

## 2. Geometric & Chemical Parameter Encoding

At each autoregressive step $u$, the current attachment site is analyzed for local geometry, steric crowding, and valency. Each dedicated qubit receives its **tailored physical compatibility parameter** $x_i \in [0, 1]$:

1. **Carbon ($x_C$)**: Evaluates alignment with tetrahedral ($109.5^\circ$) or trigonal planar ($120.0^\circ$) bond angles and verifies that available valence $\ge 1$.
2. **Nitrogen ($x_N$)**: Evaluates alignment with pyramidal angle ($107.0^\circ$) and enforces an **anti-triazene check** (suppresses $N$ if two adjacent nitrogens already exist).
3. **Oxygen ($x_O$)**: Evaluates alignment with bent angle ($104.5^\circ$) and enforces a **strict anti-peroxide check** (if the anchor is already Oxygen, $x_O \to 0.0$, forbidding $O-O$ bonds).
4. **Fluorine ($x_F$)**: Evaluates terminal single-slot compatibility. If the anchor is Carbon and has **exactly 1 free valence slot left**, $x_F \to 1.0$. If the anchor is a heteroatom ($N$ or $O$), $x_F \to 0.0$ (forbidding $F-O$ and $F-N$ bonds).

---

## 3. Quantum Circuit Mechanics & Readout

### State Preparation:
$$R_Y\big((1.0 - x_i) \cdot \pi\big) |0\rangle$$
- **Favorable atom ($x_i \to 1.0$)**: Rotation is $0 \implies$ Qubit stays in state $|0\rangle \implies \langle Z_i \rangle = \mathbf{+1.0}$ (Promoted).
- **Forbidden atom ($x_i \to 0.0$)**: Rotation is $\pi \implies$ Qubit flips to state $|1\rangle \implies \langle Z_i \rangle = \mathbf{-1.0}$ (Suppressed).

### Entanglement as Atomic Competition:
Circular CNOT gates entangle the four dedicated qubits across $L = 3$ layers:
$$\text{Qubit}_C \longleftrightarrow \text{Qubit}_N \longleftrightarrow \text{Qubit}_O \longleftrightarrow \text{Qubit}_F$$
This entangling layer acts as an **atomic orbital arbitration network**, letting the 4 atom species compete on the Hilbert space based on overall GNN context and PPO rewards.

### Dedicated Readout:
The network measures the **Pauli-$Z$ expectation value** on each dedicated wire:
$$\text{Logit}(C) = \beta \cdot \langle Z_0 \rangle$$
$$\text{Logit}(N) = \beta \cdot \langle Z_1 \rangle$$
$$\text{Logit}(O) = \beta \cdot \langle Z_2 \rangle$$
$$\text{Logit}(F) = \beta \cdot \langle Z_3 \rangle$$

Softmax over the four physical expectation values directly yields the PPO action distribution:
$$P(\text{Atom} = i) = \frac{\exp(\beta \langle Z_i \rangle)}{\sum_{j=0}^3 \exp(\beta \langle Z_j \rangle)}$$

---

## 4. Execution & Quick Start

### Run Standalone Unit Test:
```bash
python molgan_qrl_qpolicy/quantum_policy.py
```

### Launch Training Run:
```bash
python molgan_qrl_qpolicy/main_qpolicy.py \
    --epochs 50 \
    --batch_size 16 \
    --save_dir Results/qpolicy_quantum_GAN \
    --temperature 1.2 \
    --entropy_coef 0.10
```
