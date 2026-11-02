# ⚛️ MolGAN-QPolicy: Dedicated Quantum Policy Networks with Scaffold Memory for Diverse De Novo Molecular Generation

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.5+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.20+-blueviolet.svg)](https://pennylane.ai/)
[![RDKit](https://img.shields.io/badge/RDKit-2020.09+-green.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official repository and research manuscript implementation for:**  
> *"Dedicated Quantum Policy Networks with Scaffold Memory for Diverse De Novo Molecular Generation"*

---

## 👥 Authors & Affiliations

- **Yukesh Naren M** $^{1,\dagger,*}$ ([yukeshnarenm@gmail.com](mailto:yukeshnarenm@gmail.com))
- **Mohithra K** $^{1,\dagger}$
- **Mokshagna Maddineni** $^{1,\dagger}$
- **Chandra Mohan Dasari** $^{1}$

$^{1}$ *Department of Computer Science and Engineering, Indian Institute of Information Technology Design and Manufacturing Kurnool (IIITDM Kurnool), Kurnool, Andhra Pradesh 518008, India.*  
$^{\dagger}$ *Equal contribution (Co-first authors).*  
$^{*}$ *Corresponding author.*

---

## 📌 Abstract & Overview

Deep generative models for molecular graphs are transforming rational drug design; however, reinforcement learning frameworks such as MolGAN frequently succumb to catastrophic mode collapse—repeatedly producing a narrow subset of trivial subgraphs while violating fundamental chemical valences. 

**MolGAN-QPolicy** is a fundamentally re-engineered quantum reinforcement learning framework for molecular graph generation. Rather than deploying quantum computing as an offline reward evaluator, we integrate a dedicated **4-Qubit Variational Quantum Policy Head ($\pi_\theta$)** directly into the decision-making loop of an autoregressive Relational Graph Convolutional Network (RGCN) trained via Proximal Policy Optimization (PPO).

### 🌟 Key Innovations
1. **Physical 1-Qubit = 1-Atom Mapping**: Explicit one-to-one correspondence ($q_0 \leftrightarrow \text{C}, q_1 \leftrightarrow \text{N}, q_2 \leftrightarrow \text{O}, q_3 \leftrightarrow \text{F}$), injecting local valence, angles, and neighborhood chemistries into quantum state rotations.
2. **Quantum Bond Policy Circuit (3-Qubit VQC)**: 3-qubit variational circuit encoding pair embeddings into Born-rule computational basis probabilities over the 5 QM9 bond types (None, Single, Double, Triple, Aromatic).
3. **PPO-Guided Closed-Loop Learning**: Quantum gradients propagate end-to-end to drive generator updates using clipped surrogate objectives without vanishing or detached gradients.
4. **Bemis-Murcko Scaffold Diversity Memory Filter**: Dynamic memory buckets penalize repeated core ring scaffolds, completely eliminating mode collapse.
5. **Dynamic Valence Masking**: Explicit physical valence checks prevent hypervalent and physically strained bond additions during graph assembly.
6. **Head-to-Head Size-Matched Classical Controls**: Pre-registered size-matched classical architectures (32-parameter atom head, 77-parameter bond head) provided directly in the codebase for rigorous baseline comparisons.
7. **Rigorous Quantum Benchmarks**: Evaluated on the QM9 dataset, achieving an NVU (Number of Valid and Unique molecules) of **288 (mean) / 447 (peak)** compared to MolGAN-QRL's **29** and QGAN-HG's **31** ($10\times$ to $15\times$ improvement).

---

## 📊 Experimental Results (QM9 Chemical Benchmark, $N = 497$)

| Model | Validity (%) ↑ | Uniqueness (%) ↑ | Novelty (%) ↑ | Diversity ↑ | Mean NVU ($N=497$) ↑ | Peak NVU ($N=497$) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MolGAN** *(Original)* | 67.20 | 2.10 | 97.40 | 0.32 | 7 | 12 |
| **MolGAN-QRL** *(Hergli & Harigua-Souiai, 2026)* | 69.70 | 8.36 | 98.20 | 0.49 | 29 | 42 |
| **MolGAN-QRL-S** *(Single-Reward Variant)* | 45.60 | 4.88 | 96.80 | 0.41 | 11 | 24 |
| **QGAN-HG** *(Quantum GAN with Hybrid Gen)* | 61.40 | 10.12 | 97.90 | 0.52 | 31 | 50 |
| **MolGAN-QPolicy (Ours)** | **99.47** | **62.37** | **98.80** | **0.88** | **288** | **447** |

> **Cheminformatics Validation**: 3-tier screening shows **89.0%** of generated unique structures are readily preparable bench compounds with standard synthetic accessibility ($\text{Raw SAS} \le 5.0$), verified 3D Euclidean embeddability, and zero ring/allene strain alerts.

---

## 📂 Repository Structure

```plaintext
MolGAN-Qpolicy/
├── molgan_qrl_qpolicy/           # Core MolGAN-QPolicy package
│   ├── circuit.py                # 4-Qubit Variational Quantum Circuit (PennyLane)
│   ├── quantum_policy.py         # Quantum Policy Head (1-Qubit = 1-Atom)
│   ├── models_qpolicy.py         # RGCN Generator, Discriminator, and Value Critic
│   ├── ppo_buffer.py             # Rollout buffer for PPO updates
│   ├── reward.py                 # Multi-objective reward engine & Scaffold Memory
│   ├── solver_qpolicy.py         # End-to-end PPO training engine
│   ├── layers.py                 # Graph convolution and tensor operations
│   └── main_qpolicy.py           # CLI entrypoint for training and evaluation
├── data/                         # Datasets & scoring models
│   ├── sparse_molecular_dataset.py # Graph data structures & loaders
│   ├── qm9_with_negatives.sparsedataset # Preprocessed QM9 sparse tensors (5.5 MB)
│   ├── qm9.smi                   # Canonical QM9 SMILES dataset (2.1 MB)
│   ├── SA_score.pkl.gz           # Synthetic Accessibility scoring weights
│   ├── NP_score.pkl.gz           # Natural Product-likeness scoring weights
│   └── download_dataset.sh       # Raw dataset fetch utility
├── paper_manuscript/             # Research paper & publication assets
│   ├── overleaf_qpolicy_paper.tex # Publication-ready LaTeX source
│   └── overleaf_qpolicy_paper.pdf # Compiled camera-ready manuscript PDF
├── environment.yml               # Complete Conda environment specification
├── .gitignore                    # Exclusion rules for large artifacts
└── README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Yukesh-Naren/MolGAN-Qpolicy.git
cd MolGAN-Qpolicy
```

### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate molgan-pt
```

*(Alternatively, ensure Python 3.7+ is installed with `torch >= 1.5`, `pennylane >= 0.20`, and `rdkit`.)*

---

## 🚀 Usage

### 🏃 Training MolGAN-QPolicy
Train the model on the preprocessed QM9 dataset:
```bash
python molgan_qrl_qpolicy/main_qpolicy.py \
    --dataset_path data/qm9_with_negatives.sparsedataset \
    --atom_policy dedicated_quantum \
    --edge_policy quantum \
    --batch_size 16 \
    --epochs 50 \
    --ppo_epochs 4 \
    --clip_eps 0.2 \
    --entropy_coef 0.10 \
    --temperature 1.2 \
    --alpha 0.4 \
    --beta 0.3 \
    --gamma 0.3 \
    --prop_metric qed \
    --scaffold_bucket_size 25
```

### 📄 Compiling the Manuscript
The LaTeX manuscript is structured for Overleaf or standard `pdflatex`:
```bash
cd paper_manuscript
pdflatex overleaf_qpolicy_paper.tex
pdflatex overleaf_qpolicy_paper.tex
```

---

## 🔬 Citation

If you find this work or code useful in your research, please cite our paper:

```bibtex
@article{naren2026dedicated,
  title={Dedicated Quantum Policy Networks with Scaffold Memory for Diverse De Novo Molecular Generation},
  author={Yukesh Naren, M and Mohithra, K and Mokshagna, Maddineni and Dasari, Chandra Mohan},
  journal={arXiv preprint},
  year={2026},
  publisher={Department of Computer Science and Engineering, IIITDM Kurnool}
}
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).
