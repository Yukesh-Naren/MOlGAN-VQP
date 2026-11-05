import os
import sys
import argparse
import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_PKG_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from molgan_qrl_qpolicy.solver_qpolicy import SolverQPolicy


def parse_args():
    parser = argparse.ArgumentParser(description='MolGAN-QRL with Dedicated 1-Qubit = 1-Atom Policy Head')

    default_dataset = os.path.join(_ROOT_DIR, 'data', 'qm9_with_negatives.sparsedataset')
    default_save_dir = os.path.join(_ROOT_DIR, 'Results', 'qpolicy_quantum_GAN')

    parser.add_argument('--dataset_path', type=str, default=default_dataset,
                        help='Path to the preprocessed QM9 dataset')
    parser.add_argument('--save_dir', type=str, default=default_save_dir,
                        help='Directory to save model checkpoints and images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for rollouts and training')
    parser.add_argument('--epochs', type=int, default=50, help='Total training epochs')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO update epochs per rollout')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.10, help='Entropy regularization coefficient')
    parser.add_argument('--temperature', type=float, default=1.2, help='Sampling temperature for policy diversity')

    # Composite Reward weights: R = alpha * R_Q + beta * sigma(D) + gamma * R_prop
    parser.add_argument('--alpha', type=float, default=0.4, help='Weight for 4-qubit Quantum reward')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for Discriminator realness')
    parser.add_argument('--gamma', type=float, default=0.3, help='Weight for classical property reward')
    parser.add_argument('--prop_metric', type=str, default='qed', choices=['qed', 'sas', 'np', 'dc', 'logp'],
                        help='Target chemical property metric')
    parser.add_argument('--scaffold_bucket_size', type=int, default=25,
                        help='REINVENT Scaffold Diversity Memory bucket size to prevent mode collapse')

    # Learning rates
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Learning rate for Actor policy')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for Value Critic')
    parser.add_argument('--lr_disc', type=float, default=1e-4, help='Learning rate for Discriminator')

    # Policy heads (Quantum vs Classical)
    parser.add_argument('--atom_policy', type=str, default='dedicated_quantum',
                        choices=['dedicated_quantum', 'quantum_2q', 'classical_2q'],
                        help='Policy architecture for heavy atom selection')
    parser.add_argument('--edge_policy', type=str, default='quantum',
                        choices=['quantum', 'classical', 'mlp'],
                        help='Policy architecture for bond formation')

    # Masking and filtering controls
    parser.add_argument('--no_valence_mask', action='store_true', default=False,
                        help='Disable physical valence masking on bond selection')
    parser.add_argument('--no_scaffold_memory', action='store_true', default=False,
                        help='Disable REINVENT Bemis-Murcko scaffold memory penalty in reward')
    parser.add_argument('--no_tanimoto', action='store_true', default=False,
                        help='Disable in-batch Tanimoto diversity penalty in reward')
    parser.add_argument('--no_filter', action='store_true', default=False,
                        help='Disable scaffold memory and chemical alert filters in reward')
    parser.add_argument('--no_connectivity_enforce', action='store_true', default=False,
                        help='Disable post-hoc connectivity enforcement fallback')
    parser.add_argument('--no_atom_filters', action='store_true', default=False,
                        help='Disable step 0 Carbon anchor and heteroatom cap filters')

    # Dynamic atom size controls (e.g. 4 to 9 heavy atoms)
    parser.add_argument('--min_atoms', type=int, default=4,
                        help='Minimum heavy atom count for dynamic generation (default: 4)')
    parser.add_argument('--max_atoms', type=int, default=9,
                        help='Maximum heavy atom count for dynamic generation (default: 9)')
    parser.add_argument('--num_edge_types', type=int, default=4,
                        help='Number of bond/edge types (default: 4 for Kekule [None, Single, Double, Triple])')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(" " * 10 + "MolGAN-QRL with Dedicated 1-Qubit = 1-Atom Policy Head")
    print("=" * 75)
    for k, v in vars(args).items():
        print(f"  {k:22s}: {v}")
    print("=" * 75)

    solver = SolverQPolicy(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        prop_metric=args.prop_metric,
        scaffold_bucket_size=args.scaffold_bucket_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_disc=args.lr_disc,
        atom_policy_type=args.atom_policy,
        edge_policy_type=args.edge_policy,
        use_valence_mask=not args.no_valence_mask,
        use_filter=not args.no_filter,
        use_scaffold_memory=not args.no_scaffold_memory,
        use_tanimoto=not args.no_tanimoto,
        enforce_connectivity=not args.no_connectivity_enforce,
        use_atom_filters=not args.no_atom_filters,
        num_edge_types=args.num_edge_types,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        device=args.device
    )

    solver.train()


if __name__ == '__main__':
    main()
