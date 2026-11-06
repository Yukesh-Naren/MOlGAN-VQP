import os
import argparse
from datetime import datetime
try:
    from molgan_vqp.solver_ import SolverJerbiVQP
except ImportError:
    from molgan_vqp.solver_vqp import SolverJerbiVQP


def parse_args():
    parser = argparse.ArgumentParser(
        description="MolGAN with Jerbi et al. (NeurIPS 2021) Variational Quantum Policies (Softmax-PQC)"
    )

    # Core Training Parameters
    parser.add_argument('--dataset_path', type=str, default='data/qm9_5k_kekulized.sparsedataset',
                        help='Path to QM9 kekulized dataset (default: data/qm9_5k_kekulized.sparsedataset)')
    parser.add_argument('--save_dir', type=str, default='Results/molgan_jerbi_softmax_pqc',
                        help='Base directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for rollouts and updates')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO surrogate update passes')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.1, help='Entropy regularization coefficient')
    parser.add_argument('--temperature', type=float, default=1.2, help='Softmax exploration temperature')

    # Tripartite Reward Weights
    parser.add_argument('--alpha', type=float, default=0.4, help='Weight for Quantum Property Evaluator (R_Q)')
    parser.add_argument('--beta', type=float, default=0.3, help='Weight for Discriminator Realism (R_D)')
    parser.add_argument('--gamma', type=float, default=0.3, help='Weight for Chemical Property (R_prop)')
    parser.add_argument('--prop_metric', type=str, default='sas', choices=['sas', 'qed', 'logp', 'np', 'dc'],
                        help='Target chemical property metric')
    parser.add_argument('--scaffold_bucket_size', type=int, default=25,
                        help='Maximum allowed occurrences per Bemis-Murcko scaffold before penalty')

    # Learning Rates
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Learning rate for Jerbi Actor policy')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate for Value Critic')
    parser.add_argument('--lr_disc', type=float, default=1e-4, help='Learning rate for Discriminator')

    # Jerbi Softmax-PQC Circuit Settings
    parser.add_argument('--n_layers_atom', type=int, default=3,
                        help='Number of Data Re-Uploading layers in Atom Softmax-PQC')
    parser.add_argument('--n_layers_edge', type=int, default=3,
                        help='Number of Data Re-Uploading layers in Bond Softmax-PQC')
    parser.add_argument('--output_scale', type=float, default=3.0,
                        help='Initial learnable Softmax inverse temperature scaling beta')

    # Chemistry and Graph Topology
    parser.add_argument('--min_atoms', type=int, default=4, help='Minimum heavy atoms per molecule')
    parser.add_argument('--max_atoms', type=int, default=9, help='Maximum heavy atoms per molecule')
    parser.add_argument('--num_edge_types', type=int, default=4,
                        help='Number of Kekule bond types (0:None, 1:Single, 2:Double, 3:Triple)')

    # Masking & Filter Controls
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

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for training (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_save_dir = os.path.join(args.save_dir, timestamp)

    print("=" * 75)
    print("      MolGAN with Jerbi et al. (NeurIPS 2021) Variational Quantum Policies")
    print("                      (Softmax-PQC + Data Re-Uploading)")
    print("=" * 75)
    for arg, val in vars(args).items():
        print(f"  {arg:<22} : {val}")
    print("=" * 75)

    solver = SolverJerbiVQP(
        dataset_path=args.dataset_path,
        save_dir=run_save_dir,
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
        n_layers_atom=args.n_layers_atom,
        n_layers_edge=args.n_layers_edge,
        output_scale=args.output_scale,
        use_valence_mask=not args.no_valence_mask,
        use_filter=not args.no_filter,
        use_scaffold_memory=not args.no_scaffold_memory,
        use_tanimoto=not args.no_tanimoto,
        enforce_connectivity=not args.no_connectivity_enforce,
        use_atom_filters=not args.no_atom_filters,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        num_edge_types=args.num_edge_types,
        device=args.device
    )

    solver.train()


if __name__ == '__main__':
    main()
