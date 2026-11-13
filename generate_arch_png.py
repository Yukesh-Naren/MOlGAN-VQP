import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_architecture_diagram():
    fig = plt.figure(figsize=(20, 12), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Background color
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Color Palette
    c_blue = '#1E3A8A'      # Deep navy
    c_blue_box = '#DBEAFE'  # Light blue
    c_blue_border = '#3B82F6'

    c_purple = '#581C87'    # Quantum purple
    c_purple_box = '#F3E8FF'
    c_purple_border = '#A855F7'

    c_green = '#064E3B'     # Valence green
    c_green_box = '#D1FAE5'
    c_green_border = '#10B981'

    c_amber = '#78350F'     # Reward/Critic amber
    c_amber_box = '#FEF3C7'
    c_amber_border = '#F59E0B'

    c_red = '#7F1D1D'       # Discriminator red
    c_red_box = '#FEE2E2'
    c_red_border = '#EF4444'

    c_cyan = '#164E63'      # PPO Engine cyan
    c_cyan_box = '#CFFAFE'
    c_cyan_border = '#06B6D4'

    def draw_box(x, y, w, h, title, subtitle="", bg='#FFFFFF', border='#000000', title_col='#000000', sub_col='#475569', lw=1.5, fontsize=11, sub_fontsize=8.5, radius=0.8):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius},rounding_size=0.8",
                             facecolor=bg, edgecolor=border, linewidth=lw, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=title_col, zorder=3)
            ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', fontsize=sub_fontsize, color=sub_col, zorder=3, linespacing=1.2)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=title_col, zorder=3)
        return box

    def draw_arrow(x1, y1, x2, y2, color='#64748B', lw=1.8, style='->', label='', label_pos=(0.5, 0.5), label_col='#334155', font_sz=8.5, rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0 else "arc3"
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                        connectionstyle=connectionstyle,
                                        arrowstyle=style, mutation_scale=14,
                                        color=color, linewidth=lw, zorder=4)
        ax.add_patch(arrow)
        if label:
            lx = x1 + (x2 - x1) * label_pos[0]
            ly = y1 + (y2 - y1) * label_pos[1]
            ax.text(lx, ly, label, ha='center', va='center', fontsize=font_sz, fontweight='semibold',
                    color=label_col, bbox=dict(boxstyle='round,pad=0.2', facecolor='#F8FAFC', edgecolor='none', alpha=0.9), zorder=5)

    # Big Title & Header
    ax.text(50, 58, "MolGAN-VQP: Hybrid Quantum-Classical Molecular Graph Generation Architecture",
            ha='center', va='center', fontsize=18, fontweight='bold', color='#0F172A')
    ax.text(50, 56, "Autoregressive Markov Decision Process (MDP) with Dual 4-Qubit Softmax-PQCs & 2-Pass PPO Optimization",
            ha='center', va='center', fontsize=11, color='#64748B')

    # =========================================================================
    # SECTION 1: AUTOREGRESSIVE ACTOR (GENERATOR)
    # =========================================================================
    sec_actor = FancyBboxPatch((2, 22), 46, 31, boxstyle="round,pad=0.5,rounding_size=1.2",
                               facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(sec_actor)
    ax.text(4, 51.5, "1. AUTOREGRESSIVE ACTOR (QUANTUM GENERATOR)", fontsize=11, fontweight='bold', color=c_blue)

    # Subgraph Input
    draw_box(4, 38, 11, 7, "Current Subgraph", "G_t = (X_t, A_t)\nNodes: 5D one-hot\nEdges: 4D one-hot",
             bg=c_blue_box, border=c_blue_border, title_col=c_blue)

    # R-GCN Encoder
    draw_box(18, 38, 12, 7, "Relational GCN", "2-Layer R-GCN\n5 -> 128 -> 64\nMasked Mean Pool",
             bg=c_blue_box, border=c_blue_border, title_col=c_blue)
    draw_arrow(15.5, 41.5, 17.5, 41.5, color=c_blue_border, label="Active\nSubgraphs")

    # Arrow split to Atom and Bond Policy
    # Atom Head
    draw_box(33, 44, 13, 6, "Atom Softmax-PQC", "4 Qubits | L=3 Re-uploading\nPauli-<Z> | Scale beta\nActions: {C, N, O, F}",
             bg=c_purple_box, border=c_purple_border, title_col=c_purple)
    draw_arrow(30.5, 43, 32.5, 46.5, color=c_purple_border, rad=-0.15, label="Graph Emb h_G")

    # Bond Head
    draw_box(33, 31, 13, 6.5, "Bond Softmax-PQC", "4 Qubits | L=3 Re-uploading\nPair: [h_u || h_v || h_G]\nActions: {0, 1, 2, 3}",
             bg=c_purple_box, border=c_purple_border, title_col=c_purple)
    draw_arrow(30.5, 40, 32.5, 35, color=c_purple_border, rad=0.15, label="Pair Feat")

    # Valence Masking Box
    draw_box(33, 23.5, 13, 5.5, "Chemical Valence Mask", "Anti-Peroxide / Triazene\nValence bounds: C:4,N:3,O:2,F:1\nIllegal Logits -> -1e9",
             bg=c_green_box, border=c_green_border, title_col=c_green)
    draw_arrow(39.5, 30.5, 39.5, 29.5, color=c_green_border, label="Raw Logits")

    # Molecule Assembler
    draw_box(20, 24, 10, 5.5, "Graph Assembly", "Guaranteed 100%\nChemical Validity\nRollout Trajectory tau",
             bg='#F1F5F9', border='#94A3B8', title_col='#0F172A')
    draw_arrow(32.5, 26, 30.5, 26, color=c_green_border, label="Masked Bond")
    draw_arrow(39.5, 43.5, 26, 30, color=c_purple_border, rad=-0.2, label="Atom u")

    # Assembled Molecule Output
    draw_box(4, 24, 12, 5.5, "Completed Molecule", "G = (V, E) in RDKit\nMorgan Fingerprint ECFP4",
             bg='#E2E8F0', border='#64748B', title_col='#0F172A')
    draw_arrow(19.5, 26.5, 16.5, 26.5, color='#64748B')

    # =========================================================================
    # SECTION 2: ADVERSARIAL DISCRIMINATOR
    # =========================================================================
    sec_disc = FancyBboxPatch((51, 38), 47, 15, boxstyle="round,pad=0.5,rounding_size=1.2",
                              facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(sec_disc)
    ax.text(53, 51.5, "2. RELATIONAL GCN ADVERSARIAL DISCRIMINATOR", fontsize=11, fontweight='bold', color=c_red)

    # Real vs Generated input
    draw_box(53, 42.5, 10, 6, "Dataset Inputs", "Real QM9 (x, A)\nvs.\nGenerated G (x, A)",
             bg=c_red_box, border=c_red_border, title_col=c_red)
    draw_arrow(10, 23.5, 53, 44, color=c_red_border, rad=-0.2, label="Generated G")

    # Discriminator RGCN
    draw_box(66, 42.5, 12, 6, "Discriminator R-GCN", "2-Layer R-GCN (128 -> 64)\nGlobal Mean Pooling\nMLP: 64 -> 64 -> 1",
             bg=c_red_box, border=c_red_border, title_col=c_red)
    draw_arrow(63.5, 45.5, 65.5, 45.5, color=c_red_border)

    # Discriminator Outputs
    draw_box(81, 44.5, 15, 6, "BCE Loss (L_D)", "L_D = -E[log sigma(D(x))]\n     - E[log(1 - sigma(D(G)))]",
             bg=c_red_box, border=c_red_border, title_col=c_red, fontsize=9.5)
    draw_arrow(78.5, 47, 80.5, 47, color=c_red_border, label="Train D")

    draw_box(81, 39, 15, 4.5, "Realness Score D(G)", "prob_real = sigma(D(G)) in [0, 1]",
             bg=c_red_box, border=c_red_border, title_col=c_red, fontsize=9.5)
    draw_arrow(78.5, 43.5, 80.5, 41.5, color=c_red_border, label="Reward Term")

    # =========================================================================
    # SECTION 3: REWARD ENGINE & CRITIC
    # =========================================================================
    sec_rew = FancyBboxPatch((51, 21), 47, 15, boxstyle="round,pad=0.5,rounding_size=1.2",
                             facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(sec_rew)
    ax.text(53, 34.5, "3. QUANTUM-PHYSICOCHEMICAL EVALUATOR & CRITIC", fontsize=11, fontweight='bold', color=c_amber)

    # Quantum Evaluator
    draw_box(53, 24, 13, 8, "Quantum Physicochem\nEvaluator (R_Q)", "4 Qubits | 3 Layers Rot\nEncodes 4 Lipinski Props:\nMW, LogP, HBD, HBA\nReadout: <Z_0> in [0, 1]",
             bg=c_purple_box, border=c_purple_border, title_col=c_purple, fontsize=9.5)
    draw_arrow(10, 23.5, 53, 27, color=c_purple_border, rad=-0.1, label="Descriptor Extraction")

    # Value Critic
    draw_box(68, 24, 12, 8, "Value Critic (V_phi)", "2-Layer R-GCN (128 -> 64)\nGlobal Readout\nMLP: 64 -> 64 -> 1\nBaseline: V_phi(G)",
             bg=c_amber_box, border=c_amber_border, title_col=c_amber, fontsize=9.5)
    draw_arrow(16, 26, 67.5, 27, color=c_amber_border, rad=-0.05, label="Graph Feat")

    # Composite Reward Engine
    draw_box(82, 24, 14, 8, "Composite Reward Engine", "R = (alpha*R_Q + beta*D(G) + gamma*R_prop)\n  * Tanimoto_Diversity\n  * Scaffold_Memory_Factor",
             bg=c_amber_box, border=c_amber_border, title_col=c_amber, fontsize=9)
    draw_arrow(66.5, 28, 81.5, 28, color=c_purple_border)
    draw_arrow(88, 38.5, 88, 32.5, color=c_red_border)

    # =========================================================================
    # SECTION 4: 2-PASS PPO OPTIMIZATION ENGINE
    # =========================================================================
    sec_ppo = FancyBboxPatch((2, 2), 96, 17, boxstyle="round,pad=0.5,rounding_size=1.2",
                             facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=2, linestyle='--', zorder=1)
    ax.add_patch(sec_ppo)
    ax.text(4, 17.5, "4. PROXIMAL POLICY OPTIMIZATION (PPO) CLOSED-LOOP TRAINING PIPELINE", fontsize=11, fontweight='bold', color=c_cyan)

    # PPO Rollout Buffer
    draw_box(4, 4, 16, 11, "PPO Trajectory Buffer", "Stores Rollouts:\n- Trajectories tau = {(s_t, a_t)}\n- Log Probs: log pi_old(tau)\n- Empirical Returns: G_t = R\n- Value Baselines: V_phi(G)",
             bg=c_cyan_box, border=c_cyan_border, title_col=c_cyan, fontsize=9.5)
    draw_arrow(10, 23.5, 10, 15.5, color=c_cyan_border, label="Trajectories")
    draw_arrow(86, 23.5, 20.5, 9, color=c_amber_border, rad=0.2, label="Reward R")
    draw_arrow(74, 23.5, 18, 12, color=c_amber_border, rad=0.15, label="Baseline V")

    # Advantage Normalization
    draw_box(24, 5.5, 18, 8, "Advantage Engine", "Advantage: A_t = R_t - V_phi(G)\nBatch Normalization:\nA_norm = (A - mu_A) / (sigma_A + 1e-8)\nDrastically reduces gradient variance",
             bg=c_cyan_box, border=c_cyan_border, title_col=c_cyan, fontsize=9.5)
    draw_arrow(20.5, 9.5, 23.5, 9.5, color=c_cyan_border)

    # PPO Policy Loss
    draw_box(46, 4, 25, 11, "PPO Clipped Surrogate Loss", "Ratio: r_t(theta) = exp(log pi_theta - log pi_old)\nL_clip = -E[ min(r_t * A, clip(r_t, 1-eps, 1+eps)*A) ]\nL_pi = L_clip - c_entropy * Entropy(pi_theta)\nPrevents barren plateaus & parameter explosion",
             bg=c_cyan_box, border=c_cyan_border, title_col=c_cyan, fontsize=9.5)
    draw_arrow(42.5, 9.5, 45.5, 9.5, color=c_cyan_border)

    # Critic Value Loss
    draw_box(74, 4, 22, 11, "Critic Value Update", "L_V = MSE( V_phi(G), R )\n\nUpdates R-GCN Critic baseline\nto track moving chemical returns.\n2-Pass Adam Optimization (lr=1e-3)",
             bg=c_amber_box, border=c_amber_border, title_col=c_amber, fontsize=9.5)
    draw_arrow(40, 7, 73.5, 7, color=c_cyan_border, rad=-0.1)

    # Feedback Loop back to Actor
    draw_arrow(58, 15.5, 39.5, 22, color=c_purple_border, rad=0.3, label="Update Variational Angles (theta)")
    draw_arrow(85, 15.5, 74, 23.5, color=c_amber_border, rad=0.2, label="Update Value Weights (phi)")

    plt.tight_layout()
    output_path = "molgan_vqp_architecture.png"
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
