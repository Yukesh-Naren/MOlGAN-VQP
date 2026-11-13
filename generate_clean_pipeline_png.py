import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_clean_pipeline():
    # 16:9 ratio, ultra-high resolution
    fig = plt.figure(figsize=(18, 9.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis('off')

    # Minimalist crisp background
    bg_canvas = '#F8FAFC'
    fig.patch.set_facecolor(bg_canvas)
    ax.set_facecolor(bg_canvas)

    # Cohesive Design Palette
    card_bg = '#FFFFFF'
    border_subtle = '#E2E8F0'
    text_dark = '#0F172A'
    text_muted = '#64748B'

    # Accent Colors for Stages
    c1 = {'accent': '#3B82F6', 'light': '#EFF6FF', 'badge': '#1D4ED8'}  # Blue: Stage 1 (Actor)
    c2 = {'accent': '#8B5CF6', 'light': '#F5F3FF', 'badge': '#6D28D9'}  # Purple: Stage 2 (Reward)
    c3 = {'accent': '#F59E0B', 'light': '#FFFBEB', 'badge': '#B45309'}  # Amber: Stage 3 (Critic)
    c4 = {'accent': '#10B981', 'light': '#ECFDF5', 'badge': '#047857'}  # Emerald: Stage 4 (PPO Update)

    # Header
    ax.text(50, 49.5, "MolGAN-VQP Training Pipeline", ha='center', va='center',
            fontsize=20, fontweight='bold', color=text_dark)
    ax.text(50, 47.2, "Autoregressive 4-Qubit Variational Quantum Policies with Two-Pass Proximal Policy Optimization",
            ha='center', va='center', fontsize=11, color=text_muted)

    # Helper: Draw Stage Card
    def draw_stage_card(x, y, w, h, stage_num, stage_title, color_dict):
        # Outer Card
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                              facecolor=card_bg, edgecolor=border_subtle, linewidth=1.5, zorder=1)
        ax.add_patch(card)

        # Stage Banner Pill
        pill = FancyBboxPatch((x + 1.2, y + h - 3.8), w - 2.4, 2.8, boxstyle="round,pad=0.2,rounding_size=0.6",
                              facecolor=color_dict['light'], edgecolor=color_dict['accent'], linewidth=1.2, zorder=2)
        ax.add_patch(pill)

        ax.text(x + 2.2, y + h - 2.4, f"STAGE {stage_num}", fontsize=8.5, fontweight='black',
                color=color_dict['badge'], va='center', zorder=3)
        ax.text(x + w/2 + 1.5, y + h - 2.4, stage_title, fontsize=10, fontweight='bold',
                color=text_dark, ha='center', va='center', zorder=3)

    # Helper: Draw Inner Block
    def draw_inner_block(x, y, w, h, title, items, badge="", bg='#F8FAFC', border='#E2E8F0', title_col=text_dark):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6",
                             facecolor=bg, edgecolor=border, linewidth=1.2, zorder=3)
        ax.add_patch(box)

        # Title
        ax.text(x + 1.0, y + h - 1.4, title, fontsize=9.5, fontweight='bold', color=title_col, va='center', zorder=4)
        if badge:
            bx = x + w - 1.0
            ax.text(bx, y + h - 1.4, badge, fontsize=7.5, fontweight='bold', color='#6D28D9',
                    ha='right', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E8FF', edgecolor='#DDD6FE', lw=0.8), zorder=4)

        # Item bullets
        start_y = y + h - 3.0
        for i, item in enumerate(items):
            ax.text(x + 1.2, start_y - i * 1.55, f"• {item}", fontsize=8.2, color='#334155', va='center', zorder=4)

    # Helper: Smooth Arrow
    def draw_arrow(x1, y1, x2, y2, col='#94A3B8', rad=0.0, label=''):
        conn = f"arc3,rad={rad}" if rad != 0 else "arc3"
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                        connectionstyle=conn, arrowstyle='->',
                                        mutation_scale=13, color=col, linewidth=1.8, zorder=5)
        ax.add_patch(arrow)
        if label:
            mx = (x1 + x2)/2
            my = (y1 + y2)/2
            ax.text(mx, my + 0.8, label, fontsize=8, fontweight='bold', color='#475569',
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.8), zorder=6)

    # =========================================================================
    # STAGE 1: AUTOREGRESSIVE ACTOR (X: 3 to 24)
    # =========================================================================
    draw_stage_card(3, 4, 21.5, 40.5, "1", "Quantum Actor Rollout", c1)

    draw_inner_block(4.5, 33, 18.5, 6.5, "R-GCN Graph Encoder", [
        "Encodes active subgraph (X_t, A_t)",
        "2-Layer Relational GCN (128 -> 64)",
        "Masked Global Mean Pooling -> h_G"
    ], bg='#FFFFFF', border='#BFDBFE')

    draw_inner_block(4.5, 22.5, 18.5, 9.2, "Dual 4-Qubit PQCs", [
        "Atom PQC: Re-uploads h_G (L=3)",
        "  Selects atom u in {C, N, O, F}",
        "Bond PQC: Re-uploads [h_u || h_v || h_G]",
        "  Selects bond (u, v) in {0, 1, 2, 3}"
    ], badge="Ansatz L=3", bg='#FFFFFF', border='#DDD6FE')

    draw_inner_block(4.5, 13.5, 18.5, 7.8, "Valence Masking Engine", [
        "Monitors open atomic valencies",
        "Enforces C:4, N:3, O:2, F:1 bounds",
        "Filters anti-peroxides & triazenes",
        "Illegal bond logits forced to -1e9"
    ], badge="100% Valid", bg='#F0FDF4', border='#86EFAC', title_col='#166534')

    draw_inner_block(4.5, 5.5, 18.5, 6.8, "Trajectory Output", [
        "Assembles complete chemical graph",
        "Records log pi_old(tau) & actions a_t",
        "100.0% valid molecules by design"
    ], bg='#F8FAFC', border='#E2E8F0')

    # Connect internal blocks in Stage 1
    draw_arrow(13.75, 33, 13.75, 31.7, col=c1['accent'])
    draw_arrow(13.75, 22.5, 13.75, 21.3, col=c1['accent'])
    draw_arrow(13.75, 13.5, 13.75, 12.3, col=c1['accent'])

    # =========================================================================
    # STAGE 2: MULTI-OBJECTIVE REWARD (X: 27.5 to 49)
    # =========================================================================
    draw_stage_card(27.5, 4, 21.5, 40.5, "2", "Multi-Objective Reward", c2)

    draw_inner_block(29, 31.5, 18.5, 8.0, "Adversarial Realness", [
        "R-GCN Discriminator evaluates G",
        "Trained on Real QM9 vs Generated",
        "Output: Realness prob sigma(D(G))",
        "Weight: beta = 0.30"
    ], badge="GAN Prior", bg='#FEF2F2', border='#FECACA', title_col='#991B1B')

    draw_inner_block(29, 21.5, 18.5, 8.8, "Quantum PhysChem (R_Q)", [
        "4-Qubit Lipinski Evaluator circuit",
        "Encodes MW, LogP, HBD, and HBA",
        "Ansatz: 3-layer Rot + circular CNOT",
        "Readout: <Z_0> in [0, 1] | alpha = 0.40"
    ], badge="4 Qubits", bg='#FAF5FF', border='#E9D5FF', title_col='#6B21A8')

    draw_inner_block(29, 13, 18.5, 7.3, "Chemical Property (R_prop)", [
        "Calculates target property heuristic",
        "Options: QED, SAS, LogP, or DC",
        "Normalised score in [0, 1]",
        "Weight: gamma = 0.30"
    ], bg='#FFFFFF', border='#E2E8F0')

    draw_inner_block(29, 5.5, 18.5, 6.3, "Anti-Collapse Multiplier", [
        "In-batch Morgan Tanimoto penalty",
        "Factor: max(0, 1 - sim_max^2)",
        "Bemis-Murcko scaffold memory"
    ], badge="High Div", bg='#FFFBEB', border='#FDE68A', title_col='#92400E')

    # Inter-stage arrow: Stage 1 -> Stage 2
    draw_arrow(24.5, 8.9, 27.5, 8.9, col='#64748B', label="Mol G")
    draw_arrow(24.5, 36.2, 27.5, 36.2, col='#64748B')
    draw_arrow(24.5, 25.9, 27.5, 25.9, col='#64748B')

    # =========================================================================
    # STAGE 3: CRITIC & ADVANTAGE (X: 52 to 73.5)
    # =========================================================================
    draw_stage_card(52, 4, 21.5, 40.5, "3", "Critic & Advantage Baseline", c3)

    draw_inner_block(53.5, 30.5, 18.5, 9.0, "Composite Reward Synthesis", [
        "Combines all feedback signals:",
        "  R = (alpha*R_Q + beta*D + gamma*R_p)",
        "      * Tanimoto_Diversity_Factor",
        "Returns empirical target G_t = R",
        "Eliminates reward hacking"
    ], bg='#FFFBEB', border='#FDE68A', title_col='#92400E')

    draw_inner_block(53.5, 19, 18.5, 10.0, "R-GCN Value Critic", [
        "Inspects final generated graph G",
        "2-Layer R-GCN + Mean Pool Readout",
        "MLP Head: Linear(64 -> 64 -> 1)",
        "Estimates state value baseline: V_phi(G)",
        "Trained via MSE Loss against returns"
    ], badge="Variance Red.", bg='#FFFFFF', border='#FED7AA')

    draw_inner_block(53.5, 5.5, 18.5, 12.0, "Batch Advantage Engine", [
        "Computes raw advantage difference:",
        "  A_t = R_t - V_phi(G_t)",
        "Standardizes across batch:",
        "  A_norm = (A - mu_A) / (sigma_A + 1e-8)",
        "Critical stabilization step for",
        "quantum gradient propagation"
    ], badge="Stable PPO", bg='#F0FDFA', border='#99F6E4', title_col='#0F766E')

    # Arrows inside Stage 3
    draw_arrow(49, 35, 52, 35, col=c2['accent'], label="Reward Terms")
    draw_arrow(62.75, 30.5, 62.75, 29, col=c3['accent'])
    draw_arrow(62.75, 19, 62.75, 17.5, col=c3['accent'])

    # =========================================================================
    # STAGE 4: PPO OPTIMIZATION & UPDATE (X: 76.5 to 97)
    # =========================================================================
    draw_stage_card(76.5, 4, 20.5, 40.5, "4", "2-Pass PPO Updates", c4)

    draw_inner_block(77.8, 30.5, 17.9, 9.0, "Importance Ratio r(theta)", [
        "Evaluates new trajectory log probs:",
        "  log pi_theta(tau)",
        "Calculates policy ratio:",
        "  r(theta) = exp(log pi_new - log pi_old)",
        "Clamped to [-5.0, 5.0] log-space"
    ], bg='#FFFFFF', border='#E2E8F0')

    draw_inner_block(77.8, 17.5, 17.9, 11.5, "Clipped Surrogate Objective", [
        "PPO Clipping (eps = 0.20):",
        "  surr1 = r(theta) * A_norm",
        "  surr2 = clip(r, 1-eps, 1+eps) * A_norm",
        "  L_clip = -min(surr1, surr2)",
        "Adds entropy exploration bonus:",
        "  L_pi = L_clip - c_ent * H(pi_theta)",
        "Prevents mode collapse & barren plateaus"
    ], badge="Clipped eps=0.2", bg='#ECFDF5', border='#A7F3D0', title_col='#047857')

    draw_inner_block(77.8, 5.5, 17.9, 10.5, "Dual Parameter Updates", [
        "Actor Update (Adam lr=1e-4):",
        "  Backpropagates L_pi into",
        "  variational angles theta in PQCs",
        "Critic Update (Adam lr=1e-3):",
        "  Minimizes MSE(V_phi(G), R)",
        "Norm gradient clipping at 1.0"
    ], bg='#FFFFFF', border='#E2E8F0')

    # Stage 3 -> Stage 4
    draw_arrow(73.5, 11.5, 76.5, 23.2, col=c4['accent'], rad=0.1, label="A_norm")
    draw_arrow(86.75, 30.5, 86.75, 29, col=c4['accent'])
    draw_arrow(86.75, 17.5, 86.75, 16, col=c4['accent'])

    # Long Feedback Arrow (Stage 4 back to Stage 1)
    draw_arrow(86.75, 5.5, 13.75, 4.0, col='#10B981', rad=-0.18, label="Closed-Loop Policy Update (theta)")

    plt.tight_layout()
    output_png = "molgan_vqp_pipeline_clean.png"
    plt.savefig(output_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Clean pipeline diagram saved successfully to {output_png}")

if __name__ == "__main__":
    create_clean_pipeline()
