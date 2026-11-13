import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_names_only_pipeline():
    fig = plt.figure(figsize=(16, 7), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis('off')

    bg_color = '#F8FAFC'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Clean Modern Palette
    c_blue   = {'bg': '#EFF6FF', 'border': '#3B82F6', 'text': '#1D4ED8'}
    c_purple = {'bg': '#F5F3FF', 'border': '#8B5CF6', 'text': '#6D28D9'}
    c_green  = {'bg': '#ECFDF5', 'border': '#10B981', 'text': '#047857'}
    c_red    = {'bg': '#FEF2F2', 'border': '#EF4444', 'text': '#B91C1C'}
    c_amber  = {'bg': '#FFFBEB', 'border': '#F59E0B', 'text': '#B45309'}
    c_teal   = {'bg': '#F0FDFA', 'border': '#14B8A6', 'text': '#0F766E'}
    c_gray   = {'bg': '#FFFFFF', 'border': '#94A3B8', 'text': '#0F172A'}

    def draw_node(x, y, w, h, title, style, font_size=10.5):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6",
                             facecolor=style['bg'], edgecolor=style['border'], linewidth=1.8, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=font_size, fontweight='bold', color=style['text'], zorder=3)
        return (x, y, w, h)

    def draw_arrow(x1, y1, x2, y2, color='#64748B', rad=0.0, lw=2.0):
        conn = f"arc3,rad={rad}" if rad != 0 else "arc3"
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2),
                                        connectionstyle=conn, arrowstyle='->',
                                        mutation_scale=14, color=color, linewidth=lw, zorder=4)
        ax.add_patch(arrow)

    # Title
    ax.text(50, 39, "MolGAN-VQP Architecture Pipeline", ha='center', va='center',
            fontsize=17, fontweight='bold', color='#0F172A')

    # ROW 1: GENERATION LOOP (Left to Right)
    # 1. State
    draw_node(3, 27, 14, 5.5, "Active Subgraph\n(X_t, A_t)", c_gray, 9.5)

    # 2. R-GCN Encoder
    draw_node(20, 27, 16, 5.5, "Relational GCN\nEncoder", c_blue, 10.5)
    draw_arrow(17, 29.75, 20, 29.75, c_blue['border'])

    # 3. Quantum Policy
    draw_node(39, 27, 18, 5.5, "Dual 4-Qubit\nSoftmax-PQCs", c_purple, 10.5)
    draw_arrow(36, 29.75, 39, 29.75, c_purple['border'])

    # 4. Valence Mask
    draw_node(60, 27, 17, 5.5, "Chemical Valence\nMasking Engine", c_green, 10.5)
    draw_arrow(57, 29.75, 60, 29.75, c_green['border'])

    # 5. Generated Molecule
    draw_node(80, 27, 17, 5.5, "Generated\nMolecule G", c_gray, 10.5)
    draw_arrow(77, 29.75, 80, 29.75, '#64748B')

    # ROW 2: MULTI-OBJECTIVE EVALUATION (Branching Downwards)
    draw_arrow(88.5, 27, 88.5, 21.5, '#64748B')

    # Evaluator 1: Discriminator
    draw_node(78, 16, 19, 5, "Relational GCN\nDiscriminator", c_red, 10)

    # Evaluator 2: Quantum Evaluator
    draw_node(55, 16, 19, 5, "4-Qubit Quantum\nPhysChem Evaluator", c_purple, 10)

    # Evaluator 3: Critic
    draw_node(32, 16, 19, 5, "Relational GCN\nValue Critic", c_amber, 10)

    draw_arrow(88.5, 16, 88.5, 11, c_red['border'])
    draw_arrow(80, 27, 64.5, 21, c_purple['border'], rad=-0.1)
    draw_arrow(80, 27, 41.5, 21, c_amber['border'], rad=-0.15)

    # ROW 3: PPO REINFORCEMENT LEARNING ENGINE (Bottom Right to Left)
    # Composite Reward Engine
    draw_node(67, 5.5, 22, 5.5, "Composite Reward Engine\n(Realism + R_Q + Diversity)", c_amber, 9.5)
    draw_arrow(64.5, 16, 75, 11, c_purple['border'])
    draw_arrow(85, 11, 78, 11, c_red['border'])

    # Batch Advantage Engine
    draw_node(39, 5.5, 22, 5.5, "Batch Advantage Engine\n(A = R - V_phi)", c_teal, 10)
    draw_arrow(67, 8.25, 61, 8.25, c_amber['border'])
    draw_arrow(41.5, 16, 47, 11, c_amber['border'])

    # PPO Policy Update
    draw_node(10, 5.5, 23, 5.5, "Two-Pass Clipped PPO\nOptimization Engine", c_green, 10)
    draw_arrow(39, 8.25, 33, 8.25, c_teal['border'])

    # Feedback loop: PPO back to Dual PQCs
    draw_arrow(21.5, 11, 48, 27, c_green['border'], rad=0.25, lw=2.2)
    ax.text(28, 20, "Policy Update (theta)", fontsize=9, fontweight='bold',
            color=c_green['text'], bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor=c_green['border'], lw=1))

    # Feedback loop: Critic Update
    draw_arrow(21.5, 11, 38, 16, c_amber['border'], rad=0.1, lw=1.8)

    plt.tight_layout()
    out_file = "molgan_vqp_pipeline_names_only.png"
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Names-only pipeline saved to {out_file}")

if __name__ == "__main__":
    create_names_only_pipeline()
