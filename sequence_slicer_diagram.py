# make_sequence_slicer_panels.py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

mpl.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})

def draw_window(ax, mask, title):
    """Draw a W-step window with a mask row under it."""
    W = len(mask)
    y0, h, w, gap, x0 = 0.6, 0.5, 1.0, 0.15, 0.3
    labels = [r"$t{+}%d$" % i if i>0 else r"$t$" for i in range(W)]

    # boxes + labels
    for i in range(W):
        xi = x0 + i*(w+gap)
        face = "white" if mask[i]==1 else "#e6e6e6"
        ax.add_patch(Rectangle((xi, y0), w, h, facecolor=face,
                               edgecolor="black", linewidth=1.2))
        ax.text(xi + w/2, y0 - 0.15, labels[i], ha="center", va="top", fontsize=9)
        ax.text(xi + w/2, y0 + h/2, "valid" if mask[i] else "pad",
                ha="center", va="center", fontsize=8, color="#333")

    # mask row
    for i in range(W):
        xi = x0 + i*(w+gap)
        ax.text(xi + w/2, 0.15, str(mask[i]), ha="center", va="center", fontsize=10)

    # annotations
    ax.text(x0 - 0.2, y0 + h/2, "window", ha="right", va="center", fontsize=10)
    ax.text(x0 - 0.2, 0.15,      "mask",   ha="right", va="center", fontsize=10)
    ax.text(x0 + W*(w+gap) - 0.2, y0 + h + 0.2,
            r"$W{=}5$ (0.25\,s @ 20\,Hz), stride $=1$",
            ha="right", fontsize=10)

    # mini legend
    ax.add_patch(Rectangle((x0 + 5.8, 0.05), 0.4, 0.22, facecolor="white",
                           edgecolor="black", linewidth=1.0))
    ax.text(x0 + 6.3, 0.16, "valid (mask=1)", va="center", fontsize=9)
    ax.add_patch(Rectangle((x0 + 7.9, 0.05), 0.4, 0.22, facecolor="#e6e6e6",
                           edgecolor="black", linewidth=1.0))
    ax.text(x0 + 8.4, 0.16, "pad (mask=0)",   va="center", fontsize=9)

    ax.set_title(title, pad=4, fontsize=10.5)
    ax.set_xlim(0, x0 + W*(w+gap) + 1.2)
    ax.set_ylim(0, 1.6)
    ax.axis("off")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

# (a) right-padded: [1,1,1,0,0]
fig1, ax1 = plt.subplots(figsize=(7.2, 2.2))
draw_window(ax1, [1,1,1,0,0], "Right edge padding")
save(fig1, "sequence_slicer_w5_rightpad")

# (b) left-padded: [0,0,1,1,1]
fig2, ax2 = plt.subplots(figsize=(7.2, 2.2))
draw_window(ax2, [0,0,1,1,1], "Left edge padding")
save(fig2, "sequence_slicer_w5_leftpad")

print("Wrote: sequence_slicer_w5_rightpad.(pdf|png), sequence_slicer_w5_leftpad.(pdf|png)")
