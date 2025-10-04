# make_pipeline_flowchart.py
import textwrap
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 100,
})

# ---------- helpers ----------
def wrap_text(s: str, max_chars: int = 18) -> str:
    """Hard-wrap label text so it fits inside boxes."""
    return "\n".join(textwrap.fill(line, width=max_chars) for line in s.split("\n"))

def node(ax, xy, text, w=4.4, h=1.35, max_chars=18, fc="#ffffff"):
    """
    Draw a rounded box with centered wrapped text.
    Returns a dict with geometry for connectors.
    """
    x, y = xy
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.10,rounding_size=0.16",
        linewidth=1.2, edgecolor="black", facecolor=fc, zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, wrap_text(text, max_chars=max_chars),
            ha="center", va="center", zorder=3)
    return {"x": x, "y": y, "w": w, "h": h}

def arrow(ax, p1, p2):
    """Straight arrow with consistent styling (drawn beneath boxes)."""
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="->", linewidth=1.2,
        mutation_scale=12, shrinkA=0, shrinkB=0,
        color="black", zorder=1
    ))

def connect_lr(ax, A, B, gap=0.22):
    """Left→Right connector: right edge of A → left edge of B (forces L→R)."""
    start = (A["x"] + A["w"]/2 + gap, A["y"])
    end   = (B["x"] - B["w"]/2 - gap, B["y"])
    if end[0] <= start[0]:
        start, end = end, start
    arrow(ax, start, end)

def connect_down(ax, A, B, gap=0.22):
    """Vertical connector: bottom of A → top of B (forces downward)."""
    start = (A["x"], A["y"] - A["h"]/2 - gap)
    end   = (B["x"], B["y"] + B["h"]/2 + gap)
    if end[1] >= start[1]:
        start, end = end, start
    arrow(ax, start, end)

# ---------- canvas ----------
# Wider to accommodate extra nodes
fig, ax = plt.subplots(figsize=(20.0, 6.0))
ax.axis("off")

# Layout parameters (kept style; just extended)
w_top = 4.4
w_bot = 4.4
h_box = 1.35
step  = w_top + 0.9     # horizontal spacing so arrows don't overlap boxes
x0    = 1.6
y_top = 4.0
y_bot = 1.8

# Top-row positions (centers) — now 8 nodes
x = [x0 + i*step for i in range(8)]

# ---------- nodes ----------
# Top row (original + inserted steps)
n1 = node(ax, (x[0], y_top),
          "Raw CSVs\n(pressure, joint states,\nforce/torque, twist)", w=w_top, h=h_box)
n2 = node(ax, (x[1], y_top),
          "Pressure-ROI filter\n[t_min, t_max]", w=w_top, h=h_box)

# NEW: sort & de-dup per episode before binning
n3 = node(ax, (x[2], y_top),
          "Sort & de-dup\n(trial_id, t_sec)", w=w_top, h=h_box)

n4 = node(ax, (x[3], y_top),
          "20 Hz binning\n(per-bin MEDIAN)", w=w_top, h=h_box)
n5 = node(ax, (x[4], y_top),
          "Sanitize / clip\n(bounds per stream)", w=w_top, h=h_box)
n6 = node(ax, (x[5], y_top),
          "Time-align streams\n(synchronize)", w=w_top, h=h_box)

# NEW: explicit interpolation step
n7 = node(ax, (x[6], y_top),
          "Interpolate inside gaps\n(index-based; no ends)", w=w_top, h=h_box)

n8 = node(ax, (x[7], y_top),
          "Mask missing\n(no zero fill)", w=w_top, h=h_box)

# Bottom row (to the RIGHT of top row to keep flow L→R)
dx = w_bot + 0.9
n9  = node(ax, (x[7] + 0.0,   y_bot), "Imputed flag\n(per-tick)", w=w_bot, h=h_box)

# NEW: optional Gaussian smoothing (figures only)
n10 = node(ax, (x[7] + dx,    y_bot), "1-D Gaussian smoothing (σ=0.05,s, radius 3σ)", w=w_bot, h=h_box)

n11 = node(ax, (x[7] + 2*dx,  y_bot), "Normalize\n(train-split stats)", w=w_bot, h=h_box)
n12 = node(ax, (x[7] + 3*dx,  y_bot), "Model input\nW×D tensor", w=w_bot, h=h_box)

# ---------- connectors (arrow out of EVERY box except final sink) ----------
# Top row L→R
connect_lr(ax, n1, n2)
connect_lr(ax, n2, n3)   # new connection
connect_lr(ax, n3, n4)
connect_lr(ax, n4, n5)
connect_lr(ax, n5, n6)
connect_lr(ax, n6, n7)   # new connection
connect_lr(ax, n7, n8)

# Drop down and continue L→R
connect_down(ax, n8, n9)
connect_lr(ax, n9, n10)  # new smoothing step
connect_lr(ax, n10, n11)
connect_lr(ax, n11, n12)

# View limits (auto-fit all boxes + margins)
nodes = [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12]
xmin = min(n["x"] - n["w"]/2 for n in nodes) - 0.8
xmax = max(n["x"] + n["w"]/2 for n in nodes) + 0.8
ymin = min(n["y"] - n["h"]/2 for n in nodes) - 0.8
ymax = max(n["y"] + n["h"]/2 for n in nodes) + 0.8
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
out = "pipeline_flowchart.png"
plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {out}")
