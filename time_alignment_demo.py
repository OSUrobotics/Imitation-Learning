# make_time_alignment_demo_simple.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "lines.linewidth": 1.2,
})

# ---- shared 20 Hz grid (zoomed window) ----
t0, t1, dt = 4.20, 4.60, 0.05
grid = np.arange(t0, t1 + 1e-12, dt)
hi = 4                      # highlight this bin
hi_t = grid[hi]

# ---- toy raw timestamps (jitter around grid) ----
rng = np.random.default_rng(3)
pressure = grid + rng.normal(0, 0.011, size=len(grid))          # dense
twist    = grid[::2] + rng.normal(0, 0.015, size=len(grid[::2]))# half-rate
wrench   = grid[1::3] + rng.normal(0, 0.020, size=len(grid[1::3])) # sparse

def nearest_bin_indices(t_raw, grid, dt):
    idx = np.round((t_raw - grid[0]) / dt).astype(int)
    valid = (idx >= 0) & (idx < len(grid))
    return idx[valid], t_raw[valid]

p_idx, p_raw = nearest_bin_indices(pressure, grid, dt)
t_idx, t_raw = nearest_bin_indices(twist,    grid, dt)
w_idx, w_raw = nearest_bin_indices(wrench,   grid, dt)

# ---- figure ----
fig, ax = plt.subplots(figsize=(10.5, 3.8))
fig.subplots_adjust(top=0.88)  # prevent title clipping
ax.set_title("Alignment by rounding to the shared 20 Hz grid (per-bin median)")

# rows
rows = [("pressure", 3, p_idx, p_raw),
        ("twist cmd",2, t_idx, t_raw),
        ("wrench",   1, w_idx, w_raw)]

# light grid
for g in grid:
    ax.axvline(g, color="k", linestyle="--", alpha=0.12, linewidth=0.9)

# highlight one bin
ax.axvline(hi_t, color="k", linestyle="--", alpha=0.6)
ax.text(hi_t, 3.35, f"grid tick {hi_t:.2f}s", ha="center", va="bottom")

def draw_row(name, y, idx, raw):
    # raw stems + dots
    for ti in raw:
        ax.vlines(ti, y-0.20, y+0.20, alpha=0.45)
    ax.scatter(raw, np.full_like(raw, y), s=26, label=None, zorder=3)

    # hits in the highlighted bin
    hits = raw[idx == hi]
    if hits.size > 0:
        # arrows from hits → grid tick
        for ti in hits:
            ax.annotate("",
                        xy=(hi_t, y+0.26), xytext=(ti, y+0.26),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, alpha=0.9))
        # “assigned to bin” marker (represents per-bin aggregation -> median)
        ax.scatter([hi_t], [y+0.26], s=52, marker="s",
                   facecolor="none", edgecolor="k", zorder=5)
    else:
        # empty bin → NaN (later filled only by interior interpolation)
        ax.scatter([hi_t], [y+0.26], s=52, marker="x",
                   color="crimson", zorder=5)
        ax.text(hi_t, y+0.48, "empty → NaN", color="crimson",
                ha="center", va="bottom", fontsize=9)

    # row label
    ax.text(grid[0]-0.012, y, name, ha="right", va="center")

for name, y, idx, raw in rows:
    draw_row(name, y, idx, raw)

# cosmetics
ax.set_xlim(t0-0.01, t1+0.01)
ax.set_ylim(0.7, 3.6)
ax.set_yticks([])  # row labels already drawn
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.grid(which="minor", axis="x", alpha=0.12, linewidth=0.6)
ax.set_xlabel("time (s)")

# minimal legend
legend_handles = [
    mpl.lines.Line2D([], [], linestyle="", marker="o", markersize=6, color="k", label="raw timestamps"),
    mpl.lines.Line2D([], [], linestyle="", marker="s", markersize=7, markerfacecolor="none",
                     markeredgecolor="k", label="assigned to grid (per-bin median)"),
    mpl.lines.Line2D([], [], linestyle="", marker="x", markersize=7, color="crimson", label="empty bin → NaN"),
]
ax.legend(handles=legend_handles, loc="upper right", frameon=False)

plt.tight_layout()
out = "time_alignment_demo_simple.png"
plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"Saved {out}")
