# visualize_gripper_pressure_csv.py
# Plot 3-cup pressures vs time for a SINGLE trial_id to avoid over-plotting.

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

# --- readability tweaks ---
mpl.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 6.5,
})


CSV_PATH   = Path("/home/aryan/IL_Workspace/srl_il/kept_pressure_windows_20hz.csv")  # or *_20hz.csv
TRIAL_ID   = None                 # e.g., "pressure_servo_20250730_123537.db3_export"; if None, first trial is used
XLIM       = (3.5, 14.0)           # zoom range in seconds (or set to None)
THRESH_HPA = 600.0                # 600 hPa == 60 kPa
# ========================

df = pd.read_csv(CSV_PATH)

# --- pick time column ---
time_col = next((c for c in ["t_sec", "time", "t", "sec"] if c in df.columns), None)
if time_col is None:
    df["t_sec"] = (df.index - df.index.min()) * 1.0
    time_col = "t_sec"

# --- pick pressure columns (data_0/1/2 expected) ---
press_cols = [c for c in df.columns if c.startswith("data_")]
if not press_cols:
    press_cols = [c for c in df.select_dtypes("number").columns if c not in {time_col, "imputed"}]

# coerce numeric
df[press_cols] = df[press_cols].apply(pd.to_numeric, errors="coerce")

# --- select ONE trial to plot if trial_id exists ---
if "trial_id" in df.columns:
    trials = list(dict.fromkeys(df["trial_id"].astype(str)))  # preserve order, unique
    print(f"Numbers of trials: {len(trials)}")
    if TRIAL_ID is None:
        TRIAL_ID = trials[0]
        print(f"[info] trial_id not provided; using first trial: {TRIAL_ID}")
    elif TRIAL_ID not in trials:
        raise ValueError(f"trial_id={TRIAL_ID!r} not found. Available: {trials[:10]}{' ...' if len(trials)>10 else ''}")
    df = df[df["trial_id"].astype(str) == str(TRIAL_ID)].copy()
    trial_idx = trials.index(str(TRIAL_ID)) + 1
    title_suffix = f" — trial {trial_idx}"
    trial_num_suffix = f"_trial{trial_idx}"
else:
    title_suffix = ""

# sort by time inside this trial
df = df.sort_values(time_col).reset_index(drop=True)

# --- legend renaming ---
legend_name = {
    "data_0": "Suction Cup A",
    "data_1": "Suction Cup B",
    "data_2": "Suction Cup C",
}
series_labels = [legend_name.get(c, c) for c in press_cols]

# --- plotting ---
fig, ax = plt.subplots(figsize=(9, 5.4))

# minor grid every 50 ms (20 Hz)
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.grid(which="minor", axis="x", alpha=0.15, linewidth=0.5)

has_imputed = "imputed" in df.columns and df["imputed"].any()

# plot each cup as a line
for c, lbl in zip(press_cols, series_labels):
    line, = ax.plot(df[time_col], df[c], label=lbl)
    # Only add hollow markers if this CSV has imputed==1 flags (i.e., 20 Hz file)
    if has_imputed:
        imp = df["imputed"].eq(1)
        ax.scatter(df.loc[imp, time_col], df.loc[imp, c],
                   s=28, marker="o", facecolors="none",
                   edgecolors=line.get_color(), linewidths=0.8, zorder=3)

# Legend entry for “interpolated gaps” only when applicable
if has_imputed:
    edge = ax.lines[0].get_color() if ax.lines else "k"
    ax.scatter([], [], s=28, marker="o", facecolors="none",
               edgecolors=edge, linewidths=0.8, label="Interpolated gap (imputed)")

# threshold line
ax.axhline(THRESH_HPA, linestyle="--", linewidth=1, label="600 hPa threshold")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Pressure (hPa)")
ax.set_title(f"Cup pressures vs time{title_suffix}")

# if XLIM is not None:
#     ax.set_xlim(XLIM)

ax.legend(loc="best")
plt.tight_layout()
suffix = f"_{TRIAL_ID}" if "trial_id" in df.columns else ""
stem = CSV_PATH.stem.lower()
out_name = (
    f"kept_zoom{suffix}.png"
    if not stem.endswith("_20hz")           
    else f"pressure_20hz_zoom{suffix}.png"  
)
out_path = CSV_PATH.with_name(out_name)

out_path = CSV_PATH.with_name(out_name)
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f"Saved plot to: {out_path}")
plt.show()
