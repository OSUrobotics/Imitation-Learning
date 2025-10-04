import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

SMOOTH_SIGMA_SEC = 0.05
PRE_ENGAGE_PADDING_BINS = 2

def _gaussian_kernel(sigma_sec: float, dt_sec: float, radius: float = 3.0) -> np.ndarray:
    """Return normalized 1D Gaussian kernel; radius≈3σ covers ~99% mass."""
    if sigma_sec <= 0:
        return np.array([1.0], dtype=float)
    sigma_samples = max(sigma_sec / max(dt_sec, 1e-9), 1e-6)
    half = int(max(1, round(radius * sigma_samples)))
    x = np.arange(-half, half + 1, dtype=float)
    w = np.exp(-(x**2) / (2.0 * sigma_samples**2))
    return w / w.sum()

def _smooth_columns(df: pd.DataFrame, cols: list[str], sigma_sec: float, dt_sec: float) -> pd.DataFrame:
    """Convolve numeric columns with a Gaussian kernel; do not create values where original was NaN."""
    if sigma_sec <= 0 or not cols:
        return df
    ker = _gaussian_kernel(sigma_sec, dt_sec)
    out = df.copy()
    for c in cols:
        s = out[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        valid = s.notna().values
        if valid.sum() == 0:
            continue
        v = s.interpolate(limit_direction="both").values
        sm = np.convolve(v, ker, mode="same")
        sm[~valid] = np.nan  # keep leading/trailing NaNs
        out[c] = sm
    return out


def _normalize_joint_states_20hz(df_20hz: pd.DataFrame) -> pd.DataFrame:
    out = df_20hz.copy()
    pos_cols = [c for c in out.columns if c.startswith("position_")]
    vel_cols = [c for c in out.columns if c.startswith("velocity_")]
    eff_cols = [c for c in out.columns if c.startswith("effort_")]

    if pos_cols:
        X = out[pos_cols].to_numpy(float)
        out[[f'{c}_sin' for c in pos_cols]] = np.sin(X)
        out[[f'{c}_cos' for c in pos_cols]] = np.cos(X)
        # out.drop(columns=pos_cols, inplace=True)  # if you don’t want raw angles


    if vel_cols:
        V = out[vel_cols].astype(float).to_numpy()
        V = np.clip(V / 6.0, -1.0, 1.0)
        out[vel_cols] = V

    if eff_cols:
        E = out[eff_cols].astype(float).to_numpy()
        E = np.clip(E / 300.0, -1.0, 1.0)
        out[eff_cols] = E

    return out

def _pre_and_engaged_from_raw(
    df_p: pd.DataFrame,
    threshold: float,
    pre_rows: int,
    rate_hz: float = 20.0,
) -> tuple[pd.DataFrame, float | None]:
    """
    Pick raw rows so that, after 20 Hz nearest-bin snapping, you get
    EXACTLY `pre_rows` distinct pre-engaged bins immediately before
    the first engaged bin.

    Returns:
      df_out: pre + engaged rows
      first_engaged_t: raw t_sec of first engaged (or None)
    """
    pressure_cols = ["data_0", "data_1", "data_2"]
    if not set(pressure_cols).issubset(df_p.columns) or df_p.empty:
        return df_p.iloc[0:0].copy(), None

    df_p = df_p.sort_values("t_sec").reset_index(drop=True)
    engaged_mask = df_p[pressure_cols].le(threshold).any(axis=1).to_numpy()
    if not engaged_mask.any():
        return df_p.iloc[0:0].copy(), None

    dt = 1.0 / float(rate_hz)
    first_idx = int(np.flatnonzero(engaged_mask)[0])
    t_eng = float(df_p.loc[first_idx, "t_sec"])

    # Walk backward and only accept a raw row if,
    # under the *current* earliest-pre-time (future t_start),
    # its rounded 20 Hz bin index is (a) < engaged bin and (b) new.
    selected = []          # raw indices for pre rows (closest → farthest)
    selected_times = []    # their times (for quick min() and bin recompute)

    def ok_if_added(idx: int) -> bool:
        times = selected_times + [float(df_p.loc[idx, "t_sec"])]
        t_start = min(times)               # will become ROI start
        k_eng = int(np.rint((t_eng - t_start) / dt))
        k_list = [int(np.rint((t - t_start) / dt)) for t in times]
        # all pre bins must be strictly before engaged, and unique
        return all(k < k_eng for k in k_list) and (len(set(k_list)) == len(k_list))

    for j in range(first_idx - 1, -1, -1):  # nearest → farthest
        if len(selected) >= int(max(0, pre_rows)):
            break
        if ok_if_added(j):
            selected.append(j)
            selected_times.append(float(df_p.loc[j, "t_sec"]))

    # Build masks
    pre_mask = np.zeros(len(df_p), dtype=bool)
    if selected:
        pre_mask[selected] = True

    keep_mask = pre_mask | engaged_mask
    df_out = df_p.loc[keep_mask].copy()
    return df_out, t_eng



class FilterDataPipeline:
    def __init__(
        self,
        pressure_csv_path: str,
        joint_states_csv_path: str,
        wrench_csv_path: str,
        twist_csv_path: str,
        engagement_threshold: float = 600.0,
        trial_id=None,
        pre_engage_padding_bins: int = PRE_ENGAGE_PADDING_BINS,
        normalize_joints: bool = True,
    ):
        self.pressure_csv_path = pressure_csv_path
        self.joint_states_csv_path = joint_states_csv_path
        self.wrench_csv_path = wrench_csv_path
        self.twist_csv_path = twist_csv_path
        self.engagement_threshold = engagement_threshold

        self.t_start = None
        self.t_end = None

        self.out_paths = {
            "pressure": "kept_pressure_windows.csv",
            "joint_states": "kept_joint_states_windows.csv",
            "wrench": "kept_force_torque_sensor_broadcaster_wrench_windows.csv",
            "twist": "kept_servo_node_delta_twist_cmds_windows.csv",
        }
        self.trial_id = trial_id or os.path.basename(os.path.dirname(pressure_csv_path))

        self.pre_engage_padding_bins = int(max(0, pre_engage_padding_bins))
        self.normalize_joints = bool(normalize_joints)
        self.first_engaged_20hz_tsec = None


    # --- helper: append then sort by t_sec (and drop exact duplicate rows)
    def _append_sorted(self, out_path: str, df_new: pd.DataFrame):
        if df_new.empty:
            if not os.path.exists(out_path):
                df_new.to_csv(out_path, index=False)
            print("Wrote (no new rows):", out_path)
            return out_path

        df_new = df_new.copy()
        if "trial_id" not in df_new.columns:
            df_new["trial_id"] = self.trial_id

        if "t_sec" in df_new.columns:
            df_new["t_sec"] = np.round(df_new["t_sec"].astype(float), 3)  # round to 1 ms for stability

        if os.path.exists(out_path):
            df_old = pd.read_csv(out_path)
        else:
            df_old = pd.DataFrame(columns=df_new.columns)

        df_all = pd.concat([df_old, df_new], ignore_index=True, sort=False)

        # ensure cols exist even if earlier files lacked them
        for c in df_new.columns:
            if c not in df_all.columns:
                df_all[c] = np.nan

        df_all = (
            df_all
            .sort_values(["trial_id", "t_sec"])
            .drop_duplicates(subset=["trial_id", "t_sec"], keep="last")
            .reset_index(drop=True)
        )
        df_all.to_csv(out_path, index=False)
        print("Appended:", out_path)
        return out_path




    # --- helper: downsample a kept_* CSV to 20 Hz via per-bin mean, using the ROI [t_start, t_end]
    def _downsample_to_20hz(
        self,
        kept_csv_path: str,
        out_csv_path: str,
        t_start: float,
        t_end: float,
        rate_hz: float = 20.0,
    ):
        if t_start is None or t_end is None:
            pd.DataFrame().to_csv(out_csv_path, index=False); print("Wrote (empty 20Hz):", out_csv_path); return out_csv_path

        df = pd.read_csv(kept_csv_path)
        if df.empty:
            df.to_csv(out_csv_path, index=False); print("Wrote (empty 20Hz):", out_csv_path); return out_csv_path
        

        # restrict to this trial only
        if "trial_id" in df.columns:
            df = df[df["trial_id"] == self.trial_id]

        # --- keep: basic bounds for pressure, if present
        VALID_MIN, VALID_MAX = 0.0, 1100.0
        if {"data_0","data_1","data_2"}.issubset(df.columns):
            for c in ("data_0","data_1","data_2"):
                df.loc[(df[c] < VALID_MIN) | (df[c] > VALID_MAX), c] = np.nan

        df = df.sort_values("t_sec").reset_index(drop=True)

        # --- 20 Hz grid
        dt = 1.0 / rate_hz
        n_bins = int(np.floor((t_end - t_start) / dt)) + 1
        grid = pd.DataFrame({"__bin": np.arange(n_bins)})
        grid["t_sec"] = np.round(t_start + grid["__bin"] * dt, 3)  # snap to 3 decimals (50 ms)

        # --- nearest-bin assignment (fixes left bias)
        binf = (df["t_sec"] - t_start) / dt
        df["__bin"] = np.rint(binf).astype("int64")
        df = df[(df["__bin"] >= 0) & (df["__bin"] < n_bins)].copy()

        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in ("__bin","t_sec")]

        # robust aggregate but KEEP singletons; only empty bins stay NaN
        grouped = df.groupby("__bin")[num_cols].median().reset_index()
        out = grid.merge(grouped, on="__bin", how="left").drop(columns="__bin")
        out = out[["t_sec"] + [c for c in out.columns if c != "t_sec"]]

        # time-aware small-gap interpolation only (no extrapolation)
        df_20hz = out.set_index("t_sec")
        if len(num_cols):
            imputed_mask = df_20hz[num_cols].isna().any(axis=1)
            df_20hz[num_cols] = df_20hz[num_cols].interpolate(method="index", limit_area="inside")

            out_interp = df_20hz.reset_index()
            # mark any row that required interpolation 
            out_interp["imputed"] = (imputed_mask).astype(int).values

        else:
            out_interp = df_20hz.reset_index(); out_interp["imputed"] = 0

        out_interp["trial_id"] = self.trial_id


        # Gaussian smoothing
        if SMOOTH_SIGMA_SEC > 0:
            smooth_cols = [c for c in num_cols if c in out_interp.columns]
            out_interp = _smooth_columns(out_interp, smooth_cols, SMOOTH_SIGMA_SEC, dt)
            out_interp["smoothed_sigma_s"] = SMOOTH_SIGMA_SEC


        # normalize joint states in the 20 Hz output only
        if self.normalize_joints and os.path.basename(kept_csv_path).startswith("kept_joint_states_windows"):
            out_interp = _normalize_joint_states_20hz(out_interp)


        # --- enforce at most N pre-engaged bins in the 20 Hz outputs ---
        # For pressure: define the 20 Hz "first engaged bin" by thresholding.
        is_pressure_20hz = os.path.basename(kept_csv_path).startswith("kept_pressure_windows")
        if is_pressure_20hz and {"data_0","data_1","data_2"}.issubset(out_interp.columns):
            all_nan = out_interp[["data_0","data_1","data_2"]].isna().all(axis=1)
            out_interp = out_interp[~all_nan].reset_index(drop=True)
            engaged20 = out_interp[["data_0","data_1","data_2"]].le(self.engagement_threshold).any(axis=1)
            if engaged20.any():
                first_idx_20 = int(np.flatnonzero(engaged20)[0])
                start_idx_20 = max(0, first_idx_20 - self.pre_engage_padding_bins)
                # keep N pre-bins (unconditionally)
                pre_df  = out_interp.iloc[start_idx_20:first_idx_20]

                # keep only engaged from the onset forward
                post_df = out_interp.iloc[first_idx_20:]
                post_df = post_df[engaged20.iloc[first_idx_20:].values]

                out_interp = pd.concat([pre_df, post_df], ignore_index=True)

                # first engaged t in the *new* frame = first row after the N pre-bins
                self.first_engaged_20hz_tsec = float(out_interp.iloc[len(pre_df)]["t_sec"])
            else:
                self.first_engaged_20hz_tsec = None

        # For all other streams: align to the pressure's first engaged bin and allow only N pre bins.
        is_other_stream_20hz = not is_pressure_20hz
        if is_other_stream_20hz and self.first_engaged_20hz_tsec is not None and "t_sec" in out_interp.columns:
            # find the first index whose t_sec >= pressure first-engaged time
            idx = int(np.searchsorted(out_interp["t_sec"].to_numpy(), self.first_engaged_20hz_tsec, side="left"))
            start_idx_20 = max(0, idx - self.pre_engage_padding_bins)
            out_interp = out_interp.iloc[start_idx_20:].reset_index(drop=True)


        # final clean sort for append semantics
        out_interp = out_interp.sort_values(["trial_id","t_sec"]).reset_index(drop=True)


        # Write interpolated, masked result
        self._append_sorted(out_csv_path, out_interp)
        print("Wrote 20Hz:", out_csv_path)
        return out_csv_path

    # helper: filter any CSV by time, then append+sort (uses self.t_start/self.t_end)
    def _filter_by_time(self, csv_path: str, out_path: str):
        df = pd.read_csv(csv_path)
        df["trial_id"] = self.trial_id
        if self.t_start is None or self.t_end is None:
            df_out = df.iloc[0:0].copy()  # empty
        else:
            mask = df["t_sec"].between(self.t_start, self.t_end, inclusive="both")
            df_out = df.loc[mask].copy()
        return self._append_sorted(out_path, df_out)

    def run(self):
        # ---- 1) PRESSURE: keep N pre rows (raw) + all engaged ----
        df_p = pd.read_csv(self.pressure_csv_path)
        df_p["trial_id"] = self.trial_id

        df_p_out, first_engaged_t_raw = _pre_and_engaged_from_raw(
            df_p,
            threshold=self.engagement_threshold,
            pre_rows=self.pre_engage_padding_bins,
            rate_hz=20.0,  # must match your downsample rate
        )


        self._append_sorted(self.out_paths["pressure"], df_p_out)

        # time ROI from the kept pressure rows (includes the pre rows we just added)
        if df_p_out.empty:
            print("No engaged pressure rows found; downstream filters will be empty.")
            self.t_start = self.t_end = None
        else:
            self.t_start = float(df_p_out["t_sec"].min())
            self.t_end   = float(df_p_out["t_sec"].max())


        # ---- 2) Apply the auto time-ROI to the other logs ----
        self._filter_by_time(self.joint_states_csv_path, self.out_paths["joint_states"])
        self._filter_by_time(self.wrench_csv_path, self.out_paths["wrench"])
        self._filter_by_time(self.twist_csv_path, self.out_paths["twist"])

        # ---- 3) Downsample all kept_* to a common 20 Hz grid (per-bin mean) ----
        self._downsample_to_20hz(
            self.out_paths["pressure"],
            self.out_paths["pressure"].replace(".csv", "_20hz.csv"),
            self.t_start, self.t_end, rate_hz=20.0
        )
        self._downsample_to_20hz(
            self.out_paths["joint_states"],
            self.out_paths["joint_states"].replace(".csv", "_20hz.csv"),
            self.t_start, self.t_end, rate_hz=20.0
        )
        self._downsample_to_20hz(
            self.out_paths["wrench"],
            self.out_paths["wrench"].replace(".csv", "_20hz.csv"),
            self.t_start, self.t_end, rate_hz=20.0
        )
        self._downsample_to_20hz(
            self.out_paths["twist"],
            self.out_paths["twist"].replace(".csv", "_20hz.csv"),
            self.t_start, self.t_end, rate_hz=20.0
        )


        return self.out_paths


# ------ Public API kept EXACTLY the same ------
def filter_data(
    pressure_csv_path: str,
    joint_states_csv_path: str,
    wrench_csv_path: str,
    twist_csv_path: str,
    engagement_threshold: float = 600.0,
    trial_id=None,
):
    """
    Wrapper preserved for compatibility; internally uses FilterDataPipeline.
    """
    pipeline = FilterDataPipeline(
        pressure_csv_path,
        joint_states_csv_path,
        wrench_csv_path,
        twist_csv_path,
        engagement_threshold,
        trial_id=trial_id,
    )
    return pipeline.run()


if __name__ == '__main__':
    REQ = {
        "pressure": "_gripper_pressure.csv",
        "joint_states": "_joint_states.csv",
        "wrench": "_force_torque_sensor_broadcaster_wrench.csv",
        "twist": "_servo_node_delta_twist_cmds.csv",
    }

    def find_trials(root: Path):
        """
        Yield dicts with the 4 csv paths for every export folder under `root`
        that contains all required files.
        We anchor on _gripper_pressure.csv to avoid false positives.
        """
        for p in root.rglob(REQ["pressure"]):   
            export_dir = p.parent
            files = {k: export_dir / fname for k, fname in REQ.items()}
            if all(f.exists() for f in files.values()):
                files["trial_id"] = export_dir.name  
                yield files

    parser = argparse.ArgumentParser(
        description="Batch process all export folders under one or more roots."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=[str(Path.home() / "Documents")],
        help="One or more root directories to scan (default: ~/Documents)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List candidate trials without running the pipeline."
    )
    args = parser.parse_args()

    total = 0
    for root_str in args.roots:
        root = Path(root_str).expanduser().resolve()
        print(f"\n[scan] {root}")
        for files in find_trials(root):
            trial_id = files["trial_id"]
            if args.dry_run:
                print(f"  - {trial_id}")
                continue

            try:
                filter_data(
                    str(files["pressure"]),
                    str(files["joint_states"]),
                    str(files["wrench"]),
                    str(files["twist"]),
                    trial_id=trial_id,                 
                )
                total += 1
            except Exception as e:
                print(f"[ERROR] trial {trial_id}: {e}")

    if not args.dry_run:
        print(f"\nDone. Processed {total} trials.")
