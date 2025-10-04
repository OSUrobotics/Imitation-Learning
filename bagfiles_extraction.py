#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, glob, heapq, sqlite3, tempfile, shutil, math, struct
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any
from datetime import timezone, datetime

# ========== SETTINGS ==========
VERBOSE = True
CHUNK_FETCH = 1000      # rows per fetch from SQLite
MERGE_BY_TIMESTAMP = True  # merge rows across db3 chunks by timestamp
# ==============================

# ---------- small utils ----------
def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.replace("/", "_"))

def ns_to_sec_rel(ts_ns: int, t0_ns: int) -> float:
    return (ts_ns - t0_ns) / 1e9

def export_dir_for(path_like: str) -> Path:
    p = Path(path_like)
    out = p.with_name(p.name + "_export")
    out.mkdir(parents=True, exist_ok=True)
    return out

# ---------- list *.db3 files ----------
def list_db_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() != ".db3":
            raise ValueError(f"Expected a .db3 file, got: {p}")
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(p)
    files = list(map(Path, glob.glob(str(p / "*.db3"))))
    if not files:
        raise FileNotFoundError(f"No .db3 files in: {p}")
    def key_fn(pp: Path):
        m = re.search(r"_(\d+)\.db3$", pp.name)
        return int(m.group(1)) if m else 0
    return sorted(files, key=key_fn)

# ---------- SQLite connect (read-only & robust) ----------
def connect_ro_file(db_file: Path) -> sqlite3.Connection:
    uri = f"file:{db_file.as_posix()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=30)

def connect_via_temp_copy(db_file: Path) -> sqlite3.Connection:
    tmpdir = Path(tempfile.mkdtemp(prefix="db3tmp_"))
    tmpdb = tmpdir / db_file.name
    shutil.copy2(db_file, tmpdb)
    for suf in ("-wal", "-shm"):
        side = Path(str(db_file) + suf)
        if side.exists():
            shutil.copy2(side, Path(str(tmpdb) + suf))
    return sqlite3.connect(tmpdb.as_posix(), timeout=30)

def safe_connect_file(db_file: Path) -> sqlite3.Connection:
    if not db_file.exists():
        raise FileNotFoundError(db_file)
    if not os.access(db_file.parent, os.X_OK):
        raise PermissionError(f"No execute permission on: {db_file.parent}")
    try:
        return connect_ro_file(db_file)
    except sqlite3.OperationalError as e:
        if "disk i/o" in str(e).lower() or "readonly" in str(e).lower():
            return connect_via_temp_copy(db_file)
        raise

# ---------- CDR reader (minimal) ----------
class CDRReader:
    """
    Minimal CDR reader for ROS 2 (Fast-CDR / XCDR1):
      - string, float32, float64, uint32, int32
      - sequences of those
    Assumes 4-byte encapsulation header; little-endian if rep-id is 0x0001 or 0x0003.
    """
    def __init__(self, data: bytes):
        if len(data) < 4:
            raise ValueError("CDR buffer too small")
        rep_id = int.from_bytes(data[0:2], "big", signed=False)
        self.le = rep_id in (0x0001, 0x0003)  # CDR_LE, PL_CDR_LE
        self.data = data
        self.o = 4  # skip encapsulation
        self._pref = "<" if self.le else ">"

    def bytes_left(self) -> int:
        return len(self.data) - self.o

    def _align(self, a: int):
        self.o += (-self.o) & (a - 1)

    def _unpack(self, fmt: str):
        s = struct.Struct(self._pref + fmt)
        v = s.unpack_from(self.data, self.o)
        self.o += s.size
        return v

    def read_uint32(self) -> int:
        self._align(4)
        (x,) = self._unpack("I")
        return x

    def read_int32(self) -> int:
        self._align(4)
        (x,) = self._unpack("i")
        return x

    def read_float32(self) -> float:
        self._align(4)
        (x,) = self._unpack("f")
        return float(x)

    def read_float64(self) -> float:
        self._align(8)
        (x,) = self._unpack("d")
        return float(x)

    def read_string(self) -> str:
        # XCDR1: length (u32, 4-aligned) INCLUDES the terminating NUL byte.
        self._align(4)
        (n,) = struct.Struct(self._pref + "I").unpack_from(self.data, self.o)
        self.o += 4
        if n < 0 or n > self.bytes_left():
            raise ValueError(f"string length invalid: {n} at {self.o}")
        raw = self.data[self.o : self.o + n]
        self.o += n
        # Trim the in-band NUL (no extra byte is consumed beyond 'n')
        if raw and raw[-1] == 0:
            raw = raw[:-1]
        return raw.decode("utf-8", errors="ignore")

    def read_seq_float32(self) -> List[float]:
        # align(4) count -> align(4) payload
        self._align(4)
        n = self.read_uint32()
        self._align(4)
        need = n * 4
        if need > self.bytes_left():
            raise ValueError(f"float32[] overruns buffer: need {need} from {self.o}, have {self.bytes_left()}")
        vals = list(struct.unpack_from(self._pref + f"{n}f", self.data, self.o))
        self.o += need
        return [float(v) for v in vals]

    def read_seq_float64(self) -> List[float]:
        """
        Robust reader for sequence<double> supporting:
            1) XCDR1:  align(4), count(u32), align(8), n*8 bytes
            2) XCDR1 with extra pad-to-8 BEFORE count (non-standard seen in logs)
            3) XCDR2 'delimited': align(4), DHEADER(u32=block_size), [inside block]:
                                    align(4), count(u32), align(8), n*8 bytes, then jump to block end
        """
        def try_xcdr1_at(offset: int) -> Optional[List[float]]:
            o0 = self.o
            self.o = offset
            self._align(4)
            try:
                (n,) = struct.Struct(self._pref + "I").unpack_from(self.data, self.o)
                self.o += 4
            except struct.error:
                self.o = o0
                return None
            self._align(8)
            need = n * 8
            if need < 0 or need > self.bytes_left():
                self.o = o0
                return None
            try:
                vals = list(struct.unpack_from(self._pref + f"{n}d", self.data, self.o))
            except struct.error:
                self.o = o0
                return None
            self.o += need
            return [float(v) for v in vals]

        def try_xcdr2_delimited_at(offset: int) -> Optional[List[float]]:
            o0 = self.o
            self.o = offset
            self._align(4)
            try:
                (block_size,) = struct.Struct(self._pref + "I").unpack_from(self.data, self.o)
            except struct.error:
                self.o = o0
                return None
            self.o += 4
            if block_size < 4 or block_size > self.bytes_left():
                self.o = o0
                return None
            block_start = self.o
            block_end   = block_start + block_size

            self._align(4)
            try:
                (n,) = struct.Struct(self._pref + "I").unpack_from(self.data, self.o)
            except struct.error:
                self.o = o0
                return None
            self.o += 4
            self._align(8)
            need = n * 8
            if need < 0 or self.o + need > block_end:
                self.o = o0
                return None
            try:
                vals = list(struct.unpack_from(self._pref + f"{n}d", self.data, self.o))
            except struct.error:
                self.o = o0
                return None
            self.o += need
            self.o = block_end
            return [float(v) for v in vals]

        base = self.o
        v = try_xcdr1_at(base)
        if v is not None:
            return v

        self.o = base
        self._align(8)
        v = try_xcdr1_at(self.o)
        if v is not None:
            return v

        v = try_xcdr2_delimited_at(base)
        if v is not None:
            return v

        self.o = base
        self._align(4)
        try:
            (raw_u32,) = struct.Struct(self._pref + "I").unpack_from(self.data, self.o)
        except struct.error:
            raw_u32 = None
        raise ValueError(
            f"float64[] overruns buffer: unable to parse at {base} "
            f"(bytes_left={len(self.data)-base}, peek_u32={raw_u32})"
        )

    def read_seq_string(self) -> List[str]:
        self._align(4)
        n = self.read_uint32()
        out = []
        for _ in range(n):
            out.append(self.read_string())
        return out

    # --- small helpers for speculative parsing ---
    def tell(self) -> int:
        return self.o

    def seek(self, off: int):
        if not (0 <= off <= len(self.data)):
            raise ValueError("seek out of range")
        self.o = off

    def peek_u32(self) -> Optional[int]:
        save = self.o
        try:
            self._align(4)
            (v,) = struct.unpack_from(self._pref + "I", self.data, self.o)
            return int(v)
        except Exception:
            return None
        finally:
            self.o = save


# ---------- adaptive numeric sequence readers ----------
def _try_seq_counted(r: CDRReader, dtype: str) -> Optional[List[float]]:
    """
    Attempt: align(4) read count, align(sizeof(dtype)), then read n elements.
    dtype: 'f32' or 'f64'
    """
    save = r.tell()
    try:
        r._align(4)
        n = r.read_uint32()
        if n > 1_000_000:
            raise ValueError("unreasonable length")
        if dtype == "f64":
            r._align(8)
            need = n * 8
            if need > r.bytes_left():
                raise ValueError("not enough bytes for f64 payload")
            vals = list(struct.unpack_from(r._pref + f"{n}d", r.data, r.o))
            r.seek(r.o + need)
            return [float(v) for v in vals]
        else:
            r._align(4)
            need = n * 4
            if need > r.bytes_left():
                raise ValueError("not enough bytes for f32 payload")
            vals = list(struct.unpack_from(r._pref + f"{n}f", r.data, r.o))
            r.seek(r.o + need)
            return [float(v) for v in vals]
    except Exception:
        r.seek(save)
        return None

def _try_seq_plain_known_len(r: CDRReader, dtype: str, want: int) -> Optional[List[float]]:
    """
    Attempt: no count; just aligned payload with exactly 'want' elements.
    """
    save = r.tell()
    try:
        if dtype == "f64":
            r._align(8)
            need = want * 8
            if need > r.bytes_left():
                raise ValueError("not enough bytes for plain f64")
            vals = list(struct.unpack_from(r._pref + f"{want}d", r.data, r.o))
            r.seek(r.o + need)
            return [float(v) for v in vals]
        else:
            r._align(4)
            need = want * 4
            if need > r.bytes_left():
                raise ValueError("not enough bytes for plain f32")
            vals = list(struct.unpack_from(r._pref + f"{want}f", r.data, r.o))
            r.seek(r.o + need)
            return [float(v) for v in vals]
    except Exception:
        r.seek(save)
        return None

def read_seq_numeric_smart(r: CDRReader, expect_len: int) -> List[float]:
    """
    Read a numeric array that might be encoded as:
      - counted float64 sequence (XCDR)
      - plain float64 array (no count)
      - counted float32 sequence
      - plain float32 array
    Returns [] if every attempt fails (so we can keep going).
    """
    vals = _try_seq_counted(r, "f64")
    if vals is not None:
        return vals
    vals = _try_seq_plain_known_len(r, "f64", expect_len)
    if vals is not None:
        return vals
    vals = _try_seq_counted(r, "f32")
    if vals is not None:
        return vals
    vals = _try_seq_plain_known_len(r, "f32", expect_len)
    if vals is not None:
        return vals
    return []

# ---------- type-specific decoders ----------
def decode_std_msgs__msg__String(buf: bytes) -> Dict[str, Any]:
    r = CDRReader(buf)
    s = r.read_string()
    try:
        val = float(s.strip())
    except Exception:
        val = None
    return {"value": val}

def decode_std_msgs__msg__Float32MultiArray(buf: bytes) -> Dict[str, Any]:
    r = CDRReader(buf)
    r._align(4)
    dim_n = r.read_uint32()
    for _ in range(dim_n):
        _label = r.read_string()
        _size  = r.read_uint32()
        _stride= r.read_uint32()
    _data_offset = r.read_uint32()  # not used
    vals = r.read_seq_float32()
    return {"data": vals}

def _read_header_skip(r: CDRReader):
    _sec = r.read_int32()
    _nsec = r.read_uint32()
    _frame_id = r.read_string()

def _read_vector3(r: CDRReader) -> Tuple[float, float, float]:
    x = r.read_float64()
    y = r.read_float64()
    z = r.read_float64()
    return x, y, z


def _read_vector3_f32(r: CDRReader) -> Tuple[float, float, float]:
    x = r.read_float32()
    y = r.read_float32()
    z = r.read_float32()
    return x, y, z


def decode_geometry_msgs__msg__TwistStamped(buf: bytes) -> Dict[str, Any]:
    r = CDRReader(buf)
    _read_header_skip(r)
    lx, ly, lz = _read_vector3(r)
    ax, ay, az = _read_vector3(r)
    return {
        "linear_x": lx, "linear_y": ly, "linear_z": lz,
        "angular_x": ax, "angular_y": ay, "angular_z": az,
    }

def decode_geometry_msgs__msg__WrenchStamped(buf: bytes) -> Dict[str, Any]:
    r = CDRReader(buf)
    _read_header_skip(r)
    fx, fy, fz = _read_vector3(r)
    tx, ty, tz = _read_vector3(r)
    return {
        "force_x": fx, "force_y": fy, "force_z": fz,
        "torque_x": tx, "torque_y": ty, "torque_z": tz,
    }

def decode_sensor_msgs__msg__JointState(buf: bytes) -> Dict[str, Any]:
    r = CDRReader(buf)
    _read_header_skip(r)
    names = r.read_seq_string()
    N = len(names)

    # --- light diagnostics: show state right after names (print up to 3 times) ---
    if VERBOSE:
        try:
            decode_sensor_msgs__msg__JointState._dbg_seen += 1
        except AttributeError:
            decode_sensor_msgs__msg__JointState._dbg_seen = 1
        if decode_sensor_msgs__msg__JointState._dbg_seen <= 3:
            print(f"[dbg] JointState after names: count={N}, bytes_left={r.bytes_left()}, peek_u32={r.peek_u32()}")

    # positions / velocities / effort with adaptive parsing
    pos = read_seq_numeric_smart(r, N)
    vel = read_seq_numeric_smart(r, N)
    eff = read_seq_numeric_smart(r, N)

    return {
        "_names": names,
        "_pos":   pos,
        "_vel":   vel,
        "_eff":   eff,
    }

def decode_twist_lenient(buf: bytes) -> Dict[str, Any]:
    """
    Lenient decoder for /servo_node/delta_twist_cmds:
      try TwistStamped f64 -> TwistStamped f32 -> Twist f64 -> Twist f32
    Returns the same flat keys as the normal TwistStamped decoder.
    """
    last_e: Optional[Exception] = None

    def build(lx, ly, lz, ax, ay, az) -> Dict[str, Any]:
        return {
            "linear_x": lx, "linear_y": ly, "linear_z": lz,
            "angular_x": ax, "angular_y": ay, "angular_z": az,
        }

    # 1) TwistStamped (float64)
    try:
        r = CDRReader(buf)
        _read_header_skip(r)
        lx, ly, lz = _read_vector3(r)
        ax, ay, az = _read_vector3(r)
        if VERBOSE:
            try: decode_twist_lenient._dbg += 1
            except AttributeError: decode_twist_lenient._dbg = 1
            if decode_twist_lenient._dbg <= 3:
                print("[dbg] /servo_node/delta_twist_cmds parsed as TwistStamped f64")
        return build(lx, ly, lz, ax, ay, az)
    except Exception as e:
        last_e = e

    # 2) TwistStamped (float32)
    try:
        r = CDRReader(buf)
        _read_header_skip(r)
        lx, ly, lz = _read_vector3_f32(r)
        ax, ay, az = _read_vector3_f32(r)
        if VERBOSE:
            try: decode_twist_lenient._dbg += 1
            except AttributeError: decode_twist_lenient._dbg = 1
            if decode_twist_lenient._dbg <= 3:
                print("[dbg] /servo_node/delta_twist_cmds parsed as TwistStamped f32")
        return build(lx, ly, lz, ax, ay, az)
    except Exception as e:
        last_e = e

    # 3) Twist (no header), float64
    try:
        r = CDRReader(buf)
        lx, ly, lz = _read_vector3(r)
        ax, ay, az = _read_vector3(r)
        if VERBOSE:
            try: decode_twist_lenient._dbg += 1
            except AttributeError: decode_twist_lenient._dbg = 1
            if decode_twist_lenient._dbg <= 3:
                print("[dbg] /servo_node/delta_twist_cmds parsed as Twist f64")
        return build(lx, ly, lz, ax, ay, az)
    except Exception as e:
        last_e = e

    # 4) Twist (no header), float32
    try:
        r = CDRReader(buf)
        lx, ly, lz = _read_vector3_f32(r)
        ax, ay, az = _read_vector3_f32(r)
        if VERBOSE:
            try: decode_twist_lenient._dbg += 1
            except AttributeError: decode_twist_lenient._dbg = 1
            if decode_twist_lenient._dbg <= 3:
                print("[dbg] /servo_node/delta_twist_cmds parsed as Twist f32")
        return build(lx, ly, lz, ax, ay, az)
    except Exception as e:
        last_e = e

    # If nothing worked, bubble the last error so your existing warning path prints it.
    raise last_e if last_e else RuntimeError("decode_twist_lenient: unknown failure")


# registry
DECODERS = {
    "std_msgs/msg/String": decode_std_msgs__msg__String,
    "std_msgs/msg/Float32MultiArray": decode_std_msgs__msg__Float32MultiArray,
    "geometry_msgs/msg/TwistStamped": decode_geometry_msgs__msg__TwistStamped,
    "geometry_msgs/msg/WrenchStamped": decode_geometry_msgs__msg__WrenchStamped,
    "sensor_msgs/msg/JointState": decode_sensor_msgs__msg__JointState,
}

# ---------- per-topic CSV writer ----------
class TopicCSV:
    def __init__(self, topic_name: str, ros_type: str, outdir: Path):
        self.topic = topic_name
        self.ros_type = ros_type
        self.outpath = outdir / f"{sanitize_filename(topic_name)}.csv"
        self.fh = self.outpath.open("w", newline="", encoding="utf-8")
        self.w = csv.writer(self.fh)
        self.header_written = False
        self.header_cols: List[str] = []
        self.jointstate_names: Optional[List[str]] = None  # locked on first JS message

    def _write_header(self, col_keys: List[str]):
        self.header_cols = ["t_sec"] + col_keys
        self.w.writerow(self.header_cols)
        self.header_written = True

    def _columns_for_payload(self, payload: Dict[str, Any]) -> Tuple[List[str], List[Any]]:
        if "data" in payload:
            vals = payload["data"]
            keys = [f"data_{i}" for i in range(len(vals))]
            return keys, vals

        if any(k.startswith(("linear_", "angular_", "force_", "torque_")) for k in payload.keys()):
            keys = sorted(payload.keys())
            vals = [payload[k] for k in keys]
            return keys, vals

        if "value" in payload:
            return ["value"], [payload["value"]]

        if all(k in payload for k in ("_names", "_pos", "_vel", "_eff")):
            names = payload["_names"]
            if self.jointstate_names is None:
                self.jointstate_names = names[:]
                keys = (
                    [f"position_{n}" for n in names] +
                    [f"velocity_{n}" for n in names] +
                    [f"effort_{n}"   for n in names]
                )
                vals = list(payload["_pos"]) + list(payload["_vel"]) + list(payload["_eff"])
                return keys, vals
            else:
                L = len(self.jointstate_names)
                pos = (payload["_pos"] + [""] * L)[:L]
                vel = (payload["_vel"] + [""] * L)[:L]
                eff = (payload["_eff"] + [""] * L)[:L]
                keys = (
                    [f"position_{n}" for n in self.jointstate_names] +
                    [f"velocity_{n}" for n in self.jointstate_names] +
                    [f"effort_{n}"   for n in self.jointstate_names]
                )
                vals = list(pos) + list(vel) + list(eff)
                return keys, vals

        return [], []

    def write_row(self, t_sec: float, payload: Dict[str, Any]):
        keys, vals = self._columns_for_payload(payload)
        if not self.header_written:
            self._write_header(keys)
        if len(keys) != len(self.header_cols) - 1:
            width = len(self.header_cols) - 1
            vals = (vals + [""] * width)[:width]
        self.w.writerow([t_sec] + vals)

    def close(self):
        try:
            self.fh.flush()
            self.fh.close()
        except Exception:
            pass

# ---------- iterate messages from SQLite ----------
def topics_map(conn: sqlite3.Connection) -> Dict[int, Tuple[str, str]]:
    cur = conn.cursor()
    cur.execute("SELECT id, name, type FROM topics;")
    return {tid: (name, typ) for tid, name, typ in cur.fetchall()}

def iter_messages_for_topic(conn: sqlite3.Connection, topic_id: int) -> Iterator[Tuple[int, bytes]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp;",
        (topic_id,)
    )
    while True:
        rows = cur.fetchmany(CHUNK_FETCH)
        if not rows:
            break
        for ts_ns, blob in rows:
            yield ts_ns, blob

def merged_iters(iters: List[Iterator[Tuple[int, bytes]]]) -> Iterator[Tuple[int, bytes]]:
    heap = []
    for idx, it in enumerate(iters):
        try:
            ts, blob = next(it)
            heap.append((ts, idx, blob, it))
        except StopIteration:
            pass
    heapq.heapify(heap)
    while heap:
        ts, idx, blob, it = heapq.heappop(heap)
        yield ts, blob
        try:
            nts, nblob = next(it)
            heapq.heappush(heap, (nts, idx, nblob, it))
        except StopIteration:
            pass

# ---------- export logic ----------
def export_all_topics(bag_or_db_path: str):
    db_files = list_db_files(bag_or_db_path)
    outdir = export_dir_for(bag_or_db_path)
    if VERBOSE:
        print(f"[info] Export dir: {outdir}")
        for i, dbf in enumerate(db_files):
            print(f"  - DB[{i}]: {dbf}")

    # Build topic roster (name -> type) from first file that has it
    roster: Dict[str, str] = {}
    for dbf in db_files:
        with safe_connect_file(dbf) as conn:
            for tid, (name, typ) in topics_map(conn).items():
                roster.setdefault(name, typ)

    # --- diagnostic: show the recorded ROS type for /joint_states, if present ---
    if VERBOSE and "/joint_states" in roster:
        print(f"[info] /joint_states type: {roster['/joint_states']}")

    # Writers per topic
    writers: Dict[str, TopicCSV] = {
        name: TopicCSV(name, typ, outdir) for name, typ in roster.items()
    }

    # Decoder per topic
    decoders: Dict[str, Any] = {}
    for name, typ in roster.items():
        dec = DECODERS.get(typ)
        # ---- special case: be lenient for delta_twist_cmds only ----
        if name == "/servo_node/delta_twist_cmds":
            dec = decode_twist_lenient
            if VERBOSE:
                print(f"[info] Using lenient decoder for {name} (recorded type: {typ})")
        # -------------------------------------------------------------
        if not dec and VERBOSE:
            print(f"[warn] No decoder for {name} ({typ}), skipping.")
        decoders[name] = dec


    # Build per-file iterators for each topic
    per_topic_iters: Dict[str, List[Iterator[Tuple[int, bytes]]]] = {name: [] for name in roster}
    for dbf in db_files:
        conn = safe_connect_file(dbf)
        tmap = topics_map(conn)  # id -> (name, type)
        for tid, (name, _typ) in tmap.items():
            per_topic_iters[name].append(iter_messages_for_topic(conn, tid))

    # Determine t0 across all files (first timestamp)
    t0: Optional[int] = None
    for dbf in db_files:
        with safe_connect_file(dbf) as conn:
            cur = conn.cursor()
            cur.execute("SELECT MIN(timestamp) FROM messages;")
            row = cur.fetchone()
            if row and row[0] is not None:
                t0 = row[0] if t0 is None else min(t0, row[0])
    if t0 is None:
        print("[info] No messages found.")
        return

    # Stream & write per topic
    for name, iters in per_topic_iters.items():
        if not iters:
            continue
        it = merged_iters(iters) if (MERGE_BY_TIMESTAMP and len(iters) > 1) else \
             (r for it_ in iters for r in it_)
        decoder = decoders.get(name)
        w = writers[name]
        count = 0
        for ts_ns, blob in it:
            if not decoder:
                continue
            try:
                payload = decoder(blob)
            except Exception as e:
                if VERBOSE:
                    print(f"[warn] decode failed for {name}: {e}")
                continue
            t_sec = ns_to_sec_rel(ts_ns, t0)
            w.write_row(t_sec, payload)
            count += 1
        w.close()
        if VERBOSE:
            print(f"[ok] {name}: {count} rows -> {w.outpath}")

# ---------- main ----------
if __name__ == "__main__":
    # Bag directory (contains metadata.yaml) OR a .db3 inside it
    BAG_PATH = "/media/aryan/Pruning25/02_Harvesting_bagfiles/Carmen_apple_picks_data/data/batch_73/apple_1/pressure_servo_20250805_155545.db3"
    export_all_topics(BAG_PATH)
