import os, json, shutil, logging
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path

# ---------- optional deps ----------
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

import numpy as np
import librosa                  # pip install librosa==0.8.1
import soundfile as sf          # pip install soundfile
from spleeter.separator import Separator  # spleeter==2.4.0 (TF<2.10)

# ---------- optional Supabase (no-op if not configured) ----------
try:
    from supabase import create_client  # pip install supabase
except Exception:
    create_client = None

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://vdbjltfyoxmiijuwjlur.supabase.co").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZkYmpsdGZ5b3htaWlqdXdqbHVyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjkxMDkwNywiZXhwIjoyMDY4NDg2OTA3fQ.XB8bYvzNi1CfDb0sF7eKPb_2EuebppyulVwQXVFYS50").strip()
sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if (create_client and SUPABASE_URL and SUPABASE_SERVICE_KEY) else None

def sb_log(row: dict):
    if not sb: return
    try:
        sb.table("vocal_jobs").insert(row).execute()
    except Exception as e:
        logging.warning(f"Supabase insert failed: {e}")

def sb_event(job_id: str, status: str, **extra):
    if not sb: return
    sb_log({
        "job_id": job_id,
        "status": status,  # queued|processing|done|improved|error|housekeeping
        "note": extra.get("note"),
        "score": extra.get("score"),
        "filename": extra.get("filename"),
        "completed_at": datetime.utcnow().isoformat() + "Z",
    })

# ---------- config / paths ----------
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "automation_config.yaml"

DEFAULT_CONFIG = {
    "paths": {
        "outputs_dir": "../static/outputs",   # fixed: 'outputs' not 'output'
        "feedback_file": "../learning_data.json",
        "flags_file": "../outputs_to_improve.json",
        "log_dir": "./logs",
    },
    "thresholds": {
        "negatives_to_flag": 2,
        "min_improvement_margin": 0.02,
    },
    "enhance": {
        "sample_rate": 22050,
        "duration_limit_sec": 180,
        "try_4stems": False,
    },
    "housekeeping": {
        "keep_days_outputs": 30,
        "max_total_outputs_gb": 10.0,
        "max_log_files": 30,
    },
    "jobs": {
        "rebuild_flags": True,
        "enhance_flagged": True,
        "prune_outputs": True,
        "rotate_logs": True,
    },
}

def _deep_update(dst: dict, src: dict):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

CONFIG = DEFAULT_CONFIG.copy()
if yaml and CONFIG_PATH.exists():
    try:
        file_cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        CONFIG = _deep_update(CONFIG, file_cfg)
    except Exception as e:
        # fall back to defaults if YAML bad
        logging.warning(f"Config parse failed: {e}")

OUTPUTS_DIR = (HERE / CONFIG["paths"]["outputs_dir"]).resolve()
FEEDBACK_FILE = (HERE / CONFIG["paths"]["feedback_file"]).resolve()
FLAGS_FILE = (HERE / CONFIG["paths"]["flags_file"]).resolve()
LOG_DIR = (HERE / CONFIG["paths"]["log_dir"]).resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=str(log_file),
                    level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
print("Automation running. Log:", log_file)

# ---------- utils ----------
def _atomic_write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def dir_size_gb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except Exception:
                pass
    return total / (1024**3)

def read_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

# ---------- derive flags from feedback ----------
def _is_negative(rating: str) -> bool:
    r = (rating or "").strip().lower()
    return r in {"bad", "negative", "0", "1", "1-star", "poor", "no"}

def _id_from_feedback(row: dict) -> str:
    # support both "job_id" and "output_id"
    oid = str(row.get("output_id") or row.get("job_id") or "").strip()
    return oid

def rebuild_flags():
    fb = read_json(FEEDBACK_FILE, [])
    neg_ids = [ _id_from_feedback(x) for x in fb if _is_negative(x.get("rating")) ]
    pos_ids = [ _id_from_feedback(x) for x in fb if not _is_negative(x.get("rating")) and (x.get("rating") is not None) ]
    neg = Counter([i for i in neg_ids if i])
    pos = Counter([i for i in pos_ids if i])

    th = int(CONFIG["thresholds"]["negatives_to_flag"])
    flagged = [oid for oid, n in neg.items() if n >= th and n > pos.get(oid, 0)]
    flagged = sorted(set(flagged))
    _atomic_write_json(FLAGS_FILE, flagged)
    logging.info(f"rebuild_flags: total_flagged={len(flagged)}")
    return flagged

# ---------- audio helpers ----------
SR = int(CONFIG["enhance"]["sample_rate"])
DUR_LIMIT = int(CONFIG["enhance"]["duration_limit_sec"])
IMPROVE_MARGIN = float(CONFIG["thresholds"]["min_improvement_margin"])

def safe_load(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def peak_normalize(y, peak=0.97):
    m = np.max(np.abs(y)) + 1e-9
    return np.clip(y * (peak / m), -1.0, 1.0)

def rms_db(y):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    return 20 * np.log10(rms + 1e-12)

def loudness_normalize(y, target_rms_db=-20.0):
    gain_db = target_rms_db - rms_db(y)
    g = 10 ** (gain_db / 20.0)
    return peak_normalize(y * g)

def hpf_first_order(y, cutoff_hz=80, sr=SR):
    alpha = 1 - (cutoff_hz / (sr/2))
    return y - alpha * np.concatenate([[0], y[:-1]])

def de_ess(y, sr=SR, band=(5000, 9000), amount=0.25):
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    Y[mask] *= (1.0 - amount)
    return np.fft.irfft(Y, n=len(y))

def tilt_eq(y, sr=SR, low_boost_db=1.5, high_cut_db=1.0):
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    low_mask = (freqs > 0) & (freqs < 300)
    high_mask = (freqs > 8000)
    gain = np.ones_like(Y.real)
    gain[low_mask] *= 10 ** (low_boost_db/20)
    gain[high_mask] *= 10 ** (-high_cut_db/20)
    return np.fft.irfft(Y * gain, n=len(y))

def spectral_gate(y, sr=SR, n_fft=1024, hop=256, floor_db=-35, reduction_db=12):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag, phase = np.abs(S), np.angle(S)
    mag_db = 20 * np.log10(mag + 1e-8)
    floor = np.percentile(mag_db, 20, axis=1, keepdims=True)
    thresh = np.maximum(floor + reduction_db, floor_db)
    att = np.where(mag_db > thresh, 1.0, 10 ** (reduction_db / -20.0))
    S2 = (mag * att) * np.exp(1j * phase)
    return librosa.istft(S2, hop_length=hop, window="hann", length=len(y))

def leakage_score(vocals, accomp, sr=SR):
    def energy(x): return float(np.mean(x**2)) if x.size else 0.0
    vE, aE = energy(vocals), energy(accomp)
    Sv = librosa.feature.melspectrogram(y=vocals, sr=sr)
    Sa = librosa.feature.melspectrogram(y=accomp, sr=sr)
    Sv = (Sv - Sv.mean()) / (Sv.std() + 1e-8)
    Sa = (Sa - Sa.mean()) / (Sa.std() + 1e-8)
    corr = float(np.mean(Sv * Sa))
    return corr + (aE / (vE + 1e-8))  # lower is better

def chain_mild(v, a):
    v2 = hpf_first_order(v, 80)
    v2 = 0.85 * v2 + 0.15 * de_ess(v2, amount=0.2)
    v2 = tilt_eq(v2, low_boost_db=1.5, high_cut_db=1.0)
    v2 = loudness_normalize(v2, -21)
    a2 = hpf_first_order(a, 40)
    a2 = tilt_eq(a2, low_boost_db=0.8, high_cut_db=0.8)
    a2 = loudness_normalize(a2, -21)
    return v2, a2

def chain_strong(v, a):
    v2 = hpf_first_order(v, 100)
    v2 = de_ess(v2, amount=0.35)
    v2 = tilt_eq(v2, low_boost_db=2.0, high_cut_db=1.5)
    v2 = loudness_normalize(v2, -20)
    a_nr = spectral_gate(a, reduction_db=15)
    a2 = hpf_first_order(a_nr, 60)
    a2 = tilt_eq(a2, low_boost_db=0.8, high_cut_db=1.2)
    a2 = loudness_normalize(a2, -20)
    return v2, a2

# ---------- locate/rebuild stems ----------
def _find_existing_stems(out_dir: Path, oid: str):
    # check common subdir names
    candidates = [out_dir / f"input_{oid}", out_dir / f"{oid}"]
    for sub in candidates:
        voc = sub / "vocals.wav"
        acc = sub / "accompaniment.wav"
        if voc.exists() and acc.exists():
            return sub

    # scan any child that contains both stems
    for sub in out_dir.iterdir():
        if sub.is_dir():
            voc = sub / "vocals.wav"
            acc = sub / "accompaniment.wav"
            if voc.exists() and acc.exists():
                return sub
    return None

def ensure_current_stems(out_dir: Path, oid: str):
    sub = _find_existing_stems(out_dir, oid)
    if sub:
        return safe_load(str(sub / "vocals.wav")), safe_load(str(sub / "accompaniment.wav")), sub

    # rebuild from original mix if available
    original = None
    for ext in (".mp3", ".wav", ".aac", ".flac"):
        p = out_dir / f"original{ext}"
        if p.exists():
            original = p; break
    if not original:
        logging.info(f"[{oid}] original mix not found; skip")
        return None, None, None

    sep = Separator('spleeter:2stems', stft_backend='librosa', multiprocess=False)
    sep.separate_to_file(str(original), str(out_dir), duration=DUR_LIMIT, offset=0)
    sub = _find_existing_stems(out_dir, oid)
    if not sub:
        logging.info(f"[{oid}] stems missing after regenerate; skip")
        return None, None, None

    return safe_load(str(sub / "vocals.wav")), safe_load(str(sub / "accompaniment.wav")), sub

def write_pair(subdir: Path, v, a, sr=SR):
    subdir.mkdir(parents=True, exist_ok=True)
    sf.write(str(subdir / "vocals.wav"), v, sr)
    sf.write(str(subdir / "accompaniment.wav"), a, sr)

# ---------- core ----------
def enhance_flagged(flagged_ids):
    improved = 0
    for oid in flagged_ids:
        out_dir = OUTPUTS_DIR / f"output_{oid}"
        if not out_dir.exists():
            logging.info(f"[{oid}] output folder missing; skip")
            continue

        v0, a0, subdir = ensure_current_stems(out_dir, oid)
        if v0 is None or a0 is None:
            continue

        base = leakage_score(v0, a0)
        v1, a1 = chain_mild(v0, a0); s1 = leakage_score(v1, a1)
        v2, a2 = chain_strong(v0, a0); s2 = leakage_score(v2, a2)

        cand = [(base, v0, a0, "baseline"),
                (s1, v1, a1, "mild"),
                (s2, v2, a2, "strong+NR")]
        best = min(cand, key=lambda x: x[0])
        logging.info(f"[{oid}] scores -> base={base:.3f}, mild={s1:.3f}, strong={s2:.3f} -> pick {best[3]}")

        if best[0] < base - IMPROVE_MARGIN:
            v_out = peak_normalize(best[1])
            a_out = peak_normalize(best[2])
            write_pair(subdir, v_out, a_out, SR)
            logging.info(f"[{oid}] improved and replaced stems.")
            improved += 1
            sb_event(oid, "improved", score=float(best[0]), note=best[3], filename=f"output_{oid}")
        else:
            logging.info(f"[{oid}] no significant improvement; keep current.")
            sb_event(oid, "done", score=float(base), note="no-change", filename=f"output_{oid}")
    return improved

# ---------- housekeeping ----------
def prune_outputs():
    keep_days = int(CONFIG["housekeeping"]["keep_days_outputs"])
    max_gb = float(CONFIG["housekeeping"]["max_total_outputs_gb"])
    deadline = datetime.now() - timedelta(days=keep_days)
    removed = 0
    for p in OUTPUTS_DIR.glob("output_*"):
        try:
            if datetime.fromtimestamp(p.stat().st_mtime) < deadline:
                shutil.rmtree(p, ignore_errors=True); removed += 1
        except Exception:
            pass
    logging.info(f"prune_outputs: removed_old={removed}")

    total_gb = dir_size_gb(OUTPUTS_DIR)
    if total_gb > max_gb:
        dirs = sorted(OUTPUTS_DIR.glob("output_*"), key=lambda d: d.stat().st_mtime)
        while total_gb > max_gb and dirs:
            victim = dirs.pop(0)
            try: shutil.rmtree(victim, ignore_errors=True)
            except Exception: pass
            total_gb = dir_size_gb(OUTPUTS_DIR)
        logging.info(f"prune_outputs: enforced cap, size_now={total_gb:.2f} GB")
        sb_event("housekeeping", "housekeeping", note=f"size_now={total_gb:.2f} GB")

def rotate_logs():
    keep = int(CONFIG["housekeeping"]["max_log_files"])
    files = sorted(LOG_DIR.glob("automation_*.log"), key=lambda p: p.stat().st_mtime)
    for old in files[:max(0, len(files) - keep)]:
        try: old.unlink()
        except Exception: pass

# ---------- main ----------
if __name__ == "__main__":
    logging.info("=== Automation Run Started ===")

    flagged_ids = []
    if CONFIG["jobs"]["rebuild_flags"]:
        flagged_ids = rebuild_flags()

    if CONFIG["jobs"]["enhance_flagged"] and flagged_ids:
        n = enhance_flagged(flagged_ids)
        logging.info(f"enhance_flagged: improved={n}/{len(flagged_ids)}")

    if CONFIG["jobs"]["prune_outputs"]:
        prune_outputs()

    if CONFIG["jobs"]["rotate_logs"]:
        rotate_logs()

    logging.info("=== Automation Run Completed ===")
    print("Done. Check log:", log_file)