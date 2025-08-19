import os, sys, json, shutil, logging
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf
from spleeter.separator import Separator

# ---------- Settings ----------
SR = 22050                 # processing sample rate
DURATION_LIMIT = 180       # seconds (same as app)
LOG_DIR = Path(__file__).resolve().parent / "logs"
FLAGS_FILE = Path(__file__).resolve().parent.parent / "outputs_to_improve.json"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "static" / "outputs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"auto_improve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
print("Logging to:", log_file)

# ---------- Basic audio utils ----------
def safe_load(path, sr=SR):
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y

def peak_normalize(y, peak=0.97):
    m = np.max(np.abs(y)) + 1e-9
    return np.clip(y * (peak / m), -1.0, 1.0)

def rms_db(y):
    rms = np.sqrt(np.mean(np.square(y)) + 1e-12)
    return 20 * np.log10(rms + 1e-12)

def loudness_normalize(y, target_rms_db=-20.0):
    current = rms_db(y)
    gain_db = target_rms_db - current
    g = 10 ** (gain_db / 20.0)
    y2 = y * g
    return peak_normalize(y2)

# ---------- Simple FX ----------
def hpf_first_order(y, cutoff_hz=80, sr=SR):
    alpha = 1 - (cutoff_hz / (sr/2))
    return y - alpha * np.concatenate([[0], y[:-1]])

def de_ess(y, sr=SR, band=(5000, 9000), amount=0.2):
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
    gain = np.ones_like(Y, dtype=np.complex128)
    gain[low_mask] *= 10 ** (low_boost_db/20)
    gain[high_mask] *= 10 ** (-high_cut_db/20)
    y2 = np.fft.irfft(Y * gain, n=len(y))
    return y2

# ---------- Spectral gating noise reduction ----------
def spectral_gate(y, sr=SR, n_fft=1024, hop=256, floor_db=-35, reduction_db=12):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag, phase = np.abs(S), np.angle(S)
    mag_db = 20 * np.log10(mag + 1e-8)
    floor = np.percentile(mag_db, 20, axis=1, keepdims=True)
    thresh = np.maximum(floor + reduction_db, floor_db)
    mask = mag_db > thresh
    att = np.where(mask, 1.0, 10 ** (reduction_db / -20.0))
    mag2 = mag * att
    S2 = mag2 * np.exp(1j * phase)
    y2 = librosa.istft(S2, hop_length=hop, window="hann", length=len(y))
    return y2

# ---------- Scoring (lower is better) ----------
def leakage_score(vocals, accomp, sr=SR):
    def energy(x): return float(np.mean(x**2)) if x.size else 0.0
    vE, aE = energy(vocals), energy(accomp)
    Sv = librosa.feature.melspectrogram(y=vocals, sr=sr)
    Sa = librosa.feature.melspectrogram(y=accomp, sr=sr)
    Sv = (Sv - Sv.mean()) / (Sv.std() + 1e-8)
    Sa = (Sa - Sa.mean()) / (Sa.std() + 1e-8)
    corr = float(np.mean(Sv * Sa))
    return corr + (aE / (vE + 1e-8))

# ---------- Enhancement chains ----------
def chain_mild(v, a):
    v2 = hpf_first_order(v, 80)
    v2 = 0.85 * v2 + 0.15 * de_ess(v2)
    v2 = tilt_eq(v2)
    v2 = loudness_normalize(v2, -21)

    a2 = hpf_first_order(a, 40)
    a2 = tilt_eq(a2, high_cut_db=0.8)
    a2 = loudness_normalize(a2, -21)
    return v2, a2

def chain_strong(v, a):
    # keep vocals; denoise accompaniment only
    v2 = v
    a2 = spectral_gate(a, reduction_db=15)
    a2 = hpf_first_order(a2, 60)
    a2 = tilt_eq(a2, low_boost_db=0.8, high_cut_db=1.2)
    v2 = loudness_normalize(v2, -20)
    a2 = loudness_normalize(a2, -20)
    return v2, a2

# ---------- Spleeter helpers ----------
def ensure_current_stems(out_dir: Path, oid: str):
    sub = out_dir / f"input_{oid}"
    voc = sub / "vocals.wav"
    acc = sub / "accompaniment.wav"
    if voc.exists() and acc.exists():
        v0, a0 = safe_load(str(voc)), safe_load(str(acc))
        return v0, a0, sub

    # rebuild from archived original
    original = None
    for ext in (".mp3", ".wav", ".aac", ".flac"):
        p = out_dir / f"original{ext}"
        if p.exists():
            original = p
            break
    if not original:
        return None, None, None

    sep = Separator('spleeter:2stems')  # no stft_backend arg
    sep.separate_to_file(str(original), str(out_dir), duration=DURATION_LIMIT, offset=0)
    sub = out_dir / f"input_{oid}"
    voc = sub / "vocals.wav"
    acc = sub / "accompaniment.wav"
    if not (voc.exists() and acc.exists()):
        return None, None, None
    v0, a0 = safe_load(str(voc)), safe_load(str(acc))
    return v0, a0, sub

def write_pair(subdir: Path, v, a, sr=SR):
    subdir.mkdir(parents=True, exist_ok=True)
    sf.write(str(subdir / "vocals.wav"), v, sr)
    sf.write(str(subdir / "accompaniment.wav"), a, sr)

# ---------- Main improve routine ----------
def improve_one(oid: str) -> bool:
    out_dir = OUTPUTS_DIR / f"output_{oid}"
    if not out_dir.exists():
        logging.info(f"[{oid}] skip: output folder missing")
        return False

    v0, a0, subdir = ensure_current_stems(out_dir, oid)
    if v0 is None or a0 is None:
        logging.info(f"[{oid}] skip: stems not found and could not rebuild")
        return False

    base_score = leakage_score(v0, a0)
    logging.info(f"[{oid}] base score: {base_score:.3f}")

    # Try chains
    v_mild, a_mild = chain_mild(v0, a0)
    s_mild = leakage_score(v_mild, a_mild)

    v_strong, a_strong = chain_strong(v0, a0)
    s_strong = leakage_score(v_strong, a_strong)

    # Choose best (metric) + forced denoise fallback
    candidates = [
        (base_score, v0, a0, "baseline"),
        (s_mild, v_mild, a_mild, "mild"),
        (s_strong, v_strong, a_strong, "strong-denoise"),
    ]
    best = min(candidates, key=lambda x: x[0])
    logging.info(f"[{oid}] scores -> baseline={base_score:.3f}, mild={s_mild:.3f}, strong={s_strong:.3f} -> pick {best[3]}")

    margin = 0.0  # lenient
    if best[0] < base_score - margin:
        v_out, a_out, tag = best[1], best[2], best[3]
    else:
        # flagged as bad -> still deliver denoised accompaniment
        v_out, a_out, tag = v_strong, a_strong, "forced-strong-denoise"

    v_out = peak_normalize(v_out)
    a_out = peak_normalize(a_out)
    write_pair(subdir, v_out, a_out, sr=SR)
    logging.info(f"[{oid}] replaced stems with '{tag}'")
    return True

def main():
    if not FLAGS_FILE.exists():
        print("No outputs_to_improve.json found. Nothing to do.")
        logging.info("flags file not found")
        return

    try:
        flagged = json.load(open(FLAGS_FILE, "r", encoding="utf-8"))
    except Exception:
        flagged = []

    if not flagged:
        print("No flagged outputs. All good.")
        logging.info("no flagged outputs")
        return

    improved = 0
    for oid in flagged:
        try:
            if improve_one(oid):
                improved += 1
        except Exception as e:
            logging.exception(f"Fail on {oid}: {e}")

    print(f"Enhanced {improved}/{len(flagged)} output(s). See log:", log_file)
    logging.info(f"Enhanced {improved}/{len(flagged)} outputs.")

if __name__ == "__main__":
    main()
