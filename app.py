# app.py — FFmpeg-only backend with filter auto-detect + fallbacks

import os, json, hashlib, logging, shutil, subprocess, threading
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from collections import Counter

from flask import Flask, jsonify, render_template, request, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_ROOT   = "static/outputs"
MAX_FILE_SIZE = 30 * 1024 * 1024
ALLOWED_EXT   = {"mp3", "wav"}

FEEDBACK_FILE      = "learning_data.json"
BAD_OUTPUTS_FILE   = "outputs_to_improve.json"
THRESHOLD_NEG      = 2
REBUILD_INTERVAL_S = int(os.getenv("REBUILD_INTERVAL_SECONDS", "1800"))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_ROOT,   exist_ok=True)
for p in (FEEDBACK_FILE, BAD_OUTPUTS_FILE):
    if not os.path.exists(p):
        json.dump([], open(p, "w", encoding="utf-8"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("vocal_remover.log", encoding="utf-8")]
)

# ---------------- helpers ----------------
def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def ok(cmd: List[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False

def ffprobe_channels(path: str) -> Optional[int]:
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=channels","-of","csv=p=0", path],
            capture_output=True, text=True, timeout=20
        )
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip())
    except Exception:
        pass
    return None

def rebuild_flags_from_feedback():
    try:
        fb = json.load(open(FEEDBACK_FILE, "r", encoding="utf-8"))
        if not isinstance(fb, list): fb = []
    except Exception:
        fb = []
    neg = Counter(x.get("output_id") for x in fb if x.get("rating") == "negative")
    pos = Counter(x.get("output_id") for x in fb if x.get("rating") == "positive")
    flagged = sorted({oid for oid, n in neg.items() if oid and n >= THRESHOLD_NEG and n > pos.get(oid, 0)})
    json.dump(flagged, open(BAD_OUTPUTS_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return set(flagged)

def is_problematic_output(output_id: str) -> bool:
    try:
        return output_id in json.load(open(BAD_OUTPUTS_FILE, "r", encoding="utf-8"))
    except Exception:
        return False

# ---- FFmpeg filter detection + builders ----
_FILTER_LIST: Optional[str] = None
def ff_filters() -> str:
    global _FILTER_LIST
    if _FILTER_LIST is None:
        try:
            r = subprocess.run(["ffmpeg","-hide_banner","-filters"], capture_output=True, text=True, timeout=20)
            _FILTER_LIST = r.stdout
        except Exception:
            _FILTER_LIST = ""
    return _FILTER_LIST

def has_filter(name: str) -> bool:
    return name in ff_filters()

def build_filters() -> Tuple[str, str, Dict[str,bool]]:
    st  = has_filter("stereotools")
    dyn = has_filter("dynaudnorm")
    ac  = has_filter("acompressor")

    if st:
        vocal = "stereotools=mlev=1:slev=0"
        instr = "stereotools=mlev=0:slev=1"
    else:
        # Fallback to pan mid/side approximation
        vocal = "pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1"
        instr = "pan=stereo|c0=c0-c1|c1=c1-c0"

    # add tone-shaping + dynamics if available
    if dyn:
        vocal += ",highpass=f=120,lowpass=f=9000,dynaudnorm=f=75:s=10"
        instr += ",highpass=f=60,lowpass=f=16000,dynaudnorm=f=250:s=10"
    elif ac:
        vocal += ",highpass=f=120,lowpass=f=9000,acompressor"
        instr += ",highpass=f=60,lowpass=f=16000"
    else:
        vocal += ",highpass=f=120,lowpass=f=9000"
        instr += ",highpass=f=60,lowpass=f=16000"

    return vocal, instr, {"stereotools": st, "dynaudnorm": dyn, "acompressor": ac}

def side_energy_db(path: str) -> Optional[float]:
    try:
        cmd = ["ffmpeg","-v","error","-i",path,"-af",
               "stereotools=mlev=0:slev=1,astats=metadata=1:reset=0:measure_overall=1",
               "-f","null","-"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        import re
        vals = re.findall(r"Overall RMS level:\s*(-?\d+(?:\.\d+)?)", r.stderr)
        return float(vals[-1]) if vals else None
    except Exception:
        return None

# ---------------- separation ----------------
def separate_ffmpeg(input_path: str, output_dir: str) -> Tuple[bool, Dict[str, Any]]:
    channels = ffprobe_channels(input_path) or 0
    if channels != 2:
        return False, {"reason": "non_stereo", "channels": channels}

    s_db = side_energy_db(input_path)
    if s_db is not None and s_db < -35.0:
        return False, {"reason": "dual_mono", "side_rms_db": s_db}

    os.makedirs(output_dir, exist_ok=True)
    vocal_mp3 = os.path.join(output_dir, "vocals.mp3")
    instr_mp3 = os.path.join(output_dir, "instrumental.mp3")

    vocal_filter, instr_filter, filt_avail = build_filters()

    try:
        r1 = subprocess.run(
            ["ffmpeg","-y","-i",input_path,"-af",vocal_filter,"-codec:a","libmp3lame","-qscale:a","2", vocal_mp3],
            capture_output=True, text=True, timeout=240
        )
        if r1.returncode != 0:
            return False, {"reason": "ffmpeg_error_vocal", "stderr": r1.stderr[-4000:], "filters": filt_avail}

        r2 = subprocess.run(
            ["ffmpeg","-y","-i",input_path,"-af",instr_filter,"-codec:a","libmp3lame","-qscale:a","2", instr_mp3],
            capture_output=True, text=True, timeout=240
        )
        if r2.returncode != 0:
            return False, {"reason": "ffmpeg_error_instr", "stderr": r2.stderr[-4000:], "filters": filt_avail}

        in_sz = os.path.getsize(input_path)
        v_sz  = os.path.getsize(vocal_mp3) if os.path.exists(vocal_mp3) else 0
        i_sz  = os.path.getsize(instr_mp3) if os.path.exists(instr_mp3) else 0
        if v_sz < max(64_000, 0.05 * in_sz) or i_sz < max(64_000, 0.02 * in_sz):
            return False, {"reason": "weak_separation", "v_bytes": v_sz, "i_bytes": i_sz, "in_bytes": in_sz, "filters": filt_avail}

        return True, {"mode": "ffmpeg", "side_rms_db": s_db, "filters": filt_avail,
                      "v_bytes": v_sz, "i_bytes": i_sz, "in_bytes": in_sz}
    except subprocess.TimeoutExpired:
        return False, {"reason": "timeout"}
    except Exception as e:
        return False, {"reason": "exception", "error": str(e)}

def separate_audio(input_path: str, output_dir: str) -> Tuple[bool, Dict[str, Any]]:
    return separate_ffmpeg(input_path, output_dir)

# ---------------- automation ----------------
_stop = threading.Event()
def _loop():
    logging.info("automation loop start interval=%ss", REBUILD_INTERVAL_S)
    while not _stop.wait(REBUILD_INTERVAL_S):
        try:
            flagged = rebuild_flags_from_feedback()
            logging.info("automation rebuilt flags=%s", len(flagged))
        except Exception as e:
            logging.error("automation error: %s", e)
threading.Thread(target=_loop, daemon=True).start()

# ---------------- routes ----------------
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "ffmpeg":  ok(["ffmpeg", "-version"]),
        "ffprobe": ok(["ffprobe", "-version"]),
        "mode": "ffmpeg",
        "filters": {
            "stereotools": has_filter("stereotools"),
            "dynaudnorm": has_filter("dynaudnorm"),
            "acompressor": has_filter("acompressor"),
        },
        "automation_interval_sec": REBUILD_INTERVAL_S
    }), 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/process", methods=["POST"])
def process():
    req_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    try:
        if "file" not in request.files:
            return jsonify({"status":"error","message":"No file uploaded"}), 400
        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify({"status":"error","message":"No file selected"}), 400
        if not allowed(f.filename):
            return jsonify({"status":"error","message":"Only MP3 and WAV are supported"}), 400

        blob = f.read()
        if not blob:
            return jsonify({"status":"error","message":"Empty file"}), 400
        if len(blob) > MAX_FILE_SIZE:
            return jsonify({"status":"error","message":f"File too large (max {MAX_FILE_SIZE//1024//1024}MB)"}), 400

        file_id = hashlib.md5(blob).hexdigest()
        safe    = secure_filename(f.filename)
        in_path = os.path.join(UPLOAD_FOLDER, f"input_{file_id}_{safe}")
        out_dir = os.path.join(OUTPUT_ROOT,  f"output_{file_id}")

        try:
            if os.path.exists(out_dir): shutil.rmtree(out_dir)
        except Exception:
            pass
        os.makedirs(out_dir, exist_ok=True)
        with open(in_path, "wb") as w: w.write(blob)

        ok_sep, notes = separate_audio(in_path, out_dir)
        if not ok_sep:
            reason = notes.get("reason","processing_failed")
            msg = {
                "probe_failed": "Could not inspect audio.",
                "non_stereo": f"Stereo audio required. Channels={notes.get('channels','?')}.",
                "dual_mono":  "Track is dual-mono (L≈R). Karaoke separation can’t work on this file.",
                "ffmpeg_error_vocal": "FFmpeg failed while creating vocal track.",
                "ffmpeg_error_instr": "FFmpeg failed while creating instrumental track.",
                "weak_separation": "Separation too weak for this file.",
                "timeout": "Processing timed out.",
                "exception": "Unexpected error during processing."
            }.get(reason, f"Audio processing failed: {reason}")
            logging.error("REQ %s fail: %s %s", req_id, reason, notes)
            return jsonify({"status":"error","message":msg,"detail":notes}), 400

        voc_mp3 = os.path.join(out_dir, "vocals.mp3")
        ins_mp3 = os.path.join(out_dir, "instrumental.mp3")
        voc_url = url_for("static", filename=os.path.relpath(voc_mp3, "static").replace("\\","/"))
        ins_url = url_for("static", filename=os.path.relpath(ins_mp3, "static").replace("\\","/"))

        return jsonify({
            "status": "success",
            "message": "Processing complete",
            "vocal_path": voc_url,
            "instrumental_path": ins_url,
            "output_id": file_id,
            "flagged": is_problematic_output(file_id),
            "processing_info": notes
        }), 200

    except subprocess.TimeoutExpired:
        logging.exception("REQ %s timeout", req_id)
        return jsonify({"status":"error","message":"Processing timed out"}), 500
    except Exception as e:
        logging.exception("REQ %s unexpected", req_id)
        return jsonify({"status":"error","message":"Server error","detail":str(e)}), 500
    finally:
        try:
            if 'in_path' in locals() and os.path.exists(in_path):
                os.remove(in_path)
        except Exception:
            pass

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(silent=True) or {}
        rating = data.get("rating")
        comment = (data.get("comment") or "").strip()
        oid = (data.get("output_id") or "").strip()
        if rating not in ("positive", "negative"):
            return jsonify({"status":"error","message":"Invalid rating"}), 400
        if not oid:
            return jsonify({"status":"error","message":"Missing output_id"}), 400
        entry = {"output_id": oid, "rating": rating, "comment": comment, "timestamp": datetime.now().isoformat()}
        content = json.load(open(FEEDBACK_FILE, "r", encoding="utf-8"))
        content.append(entry)
        json.dump(content, open(FEEDBACK_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        flagged = rebuild_flags_from_feedback()
        return jsonify({"status":"success","now_flagged": oid in flagged, "flag_count": len(flagged)}), 200
    except Exception:
        logging.exception("feedback error")
        return jsonify({"status":"error","message":"Could not save feedback"}), 500

@app.route("/admin/rebuild", methods=["POST"])
def admin_rebuild():
    flagged = list(rebuild_flags_from_feedback())
    return jsonify({"status":"success","flagged_count":len(flagged),"flagged":flagged}), 200

@app.errorhandler(404)
def not_found(_):
    return jsonify({"status":"error","message":"Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(_):
    return jsonify({"status":"error","message":"Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    logging.info("Starting on 0.0.0.0:%s (mode=ffmpeg, interval=%ss)", port, REBUILD_INTERVAL_S)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
