# app.py — minimal, stable backend (MP3 output + feedback flags + 30-min automation)

import os, json, hashlib, logging, shutil, subprocess, threading
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from collections import Counter

from flask import Flask, jsonify, render_template, request, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ---------------- Flask ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ---------------- Config ----------------
UPLOAD_FOLDER = "uploads"
OUTPUT_ROOT   = "static/outputs"
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB
ALLOWED_EXT   = {"mp3", "wav"}

# Spleeter CLI path (set full path on Windows if needed)
SPLEETER_BIN  = os.getenv("SPLEETER_BIN", "spleeter")

# Feedback + automation
FEEDBACK_FILE    = "learning_data.json"
BAD_OUTPUTS_FILE = "outputs_to_improve.json"
THRESHOLD_NEG    = 2
REBUILD_INTERVAL_S = int(os.getenv("REBUILD_INTERVAL_SECONDS", "1800"))  # 30 min

# ---------------- Setup ----------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_ROOT,   exist_ok=True)

for p in (FEEDBACK_FILE, BAD_OUTPUTS_FILE):
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump([] if p == FEEDBACK_FILE else [], f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("vocal_remover.log", encoding="utf-8")]
)

# ---------------- Helpers ----------------
def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def ok(cmd: List[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False

def rebuild_flags_from_feedback():
    try:
        fb = json.load(open(FEEDBACK_FILE, "r", encoding="utf-8"))
        if not isinstance(fb, list):
            fb = []
    except Exception:
        fb = []
    neg = Counter(x.get("output_id") for x in fb if x.get("rating") == "negative")
    pos = Counter(x.get("output_id") for x in fb if x.get("rating") == "positive")
    flagged = sorted({oid for oid, n in neg.items() if oid and n >= THRESHOLD_NEG and n > pos.get(oid, 0)})
    try:
        with open(BAD_OUTPUTS_FILE, "w", encoding="utf-8") as f:
            json.dump(flagged, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return set(flagged)

def is_problematic_output(output_id: str) -> bool:
    try:
        bad = json.load(open(BAD_OUTPUTS_FILE, "r", encoding="utf-8"))
        return output_id in bad
    except Exception:
        return False

# ---------------- Automation (30 min) ----------------
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

# ---------------- Routes ----------------
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "ffmpeg":  ok(["ffmpeg", "-version"]),
        "ffprobe": ok(["ffprobe", "-version"]),
        "spleeter": ok([SPLEETER_BIN, "--help"]),
        "mode": "spleeter",
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
        tmp_root = os.path.join(out_dir, "_spleeter_tmp")

        # clean + prep
        try:
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
        except Exception:
            pass
        os.makedirs(out_dir, exist_ok=True)
        with open(in_path, "wb") as w:
            w.write(blob)

        # Spleeter 2-stems to WAVs
        cmd = [SPLEETER_BIN, "separate", "-p", "spleeter:2stems", "-o", tmp_root, in_path]
        logging.info("REQ %s spleeter: %s", req_id, " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            logging.error("REQ %s spleeter stderr: %s", req_id, r.stderr)
            return jsonify({"status":"error","message":"Spleeter failed","detail":r.stderr}), 500

        base = os.path.splitext(os.path.basename(in_path))[0]
        out_wav_dir = os.path.join(tmp_root, base)
        voc_wav = os.path.join(out_wav_dir, "vocals.wav")
        acc_wav = os.path.join(out_wav_dir, "accompaniment.wav")
        if not (os.path.exists(voc_wav) and os.path.exists(acc_wav)):
            return jsonify({"status":"error","message":"Spleeter outputs missing"}), 500

        # Encode to MP3 (libmp3lame, VBR q=2)
        voc_mp3 = os.path.join(out_dir, "vocals.mp3")
        acc_mp3 = os.path.join(out_dir, "instrumental.mp3")
        for src, dst in ((voc_wav, voc_mp3), (acc_wav, acc_mp3)):
            enc = ["ffmpeg","-y","-i",src,"-codec:a","libmp3lame","-qscale:a","2",dst]
            er = subprocess.run(enc, capture_output=True, text=True, timeout=180)
            if er.returncode != 0:
                logging.error("REQ %s encode stderr: %s", req_id, er.stderr)
                return jsonify({"status":"error","message":"Encoding failed","detail":er.stderr}), 500

        # cleanup tmp
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass

        voc_url = url_for("static", filename=os.path.relpath(voc_mp3, "static").replace("\\","/"))
        acc_url = url_for("static", filename=os.path.relpath(acc_mp3, "static").replace("\\","/"))

        return jsonify({
            "status": "success",
            "message": "Processing complete",
            "vocal_path": voc_url,
            "instrumental_path": acc_url,
            "output_id": file_id,
            "flagged": is_problematic_output(file_id)
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

# ---------------- Main ----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    logging.info("Starting on 0.0.0.0:%s (mode=spleeter, interval=%ss)", port, REBUILD_INTERVAL_S)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
