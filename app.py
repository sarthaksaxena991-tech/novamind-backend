import os, json, hashlib, logging, subprocess, sys, tempfile
from collections import Counter
from flask import Flask, jsonify, render_template, request, url_for
from flask_cors import CORS

# ---------- Flask ----------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------- Config ----------
THRESHOLD_NEG = 2
MAX_FILE_SIZE = 50 * 1024 * 1024    # 50 MB
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
FEEDBACK_FILE = "learning_data.json"
BAD_OUTPUTS_FILE = "outputs_to_improve.json"
SPLEETER_MODEL = "spleeter:2stems"  # or "spleeter:4stems" for more stems

# ---------- Setup ----------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

def rebuild_flags_from_feedback():
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            fb = json.load(f)
    except Exception:
        fb = []
    neg = Counter(x.get("output_id") for x in fb if x.get("rating") == "negative")
    pos = Counter(x.get("output_id") for x in fb if x.get("rating") == "positive")
    flagged = [oid for oid, n in neg.items() if oid and n >= THRESHOLD_NEG and n > pos.get(oid, 0)]
    try:
        with open(BAD_OUTPUTS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(set(flagged)), f, indent=2, ensure_ascii=False)
    except Exception:
        logging.exception("Could not write outputs_to_improve.json")
    return set(flagged)

def is_problematic_output(output_id: str) -> bool:
    if os.path.exists(BAD_OUTPUTS_FILE):
        try:
            with open(BAD_OUTPUTS_FILE, "r", encoding="utf-8") as f:
                return output_id in json.load(f)
        except Exception:
            logging.exception("Failed reading BAD_OUTPUTS_FILE")
    return False

def convert_to_wav_if_needed(input_path: str) -> str:
    """Convert non-WAV files to WAV format for better Spleeter compatibility"""
    if input_path.lower().endswith('.wav'):
        return input_path
        
    try:
        import librosa
        import soundfile as sf
        
        # Create temp file
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{os.path.basename(input_path)}.wav")
        
        # Load and save as WAV
        y, sr = librosa.load(input_path, sr=None)
        sf.write(temp_path, y, sr)
        
        # Remove original
        os.remove(input_path)
        return temp_path
    except Exception as e:
        logging.warning(f"Could not convert to WAV: {e}")
        return input_path

# ---------- Routes ----------
@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
            
        f = request.files["file"]
        if not f or not f.filename.lower().endswith((".mp3", ".wav")):
            return jsonify({"status": "error", "message": "Only MP3/WAV files allowed"}), 400

        # Read file in chunks to handle large files
        data = b''
        while True:
            chunk = f.stream.read(8192)
            if not chunk:
                break
            data += chunk
            
            # Check size during read to avoid memory issues
            if len(data) > MAX_FILE_SIZE:
                return jsonify({"status": "error", "message": "File too large (max 50MB)"}), 400

        if not data:
            return jsonify({"status": "error", "message": "Empty file"}), 400

        ext = os.path.splitext(f.filename)[1] or ".wav"
        file_id = hashlib.md5(data).hexdigest()

        in_path = os.path.join(UPLOAD_FOLDER, f"input_{file_id}{ext}")
        out_dir = os.path.join(OUTPUT_FOLDER, f"output_{file_id}")
        
        # Clean up any previous attempts
        try:
            if os.path.exists(out_dir):
                import shutil
                shutil.rmtree(out_dir)
        except Exception as e:
            logging.warning(f"Could not clean output dir: {e}")

        os.makedirs(out_dir, exist_ok=True)
        
        with open(in_path, "wb") as w:
            w.write(data)

        # Convert to WAV if needed (better Spleeter compatibility)
        in_path = convert_to_wav_if_needed(in_path)

        # --------- Spleeter CLI with timeout ---------
        cmd = [
            sys.executable, "-m", "spleeter", "separate",
            "-p", SPLEETER_MODEL,
            "-o", out_dir,
            "-c", "wav",  # Force WAV output
            in_path
        ]
        
        logging.info("Running: %s", " ".join(cmd))
        
        try:
            # Run with timeout (10 minutes)
            proc = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=600  # 10 minutes
            )
            
            if proc.returncode != 0:
                logging.error(f"Spleeter failed with code {proc.returncode}")
                logging.error(f"Stderr: {proc.stderr.strip()}")
                return jsonify({
                    "status": "error", 
                    "message": f"Spleeter failed: {proc.stderr.strip() or 'Unknown error'}"
                }), 500
                
        except subprocess.TimeoutExpired:
            logging.error("Spleeter processing timed out")
            return jsonify({
                "status": "error", 
                "message": "Processing timed out (took too long)"
            }), 500

        # Find output files (handling different Spleeter versions)
        base_name = os.path.splitext(os.path.basename(in_path))[0]
        possible_paths = [
            os.path.join(out_dir, base_name, "vocals.wav"),  # Newer versions
            os.path.join(out_dir, f"input_{file_id}", "vocals.wav"),  # Older versions
            os.path.join(out_dir, "vocals.wav")  # Some configurations
        ]
        
        voc = next((p for p in possible_paths if os.path.exists(p)), None)
        acc = next((p.replace("vocals.wav", "accompaniment.wav") for p in possible_paths if os.path.exists(p)), None)

        if not voc or not acc:
            logging.error(f"Output files not found. Searched: {possible_paths}")
            return jsonify({"status": "error", "message": "Output files not found"}), 500

        vocal_url = url_for("static", filename=os.path.relpath(voc, "static").replace("\\", "/"))
        acc_url = url_for("static", filename=os.path.relpath(acc, "static").replace("\\", "/"))

        try:
            with open("latest_id.txt", "w", encoding="utf-8") as _f:
                _f.write(file_id)
        except Exception:
            pass

        return jsonify({
            "status": "success",
            "message": "Processing complete",
            "vocal_path": vocal_url,
            "instrumental_path": acc_url,
            "output_id": file_id,
            "flagged": is_problematic_output(file_id)
        })
        
    except Exception as e:
        logging.exception("Processing error")
        return jsonify({"status": "error", "message": f"{type(e).__name__}: {str(e)}"}), 500
        
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
        comment = data.get("comment", "")
        oid = data.get("output_id", "")
        
        if rating not in ("positive", "negative"):
            return jsonify({"status":"error","message":"Invalid rating"}), 400
        if not oid:
            return jsonify({"status":"error","message":"Missing output_id"}), 400
            
        entry = {"output_id": oid, "rating": rating, "comment": comment}
        
        with open(FEEDBACK_FILE, "r+", encoding="utf-8") as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                content = []
            content.append(entry)
            f.seek(0)
            json.dump(content, f, indent=2, ensure_ascii=False)
            f.truncate()
            
        new_flags = rebuild_flags_from_feedback()
        return jsonify({
            "status":"success",
            "message":"Feedback received.",
            "now_flagged": oid in new_flags,
            "flag_count": len(new_flags)
        })
        
    except Exception as e:
        logging.exception("Feedback error")
        return jsonify({"status":"error","message":f"{type(e).__name__}: {str(e)}"}), 500

if __name__ == "__main__":
    if not os.path.exists(BAD_OUTPUTS_FILE):
        with open(BAD_OUTPUTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    
    # Pre-build flags
    rebuild_flags_from_feedback()
    
    logging.info("Starting Flask on http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=True, threaded=True)