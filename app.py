import os, json, hashlib, logging, shutil
from collections import Counter
from flask import Flask, jsonify, render_template, request, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from spleeter.separator import Separator

# ---------- Flask ----------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# ---------- Config ----------
THRESHOLD_NEG = 2
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
FEEDBACK_FILE = "learning_data.json"
BAD_OUTPUTS_FILE = "outputs_to_improve.json"
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac'}

# Initialize Spleeter separator
try:
    SEPARATOR = Separator("spleeter:2stems")
    logging.info("Spleeter separator initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Spleeter: {e}")
    SEPARATOR = None

# ---------- Setup ----------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rebuild_flags_from_feedback():
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            fb = json.load(f)
    except Exception as e:
        logging.warning(f"Could not read feedback file: {e}")
        fb = []
    
    neg = Counter(x.get("output_id") for x in fb if x.get("rating") == "negative")
    pos = Counter(x.get("output_id") for x in fb if x.get("rating") == "positive")
    flagged = [oid for oid, n in neg.items() if oid and n >= THRESHOLD_NEG and n > pos.get(oid, 0)]
    
    try:
        with open(BAD_OUTPUTS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(set(flagged)), f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Could not write outputs_to_improve.json: {e}")
    
    return set(flagged)

def is_problematic_output(output_id: str) -> bool:
    if os.path.exists(BAD_OUTPUTS_FILE):
        try:
            with open(BAD_OUTPUTS_FILE, "r", encoding="utf-8") as f:
                return output_id in json.load(f)
        except Exception as e:
            logging.error(f"Failed reading BAD_OUTPUTS_FILE: {e}")
    return False

def convert_to_wav_if_needed(input_path: str) -> str:
    """Convert non-WAV files to WAV for better Spleeter compatibility."""
    if input_path.lower().endswith(".wav"):
        return input_path
    
    try:
        import librosa
        import soundfile as sf
        
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{os.path.basename(input_path)}.wav")
        y, sr = librosa.load(input_path, sr=None)
        sf.write(temp_path, y, sr)
        os.remove(input_path)
        logging.info(f"Converted {input_path} to WAV format")
        return temp_path
    except Exception as e:
        logging.warning(f"Could not convert to WAV: {e}")
        return input_path

# ---------- Routes ----------
@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route("/process", methods=["POST"])
def process():
    if SEPARATOR is None:
        return jsonify({
            "status": "error", 
            "message": "Spleeter not initialized. Please check server logs."
        }), 500

    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        f = request.files["file"]
        if not f or f.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        if not allowed_file(f.filename):
            return jsonify({
                "status": "error", 
                "message": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Read file with size limit
        f.stream.seek(0, os.SEEK_END)
        file_size = f.stream.tell()
        f.stream.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "status": "error", 
                "message": f"File too large (max {MAX_FILE_SIZE//1024//1024}MB)"
            }), 400

        # Generate unique file ID
        file_content = f.read()
        file_id = hashlib.md5(file_content).hexdigest()
        ext = os.path.splitext(f.filename)[1] or ".wav"
        safe_filename = secure_filename(f.filename)
        in_path = os.path.join(UPLOAD_FOLDER, f"input_{file_id}_{safe_filename}")
        out_dir = os.path.join(OUTPUT_FOLDER, f"output_{file_id}")

        # Clean previous outputs
        try:
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
        except Exception as e:
            logging.warning(f"Could not clean output dir: {e}")
        
        os.makedirs(out_dir, exist_ok=True)

        # Save uploaded file
        with open(in_path, "wb") as w:
            w.write(file_content)

        # Convert to WAV if needed
        in_path = convert_to_wav_if_needed(in_path)

        # Process with Spleeter
        try:
            SEPARATOR.separate_to_file(
                in_path,
                out_dir,
                codec="wav",
                bitrate="320k",
                filename_format="{instrument}.{codec}",
            )
        except Exception as e:
            logging.exception("Spleeter processing failed")
            return jsonify({
                "status": "error",
                "message": f"Audio processing failed: {str(e)}"
            }), 500

        # Find output files
        base_name = os.path.splitext(os.path.basename(in_path))[0]
        possible_paths = [
            os.path.join(out_dir, base_name, "vocals.wav"),
            os.path.join(out_dir, base_name, "accompaniment.wav"),
            os.path.join(out_dir, "vocals.wav"),
            os.path.join(out_dir, "accompaniment.wav"),
        ]
        
        voc = next((p for p in possible_paths if os.path.exists(p) and "vocal" in p), None)
        acc = next((p for p in possible_paths if os.path.exists(p) and "accompaniment" in p), None)
        
        if not voc or not acc:
            logging.error(f"Output files not found. Searched: {possible_paths}")
            return jsonify({
                "status": "error", 
                "message": "Processing completed but output files not found"
            }), 500

        # Generate URLs for the frontend
        vocal_url = url_for('static', filename=os.path.relpath(voc, 'static').replace('\\', '/'))
        acc_url = url_for('static', filename=os.path.relpath(acc, 'static').replace('\\', '/'))

        # Save latest ID
        try:
            with open("latest_id.txt", "w", encoding="utf-8") as id_file:
                id_file.write(file_id)
        except Exception as e:
            logging.warning(f"Could not save latest ID: {e}")

        return jsonify({
            "status": "success",
            "message": "Processing complete",
            "vocal_path": vocal_url,
            "instrumental_path": acc_url,
            "output_id": file_id,
            "flagged": is_problematic_output(file_id),
        })

    except Exception as e:
        logging.exception("Unexpected processing error")
        return jsonify({
            "status": "error", 
            "message": f"Server error: {type(e).__name__}"
        }), 500
    
    finally:
        # Clean up input file
        try:
            if 'in_path' in locals() and os.path.exists(in_path):
                os.remove(in_path)
        except Exception as e:
            logging.warning(f"Could not remove input file: {e}")

@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(silent=True) or {}
        rating = data.get("rating")
        comment = data.get("comment", "").strip()
        oid = data.get("output_id", "").strip()

        if rating not in ("positive", "negative"):
            return jsonify({"status": "error", "message": "Invalid rating"}), 400
        
        if not oid:
            return jsonify({"status": "error", "message": "Missing output_id"}), 400

        entry = {
            "output_id": oid, 
            "rating": rating, 
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        # Thread-safe file writing
        try:
            if os.path.exists(FEEDBACK_FILE):
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    content = json.load(f)
            else:
                content = []
            
            content.append(entry)
            
            with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Failed to save feedback: {e}")
            return jsonify({
                "status": "error", 
                "message": "Could not save feedback"
            }), 500

        new_flags = rebuild_flags_from_feedback()
        
        return jsonify({
            "status": "success",
            "message": "Feedback received",
            "now_flagged": oid in new_flags,
            "flag_count": len(new_flags),
        })
        
    except Exception as e:
        logging.exception("Feedback processing error")
        return jsonify({
            "status": "error", 
            "message": f"Error processing feedback: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == "__main__":
    # Initialize bad outputs file if it doesn't exist
    if not os.path.exists(BAD_OUTPUTS_FILE):
        with open(BAD_OUTPUTS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    
    # Rebuild flags from existing feedback
    rebuild_flags_from_feedback()
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 5050))
    
    logging.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
