import os, json, hashlib, logging, shutil
import subprocess
from collections import Counter
from flask import Flask, jsonify, render_template, request, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

# Initialize Flask app first
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Allow all origins

# ---------- Config ----------
THRESHOLD_NEG = 2
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB (smaller for free tier)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"
FEEDBACK_FILE = "learning_data.json"
BAD_OUTPUTS_FILE = "outputs_to_improve.json"
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# ---------- Setup ----------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rebuild_flags_from_feedback():
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                fb = json.load(f)
        else:
            fb = []
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

def separate_audio_demucs(input_path, output_dir):
    """Use demucs for audio separation (lighter than spleeter)"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run demucs command
        cmd = [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "-o", output_dir,
            input_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logging.error(f"Demucs failed: {result.stderr}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        logging.error("Demucs processing timed out")
        return False
    except Exception as e:
        logging.error(f"Demucs error: {e}")
        return False

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
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400

        f = request.files["file"]
        if not f or f.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        if not allowed_file(f.filename):
            return jsonify({
                "status": "error", 
                "message": f"Only MP3 and WAV files are supported"
            }), 400

        # Read file with size limit
        file_content = f.read()
        if len(file_content) > MAX_FILE_SIZE:
            return jsonify({
                "status": "error", 
                "message": f"File too large (max {MAX_FILE_SIZE//1024//1024}MB)"
            }), 400

        if len(file_content) == 0:
            return jsonify({"status": "error", "message": "Empty file"}), 400

        # Generate unique file ID
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

        # Process with Demucs
        success = separate_audio_demucs(in_path, out_dir)
        if not success:
            return jsonify({
                "status": "error",
                "message": "Audio processing failed. Please try a different file."
            }), 500

        # Find output files
        # Demucs creates structure: output_dir/htdemucs/track_name/{vocals,other}.wav
        track_name = os.path.splitext(os.path.basename(in_path))[0]
        demucs_output = os.path.join(out_dir, "htdemucs", track_name)
        
        vocal_path = os.path.join(demucs_output, "vocals.wav")
        other_path = os.path.join(demucs_output, "other.wav")
        
        if not os.path.exists(vocal_path) or not os.path.exists(other_path):
            logging.error(f"Output files not found in {demucs_output}")
            return jsonify({
                "status": "error", 
                "message": "Processing completed but output files not found"
            }), 500

        # Move files to main output directory for easier access
        final_vocal_path = os.path.join(out_dir, "vocals.wav")
        final_other_path = os.path.join(out_dir, "instrumental.wav")
        
        shutil.move(vocal_path, final_vocal_path)
        shutil.move(other_path, final_other_path)
        
        # Clean up demucs directory
        shutil.rmtree(os.path.join(out_dir, "htdemucs"))

        # Generate URLs for the frontend
        vocal_url = url_for('static', filename=os.path.relpath(final_vocal_path, 'static').replace('\\', '/'))
        acc_url = url_for('static', filename=os.path.relpath(final_other_path, 'static').replace('\\', '/'))

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
            "message": f"Server error: Please try again later."
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
            "message": f"Error processing feedback"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == "__main__":
    # Initialize files if they don't exist
    for file_path in [FEEDBACK_FILE, BAD_OUTPUTS_FILE]:
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f)
    
    # Rebuild flags from existing feedback
    rebuild_flags_from_feedback()
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 10000))
    
    logging.info(f"Starting Flask server on 0.0.0.0:{port}")
    
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
