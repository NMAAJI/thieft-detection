from flask import Flask, request, jsonify, render_template, session, redirect
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect, generate_csrf
import cv2
import numpy as np
from datetime import datetime
import base64
import os
import face_recognition
from time import time
import uuid
import secrets
from threading import Lock, Thread
import time as time_module
import re

# ================= ENV CONFIG =================
UPLOAD_SECRET = os.environ.get("UPLOAD_SECRET")
if not UPLOAD_SECRET:
    raise RuntimeError("UPLOAD_SECRET not set")

MAX_UPLOAD_MB = 2
UPLOAD_INTERVAL = 0.5   # 2 FPS
MAX_HISTORY = 100
MAX_KNOWN_FACES = 100
CONFIDENCE_THRESHOLD = 0.45  # Stricter for better accuracy

# ================= WEB LOGIN CONFIG =================
WEB_USERNAME = os.environ.get("WEB_USERNAME", "admin")
WEB_PASSWORD = os.environ.get("WEB_PASSWORD")
if not WEB_PASSWORD:
    raise RuntimeError("WEB_PASSWORD must be set! Don't use default password.")

# ================= APP =================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_ENV") == "production",
    WTF_CSRF_TIME_LIMIT=None
)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
csrf = CSRFProtect(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

os.makedirs("known_faces", exist_ok=True)

# ================= GLOBALS WITH THREAD SAFETY =================
known_face_encodings = []
known_face_names = []
detection_history = []
LAST_UPLOAD = {}
face_lock = Lock()
rate_limit_lock = Lock()

# ================= HELPERS =================
def api_authorized(req):
    esp32_key = req.headers.get("X-ESP32-KEY", "").strip()
    return (
        secrets.compare_digest(esp32_key, UPLOAD_SECRET.strip())
        or session.get("web_auth")
    )

def rate_limit(ip):
    now = time()
    with rate_limit_lock:
        last_time = LAST_UPLOAD.get(ip, 0)
        if now - last_time < UPLOAD_INTERVAL:
            return False
        LAST_UPLOAD[ip] = now
    return True

def cleanup_old_rate_limits():
    """Clean up rate limit dict every 5 minutes"""
    while True:
        time_module.sleep(300)
        now = time()
        with rate_limit_lock:
            to_delete = [ip for ip, t in LAST_UPLOAD.items() if now - t > 600]
            for ip in to_delete:
                del LAST_UPLOAD[ip]

def preprocess_image(image):
    """Enhance image quality for better recognition"""
    try:
        height, width = image.shape[:2]
        
        # Resize if too large
        if width > 800:
            scale = 800 / width
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        return image
    except Exception as e:
        app.logger.error(f"Image preprocessing error: {e}")
        return image

def load_known_faces():
    with face_lock:
        known_face_encodings.clear()
        known_face_names.clear()
        
        try:
            for f in os.listdir("known_faces"):
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                
                try:
                    img_path = os.path.join("known_faces", f)
                    img = face_recognition.load_image_file(img_path)
                    encs = face_recognition.face_encodings(img)
                    
                    if encs:
                        known_face_encodings.append(encs[0])
                        known_face_names.append(os.path.splitext(f)[0].split("_")[0])
                except Exception as e:
                    app.logger.error(f"Error loading face {f}: {e}")
                    continue
        except Exception as e:
            app.logger.error(f"Error reading known_faces directory: {e}")

def recognize_faces(image):
    """Recognize faces with confidence scores"""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, locs)
        
        names = []
        confidences = []
        
        for enc in encs:
            name = "Unknown"
            confidence = 0
            
            with face_lock:
                if known_face_encodings:
                    distances = face_recognition.face_distance(known_face_encodings, enc)
                    best = np.argmin(distances)
                    
                    if distances[best] < CONFIDENCE_THRESHOLD:
                        name = known_face_names[best]
                        confidence = (1 - distances[best]) * 100
            
            names.append(name)
            confidences.append(confidence)
        
        return locs, names, confidences
    except Exception as e:
        app.logger.error(f"Face recognition error: {e}")
        return [], [], []

def validate_filename(filename):
    """Validate filename to prevent path traversal"""
    if not re.match(r'^[a-zA-Z0-9_-]+\.(jpg|jpeg|png)$', filename, re.IGNORECASE):
        return False
    return True

# ================= MIDDLEWARE =================
@app.before_request
def secure():
    if request.method == "OPTIONS":
        return "", 200
    
    # ESP32 upload endpoint - check API key
    if request.path == "/upload":
        esp32_key = request.headers.get("X-ESP32-KEY", "").strip()
        if not secrets.compare_digest(esp32_key, UPLOAD_SECRET.strip()):
            return jsonify({"error": "Unauthorized"}), 401
        return
    
    # Public endpoints
    if request.path in ["/login", "/logout"] or request.path.startswith("/static"):
        return
    
    # Protected endpoints - require web auth
    if not session.get("web_auth"):
        return redirect("/login")

@app.after_request
def set_security_headers(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# ================= ROUTES =================
@app.route("/login", methods=["GET", "POST"])
@csrf.exempt
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        
        if (secrets.compare_digest(username, WEB_USERNAME) and 
            secrets.compare_digest(password, WEB_PASSWORD)):
            session.clear()
            session.regenerate = True  # Prevent session fixation
            session["web_auth"] = True
            return redirect("/")
        
        return render_template("login.html", error="Invalid credentials")
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/")
def index():
    return render_template("index.html", csrf_token=generate_csrf())

@app.route("/csrf-token")
def get_csrf_token():
    """Provide CSRF token for AJAX requests"""
    return jsonify({"csrf_token": generate_csrf()})

@app.route("/upload", methods=["POST"])
@csrf.exempt  # ESP32 uses API key, not CSRF
def upload():
    # Auth already checked in middleware
    ip = request.remote_addr
    
    if not rate_limit(ip):
        return jsonify({"error": "Too many requests"}), 429
    
    try:
        # Get image data
        data = request.files["file"].read() if "file" in request.files else request.data
        
        if len(data) == 0:
            return jsonify({"error": "No image data"}), 400
        
        # Decode image
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Preprocess for better recognition
        image = preprocess_image(image)
        
        # Recognize faces
        locs, names, confidences = recognize_faces(image)
        
        # Draw boxes and labels
        for (t, r, b, l), name, conf in zip(locs, names, confidences):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (l, t), (r, b), color, 2)
            
            label = f"{name} ({conf:.1f}%)" if name != "Unknown" else "Unknown"
            cv2.putText(image, label, (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Encode result
        _, buf = cv2.imencode(".jpg", image)
        img_b64 = base64.b64encode(buf).decode()
        
        # Create detection record
        detection = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "faces": len(locs),
            "recognized": [{"name": n, "confidence": c} for n, c in zip(names, confidences) if n != "Unknown"],
            "unknown": names.count("Unknown"),
            "image": f"data:image/jpeg;base64,{img_b64}"
        }
        
        # Store in history
        detection_history.insert(0, detection)
        detection_history[:] = detection_history[:MAX_HISTORY]
        
        # Broadcast to web clients
        socketio.emit("new_detection", detection)
        
        return jsonify({"success": True})
    
    except Exception as e:
        app.logger.error(f"Upload error: {e}")
        return jsonify({"error": "Processing failed"}), 500

@app.route("/add_face", methods=["POST"])
def add_face():
    if not api_authorized(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.json
        name = data.get("name", "").strip()
        image_data = data.get("image", "")
        
        # Validate name
        if not name or len(name) > 50:
            return jsonify({"error": "Invalid name"}), 400
        
        if not re.match(r'^[a-zA-Z0-9\s_-]+$', name):
            return jsonify({"error": "Name contains invalid characters"}), 400
        
        # Check known faces limit
        with face_lock:
            if len(known_face_encodings) >= MAX_KNOWN_FACES:
                return jsonify({"error": f"Maximum {MAX_KNOWN_FACES} faces reached"}), 400
        
        # Decode image
        if "," in image_data:
            image_data = image_data.split(",")[-1]
        
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(image_data), np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if image is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Verify face exists
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        
        if not encodings:
            return jsonify({"error": "No face detected in image"}), 400
        
        if len(encodings) > 1:
            return jsonify({"error": "Multiple faces detected. Please upload image with single face"}), 400
        
        # Save image
        filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join("known_faces", filename)
        cv2.imwrite(filepath, image)
        
        # Reload faces
        load_known_faces()
        
        # Broadcast update
        socketio.emit("faces_updated", get_known_faces())
        
        return jsonify({"success": True})
    
    except Exception as e:
        app.logger.error(f"Add face error: {e}")
        return jsonify({"error": "Failed to add face"}), 500

@app.route("/remove_face/<filename>", methods=["DELETE"])
def remove_face(filename):
    if not api_authorized(request):
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # Validate filename
        if not validate_filename(filename):
            return jsonify({"error": "Invalid filename"}), 400
        
        # Remove file
        path = os.path.join("known_faces", filename)
        
        # Ensure path is within known_faces directory (prevent path traversal)
        real_path = os.path.realpath(path)
        real_dir = os.path.realpath("known_faces")
        
        if not real_path.startswith(real_dir):
            return jsonify({"error": "Invalid path"}), 400
        
        if os.path.exists(path):
            os.remove(path)
        
        # Reload faces
        load_known_faces()
        
        # Broadcast update
        socketio.emit("faces_updated", get_known_faces())
        
        return jsonify({"success": True})
    
    except Exception as e:
        app.logger.error(f"Remove face error: {e}")
        return jsonify({"error": "Failed to remove face"}), 500

@app.route("/known_faces")
def get_known_faces():
    try:
        faces = []
        for f in os.listdir("known_faces"):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    with open(os.path.join("known_faces", f), "rb") as img:
                        faces.append({
                            "filename": f,
                            "name": f.split("_")[0],
                            "image": "data:image/jpeg;base64," + base64.b64encode(img.read()).decode()
                        })
                except Exception as e:
                    app.logger.error(f"Error reading face {f}: {e}")
                    continue
        
        return jsonify(faces)
    
    except Exception as e:
        app.logger.error(f"Get known faces error: {e}")
        return jsonify([])

@app.route("/history")
def history():
    try:
        return jsonify(detection_history)
    except Exception as e:
        app.logger.error(f"History error: {e}")
        return jsonify([])

@app.route("/stats")
def stats():
    try:
        with face_lock:
            known_count = len(known_face_names)
        
        total_recognized = sum(len(d["recognized"]) for d in detection_history)
        
        return jsonify({
            "total_detections": len(detection_history),
            "total_recognized": total_recognized,
            "known_faces_count": known_count
        })
    
    except Exception as e:
        app.logger.error(f"Stats error: {e}")
        return jsonify({
            "total_detections": 0,
            "total_recognized": 0,
            "known_faces_count": 0
        })

# ================= SOCKET =================
@socketio.on("connect")
def socket_connect(auth):
    if not session.get("web_auth"):
        return False

# ================= START =================
if __name__ == "__main__":
    # Start cleanup thread
    cleanup_thread = Thread(target=cleanup_old_rate_limits, daemon=True)
    cleanup_thread.start()
    
    # Load known faces
    load_known_faces()
    
    # Run server
    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )