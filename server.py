from flask import Flask, request, jsonify, render_template, session
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_session import Session
import cv2
import numpy as np
from datetime import datetime
import base64
import os
import face_recognition
from time import time
import uuid

# ================= ENV CONFIG =================
UPLOAD_SECRET = os.environ.get("UPLOAD_SECRET")  # SET IN RAILWAY
if not UPLOAD_SECRET:
    raise RuntimeError("UPLOAD_SECRET not set in environment")

MAX_UPLOAD_MB = 2
UPLOAD_INTERVAL = 0.5
MAX_HISTORY = 100
MAX_KNOWN_FACES = 100
# ==============================================

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=True  # Railway uses HTTPS
)

Session(app)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

CORS(app, resources={
    r"/upload": {
        "origins": "*",
        "allow_headers": ["Content-Type", "X-ESP32-KEY"]
    },
    r"/add_face": {
        "origins": "*",
        "allow_headers": ["Content-Type", "X-ESP32-KEY"]
    },
    r"/remove_face/*": {
        "origins": "*",
        "allow_headers": ["X-ESP32-KEY"]
    },
    r"/known_faces": {
        "origins": "*",
        "allow_headers": ["X-ESP32-KEY"]
    },
    r"/history": {
        "origins": "*",
        "allow_headers": ["X-ESP32-KEY"]
    },
    r"/stats": {
        "origins": "*",
        "allow_headers": ["X-ESP32-KEY"]
    }
})

socketio = SocketIO(app, cors_allowed_origins="*")

os.makedirs("known_faces", exist_ok=True)
os.makedirs("detected_faces", exist_ok=True)

known_face_encodings = []
known_face_names = []
detection_history = []

LAST_UPLOAD = {}

# ================= HELPERS =================

def api_authorized(req):
    # ESP32 auth
    key = req.headers.get("X-ESP32-KEY", "")
    if key and key.strip() == UPLOAD_SECRET.strip():
        return True

    # Web auth (session cookie)
    if session.get("web_auth"):
        return True

    return False

def is_valid_image(data):
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img is not None
    except:
        return False

def rate_limit(ip):
    now = time()
    for k in list(LAST_UPLOAD.keys()):
        if now - LAST_UPLOAD[k] > 10:
            del LAST_UPLOAD[k]

    if now - LAST_UPLOAD.get(ip, 0) < UPLOAD_INTERVAL:
        return False

    LAST_UPLOAD[ip] = now
    return True

def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()

    for f in os.listdir("known_faces"):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = face_recognition.load_image_file(os.path.join("known_faces", f))
            enc = face_recognition.face_encodings(img)
            if enc:
                known_face_encodings.append(enc[0])
                known_face_names.append(os.path.splitext(f)[0].split("_")[0])

def recognize_faces(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(rgb, locs)

    names = []
    for enc in encs:
        matches = face_recognition.compare_faces(known_face_encodings, enc, 0.5)
        name = "Unknown"
        if True in matches:
            idx = np.argmin(face_recognition.face_distance(known_face_encodings, enc))
            name = known_face_names[idx]
        names.append(name)

    return locs, names

# ================= MIDDLEWARE =================

@app.before_request
def secure_web_access():
    # ESP32 uses header auth (only for /upload)
    if request.headers.get("X-ESP32-KEY") and request.path == "/upload":
        return

    # Allow static & homepage
    if request.path in ["/", "/socket.io/"] or request.path.startswith("/static"):
        session["web_auth"] = True
        return

    # Protect API routes
    protected = ["/add_face", "/remove_face", "/known_faces", "/history", "/stats"]
    if any(request.path.startswith(p) for p in protected):
        if not session.get("web_auth"):
            return jsonify({"error": "Unauthorized"}), 401

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    print("==== DEBUG AUTH ====")
    print("HEADER X-ESP32-KEY:", request.headers.get("X-ESP32-KEY"))
    print("ENV UPLOAD_SECRET:", UPLOAD_SECRET)
    print("====================")

    # /upload is ESP32-only (no session auth)
    key = request.headers.get("X-ESP32-KEY", "")
    if key.strip() != UPLOAD_SECRET.strip():
        return jsonify({"error": "Unauthorized"}), 401

    ip = request.remote_addr
    if not rate_limit(ip):
        return jsonify({"error": "Too many requests"}), 429

    image_bytes = request.files["file"].read() if "file" in request.files else request.data
    if not image_bytes or not is_valid_image(image_bytes):
        return jsonify({"error": "Invalid image"}), 400

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    locs = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(rgb, locs)

    names = []

    for enc in encs:
        if known_face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, enc)
            best_idx = np.argmin(distances)

            if distances[best_idx] < 0.48:
                names.append(known_face_names[best_idx])
            else:
                names.append("Unknown")
        else:
            names.append("Unknown")

    for (t, r, b, l), n in zip(locs, names):
        color = (0,255,0) if n != "Unknown" else (0,0,255)
        cv2.rectangle(image, (l,t), (r,b), color, 2)
        cv2.putText(image, n, (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buf = cv2.imencode(".jpg", image)
    img_b64 = base64.b64encode(buf).decode()

    detection = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "faces": len(locs),
        "recognized": [n for n in names if n != "Unknown"],
        "unknown": names.count("Unknown"),
        "image": f"data:image/jpeg;base64,{img_b64}"
    }

    detection_history.insert(0, detection)
    detection_history[:] = detection_history[:MAX_HISTORY]

    socketio.emit("new_detection", detection)

    return jsonify({"success": True})

@app.route("/add_face", methods=["POST"])
def add_face():
    if not api_authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    if len(os.listdir("known_faces")) >= MAX_KNOWN_FACES:
        return jsonify({"error": "Face limit reached"}), 400

    data = request.json
    name = data.get("name","").strip()
    image_data = data.get("image","").split(",")[-1]

    image = cv2.imdecode(
        np.frombuffer(base64.b64decode(image_data), np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not face_recognition.face_encodings(rgb):
        return jsonify({"error": "No face detected"}), 400

    filename = f"{name}_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(os.path.join("known_faces", filename), image)

    load_known_faces()
    socketio.emit("faces_updated", get_known_faces())

    return jsonify({"success": True})

@app.route("/remove_face/<filename>", methods=["DELETE"])
def remove_face(filename):
    if not api_authorized(request):
        return jsonify({"error": "Unauthorized"}), 401

    path = os.path.join("known_faces", filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404

    os.remove(path)
    load_known_faces()
    socketio.emit("faces_updated", get_known_faces())
    return jsonify({"success": True})

@app.route("/known_faces")
def get_known_faces():
    faces = []
    for f in os.listdir("known_faces"):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue   # skip .gitkeep and other non-image files

        with open(os.path.join("known_faces", f), "rb") as img:
            faces.append({
                "filename": f,
                "name": f.split("_")[0],
                "image": "data:image/jpeg;base64," +
                         base64.b64encode(img.read()).decode()
            })
    return faces

@app.route("/history")
def history():
    return jsonify(detection_history)

@app.route("/stats")
def stats():
    return jsonify({
        "total_detections": len(detection_history),
        "total_recognized": sum(len(d["recognized"]) for d in detection_history),
        "known_faces_count": len(known_face_names)
    })

# ================= SOCKET =================

@socketio.on("connect")
def socket_connect(auth):
    # Web UI is authenticated via session cookie
    # ESP32-CAM does NOT use socket.io
    if not session.get("web_auth"):
        return False

# ================= START =================

if __name__ == "__main__":
    load_known_faces()
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
