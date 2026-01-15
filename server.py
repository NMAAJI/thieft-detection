from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from datetime import datetime
import base64
import json
import os
import face_recognition
from time import time
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB max upload
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Directories
os.makedirs('known_faces', exist_ok=True)
os.makedirs('detected_faces', exist_ok=True)

# Store known faces
known_face_encodings = []
known_face_names = []

# Detection history
detection_history = []
MAX_HISTORY = 100

# Rate limiting
LAST_UPLOAD = {}
UPLOAD_INTERVAL = 1.5  # seconds

# Time-based face recognition (prevents memory leak in multi-worker setup)
LAST_RECOG = 0
RECOG_INTERVAL = 1.0  # seconds

def is_valid_image(image_bytes):
    """Validate if bytes represent a valid image"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img is not None
    except:
        return False

def rate_limit(ip):
    """Check if IP is rate limited"""
    now = time()
    
    # Cleanup old entries (prevent memory leak)
    for k in list(LAST_UPLOAD.keys()):
        if now - LAST_UPLOAD[k] > 10:
            del LAST_UPLOAD[k]
    
    last = LAST_UPLOAD.get(ip, 0)
    if now - last < UPLOAD_INTERVAL:
        return False
    LAST_UPLOAD[ip] = now
    return True

def load_known_faces():
    """Load all known faces from directory"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists('known_faces'):
        return
    
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join('known_faces', filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
    
    print(f"Loaded {len(known_face_names)} known faces")

def recognize_faces(image):
    """Recognize faces in image"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        
        if True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        face_names.append(name)
    
    return face_locations, face_names

def draw_faces(image, face_locations, face_names):
    """Draw rectangles and names on faces"""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Authentication
        expected = os.environ.get("UPLOAD_SECRET")
        secret = request.headers.get("X-ESP32-KEY")
        if not expected or secret != expected:
            return jsonify({"error": "Unauthorized"}), 401
        
        # Content-length safety check
        if request.content_length and request.content_length > 2 * 1024 * 1024:
            return jsonify({"error": "Payload too large"}), 413
        
        # Rate limiting
        ip = request.remote_addr
        if not rate_limit(ip):
            return jsonify({'error': 'Too many requests'}), 429
        
        # 1️⃣ Read image bytes (supports ESP32 + browser)
        if 'file' in request.files:
            image_bytes = request.files['file'].read()
        else:
            image_bytes = request.data

        if not image_bytes or len(image_bytes) == 0:
            return jsonify({'error': 'empty_image'}), 400
        
        # Validate image
        if not is_valid_image(image_bytes):
            return jsonify({'error': 'Invalid image'}), 400

        # 2️⃣ Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'decode_failed'}), 400

        # 3️⃣ Face recognition (time-based, not frame-based)
        global LAST_RECOG
        now = time()
        
        if now - LAST_RECOG > RECOG_INTERVAL:
            if known_face_encodings:
                face_locations, face_names = recognize_faces(image)
            else:
                face_locations, face_names = [], []
            LAST_RECOG = now
        else:
            face_locations, face_names = [], []
        
        image_with_faces = draw_faces(image.copy(), face_locations, face_names)

        # 4️⃣ Encode result image (FIXED)
        _, buffer = cv2.imencode('.jpg', image_with_faces)
        img_base64 = base64.b64encode(buffer).decode('utf-8')  # ✅ FIX

        detection = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'faces': len(face_locations),
            'recognized': [n for n in face_names if n != "Unknown"],
            'unknown': face_names.count("Unknown"),
            'image': f'data:image/jpeg;base64,{img_base64}'
        }

        detection_history.insert(0, detection)
        if len(detection_history) > MAX_HISTORY:
            detection_history.pop()

        if face_locations:
            filename = f"detected_faces/{uuid.uuid4().hex}.jpg"
            cv2.imwrite(filename, image_with_faces)
            
            # Auto-cleanup old detections to prevent disk filling
            try:
                files = os.listdir("detected_faces")
                if len(files) > 300:
                    for f in sorted(files)[:50]:
                        try:
                            os.remove(os.path.join("detected_faces", f))
                        except:
                            pass
            except:
                pass

        socketio.emit('new_detection', detection)

        return jsonify({
            'success': True,
            'faces': len(face_locations),
            'recognized': detection['recognized'],
            'timestamp': detection['timestamp']
        })

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    """Add a new known face"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        image_data = data.get('image', '')
        
        if not name or not image_data:
            return jsonify({'error': 'Name and image required'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if face exists in image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)
        
        if not face_encodings:
            return jsonify({'error': 'No face detected in image'}), 400
        
        # Save image
        filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join('known_faces', filename)
        cv2.imwrite(filepath, image)
        
        # Reload faces
        load_known_faces()
        
        socketio.emit('faces_updated', get_known_faces_list())
        
        return jsonify({'success': True, 'message': f'Added {name} to known faces'})
        
    except Exception as e:
        print(f"Error adding face: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/remove_face/<filename>', methods=['DELETE'])
def remove_face(filename):
    """Remove a known face"""
    try:
        filepath = os.path.join('known_faces', filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            load_known_faces()
            socketio.emit('faces_updated', get_known_faces_list())
            return jsonify({'success': True, 'message': 'Face removed'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/known_faces', methods=['GET'])
def get_known_faces():
    """Get list of known faces"""
    return jsonify(get_known_faces_list())

def get_known_faces_list():
    """Helper to get known faces with images"""
    faces = []
    for filename in os.listdir('known_faces'):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join('known_faces', filename)
            with open(filepath, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            name = os.path.splitext(filename)[0]
            # Remove timestamp from display name
            display_name = '_'.join(name.split('_')[:-2]) if len(name.split('_')) > 2 else name
            
            faces.append({
                'filename': filename,
                'name': display_name,
                'image': f'data:image/jpeg;base64,{image_data}'
            })
    return faces

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(detection_history)

@app.route('/stats', methods=['GET'])
def get_stats():
    total_detections = len(detection_history)
    detections_with_faces = sum(1 for d in detection_history if d['faces'] > 0)
    total_recognized = sum(len(d['recognized']) for d in detection_history)
    
    return jsonify({
        'total_detections': total_detections,
        'detections_with_faces': detections_with_faces,
        'total_recognized': total_recognized,
        'known_faces_count': len(known_face_names)
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'data': 'Connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("Loading known faces...")
    load_known_faces()
    print("Starting Face Recognition Server...")
    
    # Get port from environment variable (Railway uses this)
    port = int(os.environ.get('PORT', 5000))
    print(f"Server running on port {port}")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
