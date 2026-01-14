from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import face_recognition
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
KNOWN_FACES_DIR = "known_faces"
DETECTION_LOG = []

# Load known faces on startup
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Load all known faces from the known_faces directory"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"⚠️  Created {KNOWN_FACES_DIR} directory - add face images here")
        return
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                filepath = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_face_encodings.append(encodings[0])
                    # Use filename without extension as name
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    print(f"✓ Loaded face: {name}")
                else:
                    print(f"⚠️  No face found in: {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {str(e)}")
    
    print(f"\n✓ Loaded {len(known_face_names)} known faces: {known_face_names}")

# Load known faces at startup
load_known_faces()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_image():
    """Handle image upload from ESP32-CAM or simulator"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    
    try:
        # Get raw image data from request body
        image_data = request.get_data()
        
        if not image_data:
            return jsonify({
                "error": "No image data received",
                "faces": 0
            }), 400
        
        # Convert bytes to PIL Image
        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return jsonify({
                "error": f"Invalid image format: {str(e)}",
                "faces": 0
            }), 400
        
        # Convert PIL Image to numpy array for face_recognition
        image_np = np.array(image)
        
        # Detect faces
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        
        detected_faces = []
        recognized_count = 0
        unknown_count = 0
        
        # Compare with known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            # Use the known face with the smallest distance
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        recognized_count += 1
                    else:
                        unknown_count += 1
            else:
                unknown_count += 1
            
            detected_faces.append({
                "name": name,
                "status": "recognized" if name != "Unknown" else "unknown"
            })
        
        # Log detection
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "total_faces": len(face_locations),
            "recognized": recognized_count,
            "unknown": unknown_count,
            "faces": detected_faces
        }
        DETECTION_LOG.append(log_entry)
        
        # Keep only last 100 logs
        if len(DETECTION_LOG) > 100:
            DETECTION_LOG.pop(0)
        
        print(f"✓ Detected {len(face_locations)} faces - {recognized_count} recognized, {unknown_count} unknown")
        
        return jsonify({
            "status": "success",
            "faces": len(face_locations),
            "recognized": recognized_count,
            "unknown": unknown_count,
            "detections": detected_faces,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        print(f"✗ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": f"Server error: {str(e)}",
            "faces": 0
        }), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get detection logs"""
    return jsonify({
        "logs": DETECTION_LOG,
        "total": len(DETECTION_LOG)
    })

@app.route('/api/known-faces', methods=['GET'])
def get_known_faces():
    """Get list of known faces"""
    return jsonify({
        "faces": known_face_names,
        "count": len(known_face_names)
    })

@app.route('/api/reload-faces', methods=['POST'])
def reload_faces():
    """Reload known faces from directory"""
    try:
        load_known_faces()
        return jsonify({
            "status": "success",
            "count": len(known_face_names),
            "faces": known_face_names
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "known_faces": len(known_face_names),
        "detections_logged": len(DETECTION_LOG)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting Face Recognition Server on port {port}")
    print(f"📁 Known faces directory: {KNOWN_FACES_DIR}")
    print(f"👤 Loaded {len(known_face_names)} known faces")
    print("\n" + "="*50)
    app.run(host='0.0.0.0', port=port, debug=False)
