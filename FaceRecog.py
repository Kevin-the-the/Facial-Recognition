import cv2
import face_recognition
import os
import numpy as np
from flask import Flask, render_template, Response, jsonify
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import webbrowser

app = Flask(__name__)

known_encodings = []
known_names = []
database_path = "database"

def load_known_faces():
    global known_encodings, known_names
    known_encodings = []
    known_names = []

    if not os.path.exists(database_path):
        os.makedirs(database_path)
        print("Created 'database' directory. Add images of known people there.")

    for filename in os.listdir(database_path):
        filepath = os.path.join(database_path, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])

    print("Known faces database updated!")

load_known_faces()

class FolderMonitor(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            load_known_faces() 

def start_folder_monitoring():
    observer = Observer()
    event_handler = FolderMonitor()
    observer.schedule(event_handler, database_path, recursive=False)
    observer.start()
    print(f"Monitoring folder: {database_path}")

    try:
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

threading.Thread(target=start_folder_monitoring, daemon=True).start()

video_capture = None

def generate_frames():
    global video_capture
    while True:
        if video_capture is None:
            time.sleep(0.1)
            continue
        success, frame = video_capture.read()
        if not success:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            video_capture = None
            return jsonify({"status": "Failed to start camera"}), 500
        return jsonify({"status": "Camera started"}), 200
    return jsonify({"status": "Camera is already running"}), 200

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global video_capture
    if video_capture:
        video_capture.release()
        video_capture = None
        return jsonify({"status": "Camera stopped"}), 200
    return jsonify({"status": "No active camera to stop"}), 400

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture_face():
    global video_capture
    if not video_capture:
        return jsonify({"detected": "Camera is not active"}), 400

    success, frame = video_capture.read()
    if not success:
        return jsonify({"detected": "Failed to capture frame"}), 500

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    #***************CHANGE TOLERANCE TO ADJUST STRICTNESS, LOWER IS STRICTER********************

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = np.argmax(matches)
            name = known_names[match_index]

        return jsonify({"detected": name}), 200

    return jsonify({"detected": "No recognized face found"}), 200

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:3000/")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='0.0.0.0', port=3000, debug=True, use_reloader=False) 
