import cv2
import threading
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from vidgear.gears import CamGear
import torch

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {'GPU' if device=='cuda' else 'CPU'}")

model = YOLO('yolov8n.pt').to(device)
CLASS_IDS_TO_DETECT = [1, 2, 3, 5, 7]
CLASS_NAMES = model.names

CAMERAS = [
    {'id': 'camera-1', 'name': 'Camera 1', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=y-Os52eW2rg', 'status': 'online'},
    {'id': 'camera-2', 'name': 'Camera 2', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=yy4FFZBdRqo', 'status': 'online'},
    {'id': 'camera-3', 'name': 'Camera 3', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=Uuaemo4RwFU', 'status': 'online'},
    {'id': 'camera-4', 'name': 'Camera 4', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=48wtXs5okhE', 'status': 'online'}
]

output_frames = {}
lock = threading.Lock()

def process_camera_stream(camera_info):
    global output_frames, lock
    camera_id = camera_info['id']
    video_url = camera_info['streamUrl']

    try:
        stream = CamGear(source=video_url, stream_mode=True, logging=True).start()
        print(f"[{camera_id}] Successfully started stream.")
    except Exception as e:
        print(f"[{camera_id}] Error starting stream: {e}")
        return

    TARGET_FPS = 20
    FRAME_SKIP = 1
    frame_count = 0

    while True:
        frame = stream.read()
        if frame is None:
            print(f"[{camera_id}] Stream ended or failed.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        results = model.predict(frame, classes=CLASS_IDS_TO_DETECT, imgsz=(640, 384), verbose=False, device=device)

        processed_frame = frame
        if len(results) > 0:
            result = results[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = CLASS_NAMES[cls]
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        with lock:
            flag, encodedImage = cv2.imencode(".jpg", processed_frame)
            if flag:
                output_frames[camera_id] = encodedImage.tobytes()

    stream.stop()

def generate_frame_for_request(camera_id):
    global output_frames, lock
    while True:
        time.sleep(0.03)
        with lock:
            if camera_id not in output_frames:
                continue
            frame_bytes = output_frames[camera_id]

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cameras')
def get_cameras():
    return jsonify(CAMERAS)

@app.route('/video_feed')
def video_feed():
    camera_id = request.args.get('id')
    return Response(generate_frame_for_request(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    for camera in CAMERAS:
        thread = threading.Thread(target=process_camera_stream, args=(camera,), daemon=True)
        thread.start()
    
    app.run(debug=False, threaded=True)