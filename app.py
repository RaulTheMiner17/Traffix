import cv2
import threading
import time
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
from vidgear.gears import CamGear


app = Flask(__name__)
model = YOLO('yolov8n.pt')

# Class IDs for vehicles we want to detect from the COCO dataset
# bicycle: 1, car: 2, motorcycle: 3, bus: 5, truck: 7
CLASS_IDS_TO_DETECT = [1, 2, 3, 5, 7]
CLASS_NAMES = model.names

# Camera stream URLs and configurations
CAMERAS = [
    {'id': 'camera-1', 'name': 'Camera 1', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=y-Os52eW2rg', 'status': 'online'},
    {'id': 'camera-2', 'name': 'Camera 2', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=yy4FFZBdRqo', 'status': 'online'},
    {'id': 'camera-3', 'name': 'Camera 3', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=Uuaemo4RwFU', 'status': 'online'},
    {'id': 'camera-4', 'name': 'Camera 4', 'location': 'Mumbai', 'streamUrl': 'https://www.youtube.com/watch?v=48wtXs5okhE', 'status': 'online'}
]

# --- MULTI-THREADING SETUP ---
# This global dictionary will store the latest processed frame from each camera thread.
# It acts as a shared space between the processing threads and the Flask server.
output_frames = {}
#lock  to prevent race conditions
lock = threading.Lock()

def process_camera_stream(camera_info):
    """
    This is the core worker function that runs in a separate thread for each camera.
    It reads frames, performs detection, and updates the global `output_frames` dictionary.
    """
    global output_frames, lock

    camera_id = camera_info['id']
    video_url = camera_info['streamUrl']

    if not video_url:
        print(f"[{camera_id}] Error: No stream URL provided.")
        return

    try:
        stream = CamGear(source=video_url, stream_mode=True, logging=True).start()
        print(f"[{camera_id}] Successfully started stream.")
    except Exception as e:
        print(f"[{camera_id}] Error starting stream: {e}")
        return


    FRAME_SKIP = 4
    frame_count = 0

    while True:
        frame = stream.read()
        if frame is None:
            print(f"[{camera_id}] Stream ended or failed.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
     
        frame_resized = cv2.resize(frame, (640, 360))

        results = model.track(frame_resized, persist=True, classes=CLASS_IDS_TO_DETECT, verbose=False)
        processed_frame = results[0].plot() 

        with lock:
            #
            (flag, encodedImage) = cv2.imencode(".jpg", processed_frame)
            if flag:
                output_frames[camera_id] = encodedImage.tobytes()

    stream.stop()

def generate_frame_for_request(camera_id):
    """
    This is the generator function used by the Flask route.
    It continuously yields the latest processed frame for a specific camera ID.
    """
    global output_frames, lock
    while True:
        time.sleep(0.05) 
        with lock:
            if camera_id not in output_frames:
                continue
            frame_bytes = output_frames[camera_id]
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#FLASK ROUTES
@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/api/cameras')
def get_cameras():
    """API endpoint to provide the list of cameras to the frontend."""
    return jsonify(CAMERAS)

@app.route('/video_feed')
def video_feed():
    """
    The video streaming route. It now gets the camera ID from the request
    and uses the `generate_frame_for_request` generator to serve the processed frames.
    """
    camera_id = request.args.get('id')
    return Response(generate_frame_for_request(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#main
if __name__ == '__main__':
    #background thread for each cam
    for camera in CAMERAS:
        thread = threading.Thread(target=process_camera_stream, args=(camera,), daemon=True)
        thread.start()
    
    app.run(debug=False)