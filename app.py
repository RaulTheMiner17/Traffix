from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import cvzone
from vidgear.gears import CamGear

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Class IDs for car, bus, truck, bicycle, motorcycle 
# bicycle: 1, car: 2, motorcycle: 3, bus: 5, truck: 7
class_ids_to_detect = [1, 2, 3, 5, 7]
class_names = model.names

CAMERAS = [
    {
        'id': 'camera-1',
        'name': 'Camera 1',
        'location': 'Mumbai',
        'streamUrl': 'https://www.youtube.com/watch?v=y-Os52eW2rg',
        'status': 'online'
    },
    {
        'id': 'camera-2',
        'name': 'Camera 2',
        'location': 'Mumbai',
        'streamUrl': '', 
        'status': 'online'
    },
    {
        'id': 'camera-3',
        'name': 'Camera 3',
        'location': 'Mumbai',
        'streamUrl': '', 
        'status': 'online'
    },
    {
        'id': 'camera-4',
        'name': 'Camera 4',
        'location': 'Mumbai',
        'streamUrl': '',
        'status': 'online'
    }
]


def generate_frames(video_url):
    """
    Generator function to process a specific video URL and yield frames.
    """
    if not video_url:
        print("Error: No video URL provided.")
        return

    try:
        # Start video stream 
        stream = CamGear(source=video_url, stream_mode=True, logging=True).start()
        print(f"Successfully started stream for URL: {video_url}")
    except Exception as e:
        print(f"Error starting video stream for {video_url}: {e}")
        return

    while True:
        frame = stream.read()
        if frame is None:
            print(f"Stream ended or failed for {video_url}")
            break

        # Run YOLOv8 tracking 
        results = model.track(frame, persist=True, classes=class_ids_to_detect)

        # Draw boxes and labels on the frame
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                class_name = class_names[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(
                    frame,
                    f'{class_name} ID:{track_id}',
                    (x1, y1),
                    scale=1.5,
                    thickness=2,
                    offset=5
                )

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

    stream.stop()

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

#API ENDPOINT 
@app.route('/api/cameras')
def get_cameras():
    """API endpoint to provide the list of cameras to the frontend."""
    return jsonify(CAMERAS)

# VIDEO FEED 
@app.route('/video_feed')
def video_feed():
    """
    Video streaming route that takes the video URL as a query parameter.
    """
    video_url = request.args.get('url')
    return Response(generate_frames(video_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)