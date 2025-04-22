from flask import Flask, Response, render_template
import cv2
import numpy as np
from picamera2 import Picamera2
import io
import time

app = Flask(__name__)

# Initialize the Pi Camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

def find_safe_landing_spot(frame):
    """
    Detect a safe landing spot by finding a region with low variance (indicating a flat surface).
    Returns the frame with a green rectangle around the safest region, and the coordinates.
    """
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize for faster processing (optional, adjust based on Pi performance)
    gray = cv2.resize(gray, (320, 240))
    
    # Divide the image into a grid (e.g., 4x4) to analyze regions
    grid_size = 4
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    min_variance = float('inf')
    best_region = None
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Extract the region
            x, y = j * cell_w, i * cell_h
            region = gray[y:y+cell_h, x:x+cell_w]
            
            # Compute variance (low variance = uniform, likely safe)
            variance = np.var(region)
            
            if variance < min_variance:
                min_variance = variance
                best_region = (x * 2, y * 2, cell_w * 2, cell_h * 2)  # Scale back to original size
    
    # Draw a green rectangle around the safest region
    if best_region:
        x, y, w, h = best_region
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Safe Spot", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def generate_frames():
    """
    Generate frames from the camera, process them, and yield for streaming.
    """
    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convert from RGB (picamera2) to BGR (OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process frame to find safe landing spot
        frame = find_safe_landing_spot(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Control frame rate (adjust for Pi performance)
        time.sleep(0.1)

@app.route('/')
def index():
    """
    Render the homepage with the video stream.
    """
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drone Safe Landing Stream</title>
    </head>
    <body>
        <h1>Drone Camera - Safe Landing Detection</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    """
    Stream the processed video feed.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run Flask app (host='0.0.0.0' allows access from other devices on the network)
    app.run(host='0.0.0.0', port=5000, threaded=True)
