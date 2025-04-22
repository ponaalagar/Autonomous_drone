from flask import Flask, Response, render_template
import cv2
import time

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Error: Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Webcam Stream Test</title>
    </head>
    <body>
        <h1>Webcam Stream Test</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    finally:
        cap.release()
