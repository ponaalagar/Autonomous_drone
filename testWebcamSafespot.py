import cv2
import numpy as np
import time

def find_safe_landing_spot(frame):
    """
    Detect a safe landing spot by finding a region with low variance (indicating a flat surface).
    Returns the frame with a green rectangle around the safest region.
    """
    if frame is None or frame.size == 0:
        print("Warning: Empty frame received.")
        return frame

    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize for faster processing
    gray = cv2.resize(gray, (320, 240))
    
    # Divide the image into a 4x4 grid
    grid_size = 4
    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    min_variance = float('inf')
    best_region = None
    
    for i in range(grid_size):
        for j in range(grid_size):
            x, y = j * cell_w, i * cell_h
            region = gray[y:y+cell_h, x:x+cell_w]
            if region.size == 0:
                continue
            variance = np.var(region)
            if variance < min_variance:
                min_variance = variance
                best_region = (x * 2, y * 2, cell_w * 2, cell_h * 2)  # Scale back to original size
    
    if best_region:
        x, y, w, h = best_region
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Safe Spot", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    """
    Capture webcam feed, process for safe landing spot, and display in an OpenCV window.
    """
    # Try multiple webcam indices
    cap = None
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Webcam opened at index {index} with V4L2")
            break
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam opened at index {index} with default backend")
            break
    if not cap or not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            time.sleep(0.5)
            continue
        
        # Process frame for safe landing spot
        frame = find_safe_landing_spot(frame)
        
        # Display frame
        cv2.imshow("Safe Landing Detection", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Control frame rate
        time.sleep(0.1)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
