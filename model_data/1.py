import os
import cv2
from ultralytics import YOLO

video_path = os.path.join(".", "vid.mp4")

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_path = 'output_video3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID') # Or 'mp4v'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

model = YOLO("best.pt")

ret, frame = cap.read()

while ret:


    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    ret, frame = cap.read()

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_path}")