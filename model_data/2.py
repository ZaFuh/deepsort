import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# allowlist DetectionModel for torch weights-only unpickler (safe if you trust weights)
torch.serialization.add_safe_globals([DetectionModel])

# import your DeepSort implementation (adjust path if different)
from deep_sort.deep_sort.tracker import Tracker as DeepSort # Corrected import path
from deep_sort.deep_sort import nn_matching # Import nn_matching for the metric
from deep_sort.deep_sort.detection import Detection # Import Detection

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return [cx, cy, w, h]

VIDEO_IN = "vid.mp4"    # upload to Colab or mount Drive
VIDEO_OUT = "out_tracked.avi"
WEIGHTS = "best.pt"     # upload your best.pt

# use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load YOLO model
model = YOLO(WEIGHTS)
model.to(device) # Move model to the specified device

# init DeepSort (adjust args to match your DeepSort constructor)
max_cosine_distance = 0.4 # Example value, may need tuning
nn_budget = None # Example value, may need tuning
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = DeepSort(metric) # Initialize with metric

# Modified video writer configuration
cap = cv2.VideoCapture(VIDEO_IN)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

ret, frame = cap.read()
while ret:
    # inference (Ultralytics returns a Results object)
    results = model(frame)[0]

    detections = []
    if hasattr(results, "boxes") and len(results.boxes) > 0:
        for b in results.boxes:
            xyxy = b.xyxy.cpu().numpy().flatten().tolist() if hasattr(b.xyxy, "cpu") else b.xyxy.tolist()
            conf = float(b.conf.cpu().item()) if hasattr(b.conf, "cpu") else float(b.conf)
            cls = int(b.cls.cpu().item()) if hasattr(b.cls, "cpu") else int(b.cls)

            # Create a DeepSort Detection object
            # The Detection constructor typically takes bbox (xywh), confidence, and class_id
            bbox_xywh = xyxy_to_xywh(xyxy)
            detections.append(Detection(bbox_xywh, conf, cls))


    # update tracker with the list of detections
    tracks = tracker.update(detections)


    # draw tracks (expects tracks iterable of [x1,y1,x2,y2,track_id,cls_id])
    if tracks is not None:
        for track in tracks:
            bbox = track.to_tlbr() # Get the bounding box in tlbr format
            x1, y1, x2, y2 = map(int, bbox)
            track_id = int(track.track_id)

            # DeepSORT doesn't return class ID directly in the track object by default.
            # If you need the class ID, you might need to store it with the detection
            # and retrieve it based on the track ID, or modify the DeepSORT code.
            # For now, we'll just draw the bounding box and track ID.

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    writer.write(frame)

    # show in Colab (optional; use cv2_imshow from google.colab.patches)
    # from google.colab.patches import cv2_imshow
    # cv2_imshow(frame)

    ret, frame = cap.read()

cap.release()
writer.release()
print("Saved:", VIDEO_OUT)

# this has tracking unlike 1.py (remove 1st 2 lines from while loop)