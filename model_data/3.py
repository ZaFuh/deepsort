import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# allowlist DetectionModel for torch weights-only unpickler (safe if you trust weights)
torch.serialization.add_safe_globals([DetectionModel])

# DeepSORT imports
from deep_sort.deep_sort.tracker import Tracker as DeepSort
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

def xyxy_to_xywh(box):
    """Convert [x1,y1,x2,y2] box to [cx,cy,w,h]."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return [cx, cy, w, h]

# Paths
VIDEO_IN = "vid.mp4"      # input video
VIDEO_OUT = "out_tracked2.mp4"
WEIGHTS = "best.pt"       # YOLO weights

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
model = YOLO(WEIGHTS)
model.to(device)

# Init DeepSORT
max_cosine_distance = 0.4
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = DeepSort(metric)

# Open video
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video {VIDEO_IN}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Use mp4v codec for safer mp4 playback
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, verbose=False)[0]

    detections = []
    if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
        for b in results.boxes:
            xyxy = b.xyxy.cpu().numpy().flatten().tolist()
            conf = float(b.conf.cpu().item())
            cls = int(b.cls.cpu().item())
            bbox_xywh = xyxy_to_xywh(xyxy)
            detections.append(Detection(bbox_xywh, conf, cls))

    # Update tracker
    tracks = tracker.update(detections)

    # Draw tracks
    if tracks is not None:
        for track in tracks:
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            track_id = int(track.track_id)

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Ensure frame size matches writer
    frame_resized = cv2.resize(frame, (w, h))
    writer.write(frame_resized)

cap.release()
writer.release()
print("✅ Saved video:", VIDEO_OUT)
