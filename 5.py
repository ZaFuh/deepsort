# 4 : Tracked but kept changing IDs fist before the real ( first try on colab)

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# allowlist DetectionModel for torch weights-only unpickler
torch.serialization.add_safe_globals([DetectionModel])

class SimpleCentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            # Mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        # Extract centroids from detections
        input_centroids = []
        input_boxes = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append([cx, cy])
            input_boxes.append([x1, y1, x2, y2])

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            # Compute distances between existing objects and new centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_idxs = set()
            used_col_idxs = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_row_idxs.add(row)
                used_col_idxs.add(col)

            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_idxs:
                    self.register(input_centroids[col])

        # Return current tracks
        tracks = []
        for i, (object_id, centroid) in enumerate(self.objects.items()):
            if i < len(input_boxes):
                tracks.append({
                    'id': object_id,
                    'bbox': input_boxes[i],
                    'centroid': centroid
                })
        
        return tracks

# Paths
VIDEO_IN = "vid1.mp4"
VIDEO_OUT = "out_tracked4.mp4"
WEIGHTS = "best.pt"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model
model = YOLO(WEIGHTS)
model.to(device)

# Init simple tracker
tracker = SimpleCentroidTracker(max_disappeared=30, max_distance=100)

# Open video
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video {VIDEO_IN}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

frame_count = 0

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # YOLO inference
    results = model(frame, verbose=False, conf=0.3)[0]

    detections = []
    if hasattr(results, "boxes") and results.boxes is not None and len(results.boxes) > 0:
        for b in results.boxes:
            xyxy = b.xyxy.cpu().numpy().flatten().tolist()
            conf = float(b.conf.cpu().item())
            cls = int(b.cls.cpu().item())
            
            if conf >= 0.3:
                detections.append({
                    'bbox': xyxy,
                    'conf': conf,
                    'cls': cls
                })

    # Update tracker
    tracks = tracker.update(detections)
    
    print(f"Frame {frame_count}: {len(detections)} detections -> {len(tracks)} tracks")

    # Draw tracks
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        track_id = track['id']

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Write frame
    writer.write(frame)
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
writer.release()
print(f"✅ Saved: {VIDEO_OUT}")