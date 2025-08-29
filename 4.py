#tracks but really bad 
#1. use trail5.zip, !unzip trail5.zip, 
#2. !cd Trial5, install ultra alitics lib: %pip install ultralytics

#4: tryign to track with IDs not switching so quick=> still really bad

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# allowlist DetectionModel for torch weights-only unpickler
torch.serialization.add_safe_globals([DetectionModel])

class SimpleCentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.object_history = {}  # Store recent positions for better matching
        self.history_length = 5

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.object_history[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.object_history:
            del self.object_history[object_id]

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

            # Use IoU and distance for better matching
            distances = []
            for obj_id, obj_centroid in zip(object_ids, object_centroids):
                row_distances = []
                for i, input_centroid in enumerate(input_centroids):
                    # Distance between centroids
                    centroid_dist = np.linalg.norm(np.array(obj_centroid) - np.array(input_centroid))
                    
                    # Add IoU if available (using bounding boxes)
                    if obj_id < len(input_boxes):
                        # Simple overlap bonus - objects that are closer get preference
                        overlap_bonus = max(0, self.max_distance - centroid_dist) / self.max_distance * 20
                        final_distance = centroid_dist - overlap_bonus
                    else:
                        final_distance = centroid_dist
                    
                    row_distances.append(final_distance)
                distances.append(row_distances)
            
            D = np.array(distances)

            # Use Hungarian algorithm approach - assign closest matches first
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_idxs = set()
            used_col_idxs = set()

            # Match existing objects to new detections
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update history
                if object_id not in self.object_history:
                    self.object_history[object_id] = []
                self.object_history[object_id].append(input_centroids[col])
                if len(self.object_history[object_id]) > self.history_length:
                    self.object_history[object_id].pop(0)

                used_row_idxs.add(row)
                used_col_idxs.add(col)

            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)

            # Handle unmatched existing objects
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Register new objects
            else:
                for col in unused_col_idxs:
                    self.register(input_centroids[col])

        # Return current tracks with better box matching
        tracks = []
        active_objects = list(self.objects.items())
        
        # Match active objects to input boxes by proximity
        for object_id, centroid in active_objects:
            best_box = None
            min_dist = float('inf')
            
            for i, box in enumerate(input_boxes):
                box_centroid = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                dist = np.linalg.norm(np.array(centroid) - np.array(box_centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_box = box
            
            if best_box is not None and min_dist < self.max_distance:
                tracks.append({
                    'id': object_id,
                    'bbox': best_box,
                    'centroid': centroid
                })
        
        return tracks

# Paths
VIDEO_IN = "vid1.mp4"
VIDEO_OUT = "out_tracked5.mp4"
WEIGHTS = "best.pt"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model
model = YOLO(WEIGHTS)
model.to(device)

# Init improved tracker with better parameters
tracker = SimpleCentroidTracker(max_disappeared=15, max_distance=120)

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