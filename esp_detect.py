import sys
import time
import argparse
import json
import os

import cv2
from ultralytics import YOLO
from collections import defaultdict   


# ==========================
# ARGUMENTS
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)

args = parser.parse_args()

model_path = args.model
rtsp_url = args.source
min_thresh = float(args.thresh)


# ==========================
# LOAD MODEL
# ==========================
print("Loading model...")
model = YOLO(model_path)
print("Model loaded")


# ==========================
# VIDEO STREAM
# ==========================
print("Opening RTSP...")
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("ERROR: Cannot open stream")
    sys.exit(1)

print("RTSP connected")


# ==========================
# SETTINGS
# ==========================
PERSON_CLASS_ID = 0
COUNT_FILE = "count.json"

# Load saved count
if os.path.exists(COUNT_FILE):
    with open(COUNT_FILE, "r") as f:
        total_count = json.load(f).get("total", 0)
else:
    total_count = 0

# Tracking memory
crossed_ids = set()
last_positions = defaultdict(lambda: None)  


# ==========================
# MAIN LOOP
# ==========================
while True:

 ret, frame = cap.read()

 if not ret or frame is None:
    print("ESP32 stream dropped, reconnecting...")
    cap.release()
    time.sleep(1)
    cap = cv2.VideoCapture(rtsp_url)
    continue

    # Resize (Jetson optimization)
    frame = cv2.resize(frame, (480, 320))
    height, width = frame.shape[:2]

    LINE_X = int(width * 0.6)

    # Draw counting line
    cv2.line(frame, (LINE_X, 0), (LINE_X, height), (0, 0, 255), 3)

    # ==========================
    # YOLO TRACKING (KEY FIX)
    # ==========================
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            cls = int(box.cls[0])
            if cls != PERSON_CLASS_ID:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, int((y1+y2)/2)), 5, (255, 0, 0), -1)

            prev_x = last_positions[track_id]

            # ==========================
            # CROSSING LOGIC
            # ==========================
            if prev_x is not None:

                if prev_x < LINE_X and center_x >= LINE_X:

                    if track_id not in crossed_ids:

                        total_count += 1
                        crossed_ids.add(track_id)

                        print(f"Count: {total_count}")

                        # Save to file (ONLY when increment happens)
                        with open(COUNT_FILE, "w") as f:
                            json.dump({"total": total_count}, f)

            last_positions[track_id] = center_x

    # ==========================
    # DISPLAY
    # ==========================
    cv2.putText(frame, f"Count: {total_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO RTSP Counter", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
