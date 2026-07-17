import sys
import time
import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO


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

model = YOLO(
    model_path,
    task="detect"
)

print("Model loaded")



# ==========================
# RTSP
# ==========================

print("Opening RTSP...")

cap = cv2.VideoCapture(
    rtsp_url
)


if not cap.isOpened():

    print("ERROR: Cannot open RTSP")
    sys.exit(1)


print("RTSP connected")



# ==========================
# SETTINGS
# ==========================

PERSON_CLASS_ID = 0


# Vertical line position
# Increase value = move line right

LINE_X = 400



COUNT_FILE = "count.json"


if os.path.exists(COUNT_FILE):

    with open(COUNT_FILE,"r") as f:

        total_count = json.load(f).get(
            "total",
            0
        )

else:

    total_count = 0



previous_people = {}

next_id = 0



fps_buffer = []

fps_buffer_size = 50

avg_fps = 0



# ==========================
# LOOP
# ==========================

while True:


    start = time.perf_counter()


    ret, frame = cap.read()


    if not ret:

        print("RTSP frame failed")
        break



    height, width = frame.shape[:2]



    # ==========================
    # VERTICAL COUNTING LINE
    # ==========================

    cv2.line(
        frame,
        (LINE_X,0),
        (LINE_X,height),
        (0,0,255),
        3
    )



    results = model(
        frame,
        verbose=False
    )


    detections = results[0].boxes


    human_count = 0

    current_people = {}



    for i in range(len(detections)):


        class_id = int(
            detections[i].cls.item()
        )


        # Only humans

        if class_id != PERSON_CLASS_ID:

            continue



        confidence = detections[i].conf.item()


        if confidence < min_thresh:

            continue



        human_count += 1



        box = (
            detections[i]
            .xyxy
            .cpu()
            .numpy()
            .astype(int)
            .squeeze()
        )


        xmin,ymin,xmax,ymax = box



        center_x = int(
            (xmin+xmax)/2
        )

        center_y = int(
            (ymin+ymax)/2
        )



        person_id = None


        for pid,pos in previous_people.items():

            distance = (
                abs(center_x-pos[0]) +
                abs(center_y-pos[1])
            )


            if distance < 60:

                person_id = pid
                break



        if person_id is None:

            person_id = next_id
            next_id += 1



        current_people[person_id] = (
            center_x,
            center_y
        )



        # ==========================
        # VERTICAL LINE CROSSING
        # ==========================

        if person_id in previous_people:


            old_x = previous_people[person_id][0]


            # Left to right

            if old_x < LINE_X and center_x >= LINE_X:

                total_count += 1


            # Right to left

            elif old_x > LINE_X and center_x <= LINE_X:

                total_count += 1



            with open(COUNT_FILE,"w") as f:

                json.dump(
                    {"total":total_count},
                    f
                )



        cv2.rectangle(
            frame,
            (xmin,ymin),
            (xmax,ymax),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            f"Person {confidence:.2f}",
            (xmin,ymin-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )



    previous_people = current_people.copy()



    # FPS

    end=time.perf_counter()

    fps=1/(end-start)


    fps_buffer.append(fps)


    if len(fps_buffer)>fps_buffer_size:

        fps_buffer.pop(0)


    avg_fps=np.mean(fps_buffer)



    cv2.putText(
        frame,
        f"Humans: {human_count}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )


    cv2.putText(
        frame,
        f"Crossed: {total_count}",
        (10,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {avg_fps:.2f}",
        (10,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )



    cv2.imshow(
        "RTSP Human Counter",
        frame
    )



    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cap.release()

cv2.destroyAllWindows()



with open(COUNT_FILE,"w") as f:

    json.dump(
        {"total":total_count},
        f
    )


print(
    f"Final count: {total_count}"
)
