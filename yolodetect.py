import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
import glob
import time
import csv
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import psutil
import torch
from ultralytics import YOLO
import cv2


# -------------------- ARGUMENTS --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model')
parser.add_argument('--source', required=True, help='Image / folder / video / usb0')
parser.add_argument('--thresh', default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='WxH (e.g. 1280x720)')
parser.add_argument('--record', action='store_true', help='Record output video')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# -------------------- MODEL --------------------
if not os.path.exists(model_path):
    print('ERROR: Model not found')
    sys.exit(1)

model = YOLO(model_path)
labels = model.names

# -------------------- SOURCE --------------------
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    source_type = 'image' if ext.lower() in img_ext_list else 'video'
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid input source')
    sys.exit(1)

# -------------------- RESOLUTION --------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# -------------------- LOAD SOURCE --------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [
        f for f in glob.glob(img_source + '/*')
        if os.path.splitext(f)[1].lower() in img_ext_list
    ]
else:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)

# -------------------- RECORD --------------------
if record:
    if not user_res:
        print('Recording requires --resolution')
        sys.exit(1)

    recorder = cv2.VideoWriter(
        'processed_output.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH)
    )

# -------------------- CSV LOGGING --------------------
CSV_LOG_FILE = "tracking_log.csv"
CSV_FLUSH_INTERVAL = 30
csv_buffer = []

with open(CSV_LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "frame_id",
        "track_id",
        "class",
        "confidence",
        "x1", "y1", "x2", "y2"
    ])

frame_id = 0

# -------------------- METRICS --------------------
fps_buffer = []
fps_avg_len = 30
paused = False
object_counter = defaultdict(set)
img_count = 0

# -------------------- MAIN LOOP --------------------
while True:

    if not paused:
        t_start = time.perf_counter()

        if source_type in ['image', 'folder']:
            if img_count >= len(imgs_list):
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1
        else:
            ret, frame = cap.read()
            if not ret:
                break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # -------- YOLO TRACKING (BUILT-IN) --------
        results = model.track(
            frame,
            conf=min_thresh,
            persist=True,
            verbose=False
        )[0]

        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            frame_id += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
                if conf < min_thresh:
                    continue

                x1, y1, x2, y2 = map(int, box)
                label = labels[cls_id]

                object_counter[label].add(track_id)

                # Draw annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} | ID:{track_id} | {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                # Append CSV row
                csv_buffer.append([
                    timestamp,
                    frame_id,
                    track_id,
                    label,
                    round(float(conf), 3),
                    x1, y1, x2, y2
                ])

        # -------- FPS --------
        t_end = time.perf_counter()
        fps = 1 / max(t_end - t_start, 1e-6)
        fps_buffer.append(fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)

    # -------- SYSTEM METRICS --------
    cpu = psutil.cpu_percent()
    gpu = torch.cuda.utilization() if torch.cuda.is_available() else 0.0

    # -------- DASHBOARD --------
    y = 25
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"CPU: {cpu:.1f}%", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    cv2.putText(frame, f"GPU: {gpu:.1f}%", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 40

    for cls, ids_set in object_counter.items():
        cv2.putText(frame, f"{cls}: {len(ids_set)}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)
        y += 22

    # -------- CSV FLUSH --------
    if len(csv_buffer) >= CSV_FLUSH_INTERVAL:
        with open(CSV_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_buffer)
        csv_buffer.clear()

    # -------- DISPLAY --------
    cv2.imshow("Real-Time Vehicle Tracking GUI", frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5) & 0xFF
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        paused = not paused

# -------------------- FINAL CSV FLUSH --------------------
if csv_buffer:
    with open(CSV_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_buffer)

# -------------------- CLEANUP --------------------
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()

print(f"Average FPS: {avg_fps:.2f}")
print("CSV log saved to tracking_log.csv")