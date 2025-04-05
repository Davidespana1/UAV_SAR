# yolo_raft_combined.py - YOLOv10 + RAFT + Smoothed Depth + Arrows + Feedback
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import time
from pathlib import Path
from ultralytics import YOLO

# RAFT setup
import sys
sys.path.append(str(Path.home() / 'Antares/programs/RAFT/core'))
from raft import RAFT
from utils.utils import InputPadder
from argparse import Namespace

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv10n
model_path = str(Path.home() / "weights/yolov10n.pt")
yolo_model = YOLO(model_path)

# Load RAFT-small
raft_args = Namespace(small=True, mixed_precision=False, alternate_corr=False)
raft_model = torch.nn.DataParallel(RAFT(raft_args)).to(device)
raft_weights = str(Path.home() / 'Antares/programs/RAFT/models/raft-small.pth')
raft_model.load_state_dict(torch.load(raft_weights, map_location=device))
raft_model.eval()

# Flow helpers
def flow_to_image(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 15, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def draw_flow_arrows_within_boxes(image, flow, boxes, step=30):
    for (x1, y1, x2, y2) in boxes:
        for y in range(y1, y2, step):
            for x in range(x1, x2, step):
                if y >= flow.shape[0] or x >= flow.shape[1]:
                    continue
                fx, fy = flow[y, x]
                end = (int(x + fx), int(y + fy))
                cv2.arrowedLine(image, (x, y), end, (255, 255, 255), 1, tipLength=0.3)
    return image

# Average depth in a 3x3 window
def get_average_depth(depth_frame, cx, cy):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    depths = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            x = min(max(cx + dx, 0), w - 1)
            y = min(max(cy + dy, 0), h - 1)
            d = depth_frame.get_distance(x, y)
            if 0.3 < d < 30.0:  # valid range in meters
                depths.append(d)
    if depths:
        return sum(depths) / len(depths)
    return 0.0

# Classes of interest
TARGET_CLASSES = ["person", "car", "truck", "bus", "dog", "cat", "bird", "drone"]

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
pipeline.start(config)

align = rs.align(rs.stream.color)
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

last_frame = None
frame_count = 0
flow_resized = None
last_no_person_msg = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        resized_frame = cv2.resize(frame, (416, 416))

        # YOLO detection
        results = yolo_model.predict(resized_frame, conf=0.5, imgsz=416, verbose=False)
        boxes = []
        labels = []
        centers = []
        person_detected = False

        for box in results[0].boxes:
            label = yolo_model.names[int(box.cls)]
            if label in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                scale_x = frame.shape[1] / 416
                scale_y = frame.shape[0] / 416
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
                centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
                if label == "person":
                    person_detected = True

        current_time = time.time()
        if not person_detected and current_time - last_no_person_msg >= 1:
            print("No person detected in frame.")
            last_no_person_msg = current_time

        # RAFT flow every 3rd frame
        if last_frame is not None and frame_count % 3 == 0:
            image1 = cv2.resize(last_frame, (320, 240))
            image2 = cv2.resize(frame, (320, 240))

            t1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None].to(device) / 255.0
            t2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None].to(device) / 255.0
            padder = InputPadder(t1.shape)
            t1, t2 = padder.pad(t1, t2)

            with torch.no_grad():
                _, flow_up = raft_model(t1, t2, iters=12, test_mode=True)

            flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_resized = cv2.resize(flow, (frame.shape[1], frame.shape[0]))

        # Visualize output
        output = frame.copy()

        if flow_resized is not None:
            flow_img = flow_to_image(flow_resized)
            blended = cv2.addWeighted(output, 0.7, flow_img, 0.3, 0)
            draw_flow_arrows_within_boxes(blended, flow_resized, boxes)
        else:
            blended = output

        # Draw bounding boxes and depth
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            label = labels[i]
            cx, cy = centers[i]
            avg_depth = get_average_depth(depth_frame, cx, cy)
            distance_ft = avg_depth * 3.28084  # meters to feet

            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label}"
            if label == "person" and avg_depth > 0:
                label_text += f" {distance_ft:.1f} ft"
                print(f"Detected person at ~{distance_ft:.1f} feet")
            cv2.putText(blended, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("YOLO + RAFT + Depth", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        last_frame = frame.copy()
        frame_count += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
