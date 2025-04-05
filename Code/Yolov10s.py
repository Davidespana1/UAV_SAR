import os
import cv2
import time
import torch
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from urllib.request import urlretrieve

# === Configuration ===
MODEL_NAME = "yolov10s"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Running on: {device.upper()} | Model: {MODEL_NAME}")

# === Paths and download ===
model_path = os.path.expanduser(f"~/weights/{MODEL_NAME}.pt")
model_url = f"https://github.com/THU-MIG/yolov10/releases/download/v1.1/{MODEL_NAME}.pt"

if not os.path.exists(model_path):
    print(f"📥 Downloading {MODEL_NAME}.pt...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urlretrieve(model_url, model_path)
    print("✅ Model download complete!")

# === Load YOLO model ===
model = YOLO(model_path).to(device)

# === RealSense setup ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

# === Depth filters ===
spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
temporal_filter = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

# === Helper: Get average depth in area (meters) ===
def get_mean_depth(depth_frame, x, y, size=5):
    values = []
    half = size // 2
    width, height = depth_frame.get_width(), depth_frame.get_height()

    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                d = depth_frame.get_distance(nx, ny)
                if 0.2 < d < 10:  # filter invalid readings
                    values.append(d)
    return np.mean(values) if values else 0

# === Main loop ===
last_update = 0
frame_count = 0
start_time = time.time()

try:
    while True:
        # Capture and align frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Apply depth filters
        filtered_depth = spatial_filter.process(depth_frame)
        filtered_depth = temporal_filter.process(filtered_depth)
        filtered_depth = hole_filling.process(filtered_depth).as_depth_frame()

        # Convert color image
        img = np.asanyarray(color_frame.get_data())

        # Run YOLO inference
        inference_start = time.time()
        results = model.predict(img, conf=0.5, device=device, verbose=False)
        inference_end = time.time()

        # Process detections
        if results and results[0].boxes:
            current_time = time.time()
            if current_time - last_update >= 1:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    depth_m = get_mean_depth(filtered_depth, cx, cy, size=5)
                    depth_ft = depth_m * 3.28084  # Convert to feet

                    if depth_ft > 1.0:  # Ignore < 1 ft
                        label = model.names[int(box.cls)]
                        print(f"🎯 {label} | Distance: {depth_ft:.2f} ft")
                last_update = current_time

        # FPS tracking
        frame_count += 1
        if frame_count >= 30:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"⚡ FPS: {fps:.2f} | Inference: {(inference_end - inference_start) * 1000:.1f} ms")
            frame_count = 0
            start_time = time.time()

        # Display
        annotated = results[0].plot()
        cv2.imshow(f"YOLOv10s - Depth in Feet", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
