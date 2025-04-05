#git clone https://github.com/princeton-vl/RAFT.git

#Downloaded the raft-small.pth model (which you have)

#Installed dependencies (torch, opencv, realsense, etc.)

# raft_viewer.py - Visualize RAFT Optical Flow from RealSense stream
import os
import cv2
import time
import torch
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
import sys
import urllib.request

# Add RAFT path
sys.path.append(str(Path.home() / 'Antares/programs/RAFT/core'))
from raft import RAFT
from utils.utils import InputPadder
from argparse import Namespace

# Paths
model_dir = Path.home() / "Antares/programs/RAFT/models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "raft-small.pth"

# Auto-download model if missing
if not model_path.exists():
    print("Downloading RAFT-small model...")
    url = "https://dl.dropboxusercontent.com/s/6z2e3j7u5h3qybp/raft-small.pth"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# Load RAFT-small
device = 'cuda' if torch.cuda.is_available() else 'cpu'
raft_args = Namespace(small=True, mixed_precision=False, alternate_corr=False)
raft_model = torch.nn.DataParallel(RAFT(raft_args)).to(device)
raft_model.load_state_dict(torch.load(model_path, map_location=device))
raft_model.eval()

# Convert flow to HSV image
def flow_to_color(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 15, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Draw arrows
def draw_flow_arrows(image, flow, step=30):
    h, w = image.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            end = (int(x + fx), int(y + fy))
            cv2.arrowedLine(image, (x, y), end, (255, 255, 255), 1, tipLength=0.3)
    return image

# RealSense camera setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
pipeline.start(config)

last_frame = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        if last_frame is None:
            last_frame = frame
            continue

        # Prepare RAFT inputs
        image1 = torch.from_numpy(last_frame).permute(2, 0, 1).float()[None].to(device) / 255.0
        image2 = torch.from_numpy(frame).permute(2, 0, 1).float()[None].to(device) / 255.0
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        with torch.no_grad():
            _, flow_up = raft_model(image1, image2, iters=12, test_mode=True)

        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_img = flow_to_color(flow)
        blended = cv2.addWeighted(frame, 0.6, flow_img, 0.4, 0)
        draw_flow_arrows(blended, flow)

        cv2.imshow("RAFT Optical Flow", blended)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        last_frame = frame.copy()

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

