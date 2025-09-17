# pi_detect.py - YOLOv5n inference on Pi Zero 2W
import torch
import cv2
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

def main():
    print("🚀 Initializing YOLOv5n on Pi Zero 2W...")

    # Select CPU
    device = select_device('cpu')

    # Load model properly (no torch.load nonsense)
    model = DetectMultiBackend("yolov5n.pt", device=device)
    model.eval()

    # Open camera (USB or PiCam via /dev/video0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Failed to open camera (/dev/video0)")
        return

    print("✅ Camera ready, starting detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Frame grab failed")
            continue

        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.0
        img = img.to(device)

        # Run inference
        with torch.no_grad():
            pred = model(img)
            pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)[0]

        if pred is not None and len(pred):
            print("🎯 Detected objects:", pred[:, -1].cpu().numpy())

if __name__ == "__main__":
    main()
