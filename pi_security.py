# pi_security.py - Pi Zero 2W + SIM7600G-H Security System with YOLOv5n
import os
import time
import json
import cv2
import torch
import serial
import numpy as np
from datetime import datetime, timedelta

# Import YOLOv5 backend
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device


class PiSecuritySystem:
    def __init__(self):
        print("🚀 Initializing Pi Security System...")

        # Load config
        self.config = self.load_config()

        # Initialize camera
        self.cap = self.init_camera()

        # Initialize detector
        self.model, self.device, self.names = self.init_detector()

        # Initialize SMS
        self.sms = self.init_sms()

        # Alert management
        self.last_alert = {}
        self.cooldown_minutes = self.config.get("alert_cooldown_minutes", 5)

        print("✅ Pi Security System ready!")

    def load_config(self):
        """Load or create configuration"""
        config_file = "pi_security_config.json"
        default_config = {
            "phone_number": "+1234567890",
            "sim7600_port": "/dev/ttyUSB2",
            "model_path": "yolov5n.pt",
            "detection_confidence": 0.4,
            "alert_cooldown_minutes": 5,
            "target_objects": ["person", "cat", "dog"],
            "camera_resolution": [640, 480],
            "inference_size": 416,
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Config load error: {e}")

        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=4)
        print("⚠️ Created default config. Edit pi_security_config.json with your phone number!")
        return default_config

    def init_camera(self):
        """Open /dev/video0 camera (USB or PiCam driver)"""
        w, h = self.config["camera_resolution"]
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not cap.isOpened():
            raise RuntimeError("✗ Failed to open camera (/dev/video0)")
        print("✅ Camera ready (/dev/video0)")
        return cap

    def init_detector(self):
        """Load YOLOv5n model with DetectMultiBackend"""
        model_path = self.config.get("model_path", "yolov5n.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"✗ Model file not found: {model_path}\n"
                "Download with:\n"
                "wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
            )

        device = select_device("cpu")
        model = DetectMultiBackend(model_path, device=device)
        model.eval()

        names = model.names if hasattr(model, "names") else {}
        print("✅ YOLOv5n model loaded")
        return model, device, names

    def init_sms(self):
        """Initialize SIM7600G-H for SMS alerts"""
        try:
            print("📡 Connecting to SIM7600G-H...")
            ser = serial.Serial(
                port=self.config["sim7600_port"], baudrate=115200, timeout=10
            )
            time.sleep(2)

            ser.write(b"AT\r\n")
            time.sleep(1)
            if "OK" not in ser.read(ser.in_waiting).decode(errors="ignore"):
                raise Exception("AT command failed")

            ser.write(b"AT+CMGF=1\r\n")
            time.sleep(1)
            ser.read(ser.in_waiting)

            print("✅ SIM7600G-H ready")
            return ser
        except Exception as e:
            print(f"✗ SIM7600G-H init failed: {e}")
            return None

    def send_sms(self, message):
        """Send SMS alert"""
        if not self.sms:
            print("⚠️ SMS not initialized")
            return
        try:
            number = self.config["phone_number"]
            self.sms.write(f'AT+CMGS="{number}"\r'.encode())
            time.sleep(1)
            self.sms.write(message.encode() + b"\x1A")
            time.sleep(3)
            print(f"📨 SMS sent to {number}: {message}")
        except Exception as e:
            print(f"✗ SMS send failed: {e}")

    def run(self):
        """Main detection loop"""
        print("🔍 Starting detection loop...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("✗ Frame grab failed")
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img = img.to(self.device)

            with torch.no_grad():
                pred = self.model(img)
                pred = non_max_suppression(
                    pred,
                    conf_thres=self.config["detection_confidence"],
                    iou_thres=0.45,
                )[0]

            if pred is not None and len(pred):
                for *box, conf, cls in pred.cpu().numpy():
                    label = self.names.get(int(cls), "unknown")
                    if label in self.config["target_objects"]:
                        now = datetime.now()
                        last_time = self.last_alert.get(label, datetime.min)
                        if now - last_time > timedelta(
                            minutes=self.cooldown_minutes
                        ):
                            msg = f"ALERT: {label} detected ({conf:.2f}) at {now}"
                            print("🚨", msg)
                            self.send_sms(msg)
                            self.last_alert[label] = now

            time.sleep(0.2)


def main():
    try:
        system = PiSecuritySystem()
        system.run()
    except Exception as e:
        print(f"✗ System failed: {e}")


if __name__ == "__main__":
    main()
