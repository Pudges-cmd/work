# pi_security.py - Complete security system for Pi Zero 2W + SIM7600G-H
import os
import time
import json
import torch
import cv2
import serial
import numpy as np
from datetime import datetime, timedelta
import subprocess
import tempfile


class PiSecuritySystem:
    def __init__(self):
        print("🚀 Initializing Pi Security System...")

        # Load config
        self.config = self.load_config()

        # Initialize camera
        self.camera = self.init_camera()

        # Initialize detector
        self.detector = self.init_detector()

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
            "detection_interval": 5,
            "camera_resolution": [640, 480],
            "inference_size": 416,
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                print("✓ Config loaded")
                return config
            except Exception as e:
                print(f"Config load error: {e}")

        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=4)

        print(f"✓ Created default config: {config_file}")
        print("⚠️ Edit pi_security_config.json with your phone number!")
        return default_config

    def init_camera(self):
        """Initialize Pi Camera using rpicam-still"""
        try:
            print("Initializing Pi Camera with rpicam-still...")

            result = subprocess.run(["which", "rpicam-still"], capture_output=True)
            if result.returncode != 0:
                raise Exception("rpicam-still not found. Install with: sudo apt install rpicam-apps")

            test_cmd = ["rpicam-still", "-t", "1", "--nopreview", "-o", "/dev/null"]
            result = subprocess.run(test_cmd, capture_output=True, timeout=15)

            if result.returncode == 0:
                print("✓ Pi Camera ready (rpicam-still)")
                return {"type": "rpicam"}
            else:
                raise Exception("Camera test failed")

        except Exception as e:
            print(f"✗ Camera init failed: {e}")
            raise

    def init_detector(self):
        """Initialize offline YOLOv5n detector"""
        try:
            print("Loading offline YOLOv5n...")

            model_path = self.config.get("model_path", "yolov5n.pt")
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                print("wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt")
                raise FileNotFoundError(f"Model file {model_path} not found")

            device = torch.device("cpu")

            # Try safe load first
            try:
                model = torch.load(model_path, map_location=device, weights_only=True)
            except Exception:
                print("⚠️ Falling back: loading model with full unpickling...")
                model = torch.load(model_path, map_location=device)

            if isinstance(model, dict):
                if "model" in model:
                    model = model["model"]
                elif "ema" in model:
                    model = model["ema"]

            model.eval()
            model.to(device)
            torch.set_num_threads(2)

            # Subset of COCO class names
            self.names = {0: "person", 15: "cat", 16: "dog"}

            print("✓ Offline YOLOv5n ready")
            return model

        except Exception as e:
            print(f"✗ Detector init failed: {e}")
            raise

    def init_sms(self):
        """Initialize SIM7600G-H SMS"""
        try:
            print("Connecting to SIM7600G-H...")
            port = self.config["sim7600_port"]

            ser = serial.Serial(port=port, baudrate=115200, timeout=10)
            time.sleep(2)

            ser.write(b"AT\r\n")
            time.sleep(1)
            response = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")

            if "OK" not in response:
                raise Exception("AT command failed")

            ser.write(b"AT+CMGF=1\r\n")
            time.sleep(1)
            ser.read(ser.in_waiting)

            print("✓ SIM7600G-H ready")
            return ser

        except Exception as e:
            print(f"✗ SIM7600G-H init failed: {e}")
            return None

    def capture_frame(self):
        """Capture frame using rpicam-still"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_name = tmp.name

            w, h = self.config["camera_resol_]()
