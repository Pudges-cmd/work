# pi_security_2025.py - Pi Zero 2W + SIM7600G-H Security System with Robust Human Detection
import os
import time
import json
import cv2
import serial
from datetime import datetime, timedelta
from ultralytics import YOLO

class PiSecuritySystem2025:
    def __init__(self):
        print("🚀 Initializing Pi Security System 2025...")

        # Load config
        self.config = self.load_config()

        # Initialize SMS
        self.sms = self.init_sms()

        # Human Detector
        self.detector = self.init_detector()

        # Alert management
        self.last_alert = {}
        self.cooldown_minutes = self.config.get("alert_cooldown_minutes", 5)

        print("✅ Pi Security System 2025 ready!")

    def load_config(self):
        """Load or create configuration"""
        config_file = "pi_security_config_2025.json"
        default_config = {
            "phone_number": "+1234567890",
            "sim7600_port": "/dev/ttyUSB2",
            "detection_confidence": 0.5,
            "alert_cooldown_minutes": 5,
            "camera_resolution": [640, 480],
            "detection_duration_sec": 300
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Config load error: {e}")

        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=4)
        print("⚠️ Created default config. Edit pi_security_config_2025.json with your phone number!")
        return default_config

    def init_detector(self):
        """Initialize robust human detector (YOLOv5n)"""
        try:
            print("📥 Loading YOLOv5n human detector...")
            detector = YOLO("yolov5n.pt")
            return detector
        except Exception as e:
            raise RuntimeError(f"✗ Failed to load YOLO model: {e}")

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

    def capture_frame(self):
        """Try multiple camera methods (OpenCV or libcamera)"""
        # Use OpenCV
        try:
            cap = cv2.VideoCapture(0)
            w, h = self.config["camera_resolution"]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
        except:
            pass

        # Could add libcamera fallback here if needed
        return None

    def detect_humans(self, frame):
        """Detect humans using YOLOv5n"""
        if frame is None:
            return 0, []

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector(frame_rgb, verbose=False)

            human_count = 0
            human_boxes = []
            for box in results[0].boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.config["detection_confidence"]:
                    human_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    human_boxes.append((x1, y1, x2, y2, confidence))
            return human_count, human_boxes
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return 0, []

    def run_detection_loop(self):
        """Main detection loop"""
        duration = self.config.get("detection_duration_sec", 300)
        start_time = time.time()
        print(f"🔍 Starting human detection for {duration} seconds...")

        try:
            while (time.time() - start_time) < duration:
                frame = self.capture_frame()
                human_count, boxes = self.detect_humans(frame)

                if human_count > 0:
                    now = datetime.now()
                    last_alert = self.last_alert.get("human", datetime.min)
                    if now - last_alert > timedelta(minutes=self.cooldown_minutes):
                        msg = f"ALERT: {human_count} human(s) detected at {now}"
                        print("🚨", msg)
                        self.send_sms(msg)
                        self.last_alert["human"] = now

                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n👋 Detection stopped by user")

def main():
    try:
        system = PiSecuritySystem2025()
        system.run_detection_loop()
    except Exception as e:
        print(f"💥 System failed: {e}")

if __name__ == "__main__":
    main()
