#!/usr/bin/env python3
# pi_security_2025.py - Pi Zero 2W + SIM7600G-H Rescue Bot with rpicam-still Human Detection

import os
import time
import cv2
import serial
import subprocess
from datetime import datetime, timedelta
from ultralytics import YOLO

class PiSecurityRescueBot:
    def __init__(self):
        print("🚀 Initializing Rescue Bot Pi Security System...")

        # Hardcoded phone number for SMS alerts
        self.phone_number = "+639514343942"

        # SMS / SIM7600G-H initialization
        self.sms = self.init_sms()

        # Human Detector (YOLO + rpicam-still)
        self.detector_model = YOLO("yolov5n.pt")
        self.human_class_id = 0  # COCO 'person'
        self.confidence_threshold = 0.5
        self.libcamera_cmd = "rpicam-still"

        # Alert cooldown
        self.last_alert_time = datetime.min
        self.cooldown_minutes = 5

        # Detection log
        self.log_file = "human_detections.txt"

        print("✅ Rescue Bot ready!")

    def init_sms(self):
        """Initialize SIM7600G-H for SMS alerts"""
        try:
            print("📡 Connecting to SIM7600G-H...")
            ser = serial.Serial(
                port="/dev/ttyUSB2", baudrate=115200, timeout=10
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
        """Send SMS alert to the hardcoded number"""
        if not self.sms:
            print("⚠️ SMS not initialized")
            return
        try:
            self.sms.write(f'AT+CMGS="{self.phone_number}"\r'.encode())
            time.sleep(1)
            self.sms.write(message.encode() + b"\x1A")
            time.sleep(3)
            print(f"📨 SMS sent to {self.phone_number}: {message}")
        except Exception as e:
            print(f"✗ SMS send failed: {e}")

    def capture_frame(self):
        """Capture frame using rpicam-still"""
        filename = f"temp_{int(time.time()*1000)}.jpg"
        try:
            subprocess.run(
                [self.libcamera_cmd, "-o", filename, "--timeout", "0.5"],
                capture_output=True,
                timeout=2
            )
            if os.path.exists(filename):
                frame = cv2.imread(filename)
                os.remove(filename)
                return frame
        except Exception as e:
            print(f"⚠️ Capture error: {e}")
        return None

    def detect_humans(self, frame):
        """Detect humans in a frame using YOLOv5n"""
        if frame is None:
            return 0, []

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector_model(frame_rgb, verbose=False)
            human_count = 0
            human_boxes = []

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) == self.human_class_id and float(box.conf[0]) >= self.confidence_threshold:
                        human_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        human_boxes.append((x1, y1, x2, y2, confidence))

            return human_count, human_boxes
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return 0, []

    def log_detection(self, count):
        """Log human detections to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp}: {count} humans detected\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        print(f"📝 {log_entry.strip()}")

    def run_detection_loop(self):
        """Run detection indefinitely until stopped"""
        print("🔍 Starting human detection loop (press Ctrl+C to stop)...")
        try:
            while True:
                frame = self.capture_frame()
                if frame is not None:
                    human_count, boxes = self.detect_humans(frame)
                    if human_count > 0:
                        self.log_detection(human_count)
                        now = datetime.now()
                        if now - self.last_alert_time > timedelta(minutes=self.cooldown_minutes):
                            time_str = now.strftime("%H:%M:%S")
                            msg = f"Human Detected at {time_str} at [GPS]"
                            print("🚨", msg)
                            self.send_sms(msg)
                            self.last_alert_time = now
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n👋 Detection stopped by user")

def main():
    try:
        bot = PiSecurityRescueBot()
        bot.run_detection_loop()
    except Exception as e:
        print(f"💥 System failed: {e}")

if __name__ == "__main__":
    main()
