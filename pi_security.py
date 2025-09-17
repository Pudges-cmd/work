# pi_security.py - Complete security system for Pi Zero 2W + SIM7600G-H
import os
import time
import json
import torch
import cv2
import serial
import numpy as np
from datetime import datetime, timedelta

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
        self.cooldown_minutes = self.config.get('alert_cooldown_minutes', 5)
        
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
            "inference_size": 416
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print("✓ Config loaded")
                return config
            except Exception as e:
                print(f"Config load error: {e}")
        
        # Create default config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"✓ Created default config: {config_file}")
        print("⚠️  Edit pi_security_config.json with your phone number!")
        return default_config
    
    def init_camera(self):
        """Initialize Pi Camera using rpicam-still"""
        try:
            print("Initializing Pi Camera with rpicam-still...")
            import subprocess
            
            # Check if rpicam-still exists
            result = subprocess.run(['which', 'rpicam-still'], capture_output=True)
            if result.returncode != 0:
                raise Exception("rpicam-still not found. Install with: sudo apt install rpicam-apps")
            
            # Quick test
            test_cmd = ['rpicam-still', '-t', '1', '--nopreview', '-o', '/dev/null']
            result = subprocess.run(test_cmd, capture_output=True, timeout=15)
            
            if result.returncode == 0:
                print("✓ Pi Camera ready (rpicam-still)")
                return {'type': 'rpicam'}
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                raise Exception(f"Camera test failed: {error_msg}")
                
        except Exception as e:
            print(f"✗ Camera init failed: {e}")
            print("Troubleshooting:")
            print("1. sudo raspi-config -> Interface Options -> Camera -> Enable")
            print("2. sudo reboot")
            print("3. Test: rpicam-hello")
            raise
    
    def init_detector(self):
        """Initialize offline YOLOv5n detector"""
        try:
            print("Loading offline YOLOv5n...")
            
            model_path = self.config.get('model_path', 'yolov5n.pt')
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                print("Download YOLOv5n model with:")
                print("wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt")
                raise FileNotFoundError(f"Model file {model_path} not found")
            
            # Load local model file (OFFLINE)
            device = torch.device('cpu')
            model = torch.load(model_path, map_location=device)
            
            # Handle different formats
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'ema' in model:
                    model = model['ema']
            
            model.eval()
            model.to(device)
            
            # CPU optimizations for Pi Zero 2W
            torch.set_num_threads(2)
            
            # COCO class names (subset for demo)
            self.names = {
                0: 'person', 15: 'cat', 16: 'dog'
            }
            
            print("✓ Offline YOLOv5n ready")
            return model
            
        except Exception as e:
            print(f"✗ Detector init failed: {e}")
            raise
    
    def init_sms(self):
        """Initialize SIM7600G-H SMS"""
        try:
            print("Connecting to SIM7600G-H...")
            port = self.config['sim7600_port']
            
            ser = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=10
            )
            time.sleep(2)
            
            # Test AT command
            ser.write(b'AT\r\n')
            time.sleep(1)
            response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            
            if "OK" not in response:
                raise Exception("AT command failed")
            
            # Setup SMS mode
            ser.write(b'AT+CMGF=1\r\n')
            time.sleep(1)
            ser.read(ser.in_waiting)  # Clear buffer
            
            print("✓ SIM7600G-H ready")
            return ser
            
        except Exception as e:
            print(f"✗ SIM7600G-H init failed: {e}")
            print("Check:")
            print("- USB connection")
            print("- SIM card inserted")
            print("- Module powered")
            return None
    
    def capture_frame(self):
        """Capture frame using rpicam-still"""
        try:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_name = tmp.name
            
            w, h = self.config['camera_resolution']
            cmd = [
                'rpicam-still',
                '-o', tmp_name,
                '--width', str(w),
                '--height', str(h),
                '-t', '1',
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            
            if result.returncode == 0:
                frame = cv2.imread(tmp_name)
                os.unlink(tmp_name)
                if frame is not None:
                    return True, frame
                else:
                    print("Failed to read captured image")
                    return False, None
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                print(f"rpicam-still error: {error_msg}")
                return False, None
                
        except subprocess.TimeoutExpired:
            print("Camera capture timeout")
            return False, None
        except Exception as e:
            print(f"Frame capture error: {e}")
            return False, None
    
    # (Detection, NMS, SMS, Logging, Run loop remain unchanged)
    # ...

def main():
    try:
        system = PiSecuritySystem()
        system.run()
    except Exception as e:
        print(f"✗ System failed to start: {e}")
        print("\n💡 Troubleshooting:")
        print("1. sudo raspi-config -> Interface Options -> Camera -> Enable")
        print("2. Check SIM7600G-H USB connection")
        print("3. Verify SIM card is inserted")
        print("4. Edit pi_security_config.json with your phone number")

if __name__ == "__main__":
    main()
