# pi_security.py - Complete security system for Pi Zero 2W + SIM7600G-H
import os
import time
import json
import torch
import cv2
import serial
from datetime import datetime, timedelta
from picamera2 import Picamera2

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
        """Initialize Pi Camera v2 with rpicam only"""
        try:
            print("Initializing Pi Camera v2 with rpicam...")
            import subprocess
            
            # Check if rpicam-still exists
            result = subprocess.run(['which', 'rpicam-still'], capture_output=True)
            if result.returncode != 0:
                raise Exception("rpicam-still not found. Install with: sudo apt install rpicam-apps")
            
            # Test camera with quick capture
            test_cmd = ['rpicam-still', '-t', '1', '--nopreview', '-o', '/dev/null']
            result = subprocess.run(test_cmd, capture_output=True, timeout=15)
            
            if result.returncode == 0:
                print("✓ Pi Camera ready (rpicam)")
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
                print("Download YOLOv5n model:")
                print("wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt")
                raise FileNotFoundError(f"Model file {model_path} not found")
            
            # Load local model file (OFFLINE)
            device = torch.device('cpu')
            model = torch.load(model_path, map_location=device)
            
            # Handle different model formats
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'ema' in model:
                    model = model['ema']
            
            model.eval()
            model.to(device)
            
            # CPU optimizations for Pi Zero 2W
            torch.set_num_threads(2)
            
            # COCO class names (hardcoded - no internet needed)
            self.names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'
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
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_name = tmp.name
            
            # Capture with rpicam-still
            w, h = self.config['camera_resolution']
            cmd = [
                'rpicam-still',
                '-o', tmp_name,
                '--width', str(w),
                '--height', str(h),
                '-t', '1',  # 1ms timeout for speed
                '--nopreview'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            
            if result.returncode == 0:
                # Read captured image
                frame = cv2.imread(tmp_name)
                
                # Clean up
                try:
                    os.unlink(tmp_name)
                except:
                    pass
                
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
    
    def letterbox(self, img, new_shape=(416, 416)):
        """Resize and pad image maintaining aspect ratio"""
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return img

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
        """Non-Maximum Suppression (NMS)"""
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]  # confidence
            
            if not x.shape[0]:
                continue
                
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            
            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            
            # Apply finite constraint
            if not x.shape[0]:
                continue
                
            # Sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_det]]
            
            # Batched NMS
            c = x[:, 5:6] * 4096  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torch.ops.torchvision.nms(boxes, scores, iou_thres)  # NMS
            
            output[xi] = x[i]
            
        return output
    
    def xywh2xyxy(self, x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def detect_objects(self, frame):
        """Detect objects in frame - OFFLINE VERSION"""
        try:
            # Preprocess image
            img = self.letterbox(frame, new_shape=(416, 416))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
            
            # Convert to tensor
            device = torch.device('cpu')
            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # 0-255 to 0.0-1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                pred = self.detector(img)[0]
            
            # Apply NMS
            pred = self.non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)[0]
            
            # Parse results
            detections = []
            confidence_threshold = self.config['detection_confidence']
            target_objects = self.config['target_objects']
            
            if pred is not None:
                for *box, conf, cls in pred.cpu().numpy():
                    if conf > confidence_threshold:
                        label = self.names.get(int(cls), 'unknown')
                        if label in target_objects:
                            detections.append({
                                'label': label,
                                'confidence': float(conf),
                                'time': datetime.now().isoformat()
                            })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def should_send_alert(self, detections):
        """Check if alert should be sent (cooldown logic)"""
        now = datetime.now()
        
        for detection in detections:
            label = detection['label']
            
            if label in self.last_alert:
                time_diff = now - self.last_alert[label]
                if time_diff.seconds < (self.cooldown_minutes * 60):
                    continue
            
            # Update last alert time
            self.last_alert[label] = now
            return True
        
        return False
    
    def send_sms_alert(self, detections):
        """Send SMS alert via SIM7600G-H"""
        if not self.sms:
            print("✗ SMS not available")
            return False
        
        try:
            # Create message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            objects = ", ".join(set([d['label'] for d in detections]))
            message = f"🚨 SECURITY ALERT!\nDetected: {objects}\nTime: {timestamp}"
            
            phone = self.config['phone_number']
            
            # Clear buffers
            self.sms.reset_input_buffer()
            
            # Send SMS command
            sms_cmd = f'AT+CMGS="{phone}"'
            self.sms.write((sms_cmd + '\r\n').encode())
            time.sleep(1)
            
            # Send message
            self.sms.write((message + '\r\n').encode())
            time.sleep(1)
            
            # Send Ctrl+Z
            self.sms.write(bytes([26]))
            
            # Wait for response
            start_time = time.time()
            response = ""
            
            while (time.time() - start_time) < 30:
                if self.sms.in_waiting > 0:
                    data = self.sms.read(self.sms.in_waiting).decode('utf-8', errors='ignore')
                    response += data
                    if "+CMGS:" in response or "ERROR" in response:
                        break
                time.sleep(0.5)
            
            if "+CMGS:" in response:
                print(f"✓ SMS sent: {objects}")
                return True
            else:
                print(f"✗ SMS failed: {response}")
                return False
                
        except Exception as e:
            print(f"✗ SMS error: {e}")
            return False
    
    def log_detection(self, detections):
        """Log detection to file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "detections": detections
            }
            
            with open("detections.log", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def run(self):
        """Main security monitoring loop"""
        print("🛡️  Pi Security Monitor Starting...")
        print(f"📱 Alerts to: {self.config['phone_number']}")
        print(f"🎯 Watching: {', '.join(self.config['target_objects'])}")
        print(f"⏰ Cooldown: {self.cooldown_minutes} minutes")
        print("🔍 Monitoring started... (Ctrl+C to stop)\n")
        
        frame_count = 0
        detection_interval = self.config['detection_interval']
        
        try:
            while True:
                # Capture frame
                ret, frame = self.capture_frame()
                if not ret:
                    print("Camera capture failed")
                    time.sleep(1)
                    continue
                
                frame_count += 1
                
                # Run detection every N frames to save CPU
                if frame_count % detection_interval == 0:
                    detections = self.detect_objects(frame)
                    
                    if detections:
                        objects = [f"{d['label']} ({d['confidence']:.2f})" for d in detections]
                        print(f"🎯 DETECTED: {', '.join(objects)}")
                        
                        # Log detection
                        self.log_detection(detections)
                        
                        # Send alert if cooldown passed
                        if self.should_send_alert(detections):
                            print("📱 Sending alert...")
                            self.send_sms_alert(detections)
                    
                    else:
                        # Status update every 100 frames
                        if frame_count % 100 == 0:
                            print(f"👁️  Monitoring... (Frame {frame_count})")
                
                # Small delay to prevent Pi Zero 2W overheating
                time.sleep(0.3)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n✗ System error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("✓ Camera cleanup (rpicam - nothing to clean)")
        
        try:
            if self.sms and self.sms.is_open:
                self.sms.close()
                print("✓ SMS module disconnected")
        except Exception as e:
            print(f"SMS cleanup error: {e}")

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
