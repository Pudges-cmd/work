# pi_detect.py - Optimized for Pi Zero 2W + Pi Camera v2
import os
import time
import torch
import numpy as np
from picamera2 import Picamera2
import cv2

class PiDetector:
    def __init__(self, model_path="yolov5n.pt"):
        print("Loading offline YOLOv5n model...")
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            print("Download YOLOv5n model:")
            print("wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt")
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        try:
            # Load local model file (OFFLINE)
            self.device = torch.device('cpu')
            self.model = torch.load(model_path, map_location=self.device)
            
            # Handle different model formats
            if isinstance(self.model, dict):
                if 'model' in self.model:
                    self.model = self.model['model']
                elif 'ema' in self.model:
                    self.model = self.model['ema']
            
            # Set to eval mode
            self.model.eval()
            self.model.to(self.device)
            
            # CPU optimizations for Pi Zero 2W
            torch.set_num_threads(2)
            
            # COCO class names (hardcoded - no internet needed)
            self.names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow'
            }
                
            print("✓ Offline YOLOv5n model loaded")
            
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            raise
    
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
    
    def detect(self, frame):
        """Detect objects - optimized for low memory"""
        # Preprocess image
        img = self.letterbox(frame, new_shape=(416, 416))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  # 0-255 to 0.0-1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # Apply NMS
        pred = self.non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)[0]
        
        detections = []
        if pred is not None:
            for *box, conf, cls in pred.cpu().numpy():
                label = self.names.get(int(cls), 'unknown')
                if label in ['person', 'cat', 'dog']:
                    detections.append({
                        'label': label,
                        'confidence': float(conf),
                        'box': box
                    })
        
        return detections
    
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

class PiCamera:
    def __init__(self):
        print("Initializing Pi Camera v2...")
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            
            # Configure for low resolution to save processing power
            config = self.picam2.create_still_configuration(
                main={"size": (640, 480)},  # Lower resolution for Pi Zero 2W
                lores={"size": (320, 240)}, # Even lower for preview
                display="lores"
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            
            # Let camera warm up
            time.sleep(2)
            print("✓ Pi Camera v2 ready")
            
        except ImportError:
            print("picamera2 not available, trying rpicam method...")
            self.use_rpicam = True
            self.picam2 = None
            print("✓ Using rpicam method")
        except Exception as e:
            print(f"✗ Camera init failed: {e}")
            print("Make sure camera is enabled: sudo raspi-config")
            print("Or install: sudo apt install python3-picamera2")
            raise
    
    def capture_frame(self):
        """Capture frame from Pi camera"""
        try:
            if hasattr(self, 'use_rpicam') and self.use_rpicam:
                # Use rpicam-still command
                import subprocess
                import tempfile
                
                # Capture to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cmd = [
                        'rpicam-still',
                        '-o', tmp.name,
                        '--width', '640',
                        '--height', '480',
                        '--timeout', '1',
                        '--nopreview'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True)
                    if result.returncode == 0:
                        # Read image file
                        frame = cv2.imread(tmp.name)
                        os.unlink(tmp.name)  # Clean up temp file
                        
                        if frame is not None:
                            return True, frame
                    else:
                        print(f"rpicam-still error: {result.stderr.decode()}")
                        return False, None
            else:
                # Use picamera2 method
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                return True, frame
            
        except Exception as e:
            print(f"Camera capture failed: {e}")
            return False, None
    
    def release(self):
        """Clean up camera"""
        try:
            if hasattr(self, 'picam2') and self.picam2:
                self.picam2.stop()
                self.picam2.close()
            print("✓ Camera released")
        except:
            pass

def main():
    """Test detection with Pi camera"""
    try:
        # Initialize camera first
        camera = PiCamera()
        
        # Initialize detector
        detector = PiDetector()
        
        print("🔍 Detection test started (Pi Zero 2W optimized)")
        print("Press Ctrl+C to stop\n")
        
        frame_count = 0
        last_time = time.time()
        
        while True:
            # Capture frame
            ret, frame = camera.capture_frame()
            if not ret:
                print("Frame capture failed")
                time.sleep(1)
                continue
            
            frame_count += 1
            
            # Process every 5th frame to save CPU (Pi Zero 2W is slow)
            if frame_count % 5 == 0:
                detections = detector.detect(frame)
                
                if detections:
                    objects = [f"{d['label']} ({d['confidence']:.2f})" for d in detections]
                    print(f"🎯 DETECTED: {', '.join(objects)}")
                else:
                    if frame_count % 50 == 0:  # Less frequent "no detection" messages
                        current_time = time.time()
                        fps = 5 / (current_time - last_time) if frame_count > 5 else 0
                        print(f"👁️  Monitoring... (FPS: {fps:.1f})")
                        last_time = current_time
            
            # Small delay to prevent overheating Pi Zero 2W
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped")
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        try:
            camera.release()
        except:
            pass

if __name__ == "__main__":
    main()
