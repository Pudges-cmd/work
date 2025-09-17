# pi_detect.py - Optimized for Pi Zero 2W + Pi Camera v2
import os
import time
import torch
import numpy as np
import cv2
import subprocess
import tempfile


class PiDetector:
    def __init__(self, model_path="yolov5n.pt"):
        print("Loading offline YOLOv5n model...")

        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            print("Download YOLOv5n model:")
            print("wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt")
            raise FileNotFoundError(f"Model file {model_path} not found")

        try:
            self.device = torch.device("cpu")

            # Try safe load first
            try:
                self.model = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception:
                print("⚠️ Falling back: loading model with full unpickling...")
                self.model = torch.load(model_path, map_location=self.device)

            # Handle checkpoint formats
            if isinstance(self.model, dict):
                if "model" in self.model:
                    self.model = self.model["model"]
                elif "ema" in self.model:
                    self.model = self.model["ema"]

            self.model.eval()
            self.model.to(self.device)

            # CPU optimizations
            torch.set_num_threads(2)

            # Hardcoded COCO class names
            self.names = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                14: 'bird', 15: 'cat', 16: 'dog'
            }

            print("✓ Offline YOLOv5n model loaded")

        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            raise

    def letterbox(self, img, new_shape=(416, 416)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
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
        img = self.letterbox(frame, new_shape=(416, 416))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img)[0]

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
        nc = prediction.shape[2] - 5
        xc = prediction[..., 4] > conf_thres

        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            if not x.shape[0]:
                continue

            x = x[x[:, 4].argsort(descending=True)[:max_det]]
            c = x[:, 5:6] * 4096
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
            output[xi] = x[i]
        return output

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y


class PiCamera:
    def __init__(self):
        print("Initializing Pi Camera v2 with rpicam...")
        result = subprocess.run(['which', 'rpicam-still'], capture_output=True)
        if result.returncode != 0:
            raise Exception("rpicam-still not found")
        test_cmd = ['rpicam-still', '-t', '1', '--nopreview', '-o', '/dev/null']
        result = subprocess.run(test_cmd, capture_output=True, timeout=10)
        if result.returncode == 0:
            print("✓ Pi Camera v2 ready (rpicam)")
        else:
            raise Exception("Camera test failed")

    def capture_frame(self):
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_name = tmp.name
            cmd = ['rpicam-still', '-o', tmp_name, '--width', '640', '--height', '480', '-t', '1', '--nopreview']
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                frame = cv2.imread(tmp_name)
                os.unlink(tmp_name)
                return (frame is not None), frame
            return False, None
        except Exception as e:
            print(f"Camera error: {e}")
            return False, None

    def release(self):
        print("✓ Camera released")


def main():
    try:
        camera = PiCamera()
        detector = PiDetector()

        print("🔍 Detection test started (Pi Zero 2W optimized)")
        frame_count, last_time = 0, time.time()

        while True:
            ret, frame = camera.capture_frame()
            if not ret:
                print("Frame capture failed")
                time.sleep(1)
                continue
            frame_count += 1

            if frame_count % 5 == 0:
                detections = detector.detect(frame)
                if detections:
                    objects = [f"{d['label']} ({d['confidence']:.2f})" for d in detections]
                    print(f"🎯 DETECTED: {', '.join(objects)}")
                else:
                    if frame_count % 50 == 0:
                        current_time = time.time()
                        fps = 5 / (current_time - last_time)
                        print(f"👁️ Monitoring... (FPS: {fps:.1f})")
                        last_time = current_time
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Detection stopped")
    finally:
        try:
            camera.release()
        except:
            pass


if __name__ == "__main__":
    main()
