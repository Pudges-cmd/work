🔒 Offline Pi Security System Setup Guide
For Raspberry Pi Zero 2W + SIM7600G-H + Pi Camera v2

📋 Prerequisites
Hardware Required:
Raspberry Pi Zero 2W
Pi Camera v2
SIM7600G-H HAT
SIM card with data/SMS plan
MicroSD card (16GB+)
Software Required:
Raspberry Pi OS Lite 64-bit
Python 3.7+
🚀 Quick Installation (5 minutes)
Step 1: Download Model File (REQUIRED for offline operation)
bash
# Download YOLOv5n model (14MB - smallest YOLO model)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
Step 2: Install Python Dependencies
bash
sudo apt update
sudo apt install python3-pip python3-opencv libatlas-base-dev -y

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install picamera2 pyserial numpy
Step 3: Enable Pi Camera
bash
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
sudo reboot
Step 4: Create Project Directory
bash
mkdir ~/pi_security
cd ~/pi_security

# Copy your 4 scripts here:
# - pi_detect.py
# - sim_sms.py  
# - pi_security.py
# - pi_startup.py

# Move the model file here:
mv ~/yolov5n.pt ./
Step 5: Configure Your Settings
bash
# Edit the config file that gets created automatically
nano pi_security_config.json
Edit these settings:

json
{
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
Step 6: Test Everything
bash
# Test detection only
python3 pi_detect.py

# Test SMS only
python3 sim_sms.py

# Test complete system
python3 pi_security.py
Step 7: Setup Auto-Start
bash
python3 pi_startup.py
bash install_service.sh
🔧 Hardware Connections
Pi Camera v2:
Connect ribbon cable to Pi's camera port
Make sure contacts face correctly
SIM7600G-H:
Mount on Pi GPIO pins
Insert SIM card
Connect antenna
Power via USB or external supply
📱 SMS Setup
The system uses the SIM7600G-H to send SMS directly through your cellular carrier. No internet required!

Common SIM7600G-H Ports:
/dev/ttyUSB0 - GPS
/dev/ttyUSB1 - Audio
/dev/ttyUSB2 - AT Commands (SMS) ← Use this one
/dev/ttyUSB3 - Modem
🚨 Troubleshooting
"Model file not found"
bash
# Make sure yolov5n.pt is in the same directory
ls -la yolov5n.pt
"Camera not found"
bash
# Enable camera interface
sudo raspi-config

# Test camera with rpicam (newer method)
rpicam-hello --preview

# Test still capture
rpicam-still -o test.jpg

# If rpicam doesn't work, try legacy method:
sudo apt install python3-picamera2

# Test picamera2
python3 -c "from picamera2 import Picamera2; print('Camera OK')"
"SIM7600G-H not found"
bash
# Check USB devices
lsusb

# Check serial ports
ls -la /dev/ttyUSB*

# Test AT commands manually
screen /dev/ttyUSB2 115200
# Type: AT (should respond OK)
"SMS not sending"
bash
# Check signal strength
echo "AT+CSQ" > /dev/ttyUSB2

# Check network registration  
echo "AT+CREG?" > /dev/ttyUSB2

# Check SIM status
echo "AT+CPIN?" > /dev/ttyUSB2
⚡ Performance Optimization for Pi Zero 2W
The scripts are optimized for the Pi Zero 2W's limited resources:

YOLOv5n: Smallest YOLO model (14MB)
416x416: Small inference resolution
Every 5th frame: Reduced processing load
2 CPU threads: Optimized for quad-core
Letterboxing: Maintains aspect ratio
640x480 camera: Lower resolution for speed
🔄 Auto-Startup Options
Option 1: Systemd Service (Recommended)
bash
python3 pi_startup.py
bash install_service.sh

# Service commands:
sudo systemctl status pi-security
sudo systemctl start pi-security
sudo systemctl stop pi-security
sudo journalctl -u pi-security -f
Option 2: Crontab (Backup)
bash
crontab -e
# Add: @reboot /home/pi/pi_security/security_wrapper.sh
📊 System Status
Check if running:
bash
ps aux | grep pi_security
View logs:
bash
tail -f detections.log
sudo journalctl -u pi-security -f
Monitor system resources:
bash
htop
🛡️ Security Features
Offline Operation: No internet
