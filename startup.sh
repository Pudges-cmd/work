#!/bin/bash

# -------------------------
# Rescue Bot Startup Script
# -------------------------

# Absolute path to your project folder
PROJECT_DIR="/home/beris/work"

# Go to project folder
cd "$PROJECT_DIR" || { echo "❌ Project folder not found: $PROJECT_DIR"; exit 1; }

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install required packages if missing
REQUIRED_PACKAGES=("opencv-python" "ultralytics")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$pkg" > /dev/null 2>&1; then
        echo "Installing missing package: $pkg"
        pip install "$pkg"
    fi
done

# Run the bot
echo "Starting pi_security.py..."
python "$PROJECT_DIR/pi_security.py"
