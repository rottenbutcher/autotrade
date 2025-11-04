#!/bin/bash
# =============================================
# EC2 Automated Setup Script for BTC-MAE System
# =============================================

# Configurable variables
REPO_URL="https://github.com/<your-username>/<repo-name>.git"
REPO_NAME="<repo-name>"
REPO_DIR="/home/ubuntu/${REPO_NAME}"
LOG_FILE="/home/ubuntu/ec2_setup.log"
SCHEDULER_LOG="/home/ubuntu/scheduler_output.log"

# 1. Logging start
echo "🚀 Starting EC2 Auto-Setup at $(date)" >> "$LOG_FILE"

# 2. Update system
sudo apt update -y && sudo apt install -y git python3-venv awscli

# 3. Clone or update GitHub repository
cd /home/ubuntu || exit 1
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "📦 Cloning repository..." >> "$LOG_FILE"
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "🔄 Updating repository..." >> "$LOG_FILE"
    cd "$REPO_DIR" || exit 1
    git pull --ff-only origin main || git pull origin main
fi

cd "$REPO_DIR" || exit 1

# 4. Create virtual environment
if [ ! -d "venv" ]; then
    echo "🧱 Creating Python virtual environment..." >> "$LOG_FILE"
    python3 -m venv venv
fi

# shellcheck source=/dev/null
source venv/bin/activate

# 5. Install dependencies
echo "📦 Installing dependencies..." >> "$LOG_FILE"
pip install --upgrade pip
pip install -r requirements.txt

# 6. AWS CLI config check (optional prompt removal)
if [ ! -f "/home/ubuntu/.aws/credentials" ]; then
    echo "⚠️ AWS credentials not found! Run 'aws configure' manually once." >> "$LOG_FILE"
fi

# 7. Ensure Upbit config exists
if [ ! -f "bot/config.json" ]; then
    echo "⚠️ Missing bot/config.json. Create it with Upbit API keys." >> "$LOG_FILE"
fi

# 8. Run the scheduler
echo "⏱ Launching daily automation scheduler..." >> "$LOG_FILE"
nohup python automation/scheduler.py > "$SCHEDULER_LOG" 2>&1 &

echo "✅ EC2 setup completed successfully at $(date)" >> "$LOG_FILE"
