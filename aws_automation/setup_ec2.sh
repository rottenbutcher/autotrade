#!/bin/bash
# =============================================
# EC2 Automated Setup Script for BTC-MAE System
# =============================================
set -euo pipefail

LOG_FILE="/home/ubuntu/ec2_setup.log"
SCHEDULER_LOG="/home/ubuntu/scheduler_output.log"
CONFIG_PATH="/home/ubuntu/aws_automation/config.json"

log() {
    local msg="$(date '+%Y-%m-%d %H:%M:%S') $*"
    echo "$msg" | tee -a "$LOG_FILE" >/dev/null
}

log "🚀 Starting EC2 Auto-Setup"

REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-}"

if [[ -f "$CONFIG_PATH" ]]; then
    mapfile -t cfg_values < <(python3 - "$CONFIG_PATH" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
print(cfg.get("REPO_URL", ""))
print(cfg.get("REPO_DIR", ""))
PY
 "$CONFIG_PATH")
    if [[ -z "$REPO_URL" ]]; then
        REPO_URL="${cfg_values[0]}"
    fi
    if [[ -z "$REPO_DIR" ]]; then
        REPO_DIR="${cfg_values[1]}"
    fi
fi

if [[ -z "$REPO_URL" || "$REPO_URL" == *"<"* ]]; then
    log "❌ REPO_URL is not configured. Set the REPO_URL environment variable or update aws_automation/config.json."
    exit 1
fi

if [[ -z "$REPO_DIR" || "$REPO_DIR" == *"<"* ]]; then
    REPO_DIR="$(basename "${REPO_URL%.git}")"
fi

case "$REPO_DIR" in
    /*) ;; # absolute path
    *) REPO_DIR="/home/ubuntu/${REPO_DIR}" ;;
esac

log "📁 Target repository directory: $REPO_DIR"

sudo apt update -y && sudo apt install -y git python3-venv awscli

cd /home/ubuntu
if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "📦 Cloning repository from $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "🔄 Updating repository in $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only origin main || git -C "$REPO_DIR" pull origin main
fi

cd "$REPO_DIR"

if [[ ! -d "venv" ]]; then
    log "🧱 Creating Python virtual environment"
    python3 -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

log "📦 Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ ! -f "/home/ubuntu/.aws/credentials" ]]; then
    log "⚠️ AWS credentials not found! Run 'aws configure' manually once."
fi

if [[ ! -f "bot/config.json" ]]; then
    log "⚠️ Missing bot/config.json. Create it with Upbit API keys."
fi

if pgrep -f "automation/scheduler.py" >/dev/null; then
    log "ℹ️ Scheduler already running; skipping restart."
else
    log "⏱ Launching daily automation scheduler"
    nohup python automation/scheduler.py > "$SCHEDULER_LOG" 2>&1 &
fi

log "✅ EC2 setup completed successfully"
