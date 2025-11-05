# 🚀 Dual-Stream MAE Bitcoin Auto-Trader

## 1️⃣ Overview
This project delivers a fully automated Bitcoin trading platform powered by deep learning and quantitative automation. Every morning the system refreshes macro-economic and cryptocurrency data, retrains a Dual-Stream Masked Autoencoder (MAE) that captures lead–lag relationships between Bitcoin (BTC-USD) and major U.S. market indices, fine-tunes a direction classifier, and deploys updated signals to an Upbit trading bot. APScheduler orchestrates daily retraining, live trading, and weekly performance reporting while Weights & Biases (W&B) records all metrics and charts for ongoing research.

The Dual-Stream MAE processes two synchronized sequences: a Bitcoin stream and a U.S. equities/commodities stream (S&P 500, Nasdaq, Gold, Dollar Index, USD/KRW). By learning cross-stream attention, the encoder extracts joint representations that help the downstream predictor anticipate Bitcoin's daily direction based on broader market context.

## 2️⃣ Installation & Environment Setup
```bash
# Clone repository
git clone <repo_url>
cd <repo_name>

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key Python libraries used**
```
torch, torchvision, torchaudio
pandas, numpy, yfinance
pyupbit, apscheduler, wandb
matplotlib, tqdm, smtplib
```

## 3️⃣ Configuration
### Upbit API
1. Visit https://upbit.com/mypage/open_api_management
2. Generate a new API key pair (access & secret).
3. Create `bot/config.json` and store your credentials:
   ```json
   {
     "ACCESS_KEY": "YOUR_ACCESS_KEY",
     "SECRET_KEY": "YOUR_SECRET_KEY",
     "TICKER": "KRW-BTC"
   }
   ```
4. Confirm `.gitignore` contains `bot/config.json` to prevent accidental commits of secrets.

### Weights & Biases (W&B)
1. Create or log into your W&B account.
2. Authenticate from the terminal:
   ```bash
   wandb login
   ```
3. The scripts log to the project named **"BTC-AutoTrader"** by default. Set `WANDB_MODE=offline` if network access is restricted.

### Email Report
1. Enable Gmail 2FA and generate an App Password (https://support.google.com/accounts/answer/185833).
2. Update the sender/recipient addresses in `utils/backtester.py` and `automation/scheduler.py`.
3. Ensure the credentials are provided via environment variables or another secure method before running the scheduler.

## 4️⃣ Usage Guide
### Step 1 – One-time initial training
```bash
python data_pipeline/data_loader.py
python train/train_dual_mae.py
python train/finetune_dual_predictor.py
```

### Step 2 – Start full automation
```bash
python automation/scheduler.py
```

| Time (KST)      | Task                                                             |
|-----------------|------------------------------------------------------------------|
| 08:50           | EC2 instance auto-starts (Lambda) and the bot closes open BTC     |
| 08:59           | Refresh data & retrain Dual-Stream MAE + predictor               |
| ≈09:10          | Generate prediction, place trade, archive artifacts to S3        |
| 09:20           | EC2 instance issues self-stop command to minimize costs          |
| Sunday 18:00    | Weekly backtest + email report (triggered during Sunday run)     |

The scheduler runs continuously; keep the process active (e.g., with `tmux` or a service supervisor) for uninterrupted automation whenever the instance is online.

## 5️⃣ Visualization & Reporting
- Matplotlib generates an equity curve and trade-return histogram after each weekly backtest.
- Charts are saved locally under `/reports/weekly_report_<YYYYMMDD>.png` (and `_hist.png` for histograms).
- Weekly email summaries include the PNG equity curve as an attachment for quick review.
- W&B logs both the numerical metrics (total return, Sharpe) and the generated figures so you can inspect performance remotely.

## 6️⃣ Safety & Testing
- **Dry-run first**: Mock or comment out `upbit.buy_market_order` / `upbit.sell_market_order` calls to verify workflow without trading capital.
- **Start small**: Configure the bot for a minimal order size (e.g., 10,000 KRW) before scaling exposure.
- **Audit logs**: Transaction history is stored in `utils/trade_log.csv`, while generated reports remain under `/reports/` for auditing and recovery.
- **Secure secrets**: Never commit API keys or passwords. Rotate credentials regularly.

## 7️⃣ Advanced Extensions
- Integrate Telegram alerts using `python-telegram-bot` for real-time notifications of trades and anomalies.
- Customize W&B dashboards to track live cumulative profit, drawdowns, or other bespoke analytics.
- Adjust APScheduler cron expressions in `automation/scheduler.py` to align with your preferred timezone or trading schedule.

## 8️⃣ Example Project Structure
```
project_root/
├── data_pipeline/
│   ├── data_loader.py
│   └── dataset.csv
├── models/
│   └── dual_mae.py
├── train/
│   ├── train_dual_mae.py
│   └── finetune_dual_predictor.py
├── bot/
│   └── upbit_bot.py
├── utils/
│   ├── backtester.py
│   └── trade_log.csv
├── automation/
│   └── scheduler.py
├── aws_automation/
│   ├── config.json
│   ├── ec2_self_shutdown.py
│   ├── s3_backup.py
│   └── setup_lambda/
│       ├── start_instance_lambda.py
│       └── stop_instance_lambda.py
└── reports/
```

## ⚡ GPU Acceleration
The training scripts automatically detect CUDA-capable devices and enable performance optimizations:
- Mixed-precision training via Automatic Mixed Precision (AMP)
- cuDNN benchmarking for faster recurrent/transformer kernels
- Pinned-memory data loading with non-blocking GPU transfers

### Verify your GPU setup
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If the output is `True` followed by your GPU name, the Dual-Stream MAE pretraining and fine-tuning pipelines will run on the GPU automatically (often yielding an 8–30× speed-up versus CPU execution).

## ☁️ AWS Auto Management
This system now runs fully autonomously on AWS infrastructure:

| Time (KST) | Task                                      |
|-----------|-------------------------------------------|
| 08:50     | EC2 auto-start via CloudWatch + Lambda     |
| 08:59     | Data refresh and model retraining begins   |
| ≈09:10    | Trading script executes and S3 backups run |
| 09:20     | EC2 instance issues self-stop command      |

### Setup Steps
1. Populate `aws_automation/config.json` with your EC2 instance ID, region, S3 bucket, and the artifacts you want to back up.
2. Create the referenced S3 bucket to store checkpoints and logs.
3. Deploy the provided Lambda functions (`start_instance_lambda.py` and `stop_instance_lambda.py`) with permissions to start/stop the instance. Schedule them via CloudWatch Events (e.g., `cron(50 23 * * ? *)` for 08:50 KST).
4. Launch the instance manually once, run `python automation/scheduler.py`, and verify that the pipeline completes, uploads artifacts to `s3://<bucket>/backups/YYYYMMDD/`, and shuts down the instance automatically.

## 🧠 Smart Training Logic
The automation layer now checks for pretrained weights before every trading day and only performs full MAE pretraining when necessary:

| Condition | Action |
|-----------|--------|
| No pretrained model found locally or in S3 | Run full Dual-Stream MAE pretraining once |
| Pretrained model already exists | Skip pretraining, run fine-tuning + trading only |

Pretrained encoder weights are stored at `s3://<bucket>/backups/latest/dual_mae_encoder.pth` so new instances can immediately bootstrap from the latest foundation model.

### Workflow Summary
| Time (KST) | Task |
|------------|------|
| 08:50 | EC2 auto-start |
| 08:59 | Check pretrained model availability |
| 09:00 | First run triggers MAE pretraining (once) |
| ≈09:10 | Fine-tune predictor + execute trading |
| 09:20 | Upload logs/model artifacts & stop EC2 |

## 🚀 EC2 Auto Setup (`setup_ec2.sh`)

You can make EC2 automatically initialize the full environment at boot:

1. Open the AWS Console → EC2 → **User data** field (under **Advanced details**).
2. Paste this:

   ```bash
   #!/bin/bash
   export REPO_URL="https://github.com/<your-username>/<repo-name>.git"
   export REPO_DIR="autotrade"  # optional override
   su ubuntu -c "bash /home/ubuntu/aws_automation/setup_ec2.sh"
   ```

Launch or reboot your instance.

When the instance starts:

* It clones/pulls the GitHub repository.
* Sets up the virtual environment.
* Installs dependencies.
* Starts the scheduler automatically (fine-tune + trade + S3 backup).

Logs are stored in:

```bash
/home/ubuntu/ec2_setup.log
/home/ubuntu/scheduler_output.log
```

> ℹ️ **Configure before first boot:** Update `aws_automation/config.json` with your repository URL and preferred directory (or
> set the `REPO_URL`/`REPO_DIR` environment variables in the user data snippet above). The bootstrap script validates these
> values to avoid cloning placeholder URLs.

## 9️⃣ Final Note
⚠️ **Disclaimer**: This system is provided for educational and research purposes only. Cryptocurrency trading carries substantial financial risk. Operate at your own discretion and responsibility.
