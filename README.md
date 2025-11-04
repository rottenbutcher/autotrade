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

| Time (KST)      | Task                                              |
|-----------------|---------------------------------------------------|
| 08:59           | Refresh data & retrain Dual-Stream MAE + predictor |
| 09:00           | Predict daily direction & buy BTC if p_up > 0.65   |
| 08:50 next day  | Sell entire BTC position                           |
| Sunday 18:00    | Run backtest, log metrics, email weekly report     |

The scheduler runs continuously; keep the process active (e.g., with `tmux` or a service supervisor) for uninterrupted automation.

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
└── reports/
```

## 9️⃣ Final Note
⚠️ **Disclaimer**: This system is provided for educational and research purposes only. Cryptocurrency trading carries substantial financial risk. Operate at your own discretion and responsibility.
