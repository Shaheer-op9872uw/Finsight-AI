# ⚡ FinSight AI — Portfolio Commander
Turn holdings into action. FinSight AI combines live market data, short-term forecasts, portfolio analytics and Groq-powered natural-language insights — built to stun recruiters and power real investing demos.

**Made by:** **Muhammad Shaheer** & **Irmak Güney**

---

## 📌 What it is
FinSight AI is a deployable Streamlit app that fetches live prices (stocks & crypto), runs short-term forecasts, computes portfolio-level metrics (CAGR, volatility, Sharpe, VaR, drawdown), generates clear buy/hold/sell signals, and delivers concise human-friendly analysis powered by Groq.  
Polished, practical, and portfolio-ready.

---

## 🚀 Key Highlights
- **Live price data** for crypto & equities via `yfinance`
- **Reliable forecasting** (SARIMAX with graceful fallback)
- **Portfolio-first view** — track holdings with P/L, weights & equity curve
- **Actionable signals** — BUY / HOLD / SELL
- **Groq LLM insights** — concise AI-powered analysis
- **Persistent demo-ready UX** — signup/login, save portfolios
- **Single-file prototype** you can run locally or deploy

---

## ⚡ Quickstart
1. **Clone repo**  
```
git clone <repo-url>
cd <repo-folder>
Create & activate a virtual environment

Windows (PowerShell)

powershell

python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux


python3 -m venv .venv
source .venv/bin/activate
Install dependencies

pip install -r requirements.txt
Set your Groq API key

Windows (PowerShell)

powershell

$env:GROQ_API_KEY="your_groq_key_here"
macOS / Linux

export GROQ_API_KEY="your_groq_key_here"
Run the app
streamlit run finsight_ai_full.py
```
🛠 How it Works
Auth & Persistence — signup/login stored in SQLite (demo use)

Portfolio Input — add ticker, units, buy price, buy date (or upload CSV)

Live Fetch — retrieves prices & history via yfinance

Forecasting — SARIMAX(1,1,1) short-term outlooks

Analytics & Signals — equity curve, CAGR, Sharpe, VaR, MDD

AI Insights — sends numeric context to Groq LLM for a concise summary

📂 Files
finsight_ai_.py — main app

requirements.txt — Python dependencies

finsight_ai.db — local SQLite DB (created at runtime)

💡 Demo Tips
Add both crypto & stocks to show cross-asset analysis

Use older buy dates for richer CAGR & backtesting

Copy Groq insights into presentations or reports

🤝 Contribute & Expand
Ideas to add:

Portfolio optimizer (MPT)

Backtesting engine

PDF/CSV export

Streamlit Cloud deployment

Pull requests welcome — tag Muhammad Shaheer or Irmak Güney.

🏆 Credits
Built by Muhammad Shaheer & Irmak Güney
Powered by Streamlit, yfinance, statsmodels, Plotly & Groq

📜 License
MIT — use it, learn from it, build something bigger.
