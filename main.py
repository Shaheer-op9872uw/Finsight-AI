# FinSight AI - Phase 2 (Full single-file app)
# - Streamlit app: signup/login (SQLite + bcrypt), portfolio input, live prices, forecasting,
#   portfolio analytics, signals, and Groq LLM insights.
#
# Install:
# pip install streamlit yfinance pandas numpy statsmodels plotly requests bcrypt
#
# Run:
# streamlit run finsight_ai_full.py

import os
import sqlite3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import plotly.graph_objs as go
import streamlit as st
import bcrypt

# -----------------------
# CONFIG
# -----------------------
APP_TITLE = "⚡ FinSight AI — Portfolio Commander"
DB_FILE = "finsight_ai.db"
GROQ_API_ENV = "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"  # prefer env var
GROQ_BASE = "https://api.groq.com/openai/v1"  # openai-compatible path
GROQ_MODEL = "openai/llama3-70b-8192"  # change if needed
FORECAST_MIN_POINTS = 60
DEFAULT_FORECAST_HORIZON = 7

# -----------------------
# DB & AUTH
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            pw_hash BLOB NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    cur.execute(
        """CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            data_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )"""
    )
    conn.commit()
    return conn

conn = init_db()

def create_user_db(username: str, password: str) -> Tuple[bool, str]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cur.fetchone():
        return False, "Username already exists"
    salt = bcrypt.gensalt()
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), salt)
    cur.execute("INSERT INTO users (username, pw_hash, created_at) VALUES (?, ?, ?)",
                (username, pw_hash, datetime.utcnow().isoformat()))
    conn.commit()
    return True, "User created"

def verify_user_db(username: str, password: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT id, pw_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return False
    user_id, pw_hash = row[0], row[1]
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False

def get_user_id(username: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return row[0] if row else None

def save_portfolio_db(user_id: int, name: str, portfolio: dict):
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    data_json = json.dumps(portfolio)
    # if exists with same name update else insert
    cur.execute("SELECT id FROM portfolios WHERE user_id = ? AND name = ?", (user_id, name))
    row = cur.fetchone()
    if row:
        pid = row[0]
        cur.execute("UPDATE portfolios SET data_json = ?, updated_at = ? WHERE id = ?",
                    (data_json, now, pid))
    else:
        cur.execute("INSERT INTO portfolios (user_id, name, data_json, updated_at) VALUES (?, ?, ?, ?)",
                    (user_id, name, data_json, now))
    conn.commit()

def load_portfolios_db(user_id: int) -> Dict[str, dict]:
    cur = conn.cursor()
    cur.execute("SELECT name, data_json FROM portfolios WHERE user_id = ?", (user_id,))
    rows = cur.fetchall()
    out = {}
    for name, data_json in rows:
        out[name] = json.loads(data_json)
    return out

# -----------------------
# GROQ / LLM
# -----------------------
def call_groq_chat(prompt: str, api_key: str, model: str = GROQ_MODEL, max_tokens: int = 512) -> str:
    """
    Call Groq (OpenAI-compatible chat completions endpoint).
    Returns assistant text or an error message.
    """
    if not api_key:
        return "(Groq key missing) Please set the Groq API key in the settings."

    url = f"{GROQ_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system_message = (
        "You are FinSight AI, a concise professional financial assistant. "
        "Given structured numeric context (portfolio holdings, current prices, forecasts, and risk metrics), "
        "produce: \n1) a short plain-English summary (2-4 sentences) of current portfolio health, \n"
        "2) top 3 action items (each 1-2 short bullets; label Buy/Hold/Sell where appropriate), \n"
        "3) key risks and confidence notes (1-2 bullets). \n"
        "Use ONLY the numbers in the context for numeric claims. If the context lacks required numbers, say which items are missing. "
        "Respond in numbered sections and compact bullets. Be factual and conservative."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        j = r.json()
        if "choices" in j and len(j["choices"]) > 0:
            content = j["choices"][0].get("message", {}).get("content")
            if content:
                return content.strip()
        return "(Groq returned unexpected response) " + str(j)
    except Exception as e:
        return f"(Groq call failed) {e}"

# -----------------------
# DATA FETCH & FORECAST
# -----------------------
@st.cache_data(ttl=60)
def fetch_current_prices(tickers: list) -> pd.Series:
    """
    Returns last price for each ticker as a pandas Series ticker->price
    """
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    try:
        df = yf.download(tickers, period="5d", interval="1d", progress=False, auto_adjust=True)
        # df may be multi-index; prefer Close
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                close = df['Close']
            else:
                close = df.xs(df.columns.levels[0][-1], axis=1, level=0)
        else:
            close = df
        # take last available
        last = close.tail(1).T.iloc[:, 0]
        last.index = [str(i).upper() for i in last.index]
        return last
    except Exception:
        # fallback per-ticker
        out = {}
        for t in tickers:
            try:
                s = yf.download(t, period="7d", interval="1d", progress=False, auto_adjust=True)
                if s is None or s.empty:
                    out[t.upper()] = np.nan
                else:
                    if 'Close' in s.columns:
                        out[t.upper()] = float(s['Close'].dropna().iloc[-1])
                    else:
                        out[t.upper()] = float(s.iloc[:, 0].dropna().iloc[-1])
            except Exception:
                out[t.upper()] = np.nan
        return pd.Series(out)

@st.cache_data(ttl=3600)
def fetch_history_series(ticker: str, period: str = "720d", interval: str = "1d") -> pd.Series:
    """
    Returns series of daily close prices for ticker
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if 'Close' in df.columns:
            s = df['Close'].dropna()
        else:
            s = df.iloc[:, 0].dropna()
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return pd.Series(dtype=float)

def fit_sarimax_forecast(series: pd.Series, steps: int = DEFAULT_FORECAST_HORIZON):
    """
    Fit SARIMAX(1,1,1) and forecast.
    Returns mean, lower, upper series or None on failure.
    """
    if series.shape[0] < FORECAST_MIN_POINTS:
        return None
    try:
        model = sm.tsa.SARIMAX(series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, method="lbfgs")
        fc = res.get_forecast(steps=steps)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.2)  # 80% CI
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
        # ensure datetime index for future days if needed
        return mean, lower, upper
    except Exception:
        return None

def naive_flat_forecast(series: pd.Series, steps: int = DEFAULT_FORECAST_HORIZON):
    if series.empty:
        return None
    last = series.iloc[-1]
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    mean = pd.Series([last]*steps, index=idx)
    lower = mean * 0.995
    upper = mean * 1.005
    return mean, lower, upper

# -----------------------
# FINANCIAL METRICS (formulas included)
# -----------------------
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    # daily pct change
    return price_df.pct_change().dropna(how='all')

def annualized_stats(returns: pd.Series, periods_per_year: int = 252) -> Tuple[float, float]:
    """
    mu = mean(return) * periods_per_year
    sigma = std(return) * sqrt(periods_per_year)
    """
    mu = returns.mean() * periods_per_year
    sigma = returns.std(ddof=1) * np.sqrt(periods_per_year)
    return mu, sigma

def sharpe_ratio_from_returns(returns: pd.Series, rf: float = 0.0) -> float:
    mu, sigma = annualized_stats(returns)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return (mu - rf) / sigma

def max_drawdown_from_series(equity: pd.Series) -> float:
    """
    max drawdown = min((equity - cummax(equity)) / cummax(equity))
    """
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())

def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    if returns.empty:
        return np.nan
    return -np.percentile(returns.dropna(), (1-level)*100)

def cagr_from_buy_to_now(buy_price: float, current_price: float, buy_date: datetime) -> Optional[float]:
    """
    CAGR formula:
      CAGR = (V_end / V_start)^(1/T) - 1
    where T = years elapsed = days / 365.25
    """
    try:
        days = (datetime.utcnow().date() - buy_date.date()).days
        years = days / 365.25 if days > 0 else 0.0001
        return (current_price / buy_price) ** (1/years) - 1 if buy_price > 0 else None
    except Exception:
        return None

# -----------------------
# SIGNALS
# -----------------------
def generate_signal_simple(current_price: float, forecast_mean: Optional[pd.Series]) -> Tuple[str, str]:
    if forecast_mean is None or forecast_mean.empty:
        return "HOLD", "No forecast"
    expected = float(forecast_mean.iloc[-1])
    pct = (expected - current_price) / current_price * 100 if current_price else 0.0
    # thresholds: adapt as desired
    if pct >= 4.0:
        return "BUY", f"Forecast indicates ~{pct:.2f}% upside in horizon"
    elif pct <= -4.0:
        return "SELL", f"Forecast indicates ~{pct:.2f}% downside in horizon"
    else:
        return "HOLD", f"Forecast change {pct:.2f}% (small)"

# -----------------------
# UI: helper plots & nice formatting
# -----------------------
def plot_history_and_forecast(history: pd.Series, forecast_mean: Optional[pd.Series],
                              lower: Optional[pd.Series], upper: Optional[pd.Series], ticker: str, title_suffix=""):
    fig = go.Figure()
    if not history.empty:
        fig.add_trace(go.Scatter(x=history.index, y=history.values, mode='lines', name='History'))
    if forecast_mean is not None:
        idx = forecast_mean.index
        fig.add_trace(go.Scatter(x=idx, y=forecast_mean.values, mode='lines', name='Forecast', line=dict(dash='dash')))
    if lower is not None and upper is not None:
        x = list(lower.index) + list(upper.index[::-1])
        y = list(lower.values) + list(upper.values[::-1])
        fig.add_trace(go.Scatter(x=x, y=y, fill='toself', fillcolor='rgba(0,100,80,0.12)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False))
    fig.update_layout(title=f"{ticker} — History + Forecast {title_suffix}", xaxis_title="Date", yaxis_title="Price (USD)", height=420)
    return fig

# -----------------------
# STREAMLIT APP
# -----------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("**Phase 2 — full app:** Sign up, add your holdings (ticker, units, buy price, buy date). The app fetches real prices, forecasts each ticker, computes portfolio metrics, and asks Groq for a concise analysis. ")

# --- SIDEBAR: Auth & Groq key ---
with st.sidebar:
    st.header("Account")
    if "user" not in st.session_state:
        auth_mode = st.radio("Account action", ["Login", "Sign up"], index=0)
        if auth_mode == "Sign up":
            new_user = st.text_input("Username (signup)", key="su_user")
            new_pw = st.text_input("Password", type="password", key="su_pw")
            if st.button("Create account"):
                ok, msg = create_user_db(new_user, new_pw)
                if ok:
                    st.success("Account created. Please login.")
                else:
                    st.error(msg)
        else:
            username = st.text_input("Username (login)", key="li_user")
            password = st.text_input("Password", type="password", key="li_pw")
            if st.button("Login"):
                if verify_user_db(username, password):
                    st.session_state["user"] = username
                    st.success("Logged in")
                else:
                    st.error("Invalid credentials")
    else:
        st.markdown(f"**Signed in as:** `{st.session_state['user']}`")
        if st.button("Logout"):
            st.session_state.pop("user", None)
            st.experimental_rerun()

    st.markdown("---")
    st.header("Groq / LLM")
    groq_key = st.text_input("Groq API key (or set env GROQ_API_KEY)", value=os.getenv(GROQ_API_ENV, ""), type="password")
    if groq_key.strip() == "" and os.getenv(GROQ_API_ENV):
        groq_key = os.getenv(GROQ_API_ENV)
    st.write("Model:", GROQ_MODEL)
    if st.button("Test Groq"):
        result = call_groq_chat("Say hello and confirm connectivity.", api_key=groq_key)
        st.text_area("Groq ping", result, height=140)

# --- Main area ---
if "user" not in st.session_state:
    st.info("Please login or sign up in the left sidebar to begin. Demo mode: you may continue without logging in but data won't persist.")
    demo_mode = st.checkbox("Continue in demo mode (no persistence)", value=False)
else:
    demo_mode = False

# Portfolio management area
st.header("Portfolio — Add / Manage Holdings")
col_a, col_b = st.columns([2, 1])
with col_a:
    with st.form("add_holding_form", clear_on_submit=False):
        tkr = st.text_input("Ticker (yfinance format e.g. BTC-USD, ETH-USD, AAPL)", value="", key="input_ticker").upper().strip()
        units = st.number_input("Units owned", min_value=0.0, value=0.0, format="%f", key="input_units")
        buy_price = st.number_input("Buy price (per unit, USD)", min_value=0.0, value=0.0, format="%f", key="input_buy_price")
        buy_date = st.date_input("Buy date", value=datetime.utcnow().date(), key="input_buy_date")
        holding_name = st.text_input("Portfolio name (save into):", value="MyPortfolio", key="portfolio_name")
        submitted = st.form_submit_button("Add / Update holding")
        if submitted:
            if not tkr:
                st.error("Ticker required")
            else:
                holding = {"ticker": tkr, "units": float(units), "buy_price": float(buy_price), "buy_date": buy_date.isoformat()}
                holdings = st.session_state.get("holdings", {})
                # merge units if same ticker
                if tkr in holdings:
                    existing = holdings[tkr]
                    # just append as new holding with aggregated units and weighted avg price:
                    total_units = existing["units"] + holding["units"]
                    if total_units > 0:
                        weighted_price = (existing["units"] * existing["buy_price"] + holding["units"] * holding["buy_price"]) / total_units
                    else:
                        weighted_price = holding["buy_price"]
                    holdings[tkr] = {"ticker": tkr, "units": total_units, "buy_price": weighted_price, "buy_date": existing["buy_date"]}
                else:
                    holdings[tkr] = holding
                st.session_state["holdings"] = holdings
                st.success(f"Saved {tkr} -> {holdings[tkr]['units']} units at avg buy ${holdings[tkr]['buy_price']:.4f}")

with col_b:
    st.markdown("**Load / Save portfolio**")
    if "user" in st.session_state and not demo_mode:
        uid = get_user_id(st.session_state["user"])
        if uid:
            existing = load_portfolios_db(uid)
            sel = st.selectbox("Load saved portfolio", options=["(none)"] + list(existing.keys()))
            if st.button("Load selected") and sel != "(none)":
                st.session_state["holdings"] = existing[sel]
                st.success(f"Loaded {sel}")
            if st.button("Save current to DB"):
                if "holdings" in st.session_state and st.session_state["holdings"]:
                    save_portfolio_db(uid, holding_name, st.session_state["holdings"])
                    st.success("Saved portfolio to DB")
                else:
                    st.error("No holdings to save")
    else:
        st.info("Sign in to save/load portfolios persistently (or check demo mode).")
    if st.button("Clear holdings"):
        st.session_state.pop("holdings", None)
        st.success("Holdings cleared")

# show holdings
holdings: Dict[str, dict] = st.session_state.get("holdings", {})
if holdings:
    st.subheader("Current holdings (session)")
    df_hold = pd.DataFrame([{
        "Ticker": v["ticker"],
        "Units": v["units"],
        "Avg Buy Price": v["buy_price"],
        "Buy Date": v["buy_date"]
    } for _, v in holdings.items()]).set_index("Ticker")
    st.table(df_hold)

# Run analysis button
st.markdown("---")
run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    forecast_horizon = st.slider("Forecast horizon (days)", 3, 30, value=DEFAULT_FORECAST_HORIZON)
    history_period = st.selectbox("History period for models", ["180d", "365d", "720d"], index=1)
    run_btn = st.button("Run analysis & insights")

# If user pressed run
if run_btn and holdings:
    tickers = list(holdings.keys())
    st.info("Fetching current prices...")
    try:
        current_prices = fetch_current_prices(tickers)
    except Exception as e:
        st.error(f"Failed to fetch current prices: {e}")
        current_prices = pd.Series(dtype=float)

    # Build current price map & compute per-holding stats
    rows = []
    for t, info in holdings.items():
        t_up = t.upper()
        units = info["units"]
        buy_price = info["buy_price"]
        buy_date = datetime.fromisoformat(info["buy_date"]) if isinstance(info["buy_date"], str) else info["buy_date"]
        cur_price = float(current_prices.get(t_up, np.nan)) if t_up in current_prices.index else np.nan
        invested = units * buy_price
        current_val = units * cur_price if not np.isnan(cur_price) else np.nan
        pl = current_val - invested if not np.isnan(current_val) else np.nan
        roi = (current_val / invested - 1) * 100 if invested and not np.isnan(current_val) else np.nan
        cagr = cagr_from_buy_to_now(buy_price, cur_price, buy_date) if not np.isnan(cur_price) else None
        rows.append({
            "Ticker": t_up,
            "Units": units,
            "Buy Price": buy_price,
            "Buy Date": buy_date.date().isoformat(),
            "Invested (USD)": invested,
            "Current Price": cur_price,
            "Current Value": current_val,
            "Unrealized P/L (USD)": pl,
            "ROI %": roi if not np.isnan(roi) else None,
            "CAGR (est.)": cagr if cagr is not None else None
        })

    df_port = pd.DataFrame(rows).set_index("Ticker")
    # nicer formatting for display
    def fmt(x):
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "N/A"
        if isinstance(x, float):
            return f"{x:,.2f}"
        return x

    # Show summary metrics
    total_invested = df_port["Invested (USD)"].sum(min_count=1)
    total_current = df_port["Current Value"].sum(min_count=1)
    total_pl = total_current - total_invested if not np.isnan(total_current) else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Invested (USD)", f"${total_invested:,.2f}")
    col2.metric("Current Portfolio Value (USD)", f"${total_current:,.2f}")
    col3.metric("Unrealized P/L (USD)", f"${total_pl:,.2f}")

    st.subheader("Per-holding summary")
    display_df = df_port.copy()
    # format numeric nicely for display
    display_df["Invested (USD)"] = display_df["Invested (USD)"].map(lambda x: fmt(x))
    display_df["Current Price"] = display_df["Current Price"].map(lambda x: fmt(x))
    display_df["Current Value"] = display_df["Current Value"].map(lambda x: fmt(x))
    display_df["Unrealized P/L (USD)"] = display_df["Unrealized P/L (USD)"].map(lambda x: fmt(x))
    display_df["ROI %"] = display_df["ROI %"].map(lambda x: f"{x:+.2f}%" if x is not None and not (isinstance(x,float) and np.isnan(x)) else "N/A")
    display_df["CAGR (est.)"] = display_df["CAGR (est.)"].map(lambda x: f"{x*100:+.2f}%" if x is not None else "N/A")
    st.table(display_df)

    # Fetch history + forecast per ticker
    st.subheader("Forecasts & Signals (per ticker)")
    combined_fig = go.Figure()
    forecast_results = {}
    signals = {}
    price_hist_df = pd.DataFrame()

    for t in tickers:
        hist = fetch_history_series(t, period=history_period)
        price_hist_df[t] = hist
        if hist.empty or hist.shape[0] < 2:
            forecast_results[t] = (None, None, None, "NO_HISTORY")
            signals[t] = ("HOLD", "No history")
            st.warning(f"No sufficient history for {t}; skipping forecast.")
            continue

        fit = fit_sarimax_forecast(hist, steps=forecast_horizon)
        if fit is None:
            fit = naive_flat_forecast(hist, steps=forecast_horizon)
            model_used = "NAIVE"
        else:
            model_used = "SARIMAX(1,1,1)"

        if fit is None:
            forecast_results[t] = (None, None, None, "NO_FORECAST")
            signals[t] = ("HOLD", "No forecast")
            continue

        mean, lower, upper = fit
        forecast_results[t] = (mean, lower, upper, model_used)
        cur_price = float(df_port.loc[t, "Current Price"]) if t in df_port.index else float(current_prices.get(t, np.nan))
        sig, note = generate_signal_simple(cur_price, mean)
        signals[t] = (sig, note)

        # add to combined plot
        combined_fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name=f"{t} history"))
        # ensure forecast index is datetime
        if not isinstance(mean.index, pd.DatetimeIndex):
            mean.index = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=len(mean), freq='D')
            lower.index = mean.index
            upper.index = mean.index
        combined_fig.add_trace(go.Scatter(x=mean.index, y=mean.values, mode='lines', name=f"{t} forecast", line=dict(dash='dash')))
        combined_fig.add_trace(go.Scatter(x=list(lower.index) + list(upper.index[::-1]), y=list(lower.values) + list(upper.values[::-1]), fill='toself', fillcolor='rgba(0,100,80,0.08)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False))

    combined_fig.update_layout(title=f"Combined history + {forecast_horizon}-day forecasts", height=600, xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(combined_fig, use_container_width=True)

    # show signals table
    signals_rows = []
    for t in tickers:
        sig, note = signals.get(t, ("HOLD","N/A"))
        model_used = forecast_results.get(t, (None, None, None, "N/A"))[3]
        next_price = None
        m = forecast_results.get(t)[0] if forecast_results.get(t) else None
        if m is not None:
            try:
                next_price = float(m.iloc[-1])
            except Exception:
                next_price = None
        signals_rows.append({"Ticker": t, "Signal": sig, "Signal Note": note, "Forecasted Price": f"${next_price:,.2f}" if next_price else "N/A", "Model": model_used})
    st.dataframe(pd.DataFrame(signals_rows).set_index("Ticker"))

    # Portfolio-level analytics
    st.subheader("Portfolio Analytics (historical)")
    # Create portfolio equity series
    if price_hist_df.dropna(how="all").empty:
        st.warning("Not enough historical pricing across holdings for portfolio analytics.")
    else:
        # align and compute equity curve
        pv_df = pd.DataFrame()
        for t, info in holdings.items():
            s = price_hist_df[t]
            pv_df[t] = s * info["units"]
        pv_df = pv_df.dropna(how='all')
        if pv_df.empty:
            st.warning("Not enough aligned history for portfolio analytics.")
        else:
            equity = pv_df.sum(axis=1)
            port_returns = equity.pct_change().dropna()
            cagr_est = None
            try:
                start_val = equity.iloc[0]
                end_val = equity.iloc[-1]
                days = (equity.index[-1] - equity.index[0]).days
                years = days / 365.25 if days > 0 else 1
                cagr_est = (end_val / start_val) ** (1/years) - 1
            except Exception:
                cagr_est = np.nan
            ann_vol = port_returns.std(ddof=1) * np.sqrt(252) if not port_returns.empty else np.nan
            sharpe = sharpe_ratio_from_returns(port_returns) if not port_returns.empty else np.nan
            mdd = max_drawdown_from_series(equity)
            var95 = historical_var(port_returns, 0.95) if not port_returns.empty else np.nan

            st.metric("Portfolio CAGR (est.)", f"{cagr_est*100:.2f}%" if not np.isnan(cagr_est) else "N/A")
            st.metric("Annualized Volatility", f"{ann_vol*100:.2f}%" if not np.isnan(ann_vol) else "N/A")
            st.metric("Sharpe (rf=0)", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
            st.metric("Max Drawdown", f"{mdd*100:.2f}%" if not np.isnan(mdd) else "N/A")
            st.metric("Historical VaR 95%", f"{var95*100:.2f}%" if not np.isnan(var95) else "N/A")
            st.subheader("Portfolio equity curve")
            st.line_chart(equity)

    # Build prompt for Groq
    st.subheader("AI Insights (Groq)")
    # Build context concisely:
    ctx_lines = [
        f"Total invested USD: {total_invested:,.2f}",
        f"Current portfolio USD: {total_current:,.2f}",
        f"Unrealized P/L USD: {total_pl:,.2f}"
    ]
    for t in tickers:
        curp = df_port.loc[t, "Current Price"]
        invested = df_port.loc[t, "Invested (USD)"]
        curval = df_port.loc[t, "Current Value"]
        model_used = forecast_results.get(t)[3] if forecast_results.get(t) else "N/A"
        nextp = None
        f_mean = forecast_results.get(t)[0] if forecast_results.get(t) else None
        if f_mean is not None:
            try:
                nextp = float(f_mean.iloc[-1])
            except Exception:
                nextp = None
        sig, note = signals.get(t, ("HOLD","N/A"))
        ctx_lines.append(f"- {t}: current=${curp:,.2f}, invested=${invested:,.2f}, cur_val=${curval:,.2f}, forecast_{forecast_horizon}d=${nextp if nextp else 'N/A'}, signal={sig}, model={model_used}")

    ctx_lines.append(f"Portfolio-level metrics: CAGR={cagr_est if cagr_est is not None else 'N/A'}, vol={ann_vol if not np.isnan(ann_vol) else 'N/A'}, sharpe={sharpe if not np.isnan(sharpe) else 'N/A'}, mdd={mdd if not np.isnan(mdd) else 'N/A'}")

    user_request = st.text_input("Tell the AI what you want (examples: 'Conservative advice', 'Top 3 risks', 'Where should I rebalance?')", value="Summarize portfolio, top 3 risks, and 3 concise action items.")
    prompt_text = "Context:\n" + "\n".join(ctx_lines) + "\n\nUser request: " + user_request + "\n\nAnswer:"

    st.code(prompt_text[:1200] + ("...\n(truncated)" if len(prompt_text) > 1200 else ""))
    if st.button("Ask Groq for a concise analysis"):
        # determine Groq key: UI input or env
        key = groq_key.strip() if groq_key.strip() else os.getenv(GROQ_API_ENV)
        if not key:
            st.error("Groq API key missing. Set environment variable or paste key in sidebar.")
        else:
            with st.spinner("Calling Groq..."):
                answer = call_groq_chat(prompt_text, api_key=key)
                st.markdown("**Groq response:**")
                st.text_area("Groq output", answer, height=360)

else:
    if not holdings:
        st.info("Add at least one holding to run full analysis.")
    else:
        st.info("Press 'Run analysis & insights' to fetch prices, forecast, and ask Groq.")

# Footer
st.markdown("---")
st.markdown("**Notes & formulas:**")
st.markdown(
    "- CAGR formula: `CAGR = (V_end / V_start)^(1/T) - 1`, where `T = years`.\n"
    "- Annualized return µ = mean(daily_returns) * 252. Annualized vol σ = std(daily_returns) * sqrt(252).\n"
    "- Sharpe = (µ - rf) / σ. VaR historical computed as percentile of daily returns."
)
st.caption("Built with ♥ by Ace. Keep iterating — we can add rebalancing suggestions, optimization, and export next.")