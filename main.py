"""
Vault Trader — Progyny Infinite
Mean-reversion trading agent with self-improvement loop.
Deploys on Railway. Connects to Alpaca for paper/live trading.

Strategy: RSI + Bollinger Bands + Volume Spike
Universe: 54 large-cap symbols
Position sizing: 15% per position, max 5 positions
Stop loss: 3% | Take profit: 5%
Trend filter: 50-day SMA
Self-improvement: EOD Claude API review, one parameter change per day
"""

import os
import json
import datetime
import asyncio
from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Optional imports (graceful fallback if not installed) ─────────────────────
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────

ALPACA_KEY = os.environ.get("ALPACA_KEY_ID", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

DATA_DIR = "/app/logs"
os.makedirs(DATA_DIR, exist_ok=True)

TRADE_LOG = f"{DATA_DIR}/trades.json"
PARAM_LOG = f"{DATA_DIR}/params.json"
AGENT_LOG = f"{DATA_DIR}/agent.log"

# Default parameters (self-improvement loop updates these)
DEFAULT_PARAMS = {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bb_period": 20,
    "bb_std": 2.0,
    "sma_period": 50,
    "volume_spike": 1.5,
    "stop_loss": 0.03,
    "take_profit": 0.05,
    "position_size": 0.15,
    "max_positions": 5,
    "score_threshold": 70
}

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", "NFLX", "CMCSA",
    "PFE", "T", "VZ", "INTC", "CSCO", "PEP", "KO", "MRK", "ABT", "TMO",
    "NKE", "MCD", "WMT", "CVX", "BAC", "GS", "MS", "C", "WFC", "AXP",
    "CAT", "DE", "BA", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX",
    "AMGN", "GILD", "BIIB", "REGN"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {msg}\n"
    print(entry.strip())
    try:
        with open(AGENT_LOG, "a") as f:
            f.write(entry)
    except Exception:
        pass


def load_params() -> dict:
    try:
        with open(PARAM_LOG, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PARAMS.copy()


def save_params(params: dict):
    with open(PARAM_LOG, "w") as f:
        json.dump(params, f, indent=2)


def load_trades() -> list:
    try:
        with open(TRADE_LOG, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_trade(trade: dict):
    trades = load_trades()
    trades.append(trade)
    with open(TRADE_LOG, "w") as f:
        json.dump(trades[-500:], f, indent=2)  # Keep last 500 trades


def load_log(n=50) -> list:
    try:
        with open(AGENT_LOG, "r") as f:
            lines = f.readlines()
        return [l.strip() for l in lines[-n:]]
    except Exception:
        return []


# ── Trading logic ─────────────────────────────────────────────────────────────

def get_alpaca():
    if not ALPACA_AVAILABLE or not ALPACA_KEY:
        return None
    try:
        return tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL)
    except Exception as e:
        log(f"Alpaca connection failed: {e}")
        return None


def calculate_signals(bars, params: dict) -> dict:
    """Calculate RSI, Bollinger Bands, SMA, volume spike for a symbol."""
    if not PANDAS_AVAILABLE or bars is None or len(bars) < params["sma_period"]:
        return {"signal": "SKIP", "reason": "insufficient data"}

    closes = pd.Series([b.c for b in bars])
    volumes = pd.Series([b.v for b in bars])

    # RSI
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(params["rsi_period"]).mean()
    loss = -delta.clip(upper=0).rolling(params["rsi_period"]).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # Bollinger Bands
    sma_bb = closes.rolling(params["bb_period"]).mean()
    std_bb = closes.rolling(params["bb_period"]).std()
    lower_band = sma_bb - params["bb_std"] * std_bb
    current_price = closes.iloc[-1]
    below_lower = current_price < lower_band.iloc[-1]

    # 50-day SMA trend filter
    sma50 = closes.rolling(params["sma_period"]).mean().iloc[-1]
    above_sma = current_price > sma50

    # Volume spike
    avg_vol = volumes.rolling(20).mean().iloc[-1]
    vol_spike = volumes.iloc[-1] > avg_vol * params["volume_spike"]

    # Signal logic
    buy_signal = (
        current_rsi < params["rsi_oversold"] and
        below_lower and
        above_sma and
        vol_spike
    )

    return {
        "signal": "BUY" if buy_signal else "HOLD",
        "rsi": round(float(current_rsi), 2),
        "price": round(float(current_price), 2),
        "above_sma": above_sma,
        "below_lower_band": below_lower,
        "volume_spike": vol_spike
    }


async def run_scan():
    """Scan universe, find signals, execute trades."""
    api = get_alpaca()
    if not api:
        log("Scan skipped — Alpaca not connected")
        return []

    params = load_params()
    signals = []

    try:
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        positions = {p.symbol: p for p in api.list_positions()}

        log(f"Scan started — portfolio: ${portfolio_value:,.2f}, positions: {len(positions)}")

        for symbol in UNIVERSE:
            try:
                bars = api.get_bars(symbol, "1Day", limit=params["sma_period"] + 10).df
                if bars.empty:
                    continue
                # Convert to simple object
                bar_list = [type('Bar', (), {'c': row['close'], 'v': row['volume']})()\
                           for _, row in bars.iterrows()]
                result = calculate_signals(bar_list, params)
                result["symbol"] = symbol
                signals.append(result)

                if result["signal"] == "BUY" and symbol not in positions:
                    if len(positions) < params["max_positions"]:
                        qty = int((portfolio_value * params["position_size"]) / result["price"])
                        if qty > 0:
                            try:
                                order = api.submit_order(
                                    symbol=symbol,
                                    qty=qty,
                                    side="buy",
                                    type="market",
                                    time_in_force="day"
                                )
                                trade = {
                                    "date": datetime.datetime.now().isoformat(),
                                    "symbol": symbol,
                                    "action": "BUY",
                                    "qty": qty,
                                    "price": result["price"],
                                    "rsi": result["rsi"],
                                    "order_id": order.id
                                }
                                save_trade(trade)
                                log(f"BUY {qty} {symbol} @ ${result['price']}")
                            except Exception as e:
                                log(f"Order failed {symbol}: {e}")

            except Exception as e:
                log(f"Error scanning {symbol}: {e}")
                continue

        # Check stop loss / take profit on existing positions
        for symbol, pos in positions.items():
            try:
                entry = float(pos.avg_entry_price)
                current = float(pos.current_price)
                pnl_pct = (current - entry) / entry

                if pnl_pct <= -params["stop_loss"] or pnl_pct >= params["take_profit"]:
                    reason = "STOP_LOSS" if pnl_pct <= -params["stop_loss"] else "TAKE_PROFIT"
                    api.submit_order(
                        symbol=symbol,
                        qty=abs(int(pos.qty)),
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )
                    trade = {
                        "date": datetime.datetime.now().isoformat(),
                        "symbol": symbol,
                        "action": "SELL",
                        "reason": reason,
                        "pnl_pct": round(pnl_pct * 100, 2)
                    }
                    save_trade(trade)
                    log(f"SELL {symbol} — {reason} ({pnl_pct*100:.1f}%)")
            except Exception as e:
                log(f"Exit check failed {symbol}: {e}")

    except Exception as e:
        log(f"Scan error: {e}")

    log(f"Scan complete — {len([s for s in signals if s['signal'] == 'BUY'])} buy signals")
    return signals


async def run_self_improvement():
    """EOD self-improvement loop — ask Claude for one parameter change."""
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        log("Self-improvement skipped — no Anthropic key")
        return

    trades = load_trades()
    if len(trades) < 3:
        log("Self-improvement skipped — need at least 3 trades")
        return

    params = load_params()
    recent = trades[-20:]

    wins = [t for t in recent if t.get("pnl_pct", 0) > 0]
    losses = [t for t in recent if t.get("pnl_pct", 0) < 0]
    win_rate = len(wins) / len(recent) if recent else 0

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""You are the Vault Trader self-improvement engine for Randy Wain Nutt.

Current parameters: {json.dumps(params, indent=2)}

Recent performance (last {len(recent)} trades):
- Win rate: {win_rate:.1%}
- Wins: {len(wins)}, Losses: {len(losses)}
- Recent trades: {json.dumps(recent[-5:], indent=2)}

Suggest ONE parameter change to improve performance. Scientific method — one variable at a time.

Respond ONLY with valid JSON:
{{"parameter": "param_name", "old_value": old_val, "new_value": new_val, "reason": "one sentence reason"}}"""
            }]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        suggestion = json.loads(raw)
        param = suggestion["parameter"]

        if param in params:
            params[param] = suggestion["new_value"]
            save_params(params)
            log(f"Self-improvement: {param} {suggestion['old_value']} → {suggestion['new_value']} | {suggestion['reason']}")
        else:
            log(f"Self-improvement: invalid parameter suggested: {param}")

    except Exception as e:
        log(f"Self-improvement error: {e}")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Vault Trader", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    api = get_alpaca()
    account_info = {}
    if api:
        try:
            acc = api.get_account()
            account_info = {
                "portfolio_value": float(acc.portfolio_value),
                "cash": float(acc.cash),
                "buying_power": float(acc.buying_power),
                "status": acc.status
            }
        except Exception:
            account_info = {"error": "Could not fetch account"}

    return {
        "status": "Vault Trader online",
        "version": "1.0.0",
        "operator": "Randy Wain Nutt",
        "alpaca_connected": bool(api),
        "paper_trading": "paper-api" in ALPACA_BASE_URL,
        "account": account_info
    }


@app.get("/params")
def get_params():
    return load_params()


@app.post("/params")
def update_params(params: dict):
    current = load_params()
    current.update(params)
    save_params(current)
    log(f"Parameters manually updated: {params}")
    return {"status": "updated", "params": current}


@app.post("/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scan)
    return {"status": "scan started"}


@app.post("/improve")
async def trigger_improvement(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_self_improvement)
    return {"status": "self-improvement started"}


@app.get("/trades")
def get_trades(limit: int = 50):
    trades = load_trades()
    return {"trades": trades[-limit:], "total": len(trades)}


@app.get("/positions")
def get_positions():
    api = get_alpaca()
    if not api:
        return {"error": "Alpaca not connected"}
    try:
        positions = api.list_positions()
        return {
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "pnl": float(p.unrealized_pl),
                    "pnl_pct": float(p.unrealized_plpc) * 100
                }
                for p in positions
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/performance")
def get_performance():
    trades = load_trades()
    if not trades:
        return {"message": "No trades yet — paper trading not started"}

    closed = [t for t in trades if "pnl_pct" in t]
    if not closed:
        return {"message": "No closed trades yet"}

    wins = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] <= 0]
    avg_gain = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "avg_gain_pct": round(avg_gain, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expectancy": round((len(wins)/len(closed)) * avg_gain + (len(losses)/len(closed)) * avg_loss, 2)
    }


@app.get("/log")
def get_log(lines: int = 50):
    return {"log": load_log(lines)}


@app.get("/dashboard", response_class=HTMLResponse)
def trading_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vault Trader — Dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0a0a0a; color:#e0e0e0; font-family:'Segoe UI',sans-serif; padding:24px; }
  h1 { color:#c9a84c; font-size:20px; letter-spacing:2px; text-transform:uppercase; margin-bottom:24px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; margin-bottom:24px; }
  .card { background:#111; border:1px solid #1e1e1e; border-radius:8px; padding:20px; }
  .card h2 { font-size:11px; color:#c9a84c; text-transform:uppercase; letter-spacing:1px; margin-bottom:14px; }
  .stat { font-size:28px; font-weight:700; color:#c9a84c; }
  .label { font-size:12px; color:#555; margin-top:4px; }
  .btn { background:#c9a84c; border:none; border-radius:6px; padding:10px 20px; color:#0a0a0a; font-size:13px; font-weight:700; cursor:pointer; margin-right:8px; margin-top:8px; }
  .btn:hover { opacity:0.85; }
  .btn-ghost { background:transparent; border:1px solid #c9a84c; color:#c9a84c; }
  .log { background:#060606; border:1px solid #1a1a1a; border-radius:6px; padding:14px; font-size:12px; color:#555; font-family:monospace; height:200px; overflow-y:auto; line-height:1.6; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; color:#555; font-size:11px; text-transform:uppercase; padding:8px 0; border-bottom:1px solid #1a1a1a; }
  td { padding:8px 0; border-bottom:1px solid #111; color:#888; }
  .green { color:#4caf50; }
  .red { color:#e05555; }
</style>
</head>
<body>
<h1>⚡ Vault Trader</h1>
<div class="grid" id="stats"></div>
<div class="card" style="margin-bottom:16px;">
  <h2>Controls</h2>
  <button class="btn" onclick="triggerScan()">Run Scan</button>
  <button class="btn btn-ghost" onclick="triggerImprove()">Self-Improve</button>
  <button class="btn btn-ghost" onclick="loadAll()">Refresh</button>
</div>
<div class="grid">
  <div class="card">
    <h2>Positions</h2>
    <div id="positions"><div style="color:#555;font-size:13px;">Loading...</div></div>
  </div>
  <div class="card">
    <h2>Recent Trades</h2>
    <div id="trades"><div style="color:#555;font-size:13px;">Loading...</div></div>
  </div>
</div>
<div class="card" style="margin-top:16px;">
  <h2>Agent Log</h2>
  <div class="log" id="log">Loading...</div>
</div>
<script>
async function get(url) { const r = await fetch(url); return r.json(); }
async function post(url) { const r = await fetch(url, {method:'POST'}); return r.json(); }

async function loadAll() {
  const [health, perf, pos, trades, log] = await Promise.all([
    get('/'), get('/performance'), get('/positions'), get('/trades?limit=10'), get('/log')
  ]);

  document.getElementById('stats').innerHTML = `
    <div class="card"><h2>Portfolio</h2><div class="stat">$${(health.account?.portfolio_value||0).toLocaleString()}</div><div class="label">Total Value</div></div>
    <div class="card"><h2>Cash</h2><div class="stat">$${(health.account?.cash||0).toLocaleString()}</div><div class="label">Available</div></div>
    <div class="card"><h2>Win Rate</h2><div class="stat">${perf.win_rate||0}%</div><div class="label">${perf.total_trades||0} closed trades</div></div>
    <div class="card"><h2>Expectancy</h2><div class="stat ${(perf.expectancy||0)>0?'green':'red'}">${perf.expectancy||0}%</div><div class="label">Per trade avg</div></div>
    <div class="card"><h2>Mode</h2><div class="stat" style="font-size:16px;">${health.paper_trading?'📄 PAPER':'💰 LIVE'}</div><div class="label">${health.alpaca_connected?'Connected':'Not connected'}</div></div>
  `;

  const posHtml = (pos.positions||[]).length ? `<table><tr><th>Symbol</th><th>Qty</th><th>P&L</th></tr>${
    pos.positions.map(p=>`<tr><td>${p.symbol}</td><td>${p.qty}</td><td class="${p.pnl_pct>0?'green':'red'}">${p.pnl_pct.toFixed(1)}%</td></tr>`).join('')
  }</table>` : '<div style="color:#555;font-size:13px;">No open positions</div>';
  document.getElementById('positions').innerHTML = posHtml;

  const tradeHtml = (trades.trades||[]).length ? `<table><tr><th>Symbol</th><th>Action</th><th>P&L</th></tr>${
    trades.trades.slice(-10).reverse().map(t=>`<tr><td>${t.symbol}</td><td>${t.action}</td><td class="${(t.pnl_pct||0)>0?'green':'red'}">${t.pnl_pct?t.pnl_pct.toFixed(1)+'%':'-'}</td></tr>`).join('')
  }</table>` : '<div style="color:#555;font-size:13px;">No trades yet</div>';
  document.getElementById('trades').innerHTML = tradeHtml;

  document.getElementById('log').innerHTML = (log.log||[]).join('<br>');
}

async function triggerScan() { await post('/scan'); setTimeout(loadAll, 2000); }
async function triggerImprove() { await post('/improve'); setTimeout(loadAll, 3000); }

loadAll();
setInterval(loadAll, 15000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    log("Vault Trader v1.0.0 starting up")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
