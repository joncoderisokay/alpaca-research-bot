import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

LEADER = "BTC/USD"
FOLLOWERS = ["ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"]
NOTIONAL_PER_TRADE = 500.0
MAX_ENTRY_LAG_SECONDS = 90
HOLD_MINUTES = 30

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = CryptoHistoricalDataClient()

def get_indicators(closes: pd.Series):
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist_bps = ((macd - signal) / closes) * 10000.0

    return float(rsi), float(hist_bps.iloc[-1]), float(hist_bps.iloc[-2])

now_utc = datetime.now(timezone.utc)
start = now_utc - timedelta(hours=6)

all_symbols = [LEADER] + FOLLOWERS
req = CryptoBarsRequest(
    symbol_or_symbols=all_symbols,
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=start
)
bars = data_client.get_crypto_bars(req).df

# Measure execution latency from BTC's most recent completed 5-minute candle
btc_df = bars.xs(LEADER)
latest_bar_open = btc_df.index[-1]
seconds_since_open = (now_utc - latest_bar_open).total_seconds()

def get_closed_series(sym: str) -> pd.Series:
    df = bars.xs(sym)
    return df["close"].iloc[:-1].tail(60) if seconds_since_open < 300 else df["close"].tail(60)

closed_bar_close_time = (
    btc_df.index[-2] + timedelta(seconds=300)
    if seconds_since_open < 300
    else latest_bar_open + timedelta(seconds=300)
)
lag_seconds = (now_utc - closed_bar_close_time).total_seconds()

# 1. Manage exits for all open follower positions first (even if trigger was delayed)
positions = {p.symbol: p for p in trading_client.get_all_positions()}
for sym in FOLLOWERS:
    clean_sym = sym.replace("/", "")
    if clean_sym in positions:
        closed_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[sym], limit=1)
        )
        if closed_orders and closed_orders.filled_at:
            mins_held = (now_utc - closed_orders.filled_at).total_seconds() / 60.0
            if mins_held >= HOLD_MINUTES:
                trading_client.close_position(clean_sym)
                print(f"CLOSED {sym} after {mins_held:.1f}m (lag: {lag_seconds:.0f}s)")
            else:
                print(f"HOLDING {sym} ({mins_held:.1f}m / {HOLD_MINUTES}m)")

# 2. Enforce staleness guard before evaluating any new entries
if lag_seconds > MAX_ENTRY_LAG_SECONDS:
    print(f"STALE_SKIP | Trigger delayed by {lag_seconds:.0f}s (max {MAX_ENTRY_LAG_SECONDS}s)")
    raise SystemExit(0)

# 3. Evaluate BTC leader momentum
btc_rsi, btc_hist, btc_prev_hist = get_indicators(get_closed_series(LEADER))
btc_surging = (btc_hist > btc_prev_hist) and (btc_hist > 1.5) and (btc_rsi > 58)

print(f"SCAN (lag: {lag_seconds:.0f}s) | BTC RSI: {btc_rsi:.1f} ({btc_hist:.2f}bps) | Surging: {btc_surging}")

# 4. Check each follower for lag divergence
for sym in FOLLOWERS:
    clean_sym = sym.replace("/", "")
    if clean_sym in positions:
        continue

    f_rsi, f_hist, _ = get_indicators(get_closed_series(sym))
    f_lagging = (f_rsi < 55) and (f_hist < btc_hist * 0.5)

    if btc_surging and f_lagging:
        trading_client.submit_order(MarketOrderRequest(
            symbol=sym,
            notional=NOTIONAL_PER_TRADE,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        ))
        print(f"  -> BUY {sym} ($500) | {sym} RSI: {f_rsi:.1f} ({f_hist:.2f}bps)")
    else:
        print(f"  -> IDLE {sym} | RSI: {f_rsi:.1f} ({f_hist:.2f}bps)")
