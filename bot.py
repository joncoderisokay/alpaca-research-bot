import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

NOTIONAL_PER_TRADE = 500.0
MAX_ENTRY_LAG_SECONDS = 240
HOLD_MINUTES = 30

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
crypto_data_client = CryptoHistoricalDataClient()
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def get_indicators(df: pd.DataFrame):
    closes = df["close"]
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

    is_green = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
    return float(rsi), float(hist_bps.iloc[-1]), float(hist_bps.iloc[-2]), is_green

def run_sleeve(name: str, leader: str, followers: list[str], bars_df: pd.DataFrame, positions: dict, now_utc: datetime, force_close_all: bool = False):
    leader_df = bars_df.xs(leader)
    latest_bar_open = leader_df.index[-1]
    seconds_since_open = (now_utc - latest_bar_open).total_seconds()

    def get_closed_df(sym: str) -> pd.DataFrame:
        df = bars_df.xs(sym)
        return df.iloc[:-1].tail(60) if seconds_since_open < 300 else df.tail(60)

    closed_bar_close_time = (
        leader_df.index[-2] + timedelta(seconds=300)
        if seconds_since_open < 300
        else latest_bar_open + timedelta(seconds=300)
    )
    lag_seconds = (now_utc - closed_bar_close_time).total_seconds()

    # 1. Manage exits for open positions in this sleeve
    for sym in followers:
        clean_sym = sym.replace("/", "")
        if clean_sym in positions:
            if force_close_all:
                trading_client.close_position(clean_sym)
                print(f"[{name}] EOD_CLOSE {sym} before market close")
                continue

            closed_orders = trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[sym], limit=1)
            )
            if closed_orders and closed_orders.filled_at:
                mins_held = (now_utc - closed_orders.filled_at).total_seconds() / 60.0
                if mins_held >= HOLD_MINUTES:
                    trading_client.close_position(clean_sym)
                    print(f"[{name}] CLOSED {sym} after {mins_held:.1f}m (lag: {lag_seconds:.0f}s)")
                else:
                    print(f"[{name}] HOLDING {sym} ({mins_held:.1f}m / {HOLD_MINUTES}m)")

    if force_close_all:
        return

    # 2. Staleness guard
    if lag_seconds > MAX_ENTRY_LAG_SECONDS:
        print(f"[{name}] STALE_SKIP | Trigger delayed by {lag_seconds:.0f}s")
        return

    # 3. Leader regime check (Surges & Oversold Snapbacks)
    l_rsi, l_hist, l_prev_hist, l_green = get_indicators(get_closed_df(leader))
    surge = (l_hist > l_prev_hist) and (l_hist > 1.5) and (l_rsi > 58)
    snapback = (l_rsi < 38) and (l_hist > l_prev_hist) and l_green
    regime = "SURGE" if surge else ("SNAPBACK" if snapback else "NONE")

    print(f"[{name}] SCAN (lag: {lag_seconds:.0f}s) | {leader} RSI: {l_rsi:.1f} ({l_hist:.2f}bps) | Regime: {regime}")

    # 4. Evaluate followers
    tif = TimeInForce.GTC if "/" in leader else TimeInForce.DAY
    for sym in followers:
        clean_sym = sym.replace("/", "")
        if clean_sym in positions:
            continue
        if sym not in bars_df.index.get_level_values(0):
            continue

        f_df = get_closed_df(sym)
        if len(f_df) < 30:
            continue

        f_rsi, f_hist, _, _ = get_indicators(f_df)
        surge_entry = surge and (f_rsi < 55) and (f_hist < l_hist * 0.5)
        snapback_entry = snapback and (f_rsi < 35) and (f_hist < l_hist)

        if surge_entry or snapback_entry:
            mode = "SURGE" if surge_entry else "SNAPBACK"
            trading_client.submit_order(MarketOrderRequest(
                symbol=sym,
                notional=NOTIONAL_PER_TRADE,
                side=OrderSide.BUY,
                time_in_force=tif
            ))
            print(f"  -> [{name}] BUY {sym} [{mode}] ($500) | RSI: {f_rsi:.1f} ({f_hist:.2f}bps)")
        else:
            print(f"  -> [{name}] IDLE {sym} | RSI: {f_rsi:.1f} ({f_hist:.2f}bps)")

now_utc = datetime.now(timezone.utc)
positions = {p.symbol: p for p in trading_client.get_all_positions()}

# --- SLEEVE 1: 24/7 CRYPTO ---
crypto_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"]
crypto_bars = crypto_data_client.get_crypto_bars(
    CryptoBarsRequest(
        symbol_or_symbols=crypto_symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=now_utc - timedelta(hours=6)
    )
).df
run_sleeve("CRYPTO", "BTC/USD", ["ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"], crypto_bars, positions, now_utc)

# --- SLEEVE 2: US EQUITIES (9:30 AM - 4:00 PM ET) ---
clock = trading_client.get_clock()
if clock.is_open:
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    near_close = (now_et.hour == 15 and now_et.minute >= 50)
    stock_symbols = ["QQQ", "IWM", "ARKK", "NVDA", "TSLA"]
    stock_bars = stock_data_client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=stock_symbols,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=now_utc - timedelta(days=3),  # Covers weekends/overnight gaps to ensure 60 bars
            feed=DataFeed.IEX
        )
    ).df
    run_sleeve("EQUITIES", "QQQ", ["IWM", "ARKK", "NVDA", "TSLA"], stock_bars, positions, now_utc, force_close_all=near_close)
else:
    print("[EQUITIES] Market closed — skipping stock scan")
