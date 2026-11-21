import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

NOTIONAL_PER_TRADE = 500.0
MAX_ENTRY_LAG_SECONDS = 360

# Stock closing bell guardrails (Eastern Time)
STOCK_NO_NEW_ENTRIES_MINUTE = 40  # Stop entering stocks at 3:40 PM ET
STOCK_FORCE_CLOSE_MINUTE = 55     # Force-close all open stocks at 3:55 PM ET

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
crypto_data_client = CryptoHistoricalDataClient()
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_indicators(df: pd.DataFrame):
    closes = df["close"]
    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist_bps = ((macd - signal) / closes) * 10000.0

    is_green = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
    return float(rsi), float(hist_bps.iloc[-1]), float(hist_bps.iloc[-2]), is_green


def is_thesis_reversed(l_rsi: float, l_hist: float, l_prev_hist: float, f_rsi: float, f_hist: float) -> tuple[bool, str]:
    """Returns (True, reason) if the catch-up / momentum thesis has played out or reversed."""
    if f_rsi >= 65.0:
        return True, f"follower overbought (RSI {f_rsi:.1f} >= 65)"
    if f_hist >= l_hist:
        return True, f"follower caught up to leader ({f_hist:.2f}bps >= {l_hist:.2f}bps)"
    if l_hist < 0 and l_hist < l_prev_hist:
        return True, f"leader momentum reversed ({l_hist:.2f}bps < {l_prev_hist:.2f}bps)"
    if l_rsi < 45.0 and l_hist < l_prev_hist:
        return True, f"leader trend broke down (RSI {l_rsi:.1f}, hist falling)"
    return False, ""


def run_sleeve(
    name: str,
    leader: str,
    followers: list[str],
    bars_df: pd.DataFrame,
    positions: dict,
    now_utc: datetime,
    min_hist_bps: float = 1.5,
    block_new_entries: bool = False,
    force_close_all: bool = False,
) -> set[str]:
    handled_clean_syms = set()
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

    l_rsi, l_hist, l_prev_hist, l_green = get_indicators(get_closed_df(leader))

    # 1. Manage exits for any open positions in this sleeve
    for sym in followers:
        clean_sym = sym.replace("/", "")
        if clean_sym not in positions:
            continue

        handled_clean_syms.add(clean_sym)

        if force_close_all:
            trading_client.close_position(clean_sym)
            print(f"[{name}] EOD_CLOSE {sym} at 3:55 PM ET cutoff")
            continue

        if sym not in bars_df.index.get_level_values(0):
            continue

        f_df = get_closed_df(sym)
        if len(f_df) < 30:
            continue

        f_rsi, f_hist, _, _ = get_indicators(f_df)
        should_exit, reason = is_thesis_reversed(l_rsi, l_hist, l_prev_hist, f_rsi, f_hist)

        if should_exit:
            trading_client.close_position(clean_sym)
            print(f"[{name}] EXIT {sym} — {reason}")
        else:
            print(
                f"[{name}] HOLDING {sym} | Leader RSI: {l_rsi:.1f} ({l_hist:.2f}bps) | Follower RSI: {f_rsi:.1f} ({f_hist:.2f}bps)"
            )

    if force_close_all or block_new_entries:
        if block_new_entries and not force_close_all:
            print(f"[{name}] CUTOFF — too close to closing bell for new entries")
        return handled_clean_syms

    # 2. Staleness guard for new entries
    if lag_seconds > MAX_ENTRY_LAG_SECONDS:
        print(f"[{name}] STALE_SKIP | Trigger delayed by {lag_seconds:.0f}s")
        return handled_clean_syms

    # 3. Leader regime check (Surges & Oversold Snapbacks)
    surge = (l_hist > l_prev_hist) and (l_hist > min_hist_bps) and (l_rsi > 58)
    snapback = (l_rsi < 38) and (l_hist > l_prev_hist) and (abs(l_hist) > min_hist_bps) and l_green
    regime = "SURGE" if surge else ("SNAPBACK" if snapback else "NONE")

    print(
        f"[{name}] SCAN (lag: {lag_seconds:.0f}s) | {leader} RSI: {l_rsi:.1f} ({l_hist:.2f}bps, min: {min_hist_bps}bps) | Regime: {regime}"
    )

    # 4. Evaluate followers for new entries
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
        surge_entry = surge and (f_rsi < 55) and ((l_hist - f_hist) > min_hist_bps * 0.5)
        snapback_entry = snapback and (f_rsi < 35) and (f_hist < l_hist)

        if surge_entry or snapback_entry:
            mode = "SURGE" if surge_entry else "SNAPBACK"
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=sym,
                    notional=NOTIONAL_PER_TRADE,
                    side=OrderSide.BUY,
                    time_in_force=tif,
                )
            )
            print(f"  -> [{name}] BUY {sym} [{mode}] (${NOTIONAL_PER_TRADE:.0f}) | RSI: {f_rsi:.1f} ({f_hist:.2f}bps)")
        else:
            print(f"  -> [{name}] IDLE {sym} | RSI: {f_rsi:.1f} ({f_hist:.2f}bps)")

    return handled_clean_syms


now_utc = datetime.now(timezone.utc)
now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
clock = trading_client.get_clock()

positions = {p.symbol: p for p in trading_client.get_all_positions()}
handled_positions = set()

# --- SLEEVE 1: 24/7 CRYPTO (Exits strictly on thesis reversal; higher bps hurdle for fees) ---
crypto_symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"]
crypto_bars = crypto_data_client.get_crypto_bars(
    CryptoBarsRequest(
        symbol_or_symbols=crypto_symbols,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=now_utc - timedelta(hours=6),
    )
).df
handled_positions |= run_sleeve(
    "CRYPTO",
    "BTC/USD",
    ["ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"],
    crypto_bars,
    positions,
    now_utc,
    min_hist_bps=30.0,
)

# --- SLEEVE 2: US EQUITIES (9:30 AM - 4:00 PM ET) ---
if clock.is_open:
    is_3pm_hour = now_et.hour == 15
    block_stock_entries = is_3pm_hour and (now_et.minute >= STOCK_NO_NEW_ENTRIES_MINUTE)
    force_close_stocks = is_3pm_hour and (now_et.minute >= STOCK_FORCE_CLOSE_MINUTE)

    stock_symbols = ["QQQ", "IWM", "ARKK", "NVDA", "TSLA"]
    stock_bars = stock_data_client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=stock_symbols,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=now_utc - timedelta(days=3),
            feed=DataFeed.IEX,
        )
    ).df
    handled_positions |= run_sleeve(
        "EQUITIES",
        "QQQ",
        ["IWM", "ARKK", "NVDA", "TSLA"],
        stock_bars,
        positions,
        now_utc,
        min_hist_bps=1.5,
        block_new_entries=block_stock_entries,
        force_close_all=force_close_stocks,
    )
else:
    print("[EQUITIES] Market closed — skipping stock scan")

# --- GLOBAL ORPHAN SWEEP (Clean out any stray/untracked open positions) ---
for clean_sym, pos in positions.items():
    if clean_sym in handled_positions:
        continue
    if pos.asset_class != AssetClass.CRYPTO and not clock.is_open:
        print(f"[ORPHAN] SKIP {clean_sym} (equity market closed)")
        continue
    trading_client.close_position(clean_sym)
    print(f"[ORPHAN] CLOSED untracked position {clean_sym}")
