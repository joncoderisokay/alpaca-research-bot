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

start = datetime.now(timezone.utc) - timedelta(hours=6)
req = CryptoBarsRequest(
    symbol_or_symbols=["BTC/USD", "ETH/USD"],
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=start
)
bars = data_client.get_crypto_bars(req).df

btc_closes = bars.xs("BTC/USD")["close"].tail(60)
eth_closes = bars.xs("ETH/USD")["close"].tail(60)

btc_rsi, btc_hist, btc_prev_hist = get_indicators(btc_closes)
eth_rsi, eth_hist, _ = get_indicators(eth_closes)

positions = {p.symbol: p for p in trading_client.get_all_positions()}

if "ETHUSD" in positions:
    closed = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=["ETH/USD"], limit=1))
    if closed and closed.filled_at:
        mins_held = (datetime.now(timezone.utc) - closed.filled_at).total_seconds() / 60.0
        if mins_held >= 30:
            trading_client.close_position("ETHUSD")
            print(f"CLOSED ETH/USD after {mins_held:.1f}m")
        else:
            print(f"HOLDING ETH/USD ({mins_held:.1f}m / 30m)")
else:
    btc_surging = (btc_hist > btc_prev_hist) and (btc_hist > 1.5) and (btc_rsi > 58)
    eth_lagging = (eth_rsi < 55) and (eth_hist < btc_hist * 0.5)

    if btc_surging and eth_lagging:
        trading_client.submit_order(MarketOrderRequest(
            symbol="ETH/USD",
            notional=1000.0,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        ))
        print(f"BUY ETH/USD | BTC RSI: {btc_rsi:.1f}, ETH RSI: {eth_rsi:.1f}")
    else:
        print(f"IDLE | BTC RSI: {btc_rsi:.1f} ({btc_hist:.2f}bps) | ETH RSI: {eth_rsi:.1f} ({eth_hist:.2f}bps)")
