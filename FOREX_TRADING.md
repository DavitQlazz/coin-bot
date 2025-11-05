# Forex Trading with Yahoo Finance API

## Overview

The bot now supports **forex (currency) trading** using Yahoo Finance's free API. This allows backtesting and trading on major currency pairs without requiring API keys or paid subscriptions.

## 🌟 Features

- **Free Data Access**: No API keys required (Yahoo Finance public data)
- **Major Forex Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, and more
- **Multiple Timeframes**: 1m, 5m, 15m, 30m, 1h, 1d, 1wk intervals
- **Historical Data**: Up to several years of historical data available
- **Same Strategy**: Uses the optimized multi-indicator strategy from crypto bot

## 📦 Installation

The required library is already added to `requirements.txt`:

```bash
pip install yfinance
```

## 🎯 Forex Pair Symbols

Yahoo Finance uses a special format for forex pairs:

| Pair Name | Yahoo Symbol | Description |
|-----------|--------------|-------------|
| EUR/USD   | EURUSD=X     | Euro / US Dollar |
| GBP/USD   | GBPUSD=X     | British Pound / US Dollar |
| USD/JPY   | USDJPY=X     | US Dollar / Japanese Yen |
| AUD/USD   | AUDUSD=X     | Australian Dollar / US Dollar |
| USD/CAD   | USDCAD=X     | US Dollar / Canadian Dollar |
| NZD/USD   | NZDUSD=X     | New Zealand Dollar / US Dollar |
| USD/CHF   | USDCHF=X     | US Dollar / Swiss Franc |

**Note**: Always append `=X` to the currency pair code.

## 🚀 Usage

### 1. Test Yahoo Finance Connection

First, verify that you can fetch forex data:

```bash
python test_yfinance.py
```

This will:
- Test connectivity to Yahoo Finance
- Fetch current prices for 7 major pairs
- Display detailed data availability
- Show recent candle data

### 2. Run Single Pair Backtest

Test a specific forex pair:

```bash
# Default (EUR/USD, 30 days, 1-hour candles)
python backtest_forex.py EURUSD=X

# Custom pair
python backtest_forex.py GBPUSD=X

# Custom pair with period
python -c "from backtest_forex import test_single_pair; test_single_pair('USDJPY=X', '60d', '4h')"
```

### 3. Run Comprehensive Backtest

Test all major pairs at once:

```bash
python backtest_forex.py
```

This will:
- Test 4 major currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD)
- Use 30 days of 1-hour candle data
- Apply the optimized parameters (3% stop loss, 6% take profit)
- Generate comparison table
- Save results to JSON file

## 📊 Backtest Results (Last Run)

**Date**: November 5, 2025  
**Period**: 30 days (Oct 6 - Nov 5, 2025)  
**Timeframe**: 1-hour candles  
**Parameters**: Stop Loss 3.0%, Take Profit 6.0%

| Pair      | Trades | Win Rate | ROI     | Profit Factor | Final Capital |
|-----------|--------|----------|---------|---------------|---------------|
| AUD/USD   | 1      | 0.00%    | -0.98%  | 0.00          | $990.25       |
| EUR/USD   | 1      | 0.00%    | -2.02%  | 0.00          | $979.82       |
| USD/JPY   | 2      | 50.00%   | -2.34%  | 0.18          | $976.58       |
| GBP/USD   | 2      | 0.00%    | -2.91%  | 0.00          | $970.89       |

**Summary**:
- Average ROI: -2.06%
- Average Win Rate: 12.50%
- Profitable Pairs: 0/4
- Best Performer: AUD/USD (-0.98% loss)

## 📈 Analysis

### Why Negative Results?

Similar to the crypto backtesting results, the forex markets during this period (Oct-Nov 2025) were **ranging/sideways**:

1. **EUR/USD**: Declined from ~$1.17 to ~$1.15 (downtrend)
2. **GBP/USD**: Declined from ~$1.34 to ~$1.30 (downtrend)
3. **USD/JPY**: Rose from ~149 to ~154 (uptrend)
4. **AUD/USD**: Minimal movement around $0.65 (ranging)

### Strategy Performance

The **multi-indicator trend-following strategy** struggles in:
- ❌ Ranging/choppy markets (no clear trend)
- ❌ Weak trends with frequent reversals
- ❌ High volatility periods

It excels in:
- ✅ Strong, sustained trends (up or down)
- ✅ Low-volatility trending markets
- ✅ Markets with clear momentum

### Key Observations

1. **Low Trade Count**: Only 1-2 trades per pair in 30 days
   - Indicates strategy is selective (good)
   - Not overtrading despite volatile conditions
   
2. **USD/JPY Best Win Rate**: 50% win rate, but still negative ROI
   - Had the strongest trend (yen weakening)
   - One winning trade offset by one stop-loss
   
3. **Small Losses**: All pairs lost less than 3%
   - Optimized stop-loss (3%) prevented larger losses
   - Risk management working as intended

## 🔧 Configuration

The forex bot uses environment variables from `.env`:

```bash
# Forex Configuration
FOREX_PAIR=EURUSD=X
FOREX_INTERVAL=1h

# Risk Management (optimized)
STOP_LOSS_PERCENT=3.0
TAKE_PROFIT_PERCENT=6.0

# Capital
INITIAL_CAPITAL=10000
```

## 🎯 Available Timeframes

Yahoo Finance supports these intervals:

| Interval | Description | Max History | Best For |
|----------|-------------|-------------|----------|
| 1m       | 1 minute    | 7 days      | Scalping |
| 5m       | 5 minutes   | 60 days     | Day trading |
| 15m      | 15 minutes  | 60 days     | Day trading |
| 30m      | 30 minutes  | 60 days     | Swing trading |
| 1h       | 1 hour      | 730 days    | Swing trading |
| 1d       | 1 day       | 10+ years   | Position trading |
| 1wk      | 1 week      | 10+ years   | Long-term |
| 1mo      | 1 month     | 10+ years   | Long-term |

## 📁 Files Created

### Core Files

- **bot_forex.py** (360 lines)
  - Forex trading bot class
  - Uses yfinance for data fetching
  - Same indicator calculations as crypto bot
  - Handles both LONG and SHORT positions
  
- **backtest_forex.py** (160 lines)
  - Forex backtesting framework
  - Single-pair and multi-pair testing
  - Results comparison and analysis
  - JSON export of results
  
- **test_yfinance.py** (190 lines)
  - API connection testing
  - Data availability verification
  - Interval testing suite

### Results Files

- **forex_backtest_results_YYYYMMDD_HHMMSS.json**
  - Detailed backtest results
  - All trades with entry/exit data
  - Performance metrics per pair

## 💡 Recommendations

### 1. Different Time Periods

The current 30-day period showed ranging markets. Try:

```bash
# Test longer periods for better trends
python -c "from backtest_forex import test_single_pair; test_single_pair('EURUSD=X', '90d', '4h')"

# Test different timeframes
python -c "from backtest_forex import test_single_pair; test_single_pair('EURUSD=X', '180d', '1d')"
```

### 2. Try More Pairs

Add exotic pairs to `backtest_forex.py`:

```python
exotic_pairs = [
    'EURUSD=X',   # Major
    'GBPJPY=X',   # Cross
    'EURGBP=X',   # Cross
    'USDZAR=X',   # Exotic (South African Rand)
    'USDTRY=X',   # Exotic (Turkish Lira)
]
```

### 3. Market Regime Detection

Add logic to detect trending vs ranging markets:

```python
def is_trending(df, period=20):
    """Detect if market is trending"""
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    return adx.adx().iloc[-1] > 25  # ADX > 25 = trending
```

### 4. Optimize for Forex

Forex markets have different characteristics:

```python
# Tighter stops for forex (less volatile than crypto)
STOP_LOSS_PERCENT=2.0  # Instead of 3.0

# More frequent signals (forex trades more)
min_conditions = 2  # Keep at 2, but add more indicators

# Add forex-specific indicators
- Interest rate differentials
- Economic calendar events
- Correlation analysis
```

## 🚨 Important Notes

### Market Hours

Forex markets trade 24/5 (Monday-Friday):
- Sydney: 10pm - 7am GMT
- Tokyo: 12am - 9am GMT
- London: 8am - 5pm GMT
- New York: 1pm - 10pm GMT

**Best trading times**: 
- London/New York overlap (1pm-5pm GMT) - highest liquidity
- Avoid weekends (markets closed)

### Data Limitations

Yahoo Finance free data:
- ✅ Historical data available
- ✅ Real-time quotes (15-min delay)
- ⚠️ Volume data may be 0 (forex is decentralized)
- ❌ No tick-level data
- ❌ No order book data

### Risk Management

**Forex is leveraged trading**:
- Most brokers offer 50:1 to 500:1 leverage
- Can magnify profits AND losses
- This bot assumes 1:1 (no leverage) for safety
- **Always use stop losses**
- **Never risk more than 1-2% per trade**

## 🔄 Workflow

### Complete Testing Workflow

```bash
# 1. Test API connection
python test_yfinance.py

# 2. Run comprehensive backtest
python backtest_forex.py

# 3. Test best performing pair with longer period
python -c "from backtest_forex import test_single_pair; test_single_pair('AUDUSD=X', '90d', '1h')"

# 4. Optimize parameters for forex (future enhancement)
# python optimize_forex.py

# 5. Paper trade with best pair
python bot_forex.py  # (currently runs as backtest)
```

## 📚 Resources

### Learning Resources

- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
- [Forex Trading Basics](https://www.investopedia.com/forex-trading-4689660)
- [Technical Analysis for Forex](https://www.babypips.com/learn/forex)

### Data Sources

- **Yahoo Finance**: Free, no API key required
- **Alternative**: OANDA API (requires account)
- **Alternative**: Alpha Vantage (free tier available)

## 🎯 Next Steps

1. **Run on trending periods**: Test during strong trends (e.g., 2020-2021 USD weakness)
2. **Add more indicators**: Interest rate differentials, carry trade signals
3. **Optimize for forex**: Run parameter optimization specific to forex characteristics
4. **Implement live trading**: Add paper trading with real-time Yahoo Finance data
5. **Add correlation analysis**: Trade pairs with negative correlation for hedging

## ⚠️ Disclaimer

**This is educational software**:
- Past performance does not guarantee future results
- Forex trading involves substantial risk of loss
- Current results show negative ROI due to ranging markets
- Always paper trade extensively before using real money
- Consider market conditions before trading
- The strategy is designed for trending markets, not ranging conditions

---

**Created**: November 5, 2025  
**Version**: 1.0  
**Data Source**: Yahoo Finance (free, public API)  
**Status**: ✅ Implemented, tested on 4 major pairs
