# 🤖 Crypto & Forex Trading Bot

A sophisticated cryptocurrency and forex trading bot with technical analysis, backtesting, and risk management features.

**Now supports FOREX trading with Yahoo Finance API!** 🌍💱

## ⚠️ DISCLAIMER

**IMPORTANT**: This bot is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Never invest more than you can afford to lose. The authors are not responsible for any financial losses incurred while using this software.

## ✨ Features

### Crypto Trading (CCXT)
- **Multiple Trading Strategies**: RSI, MACD, EMA crossovers, Bollinger Bands
- **Risk Management**: Configurable stop-loss and take-profit levels
- **Backtesting**: Test strategies on historical data before live trading
- **Paper Trading**: Practice without risking real money
- **Market Monitor**: Real-time market analysis without trading
- **Strategy Optimizer**: Find optimal parameters for your trading strategy
- **Multiple Exchanges**: Supports Binance, Coinbase, Kraken, and more via CCXT

### Forex Trading (Yahoo Finance) 🆕
- **Free Data Access**: No API keys required
- **Major Currency Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, and more
- **Historical Backtesting**: Test on years of forex data
- **Multiple Timeframes**: 1m to 1mo intervals
- **Same Proven Strategy**: Uses optimized multi-indicator approach

## 📋 Requirements

- Python 3.8+
- API keys from your chosen exchange (for live trading)

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

Configure the following in `.env`:

```bash
EXCHANGE=binance              # Exchange name (binance, coinbase, kraken, etc.)
API_KEY=your_api_key_here     # Your exchange API key
API_SECRET=your_secret_here   # Your exchange API secret

TRADING_PAIR=BTC/USDT         # Trading pair
TIMEFRAME=1h                  # Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
INITIAL_CAPITAL=1000          # Starting capital in quote currency

STOP_LOSS_PERCENT=2.0         # Stop loss percentage
TAKE_PROFIT_PERCENT=5.0       # Take profit percentage
POSITION_SIZE_PERCENT=95      # Percentage of capital per trade

PAPER_TRADING=true            # Set to 'false' for live trading
```

### 3. Usage

#### 🌍 FOREX TRADING (NEW!)

**Test Yahoo Finance connection:**
```bash
python test_yfinance.py
```

**Backtest single forex pair:**
```bash
python backtest_forex.py EURUSD=X
```

**Backtest multiple forex pairs:**
```bash
python backtest_forex.py
```

This will test 4 major pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD) and generate a comparison report.

**📖 See [FOREX_TRADING.md](FOREX_TRADING.md) for complete forex documentation.**

---

#### 💰 CRYPTO TRADING

**Market Monitor (No Trading)**

Monitor market signals in real-time without executing trades:

```bash
python monitor.py
```

**Backtesting**

Test your strategy on historical data:

```bash
python backtest.py
```

This will:
- Fetch 30 days of historical data
- Run the strategy on past data
- Show detailed performance metrics
- Generate a performance chart

**Paper Trading**

Practice with paper trading (simulated trades):

```bash
# Ensure PAPER_TRADING=true in .env
python bot.py
```

**Live Trading**

**⚠️ RISK WARNING**: Only use live trading after thorough backtesting and paper trading!

```bash
# Set PAPER_TRADING=false in .env
python bot.py
```

**Strategy Optimization**

Find optimal parameters for your strategy:

```bash
python optimize.py
```

This tests various parameter combinations and shows the best performing settings.

## 📊 Strategy Details

The bot uses a multi-indicator approach:

### Entry Signals (BUY)
- RSI < 35 and recovering (oversold bounce)
- MACD bullish crossover
- EMA(9) > EMA(21) (golden cross)
- Price near lower Bollinger Band
- Requires 3+ conditions to trigger

### Exit Signals (SELL)
- RSI > 70 (overbought)
- MACD bearish crossover
- EMA death cross
- Stop-loss triggered (-2% default)
- Take-profit triggered (+5% default)
- Requires 2+ conditions to trigger

## 📈 Performance Metrics

The bot tracks:
- Total profit/loss
- Return on Investment (ROI)
- Win rate
- Average win/loss
- Profit factor
- Risk/reward ratio
- Trade history with timestamps

## 🔧 Customization

### Modify Strategy

Edit `bot.py` to customize the trading logic:

```python
def generate_signal(self, df):
    # Add your custom indicators and logic here
    pass
```

### Add New Indicators

The bot uses the `ta` library for technical analysis. Add more indicators:

```python
# In calculate_indicators() method
df['custom_indicator'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
```

### Change Risk Parameters

Adjust in `.env` file:
- `STOP_LOSS_PERCENT`: Maximum loss per trade
- `TAKE_PROFIT_PERCENT`: Target profit per trade
- `POSITION_SIZE_PERCENT`: Capital allocation per trade

## 🛡️ Safety Features

1. **Paper Trading Mode**: Test without real money
2. **Stop Loss**: Automatic exit on excessive losses
3. **Take Profit**: Lock in gains at target levels
4. **Position Sizing**: Risk only a portion of capital
5. **Rate Limiting**: Respects exchange API limits
6. **Error Handling**: Graceful handling of network issues

## 📁 Project Structure

```
coin-bot/
├── bot.py                    # Main crypto trading bot
├── bot_forex.py              # Forex trading bot (NEW!)
├── backtest.py               # Crypto backtesting module
├── backtest_forex.py         # Forex backtesting module (NEW!)
├── backtest_binance.py       # Enhanced Binance backtesting
├── test_binance.py           # Binance API connection test
├── test_yfinance.py          # Yahoo Finance API test (NEW!)
├── monitor.py                # Market monitoring tool
├── optimize.py               # Strategy optimizer
├── config.py                 # Interactive configuration wizard
├── summary.py                # Project summary tool
├── requirements.txt          # Python dependencies
├── .env                      # Configuration (optimized parameters)
├── .gitignore               # Git ignore rules
├── README.md                # Main documentation
├── QUICKSTART.md            # Quick start guide
├── ARCHITECTURE.md          # System architecture
├── BINANCE_API.md           # Binance public API guide
├── FOREX_TRADING.md         # Forex trading guide (NEW!)
├── OPTIMIZATION_INSIGHTS.md # Optimization analysis
├── optimization_results.json # Optimization test results
└── trades.json              # Trade history
```

## 🔍 Troubleshooting

### API Connection Issues

```bash
# Test exchange connection
python -c "import ccxt; print(ccxt.binance().fetch_ticker('BTC/USDT'))"
```

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### Data Fetching Errors

- Check your internet connection
- Verify API keys are correct
- Ensure trading pair exists on the exchange
- Check exchange API status

## 📚 Resources

- [CCXT Documentation](https://docs.ccxt.com/)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [Binance API](https://binance-docs.github.io/apidocs/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Final Warning

**Cryptocurrency trading is highly risky**:
- Markets are extremely volatile
- Past performance doesn't guarantee future results
- Automated trading can amplify losses
- Always start with paper trading
- Never risk money you can't afford to lose
- Do your own research (DYOR)

**The developers are not financial advisors and are not responsible for any losses.**

---

Built with ❤️ for educational purposes only