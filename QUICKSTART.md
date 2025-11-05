# 🚀 Quick Start Guide

## Crypto Trading Bot - Get Started in 5 Minutes

### Step 1: Verify Installation ✅

Dependencies are already installed! Verify with:
```bash
python -c "import ccxt, pandas, ta; print('✅ All dependencies installed!')"
```

### Step 2: Choose Your Mode 🎯

#### Option A: Demo Mode (Recommended First!)
**No API keys needed - Safe for testing**

```bash
python demo.py
```

This runs a complete simulation with fake data to show how the bot works.

#### Option B: Real Market Data (Read-Only)
**Requires API keys but won't trade**

```bash
python monitor.py
```

Displays live market analysis and signals without executing trades.

#### Option C: Backtest on Historical Data
**Test your strategy on real past data**

```bash
python backtest.py
```

Tests the strategy on 30 days of historical data and shows performance.

#### Option D: Live Trading
**⚠️ ONLY after thorough testing!**

```bash
python bot.py
```

### Step 3: Configure Your Bot ⚙️

Run the interactive configuration wizard:

```bash
python config.py
```

Or manually edit `.env`:

```bash
nano .env
```

### Step 4: Test Before Trading! 🧪

**CRITICAL: Follow this order:**

1. ✅ Run demo: `python demo.py`
2. ✅ Configure with real API keys
3. ✅ Run backtest: `python backtest.py`
4. ✅ Monitor market: `python monitor.py`
5. ✅ Paper trade: `python bot.py` (with PAPER_TRADING=true)
6. ⚠️ Live trade only after success in all above steps

### Step 5: Getting API Keys 🔑

#### Binance (Recommended)
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create new API key
3. Enable "Enable Reading" and "Enable Spot & Margin Trading"
4. Save your keys to `.env`

#### Coinbase Pro
1. Go to https://pro.coinbase.com/profile/api
2. Create new API key with "View" and "Trade" permissions
3. Save keys to `.env`

#### Security Tips:
- Never share your API keys
- Use IP whitelist if available
- Start with small amounts
- Use separate keys for testing

### Common Commands 📝

```bash
# Quick start menu
./start.sh

# Run demo (safe)
python demo.py

# Configure bot
python config.py

# Backtest strategy
python backtest.py

# Monitor market (read-only)
python monitor.py

# Paper trading
python bot.py

# Find best parameters
python optimize.py

# Install/update dependencies
pip install -r requirements.txt --upgrade
```

### Project Structure 📁

```
coin-bot/
├── bot.py           # Main trading bot
├── backtest.py      # Backtesting module
├── monitor.py       # Market monitor (read-only)
├── demo.py          # Demo with simulated data
├── optimize.py      # Strategy optimizer
├── config.py        # Configuration wizard
├── start.sh         # Quick start menu
├── .env             # Your configuration (keep secret!)
└── README.md        # Full documentation
```

### Troubleshooting 🔧

**"Invalid API Key" error:**
- Check keys in `.env` file
- Verify keys are active on exchange
- Run demo mode instead: `python demo.py`

**"No module named..." error:**
```bash
pip install -r requirements.txt
```

**Connection timeout:**
- Check internet connection
- Try different exchange
- Use demo mode for testing

**Bot not executing trades:**
- Verify PAPER_TRADING setting
- Check if market conditions meet entry criteria
- Review strategy parameters in `.env`

### Strategy Overview 📊

The bot uses multiple technical indicators:

**Buy Signals:**
- RSI oversold and recovering
- MACD bullish crossover
- EMA golden cross
- Price near lower Bollinger Band

**Sell Signals:**
- RSI overbought
- MACD bearish crossover
- Stop-loss hit (-2% default)
- Take-profit hit (+5% default)

### Risk Management 🛡️

Built-in safety features:
- ✅ Paper trading mode
- ✅ Stop-loss protection
- ✅ Take-profit targets
- ✅ Position sizing limits
- ✅ API rate limiting

### Customization 🎨

**Change trading pair:**
```bash
# In .env
TRADING_PAIR=ETH/USDT
```

**Adjust risk parameters:**
```bash
# In .env
STOP_LOSS_PERCENT=3.0
TAKE_PROFIT_PERCENT=7.0
```

**Modify strategy:**
Edit `bot.py` - the `generate_signal()` function contains the trading logic.

### Performance Tracking 📈

The bot tracks:
- Total profit/loss
- Win rate
- ROI (Return on Investment)
- Trade history
- Equity curve (in backtesting)

Results are saved to `trades.json`

### Getting Help 🆘

1. Read the full `README.md`
2. Check `.env.example` for configuration options
3. Run demo mode to understand bot behavior
4. Start with small amounts
5. Test thoroughly before live trading

### Final Reminders ⚠️

- 🚫 Never invest more than you can lose
- ✅ Always start with paper trading
- 📊 Backtest extensively
- 📚 Understand the strategy
- 🔐 Keep API keys secure
- ⚡ Markets are highly volatile
- 📉 Past performance ≠ future results

---

**Ready to start?**

```bash
# Run the demo!
python demo.py
```

Good luck and trade safely! 🚀
