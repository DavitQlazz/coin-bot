#!/usr/bin/env python3
"""
Configuration Helper
Helps set up the bot configuration interactively
"""

import os
from pathlib import Path

def setup_config():
    """Interactive configuration setup"""
    print("\n" + "="*60)
    print("⚙️  CRYPTO TRADING BOT - CONFIGURATION WIZARD")
    print("="*60)
    print("\nThis wizard will help you configure your trading bot.")
    print("Press Enter to use default values shown in [brackets]\n")
    
    # Exchange
    print("📊 EXCHANGE CONFIGURATION")
    print("-" * 60)
    print("Supported exchanges: binance, coinbase, kraken, kucoin, bybit")
    exchange = input("Exchange [binance]: ").strip() or "binance"
    
    print("\n🔑 API CREDENTIALS")
    print("-" * 60)
    print("Get your API keys from your exchange's API management page")
    api_key = input("API Key [leave empty for demo mode]: ").strip() or "your_api_key_here"
    api_secret = input("API Secret [leave empty for demo mode]: ").strip() or "your_api_secret_here"
    
    print("\n💰 TRADING SETTINGS")
    print("-" * 60)
    trading_pair = input("Trading Pair [BTC/USDT]: ").strip() or "BTC/USDT"
    timeframe = input("Timeframe (1m, 5m, 15m, 1h, 4h, 1d) [1h]: ").strip() or "1h"
    initial_capital = input("Initial Capital [1000]: ").strip() or "1000"
    
    print("\n⚖️  RISK MANAGEMENT")
    print("-" * 60)
    stop_loss = input("Stop Loss Percent [2.0]: ").strip() or "2.0"
    take_profit = input("Take Profit Percent [5.0]: ").strip() or "5.0"
    position_size = input("Position Size Percent [95]: ").strip() or "95"
    
    print("\n🎯 TRADING MODE")
    print("-" * 60)
    print("PAPER TRADING: Simulates trades without real money (RECOMMENDED)")
    print("LIVE TRADING: Executes real trades with real money (RISKY)")
    paper_trading = input("Enable Paper Trading? (yes/no) [yes]: ").strip().lower() or "yes"
    paper_trading = "true" if paper_trading in ['yes', 'y', 'true', '1'] else "false"
    
    # Create .env content
    env_content = f"""# Exchange Configuration
EXCHANGE={exchange}
API_KEY={api_key}
API_SECRET={api_secret}

# Trading Configuration
TRADING_PAIR={trading_pair}
TIMEFRAME={timeframe}
INITIAL_CAPITAL={initial_capital}

# Risk Management
STOP_LOSS_PERCENT={stop_loss}
TAKE_PROFIT_PERCENT={take_profit}
POSITION_SIZE_PERCENT={position_size}

# Bot Settings
PAPER_TRADING={paper_trading}
"""
    
    # Save to .env
    env_path = Path(".env")
    
    if env_path.exists():
        print("\n⚠️  .env file already exists!")
        overwrite = input("Overwrite? (yes/no) [no]: ").strip().lower()
        if overwrite not in ['yes', 'y']:
            print("Configuration cancelled.")
            return
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("\n" + "="*60)
    print("✅ CONFIGURATION SAVED")
    print("="*60)
    print(f"\nYour settings have been saved to .env")
    print(f"\nExchange: {exchange}")
    print(f"Trading Pair: {trading_pair}")
    print(f"Timeframe: {timeframe}")
    print(f"Initial Capital: ${initial_capital}")
    print(f"Stop Loss: {stop_loss}%")
    print(f"Take Profit: {take_profit}%")
    print(f"Paper Trading: {paper_trading}")
    
    print("\n📚 NEXT STEPS:")
    print("-" * 60)
    
    if api_key == "your_api_key_here":
        print("1. Run demo mode: python demo.py")
        print("2. Get API keys from your exchange")
        print("3. Run this config script again with real keys")
    else:
        print("1. Test with backtest: python backtest.py")
        print("2. Monitor market: python monitor.py")
        if paper_trading == "true":
            print("3. Start paper trading: python bot.py")
        else:
            print("3. ⚠️  You enabled LIVE TRADING - be very careful!")
            print("   Start bot with: python bot.py")
    
    print("\n" + "="*60)


def show_current_config():
    """Display current configuration"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("\n❌ No .env file found. Run setup first.")
        return
    
    print("\n" + "="*60)
    print("📋 CURRENT CONFIGURATION")
    print("="*60)
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if 'SECRET' in line or 'KEY' in line:
                    key, value = line.split('=', 1)
                    print(f"{key}=***hidden***")
                else:
                    print(line)
    
    print("="*60)


if __name__ == "__main__":
    print("\n🤖 Crypto Trading Bot Configuration")
    print("\n1) Setup/Update Configuration")
    print("2) View Current Configuration")
    print("3) Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        setup_config()
    elif choice == "2":
        show_current_config()
    else:
        print("Goodbye!")
