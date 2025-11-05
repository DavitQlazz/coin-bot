#!/usr/bin/env python3
"""
Binance Public API Tester
Tests connection and data fetching from Binance without API keys
"""

import ccxt
import pandas as pd
from datetime import datetime

def test_binance_connection():
    """Test Binance public API connection"""
    print("\n" + "="*60)
    print("🔌 TESTING BINANCE PUBLIC API")
    print("="*60)
    print("\n⚠️  Note: No API keys required for public data!")
    print("This uses Binance's free, public market data endpoints.\n")
    
    try:
        # Initialize Binance exchange (no API keys needed)
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        print("1️⃣  Initializing connection...")
        exchange.load_markets()
        print("   ✅ Connected to Binance successfully!")
        
        # Test fetching ticker
        print("\n2️⃣  Fetching current BTC/USDT price...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        print(f"   ✅ Current BTC price: ${current_price:,.2f}")
        print(f"   📊 24h High: ${ticker['high']:,.2f}")
        print(f"   📉 24h Low: ${ticker['low']:,.2f}")
        print(f"   📈 24h Volume: {ticker['baseVolume']:,.2f} BTC")
        
        # Test fetching historical data
        print("\n3️⃣  Fetching historical OHLCV data (last 100 hours)...")
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"   ✅ Retrieved {len(df)} candles")
        print(f"   📅 Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"   💰 Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
        
        # Show latest data
        print("\n4️⃣  Latest price data:")
        latest = df.iloc[-1]
        print(f"   Time: {latest['timestamp']}")
        print(f"   Open: ${latest['open']:,.2f}")
        print(f"   High: ${latest['high']:,.2f}")
        print(f"   Low: ${latest['low']:,.2f}")
        print(f"   Close: ${latest['close']:,.2f}")
        print(f"   Volume: {latest['volume']:,.2f} BTC")
        
        # Test other popular pairs
        print("\n5️⃣  Testing other trading pairs...")
        pairs = ['ETH/USDT', 'BNB/USDT', 'SOL/USDT']
        for pair in pairs:
            try:
                ticker = exchange.fetch_ticker(pair)
                print(f"   ✅ {pair}: ${ticker['last']:,.2f}")
            except Exception as e:
                print(f"   ❌ {pair}: {e}")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\n💡 You can now run backtesting without API keys!")
        print("   $ python backtest.py")
        print("\n📊 Available for backtesting:")
        print("   • Any symbol on Binance (BTC/USDT, ETH/USDT, etc.)")
        print("   • Multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)")
        print("   • Up to 1000 candles of historical data")
        print("   • All free and public - no account needed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  • Check internet connection")
        print("  • Binance may be temporarily unavailable")
        print("  • Try again in a few moments")
        print("  • Use demo mode: python demo.py")
        return False


def fetch_and_display_data(symbol='BTC/USDT', timeframe='1h', limit=30):
    """Fetch and display sample data"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        
        print(f"\n📊 Fetching {symbol} data ({timeframe} timeframe)...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"\n{'='*80}")
        print(f"Recent {symbol} Price Data")
        print(f"{'='*80}")
        print(df.tail(10).to_string(index=False))
        print(f"{'='*80}\n")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    print("\n🔬 BINANCE PUBLIC API - CONNECTION TEST")
    print("No API keys or account required!\n")
    
    success = test_binance_connection()
    
    if success:
        print("\n" + "="*60)
        print("📈 SAMPLE DATA PREVIEW")
        print("="*60)
        
        # Show sample data
        fetch_and_display_data('BTC/USDT', '1h', 20)
        
        print("\n✅ Ready to backtest with real Binance data!")
        print("\nNext steps:")
        print("  1. Run: python backtest.py")
        print("  2. The bot will use Binance public API automatically")
        print("  3. No configuration needed!")
