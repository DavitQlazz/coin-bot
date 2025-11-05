"""
Test Yahoo Finance API Connection
Verify that we can fetch forex data successfully
"""

import yfinance as yf
import pandas as pd
from datetime import datetime

def test_yfinance_connection():
    """Test basic Yahoo Finance API connectivity"""
    print("="*70)
    print("YAHOO FINANCE API CONNECTION TEST")
    print("="*70)
    
    # Test forex pairs
    forex_pairs = {
        'EURUSD=X': 'EUR/USD',
        'GBPUSD=X': 'GBP/USD', 
        'USDJPY=X': 'USD/JPY',
        'AUDUSD=X': 'AUD/USD',
        'USDCAD=X': 'USD/CAD',
        'NZDUSD=X': 'NZD/USD',
        'USDCHF=X': 'USD/CHF'
    }
    
    print(f"\nTesting {len(forex_pairs)} forex pairs...\n")
    
    successful = 0
    failed = 0
    
    for symbol, name in forex_pairs.items():
        try:
            ticker = yf.Ticker(symbol)
            
            # Fetch recent data
            df = ticker.history(period='5d', interval='1h')
            
            if not df.empty:
                latest = df.iloc[-1]
                print(f"✅ {name:<12} ({symbol:<10}): ${latest['Close']:.5f} | Volume: {latest['Volume']:,.0f}")
                successful += 1
            else:
                print(f"⚠️  {name:<12} ({symbol:<10}): No data available")
                failed += 1
                
        except Exception as e:
            print(f"❌ {name:<12} ({symbol:<10}): Error - {str(e)[:50]}")
            failed += 1
    
    print(f"\n{'-'*70}")
    print(f"Results: {successful} successful, {failed} failed")
    print(f"{'-'*70}\n")
    
    return successful > 0


def fetch_detailed_data(symbol='EURUSD=X'):
    """Fetch and display detailed data for a forex pair"""
    print("="*70)
    print(f"DETAILED DATA FETCH: {symbol}")
    print("="*70)
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get info
        print(f"\n📊 Ticker Information:")
        info = ticker.info
        if info:
            print(f"   Currency: {info.get('currency', 'N/A')}")
            print(f"   Exchange: {info.get('exchange', 'N/A')}")
        
        # Fetch different timeframes
        timeframes = {
            '1d (5min)': ('1d', '5m'),
            '5d (15min)': ('5d', '15m'),
            '30d (1hour)': ('30d', '1h'),
            '90d (1day)': ('3mo', '1d')
        }
        
        print(f"\n📈 Historical Data Availability:")
        for label, (period, interval) in timeframes.items():
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                first_date = df.index[0].strftime('%Y-%m-%d %H:%M')
                last_date = df.index[-1].strftime('%Y-%m-%d %H:%M')
                print(f"   ✅ {label:<15}: {len(df):>4} candles | {first_date} → {last_date}")
            else:
                print(f"   ❌ {label:<15}: No data")
        
        # Sample recent data
        print(f"\n📊 Recent 1-Hour Candles (Last 10):")
        df = ticker.history(period='5d', interval='1h')
        
        if not df.empty:
            recent = df.tail(10)
            print(f"\n{'Timestamp':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<12}")
            print("-"*70)
            for idx, row in recent.iterrows():
                ts = idx.strftime('%Y-%m-%d %H:%M')
                print(f"{ts:<20} {row['Open']:<10.5f} {row['High']:<10.5f} {row['Low']:<10.5f} {row['Close']:<10.5f} {row['Volume']:<12,.0f}")
        
        print(f"\n{'='*70}")
        print("✅ Data fetch successful!")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return False


def test_all_intervals(symbol='EURUSD=X'):
    """Test all available intervals for a forex pair"""
    print("="*70)
    print(f"INTERVAL AVAILABILITY TEST: {symbol}")
    print("="*70)
    
    intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    print(f"\nTesting {len(intervals)} intervals...\n")
    
    for interval in intervals:
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to fetch data
            if interval in ['1m', '2m', '5m', '15m', '30m']:
                period = '1d'
            elif interval in ['60m', '90m', '1h']:
                period = '5d'
            elif interval == '1d':
                period = '1mo'
            else:
                period = '1y'
            
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                print(f"✅ {interval:<6}: {len(df):>4} candles available")
            else:
                print(f"⚠️  {interval:<6}: No data")
                
        except Exception as e:
            print(f"❌ {interval:<6}: {str(e)[:50]}")
    
    print(f"\n{'='*70}\n")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("YAHOO FINANCE FOREX API TEST SUITE")
    print("="*70)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Test 1: Basic connectivity
    print("TEST 1: Basic Connectivity")
    print("-"*70)
    if not test_yfinance_connection():
        print("❌ Basic connectivity test failed!")
        return
    
    input("\nPress Enter to continue to detailed data fetch test...")
    
    # Test 2: Detailed data fetch
    print("\nTEST 2: Detailed Data Fetch")
    print("-"*70)
    fetch_detailed_data('EURUSD=X')
    
    input("\nPress Enter to continue to interval availability test...")
    
    # Test 3: Interval availability
    print("\nTEST 3: Interval Availability")
    print("-"*70)
    test_all_intervals('EURUSD=X')
    
    print("="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)
    print("\n💡 You can now run: python backtest_forex.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
