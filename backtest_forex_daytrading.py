"""
Day Trading Strategy Testing
Test V2 strategy with intraday timeframes (15m, 30m, 1h)
"""
from backtest_forex_v2 import ForexBacktesterV2
from backtest_forex import ForexBacktester
import pandas as pd

def test_daytrading_timeframes(pair='USDJPY=X', period='60d'):
    """
    Test different intraday timeframes for day trading
    
    Args:
        pair: Forex pair to test
        period: Historical period (use shorter for intraday - max 60d recommended)
    """
    
    # Intraday timeframes for day trading
    timeframes = ['15m', '30m', '1h']
    
    print(f"\n{'='*80}")
    print(f"DAY TRADING STRATEGY TEST: {pair}")
    print(f"Period: {period}")
    print(f"Testing timeframes: {', '.join(timeframes)}")
    print(f"{'='*80}\n")
    
    results = []
    
    for tf in timeframes:
        print(f"\n{'─'*80}")
        print(f"📊 TESTING: {tf} timeframe")
        print(f"{'─'*80}")
        
        # Test V2 strategy
        bot = ForexBacktesterV2(
            pair=pair,
            stop_loss=2.6,
            take_profit=6.2,
            min_adx=25,
            min_signal_score=3
        )
        
        try:
            metrics = bot.backtest(period=period, interval=tf, verbose=False)
            
            if metrics:
                results.append({
                    'timeframe': tf,
                    'strategy': 'V2',
                    **metrics
                })
        except Exception as e:
            print(f"❌ Error testing {tf}: {e}")
    
    # Print comparison
    if results:
        print(f"\n{'='*90}")
        print(f"DAY TRADING RESULTS COMPARISON: {pair}")
        print(f"{'='*90}")
        print(f"{'Timeframe':<12} {'ROI':<10} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'Avg Win':<12} {'Avg Loss':<12}")
        print(f"{'-'*90}")
        
        for r in results:
            print(f"{r['timeframe']:<12} {r['return_pct']:>8.2f}% {r['total_trades']:>7} "
                  f"{r['win_rate']:>8.1f}% {r['profit_factor']:>14.2f} "
                  f"${r['avg_win']:>10.2f} ${r['avg_loss']:>10.2f}")
        
        # Find best timeframe
        best = max(results, key=lambda x: x['return_pct'])
        print(f"{'-'*90}")
        print(f"🏆 BEST TIMEFRAME: {best['timeframe']} ({best['return_pct']:+.2f}% ROI, {best['win_rate']:.1f}% WR)")
        print(f"{'='*90}\n")
    
    return results


def compare_daytrading_vs_swing(pair='USDJPY=X'):
    """
    Compare day trading (15m, 30m, 1h) vs swing trading (4h)
    """
    print(f"\n{'#'*80}")
    print(f"DAY TRADING vs SWING TRADING COMPARISON: {pair}")
    print(f"{'#'*80}\n")
    
    results = []
    
    # Day trading timeframes (60 days max for intraday data)
    daytrading_configs = [
        ('15m', '60d', 'Day Trading (15-min)'),
        ('30m', '60d', 'Day Trading (30-min)'),
        ('1h', '60d', 'Day Trading (1-hour)'),
    ]
    
    # Swing trading (longer period)
    swing_config = ('4h', '360d', 'Swing Trading (4-hour)')
    
    print("🔷 Testing Day Trading Timeframes...")
    for tf, period, label in daytrading_configs:
        print(f"\n📊 {label} ({period})")
        bot = ForexBacktesterV2(
            pair=pair,
            stop_loss=2.6,
            take_profit=6.2,
            min_adx=25,
            min_signal_score=3
        )
        
        try:
            metrics = bot.backtest(period=period, interval=tf, verbose=False)
            if metrics:
                results.append({
                    'type': label,
                    'timeframe': tf,
                    'period': period,
                    **metrics
                })
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🔶 Testing Swing Trading...")
    tf, period, label = swing_config
    print(f"\n📊 {label} ({period})")
    bot = ForexBacktesterV2(
        pair=pair,
        stop_loss=2.6,
        take_profit=6.2,
        min_adx=25,
        min_signal_score=3
    )
    
    try:
        metrics = bot.backtest(period=period, interval=tf, verbose=False)
        if metrics:
            results.append({
                'type': label,
                'timeframe': tf,
                'period': period,
                **metrics
            })
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Summary
    if results:
        print(f"\n{'='*100}")
        print(f"COMPARISON SUMMARY: {pair}")
        print(f"{'='*100}")
        print(f"{'Trading Style':<25} {'TF':<8} {'Period':<10} {'ROI':<10} {'Trades':<8} {'Win Rate':<10} {'PF':<8}")
        print(f"{'-'*100}")
        
        for r in results:
            print(f"{r['type']:<25} {r['timeframe']:<8} {r['period']:<10} "
                  f"{r['return_pct']:>8.2f}% {r['total_trades']:>7} "
                  f"{r['win_rate']:>8.1f}% {r['profit_factor']:>7.2f}")
        
        # Analysis
        daytrading = [r for r in results if 'Day Trading' in r['type']]
        swing = [r for r in results if 'Swing Trading' in r['type']]
        
        if daytrading and swing:
            avg_dt_roi = sum(r['return_pct'] for r in daytrading) / len(daytrading)
            avg_dt_wr = sum(r['win_rate'] for r in daytrading) / len(daytrading)
            avg_dt_trades = sum(r['total_trades'] for r in daytrading) / len(daytrading)
            
            swing_roi = swing[0]['return_pct']
            swing_wr = swing[0]['win_rate']
            swing_trades = swing[0]['total_trades']
            
            print(f"{'-'*100}")
            print(f"{'Day Trading Average':<25} {'':<8} {'':<10} "
                  f"{avg_dt_roi:>8.2f}% {avg_dt_trades:>7.1f} {avg_dt_wr:>8.1f}%")
            print(f"{'Swing Trading':<25} {'':<8} {'':<10} "
                  f"{swing_roi:>8.2f}% {swing_trades:>7} {swing_wr:>8.1f}%")
            print(f"{'='*100}\n")
            
            # Verdict
            print("💡 ANALYSIS:")
            print(f"   Day Trading:")
            print(f"   ├─ Average ROI: {avg_dt_roi:+.2f}%")
            print(f"   ├─ Average Win Rate: {avg_dt_wr:.1f}%")
            print(f"   ├─ Average Trades: {avg_dt_trades:.0f}")
            print(f"   └─ More active (higher trade frequency)")
            print(f"\n   Swing Trading:")
            print(f"   ├─ ROI: {swing_roi:+.2f}%")
            print(f"   ├─ Win Rate: {swing_wr:.1f}%")
            print(f"   ├─ Trades: {swing_trades}")
            print(f"   └─ More patient (lower frequency)")
            
            if avg_dt_roi > swing_roi:
                diff = avg_dt_roi - swing_roi
                print(f"\n✅ DAY TRADING WINS: {diff:+.2f}% better ROI")
            else:
                diff = swing_roi - avg_dt_roi
                print(f"\n✅ SWING TRADING WINS: {diff:+.2f}% better ROI")
    
    return results


def test_multiple_pairs_daytrading():
    """Test day trading on multiple pairs with 1-hour timeframe"""
    pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
    
    print(f"\n{'='*90}")
    print(f"DAY TRADING TEST: ALL MAJOR PAIRS (1-hour timeframe)")
    print(f"Period: 60 days")
    print(f"{'='*90}\n")
    
    results = []
    
    for pair in pairs:
        print(f"\n{'─'*90}")
        print(f"Testing: {pair}")
        print(f"{'─'*90}")
        
        bot = ForexBacktesterV2(
            pair=pair,
            stop_loss=2.6,
            take_profit=6.2,
            min_adx=25,
            min_signal_score=3
        )
        
        try:
            metrics = bot.backtest(period='60d', interval='1h', verbose=False)
            
            if metrics:
                results.append({
                    'pair': pair,
                    **metrics
                })
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    if results:
        print(f"\n{'='*90}")
        print(f"DAY TRADING SUMMARY: 1-HOUR TIMEFRAME (60 days)")
        print(f"{'='*90}")
        print(f"{'Pair':<12} {'ROI':<10} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<15} {'Avg Win':<12} {'Avg Loss':<12}")
        print(f"{'-'*90}")
        
        for r in results:
            print(f"{r['pair']:<12} {r['return_pct']:>8.2f}% {r['total_trades']:>7} "
                  f"{r['win_rate']:>8.1f}% {r['profit_factor']:>14.2f} "
                  f"${r['avg_win']:>10.2f} ${r['avg_loss']:>10.2f}")
        
        avg_roi = sum(r['return_pct'] for r in results) / len(results)
        avg_wr = sum(r['win_rate'] for r in results) / len(results)
        profitable = sum(1 for r in results if r['return_pct'] > 0)
        
        print(f"{'-'*90}")
        print(f"{'AVERAGE':<12} {avg_roi:>8.2f}% {'-':>7} {avg_wr:>8.1f}% {'-':>14} {'-':>11} {'-':>11}")
        print(f"{'Profitable:':<12} {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
        print(f"{'='*90}\n")
        
        # Best pair
        best = max(results, key=lambda x: x['return_pct'])
        print(f"🏆 BEST PAIR FOR DAY TRADING: {best['pair']}")
        print(f"   ROI: {best['return_pct']:+.2f}%")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
        print(f"   Trades: {best['total_trades']}")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
    
    return results


if __name__ == '__main__':
    print("\n🚀 DAY TRADING STRATEGY TESTING\n")
    
    # Test 1: Different intraday timeframes on USD/JPY
    print("\n" + "="*80)
    print("TEST 1: TIMEFRAME COMPARISON (USD/JPY)")
    print("="*80)
    test_daytrading_timeframes('USDJPY=X', period='60d')
    
    # Test 2: Day trading vs Swing trading
    print("\n" + "="*80)
    print("TEST 2: DAY TRADING vs SWING TRADING (USD/JPY)")
    print("="*80)
    compare_daytrading_vs_swing('USDJPY=X')
    
    # Test 3: All pairs with 1-hour day trading
    print("\n" + "="*80)
    print("TEST 3: ALL PAIRS DAY TRADING (1-hour)")
    print("="*80)
    test_multiple_pairs_daytrading()
