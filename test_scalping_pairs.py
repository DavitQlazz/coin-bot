#!/usr/bin/env python3
"""
Test scalping strategy on multiple forex pairs
"""

from bot_forex_scalping import ForexScalpingBot
from datetime import datetime
import json

# Pairs to test
PAIRS_TO_TEST = [
    'EURUSD=X',
    'GBPUSD=X',
    'USDJPY=X',
    'USDCHF=X',
    'AUDUSD=X',
    'NZDUSD=X',
    'USDCAD=X',
    'EURJPY=X',
    'EURGBP=X',
    'GBPJPY=X'
]

def test_pair(pair):
    """Test a single pair"""
    print(f"\n{'='*70}")
    print(f"Testing: {pair}")
    print(f"{'='*70}")
    
    try:
        bot = ForexScalpingBot(
            pair=pair,
            stop_loss=0.4,
            take_profit=0.8,
            max_hold_minutes=30
        )
        
        results = bot.backtest(period='7d', interval='5m')
        
        return {
            'pair': pair,
                'trades': results['total_trades'],
                'wins': results['wins'],
                'losses': results['losses'],
            'win_rate': results['win_rate'],
                'roi': results['roi'],
            'profit_factor': results['profit_factor'],
            'trades_per_day': results['trades_per_day'],
            'final_capital': results['final_capital']
        }
    except Exception as e:
        print(f"❌ Error testing {pair}: {e}")
        return None

def main():
    print("="*70)
    print("SCALPING STRATEGY - MULTI-PAIR COMPARISON")
    print("="*70)
    print(f"Testing {len(PAIRS_TO_TEST)} forex pairs")
    print(f"Period: 7 days | Interval: 5 minutes")
    print(f"Stop Loss: 0.4% | Take Profit: 0.8%")
    print("="*70)
    
    all_results = []
    
    for pair in PAIRS_TO_TEST:
        result = test_pair(pair)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\n❌ No successful tests")
        return
    
    # Sort by ROI
    all_results.sort(key=lambda x: x['roi'], reverse=True)
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY - SORTED BY ROI")
    print("="*70)
    print(f"{'Pair':<12} {'Trades':<8} {'Win%':<8} {'ROI%':<10} {'Trades/Day':<12} {'PF':<8}")
    print("-"*70)
    
    for r in all_results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 900 else "∞"
        print(f"{r['pair']:<12} {r['trades']:<8} {r['win_rate']:<8.1f} {r['roi']:<+10.2f} {r['trades_per_day']:<12.1f} {pf_str:<8}")
    
    # Calculate averages
    avg_roi = sum(r['roi'] for r in all_results) / len(all_results)
    avg_wr = sum(r['win_rate'] for r in all_results) / len(all_results)
    avg_trades = sum(r['trades'] for r in all_results) / len(all_results)
    avg_tpd = sum(r['trades_per_day'] for r in all_results) / len(all_results)
    
    print("-"*70)
    print(f"{'AVERAGE':<12} {avg_trades:<8.0f} {avg_wr:<8.1f} {avg_roi:<+10.2f} {avg_tpd:<12.1f}")
    print("="*70)
    
    # Best performers
    print("\n🏆 TOP 3 PERFORMERS BY ROI:")
    for i, r in enumerate(all_results[:3], 1):
        print(f"{i}. {r['pair']}: {r['roi']:+.2f}% ROI, {r['win_rate']:.1f}% WR, {r['trades']} trades")
    
    # Most active
    most_active = sorted(all_results, key=lambda x: x['trades_per_day'], reverse=True)
    print("\n📊 TOP 3 MOST ACTIVE (Trades/Day):")
    for i, r in enumerate(most_active[:3], 1):
        print(f"{i}. {r['pair']}: {r['trades_per_day']:.1f} trades/day, {r['roi']:+.2f}% ROI")
    
    # Best win rate
    best_wr = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)
    print("\n🎯 TOP 3 BEST WIN RATES:")
    for i, r in enumerate(best_wr[:3], 1):
        print(f"{i}. {r['pair']}: {r['win_rate']:.1f}% WR, {r['roi']:+.2f}% ROI")
    
    # Frequency target check
    print("\n🎯 FREQUENCY TARGET (1+ trade/day):")
    meeting_target = [r for r in all_results if r['trades_per_day'] >= 1.0]
    print(f"✅ {len(meeting_target)}/{len(all_results)} pairs meet target")
    
    # Profitability check
    profitable = [r for r in all_results if r['roi'] > 0]
    print(f"\n💰 PROFITABILITY:")
    print(f"✅ {len(profitable)}/{len(all_results)} pairs profitable ({len(profitable)/len(all_results)*100:.0f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scalping_multi_pair_results_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'test_period_days': 7,
            'pairs_tested': len(all_results),
            'results': all_results,
            'summary': {
                'avg_roi': avg_roi,
                'avg_win_rate': avg_wr,
                'avg_trades': avg_trades,
                'avg_trades_per_day': avg_tpd,
                'pairs_meeting_frequency_target': len(meeting_target),
                'profitable_pairs': len(profitable)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    print("\n" + "="*70)
    print("MULTI-PAIR TEST COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
