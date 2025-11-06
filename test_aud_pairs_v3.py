#!/usr/bin/env python3
"""
Tuned High-Win-Rate Scalping Bot - AUD Pair Testing
Optimized parameters for mean-reversion on 15m timeframe
"""

from bot_forex_scalping_v3_highwr import HighWinRateScalpingBot
import json
from datetime import datetime

def test_aud_pairs():
    """Test tuned v3 bot across 10 AUD pairs on 15m timeframe"""
    
    # AUD pairs to test
    aud_pairs = [
        'AUDUSD=X',
        'AUDNZD=X',
        'AUDCAD=X',
        'AUDJPY=X',
        'AUDCHF=X',
        'AUDSGD=X',
        'AUDHKD=X',
        'AUDGBP=X',
        'AUDEUR=X',
        'AUDZWD=X',
    ]
    
    # Tuned parameters
    test_config = {
        'atr_sl_multiplier': 1.4,       # Slightly tighter SL for mean reversion
        'atr_tp_multiplier': 2.4,       # 1:1.6 RR (vs 1:2) to hit TP more often
        'max_hold_minutes': 60,         # 1h for 15m candles
        'min_win_rate_target': 55,
        'base_required_score': 5,       # Slightly relaxed from 6 for frequency
        'relax_bars': 200,
        'period': '7d',
        'interval': '15m'
    }
    
    results = []
    
    print("="*80)
    print("TUNED HIGH-WIN-RATE SCALPING - AUD PAIR TEST")
    print("="*80)
    print(f"Config: SL={test_config['atr_sl_multiplier']}x, TP={test_config['atr_tp_multiplier']}x, " +
          f"RR=1:{test_config['atr_tp_multiplier']/test_config['atr_sl_multiplier']:.2f}")
    print(f"Timeframe: {test_config['interval']}, Hold: {test_config['max_hold_minutes']}m\n")
    
    for i, pair in enumerate(aud_pairs, 1):
        print(f"\n[{i}/{len(aud_pairs)}] Testing {pair}...")
        try:
            bot = HighWinRateScalpingBot(
                pair=pair,
                atr_sl_multiplier=test_config['atr_sl_multiplier'],
                atr_tp_multiplier=test_config['atr_tp_multiplier'],
                max_hold_minutes=test_config['max_hold_minutes'],
                min_win_rate_target=test_config['min_win_rate_target'],
                base_required_score=test_config['base_required_score'],
                relax_bars=test_config['relax_bars'],
            )
            
            result = bot.backtest(
                period=test_config['period'],
                interval=test_config['interval']
            )
            
            # Enrich result with pair info
            result['pair'] = pair
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'pair': pair,
                'error': str(e),
                'total_trades': 0,
                'win_rate': 0,
                'roi': 0
            })
    
    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY - AUD PAIRS RESULTS")
    print(f"{'='*80}")
    print(f"{'Pair':<12} {'Trades':>7} {'Win%':>7} {'ROI%':>8} {'TPD':>6} {'PF':>6} {'Status':>8}")
    print("-" * 80)
    
    passing = []
    for r in results:
        if r.get('total_trades', 0) > 0:
            status = "✅ PASS" if r['win_rate'] >= test_config['min_win_rate_target'] else "❌ FAIL"
            pf = f"{r.get('profit_factor', 0):.2f}"
            print(f"{r['pair']:<12} {r['total_trades']:>7} {r['win_rate']:>7.1f} "
                  f"{r['roi']:>8.2f} {r['trades_per_day']:>6.1f} {pf:>6} {status:>8}")
            if r['win_rate'] >= test_config['min_win_rate_target']:
                passing.append(r)
        else:
            print(f"{r['pair']:<12} {'ERROR':>7}")
    
    # Summary stats
    print("-" * 80)
    total_trades = sum(r.get('total_trades', 0) for r in results)
    avg_wr = sum(r.get('win_rate', 0) for r in results if r.get('total_trades', 0) > 0) / len([r for r in results if r.get('total_trades', 0) > 0]) if any(r.get('total_trades', 0) for r in results) else 0
    avg_roi = sum(r.get('roi', 0) for r in results if r.get('total_trades', 0) > 0) / len([r for r in results if r.get('total_trades', 0) > 0]) if any(r.get('total_trades', 0) for r in results) else 0
    
    print(f"\n📊 Portfolio Stats:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Avg Win Rate: {avg_wr:.1f}%")
    print(f"  Avg ROI: {avg_roi:.2f}%")
    print(f"  Pairs Passing (≥{test_config['min_win_rate_target']}%): {len(passing)}/{len(aud_pairs)}")
    
    if passing:
        print(f"\n✅ Best Performers:")
        for r in sorted(passing, key=lambda x: x['win_rate'], reverse=True)[:5]:
            print(f"  {r['pair']:<12} WR={r['win_rate']:.1f}% ROI={r['roi']:+.2f}% Trades/Day={r['trades_per_day']:.1f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'aud_pairs_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump({
            'config': test_config,
            'timestamp': timestamp,
            'results': results,
            'summary': {
                'total_trades': total_trades,
                'avg_win_rate': avg_wr,
                'avg_roi': avg_roi,
                'passing_pairs': len(passing),
                'total_pairs': len(aud_pairs)
            }
        }, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to {filename}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_aud_pairs()
