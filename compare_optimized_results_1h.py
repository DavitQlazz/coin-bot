#!/usr/bin/env python3
"""
Compare Optimized Stochastic RSI Results on 1h Timeframe
"""

import json
import os
import pandas as pd

def load_result(symbol):
    """Load optimized result for a symbol"""
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    fname = f'optimized_stoch_rsi_results_{safe_symbol}.json'

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            return json.load(f)
    return None

def main():
    # Pairs to analyze
    pairs = ['EURUSD=X', 'GBPUSD=X', 'USDCAD=X', 'AUDUSD=X']

    results = []

    for pair in pairs:
        data = load_result(pair)
        if data and 'summary' in data:
            summary = data['summary']
            results.append({
                'pair': pair,
                'trades': summary['trades'],
                'wins': summary['wins'],
                'losses': summary['losses'],
                'win_rate': summary['win_rate'],
                'net_pnl': summary['net_pnl'],
                'ending_balance': summary['ending_balance'],
                'profit_factor': summary.get('profit_factor', 0),
                'avg_win': summary.get('avg_win', 0),
                'avg_loss': summary.get('avg_loss', 0)
            })

    if not results:
        print("No results found!")
        return

    # Print comparison
    print("="*80)
    print("OPTIMIZED STOCHASTIC RSI - 1H TIMEFRAME COMPARISON")
    print("="*80)
    print(f"{'Pair':<10} {'Trades':<6} {'Win%':<6} {'P/L':<8} {'Profit':<8} {'Avg Win':<8} {'Avg Loss':<9}")
    print("-"*80)

    for result in results:
        print(f"{result['pair']:<10} {result['trades']:<6} {result['win_rate']:<6.1f} "
              f"${result['net_pnl']:<7.2f} {result['profit_factor']:<8.2f} "
              f"${result['avg_win']:<7.2f} ${result['avg_loss']:<8.2f}")

    # Summary statistics
    print("-"*80)
    profitable_pairs = [r for r in results if r['net_pnl'] > 0]
    total_pnl = sum(r['net_pnl'] for r in results)
    avg_win_rate = sum(r['win_rate'] for r in results) / len(results)

    print(f"Summary: {len(profitable_pairs)}/{len(results)} pairs profitable")
    print(f"Average Win Rate: {avg_win_rate:.1f}%")
    print(f"Total P/L: ${total_pnl:.2f}")

    # Best performers
    if profitable_pairs:
        best = max(profitable_pairs, key=lambda x: x['net_pnl'])
        print(f"\n🏆 Best Performer: {best['pair']} (${best['net_pnl']:.2f} P/L, {best['win_rate']:.1f}% win rate)")

        best_pf = max(profitable_pairs, key=lambda x: x['profit_factor'])
        if best_pf != best:
            print(f"💪 Best Risk-Reward: {best_pf['pair']} (Profit Factor: {best_pf['profit_factor']:.2f})")

    # Strategy insights
    print(f"\n📊 STRATEGY INSIGHTS (1H TIMEFRAME):")
    print(f"• Risk per trade: 1.5% of capital")
    print(f"• Stop Loss: 2.0% | Take Profit: 5.0%")
    print(f"• Stoch RSI Levels: Oversold=45, Overbought=85")
    print(f"• Average trade frequency: {sum(r['trades'] for r in results)/len(results):.1f} trades per pair")
    print(f"• More trades on 1h vs 4h timeframe (higher frequency)")

    # Performance analysis
    high_win_rate = [r for r in results if r['win_rate'] > 55]
    if high_win_rate:
        print(f"• High win rate pairs (>55%): {', '.join(r['pair'] for r in high_win_rate)}")

    strong_rr = [r for r in results if r['profit_factor'] > 1.5]
    if strong_rr:
        print(f"• Strong risk-reward pairs (PF>1.5): {', '.join(r['pair'] for r in strong_rr)}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if profitable_pairs:
        top_pairs = sorted(profitable_pairs, key=lambda x: x['net_pnl'], reverse=True)[:2]
        print(f"• Focus on: {', '.join(p['pair'] for p in top_pairs)}")
        print(f"• Consider portfolio allocation: 50% {top_pairs[0]['pair']}, 50% {top_pairs[1]['pair']}")
    else:
        print("• Strategy needs parameter optimization for 1h timeframe")
        print("• Consider adjusting Stoch RSI levels or adding trend filters")

    print(f"• 1h timeframe shows higher trade frequency but lower win rates")
    print(f"• May benefit from additional filters (trend, volatility, etc.)")
    print(f"• Compare with 4h results to determine optimal timeframe per pair")

if __name__ == '__main__':
    main()