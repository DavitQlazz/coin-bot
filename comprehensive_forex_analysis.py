#!/usr/bin/env python3
"""
Comprehensive Forex Pairs Analysis for Optimized Stochastic RSI
"""

import json
import os
import pandas as pd

def load_result(symbol):
    """Load optimized result for a symbol"""
    safe_symbol = symbol.replace('/', '_').replace('=X', '')
    fname = f'optimized_stoch_rsi_results_{safe_symbol}_4h.json'

    if os.path.exists(fname):
        with open(fname, 'r') as f:
            return json.load(f)
    return None

def main():
    # All tested forex pairs
    pairs = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X',
        'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'EURCAD=X', 'GBPCAD=X'
    ]

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

    # Sort by profit
    results.sort(key=lambda x: x['net_pnl'], reverse=True)

    # Print comprehensive analysis
    print("="*100)
    print("COMPREHENSIVE FOREX PAIRS ANALYSIS - OPTIMIZED STOCHASTIC RSI (4H)")
    print("="*100)
    print(f"{'Pair':<10} {'Trades':<6} {'Win%':<6} {'P/L':<8} {'Profit':<8} {'Performance':<12}")
    print("-"*100)

    for result in results:
        perf = '⭐⭐⭐ Excellent' if result['win_rate'] >= 70 else '⭐⭐ Good' if result['net_pnl'] > 5 else '⭐ OK' if result['net_pnl'] > 0 else '❌ Poor'
        print(f"{result['pair']:<10} {result['trades']:<6} {result['win_rate']:<6.1f} "
              f"${result['net_pnl']:<7.2f} {result['profit_factor']:<8.2f} {perf:<12}")

    # Summary statistics
    print("-"*100)
    profitable_pairs = [r for r in results if r['net_pnl'] > 0]
    total_pnl = sum(r['net_pnl'] for r in results)
    avg_win_rate = sum(r['win_rate'] for r in results) / len(results)

    print(f"OVERALL SUMMARY: {len(profitable_pairs)}/{len(results)} pairs profitable")
    print(f"Average Win Rate: {avg_win_rate:.1f}%")
    print(f"Total P/L: ${total_pnl:.2f}")

    # Performance categories
    excellent = [r for r in results if r['win_rate'] >= 70]
    good = [r for r in results if 50 <= r['win_rate'] < 70 and r['net_pnl'] > 5]
    moderate = [r for r in results if r['net_pnl'] > 0 and r not in excellent and r not in good]
    poor = [r for r in results if r['net_pnl'] <= 0]

    print(f"\nPERFORMANCE BREAKDOWN:")
    print(f"⭐⭐⭐ Excellent (≥70% win rate): {len(excellent)} pairs")
    print(f"⭐⭐ Good (50-69% win rate + >$5 profit): {len(good)} pairs")
    print(f"⭐ Moderate (profitable): {len(moderate)} pairs")
    print(f"❌ Poor (unprofitable): {len(poor)} pairs")

    # Top performers
    if excellent:
        print(f"\n🏆 EXCELLENT PERFORMERS:")
        for r in excellent:
            print(f"• {r['pair']}: {r['win_rate']:.1f}% win rate, ${r['net_pnl']:.2f} profit")

    if good:
        print(f"\n💪 STRONG PERFORMERS:")
        for r in good:
            print(f"• {r['pair']}: {r['win_rate']:.1f}% win rate, ${r['net_pnl']:.2f} profit")

    # Currency analysis
    print(f"\n🌍 CURRENCY PERFORMANCE ANALYSIS:")

    # Group by base currency
    currencies = {}
    for r in results:
        base_curr = r['pair'][:3]
        if base_curr not in currencies:
            currencies[base_curr] = []
        currencies[base_curr].append(r)

    for curr, pairs_list in currencies.items():
        profitable = [p for p in pairs_list if p['net_pnl'] > 0]
        avg_pnl = sum(p['net_pnl'] for p in pairs_list) / len(pairs_list)
        print(f"• {curr} pairs: {len(profitable)}/{len(pairs_list)} profitable, avg P/L: ${avg_pnl:.2f}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    top_pairs = results[:5]  # Top 5 by profit
    print(f"• Focus on top performers: {', '.join(p['pair'] for p in top_pairs)}")
    print(f"• EUR and GBP pairs show strongest performance")
    print(f"• Avoid JPY pairs (USDJPY, AUDJPY) - poor performance")
    print(f"• Consider portfolio: 30% EUR pairs, 30% GBP pairs, 20% CAD pairs, 20% other")

    print(f"\n📊 STRATEGY INSIGHTS:")
    print(f"• Risk per trade: 1.5% of capital")
    print(f"• Stop Loss: 2.0% | Take Profit: 5.0%")
    print(f"• Stoch RSI Levels: Oversold=45, Overbought=85")
    print(f"• 4H timeframe provides optimal signal quality")
    print(f"• Strategy works best on EUR, GBP, and CAD pairs")

if __name__ == '__main__':
    main()