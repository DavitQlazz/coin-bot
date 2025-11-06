#!/usr/bin/env python3
"""
Strategy Comparison: Analyzing All Tested Approaches
"""

import json
import pandas as pd

def load_strategy_results():
    """Load results from all tested strategies"""
    strategies = {}

    try:
        with open('top_pairs_focused_results.json', 'r') as f:
            strategies['Top Pairs Focused (Stoch RSI + ADX)'] = json.load(f)
    except FileNotFoundError:
        pass

    try:
        with open('macd_rsi_strategy_results.json', 'r') as f:
            strategies['MACD + RSI'] = json.load(f)
    except FileNotFoundError:
        pass

    try:
        with open('bb_mean_reversion_results.json', 'r') as f:
            strategies['Bollinger Bands Mean Reversion'] = json.load(f)
    except FileNotFoundError:
        pass

    try:
        with open('ma_crossover_results.json', 'r') as f:
            strategies['MA Crossover'] = json.load(f)
    except FileNotFoundError:
        pass

    return strategies

def create_comparison_table(strategies):
    """Create a comparison table of all strategies"""
    comparison_data = []

    for strategy_name, results in strategies.items():
        overall = results['overall_results']
        comparison_data.append({
            'Strategy': strategy_name,
            'Trades': overall['total_trades'],
            'Win Rate (%)': round(overall['win_rate'], 1),
            'Profit Factor': round(overall['profit_factor'], 2),
            'Net P/L ($)': round(overall['net_pnl'], 2),
            'Avg Win ($)': round(overall['avg_win'], 2),
            'Avg Loss ($)': round(overall['avg_loss'], 2)
        })

    return pd.DataFrame(comparison_data)

def print_strategy_comparison():
    """Print comprehensive strategy comparison"""
    print("🎯 STRATEGY COMPARISON: Alternative Approaches Tested")
    print("=" * 90)

    strategies = load_strategy_results()

    if not strategies:
        print("❌ No strategy results found. Please run the strategies first.")
        return

    # Create comparison table
    df = create_comparison_table(strategies)

    # Sort by Profit Factor (primary metric)
    df = df.sort_values('Profit Factor', ascending=False)

    print("\n📊 PERFORMANCE COMPARISON (Sorted by Profit Factor):")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    # Find best strategy
    best_strategy = df.iloc[0]

    print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy['Strategy']}")
    print(f"   ✅ {best_strategy['Trades']} trades, {best_strategy['Win Rate (%)']}% win rate")
    print(f"   ✅ Profit Factor: {best_strategy['Profit Factor']}")
    print(f"   ✅ Net P/L: ${best_strategy['Net P/L ($)']}")

    # Strategy insights
    print("\n📋 STRATEGY INSIGHTS:")
    print("=" * 90)

    for _, row in df.iterrows():
        strategy_name = row['Strategy']
        trades = row['Trades']
        win_rate = row['Win Rate (%)']
        pf = row['Profit Factor']
        pnl = row['Net P/L ($)']

        if pf > 1.4 and win_rate > 45:
            print(f"🎉 EXCELLENT: {strategy_name} - High profit factor and win rate")
        elif pf > 1.2 and win_rate > 40:
            print(f"✅ GOOD: {strategy_name} - Solid performance")
        elif pf > 1.0:
            print(f"📊 DECENT: {strategy_name} - Profitable but could be improved")
        else:
            print(f"❌ POOR: {strategy_name} - Not recommended")

    print("\n🎯 RECOMMENDATIONS:")
    print("=" * 90)
    print("1. 🔵 TREND-FOLLOWING: Stoch RSI + ADX and BB Mean Reversion performed best")
    print("2. 🔴 MOMENTUM: MACD + RSI was decent but not exceptional")
    print("3. ❌ CLASSIC MA: Simple MA crossover underperformed significantly")
    print("4. 💡 CONCLUSION: Hybrid approaches with multiple confirmations work best")

    print("\n📈 NEXT STEPS:")
    print("=" * 90)
    print("• Focus on Top Pairs Focused strategy (Stoch RSI + ADX)")
    print("• Consider BB Mean Reversion as alternative approach")
    print("• Avoid simple MA crossover strategies")
    print("• Test on longer timeframes or different market conditions")

if __name__ == "__main__":
    print_strategy_comparison()