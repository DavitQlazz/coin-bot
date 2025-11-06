#!/usr/bin/env python3
"""
50 Trading Strategy Variants
Comprehensive collection of forex trading strategies for systematic testing
"""

def print_strategy_variants():
    """Print 50 different trading strategy variants"""

    strategies = [
        # 1-10: Momentum-based strategies
        {
            "id": 1,
            "name": "RSI Divergence Momentum",
            "category": "Momentum",
            "description": "Enter long on bullish RSI divergence + MACD crossover, exit on RSI overbought",
            "indicators": ["RSI", "MACD", "Price"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["RSI divergence", "MACD signal line cross"],
            "exit_conditions": ["RSI > 70", "Profit target 2:1 RR"]
        },
        {
            "id": 2,
            "name": "Stochastic RSI Breakout",
            "category": "Momentum",
            "description": "Enter on Stochastic RSI oversold bounce + volume confirmation",
            "indicators": ["Stoch RSI", "Volume", "Support/Resistance"],
            "timeframes": ["15M", "1H"],
            "entry_conditions": ["Stoch RSI < 20", "Volume spike"],
            "exit_conditions": ["Stoch RSI > 80", "Trailing stop"]
        },
        {
            "id": 3,
            "name": "MACD Trend Following",
            "category": "Momentum",
            "description": "Follow MACD trend with ADX filter for strong trends only",
            "indicators": ["MACD", "ADX", "EMA"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["MACD histogram positive", "ADX > 25"],
            "exit_conditions": ["MACD bearish cross", "ADX < 20"]
        },
        {
            "id": 4,
            "name": "CCI Extreme Reversal",
            "category": "Momentum",
            "description": "Enter on CCI extreme readings with trend confirmation",
            "indicators": ["CCI", "Trend Direction", "ATR"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["CCI < -200", "Uptrend"],
            "exit_conditions": ["CCI > +200", "ATR-based stop"]
        },
        {
            "id": 5,
            "name": "Williams %R Bounce",
            "category": "Momentum",
            "description": "Buy on Williams %R oversold with momentum divergence",
            "indicators": ["Williams %R", "Momentum", "Bollinger Bands"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["%R < -80", "Price at lower BB"],
            "exit_conditions": ["%R > -20", "Middle BB target"]
        },
        {
            "id": 6,
            "name": "TSI Signal Line Cross",
            "category": "Momentum",
            "description": "True Strength Index signal line crossover with volume filter",
            "indicators": ["TSI", "Volume", "SMA"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["TSI signal cross up", "Above average volume"],
            "exit_conditions": ["TSI signal cross down", "Volume decrease"]
        },
        {
            "id": 7,
            "name": "ROC Acceleration",
            "category": "Momentum",
            "description": "Rate of Change acceleration with moving average filter",
            "indicators": ["ROC", "EMA", "Price"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["ROC > 5", "Price > EMA20"],
            "exit_conditions": ["ROC < -5", "Price < EMA20"]
        },
        {
            "id": 8,
            "name": "Ultimate Oscillator",
            "category": "Momentum",
            "description": "Ultimate Oscillator overbought/oversold with divergence",
            "indicators": ["Ultimate Oscillator", "Price", "Support"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["UO < 30", "Bullish divergence"],
            "exit_conditions": ["UO > 70", "Bearish divergence"]
        },
        {
            "id": 9,
            "name": "Awesome Oscillator",
            "category": "Momentum",
            "description": "Awesome Oscillator saucer pattern with trend filter",
            "indicators": ["AO", "Trend", "Fibonacci"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["AO saucer up", "Uptrend"],
            "exit_conditions": ["AO saucer down", "Fibonacci target"]
        },
        {
            "id": 10,
            "name": "Balance of Power",
            "category": "Momentum",
            "description": "Balance of Power extreme readings with volume confirmation",
            "indicators": ["BOP", "Volume", "VWAP"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["BOP < -0.5", "High volume"],
            "exit_conditions": ["BOP > 0.5", "VWAP touch"]
        },

        # 11-20: Mean Reversion strategies
        {
            "id": 11,
            "name": "Bollinger Band Squeeze",
            "category": "Mean Reversion",
            "description": "Enter on BB squeeze breakout with RSI filter",
            "indicators": ["BB", "RSI", "Volume"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["BB squeeze", "RSI neutral"],
            "exit_conditions": ["Price touches BB", "RSI extreme"]
        },
        {
            "id": 12,
            "name": "Z-Score Reversal",
            "category": "Mean Reversion",
            "description": "Z-Score based entries with mean reversion expectation",
            "indicators": ["Z-Score", "SMA", "ATR"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Z-Score < -2", "Price near SMA"],
            "exit_conditions": ["Z-Score > 2", "ATR stop"]
        },
        {
            "id": 13,
            "name": "Deviation from Mean",
            "category": "Mean Reversion",
            "description": "Price deviation from moving average with volatility filter",
            "indicators": ["Price", "SMA", "Volatility"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["Price 3SD below SMA", "Low volatility"],
            "exit_conditions": ["Price at SMA", "High volatility"]
        },
        {
            "id": 14,
            "name": "RSI Mean Reversion",
            "category": "Mean Reversion",
            "description": "RSI extreme levels with price action confirmation",
            "indicators": ["RSI", "Price Action", "Support"],
            "timeframes": ["15M", "1H"],
            "entry_conditions": ["RSI < 30", "Support level"],
            "exit_conditions": ["RSI > 70", "Resistance level"]
        },
        {
            "id": 15,
            "name": "Envelope Reversal",
            "category": "Mean Reversion",
            "description": "Price envelope touches with momentum filter",
            "indicators": ["Envelopes", "Momentum", "Volume"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["Lower envelope touch", "Bullish momentum"],
            "exit_conditions": ["Upper envelope touch", "Bearish momentum"]
        },
        {
            "id": 16,
            "name": "Donchian Channel",
            "category": "Mean Reversion",
            "description": "Donchian channel breakouts and reversals",
            "indicators": ["Donchian", "ADX", "Price"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Channel breakout", "ADX < 25"],
            "exit_conditions": ["Opposite channel edge", "ADX > 30"]
        },
        {
            "id": 17,
            "name": "Keltner Channel",
            "category": "Mean Reversion",
            "description": "Keltner channel mean reversion with ATR filter",
            "indicators": ["Keltner", "ATR", "Volume"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["Lower Keltner touch", "ATR expansion"],
            "exit_conditions": ["Middle Keltner", "ATR contraction"]
        },
        {
            "id": 18,
            "name": "Price Channel Reversal",
            "category": "Mean Reversion",
            "description": "Price channel boundaries with oscillator confirmation",
            "indicators": ["Price Channel", "Stochastic", "RSI"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["Lower channel", "Stochastic oversold"],
            "exit_conditions": ["Upper channel", "Stochastic overbought"]
        },
        {
            "id": 19,
            "name": "Moving Average Ribbon",
            "category": "Mean Reversion",
            "description": "MA ribbon squeeze and expansion patterns",
            "indicators": ["Multiple MAs", "Price", "Volume"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["MA convergence", "Volume spike"],
            "exit_conditions": ["MA divergence", "Volume decrease"]
        },
        {
            "id": 20,
            "name": "Ichimoku Cloud",
            "category": "Mean Reversion",
            "description": "Ichimoku cloud rejections and TK cross signals",
            "indicators": ["Ichimoku", "Price", "Senkou Span"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Cloud rejection", "TK cross up"],
            "exit_conditions": ["Cloud rejection opposite", "TK cross down"]
        },

        # 21-30: Trend Following strategies
        {
            "id": 21,
            "name": "ADX Trend Strength",
            "category": "Trend Following",
            "description": "ADX-based trend following with directional bias",
            "indicators": ["ADX", "DI+", "DI-"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["ADX > 25", "DI+ > DI-"],
            "exit_conditions": ["ADX < 20", "DI- > DI+"]
        },
        {
            "id": 22,
            "name": "Parabolic SAR Trend",
            "category": "Trend Following",
            "description": "Parabolic SAR trailing stops with trend filter",
            "indicators": ["Parabolic SAR", "Trend", "EMA"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["SAR below price", "Uptrend"],
            "exit_conditions": ["SAR above price", "SAR stop"]
        },
        {
            "id": 23,
            "name": "SuperTrend Breakout",
            "category": "Trend Following",
            "description": "SuperTrend trend following with volatility adjustment",
            "indicators": ["SuperTrend", "ATR", "Volume"],
            "timeframes": ["15M", "1H"],
            "entry_conditions": ["Price above SuperTrend", "High volume"],
            "exit_conditions": ["Price below SuperTrend", "Low volume"]
        },
        {
            "id": 24,
            "name": "EMA Crossover System",
            "category": "Trend Following",
            "description": "Multiple EMA crossover with trend strength filter",
            "indicators": ["EMA 9/21", "EMA 50", "ADX"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["EMA9 > EMA21", "ADX > 20"],
            "exit_conditions": ["EMA9 < EMA21", "ADX < 20"]
        },
        {
            "id": 25,
            "name": "VWAP Trend Rider",
            "category": "Trend Following",
            "description": "VWAP-based trend following with price action",
            "indicators": ["VWAP", "Price", "Volume"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["Price > VWAP", "Volume trend up"],
            "exit_conditions": ["Price < VWAP", "Volume trend down"]
        },
        {
            "id": 26,
            "name": "Heikin Ashi Trend",
            "category": "Trend Following",
            "description": "Heikin Ashi candlestick trend following",
            "indicators": ["Heikin Ashi", "Trend", "Support"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["HA bullish", "Higher highs"],
            "exit_conditions": ["HA bearish", "Lower lows"]
        },
        {
            "id": 27,
            "name": "Fractal Dimension",
            "category": "Trend Following",
            "description": "Fractal dimension for trend strength assessment",
            "indicators": ["Fractal Dimension", "Price", "Volume"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Low fractal dimension", "Strong trend"],
            "exit_conditions": ["High fractal dimension", "Weak trend"]
        },
        {
            "id": 28,
            "name": "Linear Regression",
            "category": "Trend Following",
            "description": "Linear regression slope and channel trading",
            "indicators": ["Linear Regression", "R-Squared", "Price"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["Positive slope", "High R-squared"],
            "exit_conditions": ["Negative slope", "Low R-squared"]
        },
        {
            "id": 29,
            "name": "Guppy MMA",
            "category": "Trend Following",
            "description": "Guppy Multiple Moving Average trend system",
            "indicators": ["Guppy MMA", "Price", "Volume"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Fast MAs > Slow MAs", "Separation"],
            "exit_conditions": ["Fast MAs < Slow MAs", "Compression"]
        },
        {
            "id": 30,
            "name": "Rainbow Oscillator",
            "category": "Trend Following",
            "description": "Rainbow oscillator trend strength and direction",
            "indicators": ["Rainbow", "Price", "Momentum"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["Rainbow up", "Strong momentum"],
            "exit_conditions": ["Rainbow down", "Weak momentum"]
        },

        # 31-40: Breakout strategies
        {
            "id": 31,
            "name": "Consolidation Breakout",
            "category": "Breakout",
            "description": "Identify consolidation patterns and trade breakouts",
            "indicators": ["Price", "Volume", "ATR"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Tight consolidation", "Volume increase"],
            "exit_conditions": ["Failed breakout", "ATR target"]
        },
        {
            "id": 32,
            "name": "Triangle Breakout",
            "category": "Breakout",
            "description": "Triangle pattern breakouts with volume confirmation",
            "indicators": ["Triangle", "Volume", "Price"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Triangle completion", "Volume spike"],
            "exit_conditions": ["False breakout", "Pattern height target"]
        },
        {
            "id": 33,
            "name": "Rectangle Breakout",
            "category": "Breakout",
            "description": "Rectangle pattern trading with range expansion",
            "indicators": ["Rectangle", "Volume", "Momentum"],
            "timeframes": ["2H", "4H"],
            "entry_conditions": ["Rectangle breakout", "Momentum up"],
            "exit_conditions": ["Rectangle retest", "Opposite side"]
        },
        {
            "id": 34,
            "name": "Wedge Breakout",
            "category": "Breakout",
            "description": "Wedge pattern breakouts with trend continuation",
            "indicators": ["Wedge", "Trend", "Volume"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Wedge breakout", "Trend direction"],
            "exit_conditions": ["Wedge retest", "Trend exhaustion"]
        },
        {
            "id": 35,
            "name": "Flag Pattern Breakout",
            "category": "Breakout",
            "description": "Flag and pennant pattern continuation trades",
            "indicators": ["Flag", "Volume", "Price"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["Flag breakout", "Volume confirmation"],
            "exit_conditions": ["Flag pole target", "Pattern failure"]
        },
        {
            "id": 36,
            "name": "Support/Resistance Break",
            "category": "Breakout",
            "description": "Key level breakouts with multiple timeframe confirmation",
            "indicators": ["S/R", "Volume", "Multi-timeframe"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["S/R break", "Higher TF alignment"],
            "exit_conditions": ["S/R retest", "Lower TF rejection"]
        },
        {
            "id": 37,
            "name": "Gap Fill Strategy",
            "category": "Breakout",
            "description": "Gap identification and fill probability trading",
            "indicators": ["Gap", "Volume", "Price"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Gap formation", "Volume analysis"],
            "exit_conditions": ["Gap fill", "Time decay"]
        },
        {
            "id": 38,
            "name": "Opening Range Breakout",
            "category": "Breakout",
            "description": "Opening range breakout with time-based entries",
            "indicators": ["Opening Range", "Time", "Volume"],
            "timeframes": ["15M", "1H"],
            "entry_conditions": ["ORB breakout", "Time filter"],
            "exit_conditions": ["ORB failure", "Time stop"]
        },
        {
            "id": 39,
            "name": "Volume Breakout",
            "category": "Breakout",
            "description": "Volume-based breakouts with price confirmation",
            "indicators": ["Volume", "Price", "VWAP"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["Volume spike", "Price breakout"],
            "exit_conditions": ["Volume decrease", "VWAP rejection"]
        },
        {
            "id": 40,
            "name": "News Event Breakout",
            "category": "Breakout",
            "description": "News event volatility breakouts with risk management",
            "indicators": ["News", "Volatility", "Price"],
            "timeframes": ["15M", "1H"],
            "entry_conditions": ["News trigger", "Volatility expansion"],
            "exit_conditions": ["News end", "Volatility contraction"]
        },

        # 41-50: Advanced/Combined strategies
        {
            "id": 41,
            "name": "Multi-Timeframe Momentum",
            "category": "Advanced",
            "description": "Multi-timeframe momentum alignment strategy",
            "indicators": ["RSI", "MACD", "Multi-TF"],
            "timeframes": ["15M", "1H", "4H"],
            "entry_conditions": ["All TF momentum up", "TF alignment"],
            "exit_conditions": ["TF divergence", "Momentum reversal"]
        },
        {
            "id": 42,
            "name": "Market Structure + Momentum",
            "category": "Advanced",
            "description": "Market structure analysis with momentum confirmation",
            "indicators": ["Market Structure", "Momentum", "Volume"],
            "timeframes": ["1H", "2H"],
            "entry_conditions": ["Higher high/low", "Momentum up"],
            "exit_conditions": ["Lower high/low", "Momentum down"]
        },
        {
            "id": 43,
            "name": "Order Flow + Volume",
            "category": "Advanced",
            "description": "Order flow analysis with volume profile",
            "indicators": ["Order Flow", "Volume Profile", "Price"],
            "timeframes": ["30M", "1H"],
            "entry_conditions": ["Order flow imbalance", "Volume cluster"],
            "exit_conditions": ["Order flow balance", "Volume decrease"]
        },
        {
            "id": 44,
            "name": "Intermarket Analysis",
            "category": "Advanced",
            "description": "Intermarket relationships and correlations",
            "indicators": ["Correlations", "Related Assets", "Price"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Correlation break", "Lead asset signal"],
            "exit_conditions": ["Correlation restore", "Divergence"]
        },
        {
            "id": 45,
            "name": "Sentiment + Technical",
            "category": "Advanced",
            "description": "Market sentiment combined with technical signals",
            "indicators": ["Sentiment", "Technical", "COT"],
            "timeframes": ["1H", "Daily"],
            "entry_conditions": ["Contrarian signal", "Technical confirmation"],
            "exit_conditions": ["Sentiment extreme", "Technical reversal"]
        },
        {
            "id": 46,
            "name": "Machine Learning Signals",
            "category": "Advanced",
            "description": "ML-based pattern recognition and prediction",
            "indicators": ["ML Model", "Features", "Probability"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["ML signal > 70%", "Risk filter"],
            "exit_conditions": ["ML signal < 30%", "Stop loss"]
        },
        {
            "id": 47,
            "name": "Harmonic Patterns",
            "category": "Advanced",
            "description": "Harmonic pattern recognition and trading",
            "indicators": ["Harmonic", "Fibonacci", "Price"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Pattern completion", "Fibonacci confluence"],
            "exit_conditions": ["Pattern invalidation", "Target levels"]
        },
        {
            "id": 48,
            "name": "Elliott Wave + Fibonacci",
            "category": "Advanced",
            "description": "Elliott wave theory with Fibonacci projections",
            "indicators": ["Elliott Wave", "Fibonacci", "Price"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Wave completion", "Fibonacci level"],
            "exit_conditions": ["Wave invalidation", "Next wave target"]
        },
        {
            "id": 49,
            "name": "Seasonal + Technical",
            "category": "Advanced",
            "description": "Seasonal patterns combined with technical analysis",
            "indicators": ["Seasonal", "Technical", "Calendar"],
            "timeframes": ["1H", "Daily"],
            "entry_conditions": ["Seasonal signal", "Technical alignment"],
            "exit_conditions": ["Seasonal end", "Technical reversal"]
        },
        {
            "id": 50,
            "name": "Risk Parity Momentum",
            "category": "Advanced",
            "description": "Risk parity approach with momentum allocation",
            "indicators": ["Risk Parity", "Momentum", "Volatility"],
            "timeframes": ["1H", "4H"],
            "entry_conditions": ["Risk allocation", "Momentum signal"],
            "exit_conditions": ["Risk rebalance", "Momentum reversal"]
        }
    ]

    print("🎯 50 TRADING STRATEGY VARIANTS")
    print("=" * 100)
    print(f"{'ID':<3} {'Category':<15} {'Strategy Name':<25} {'Timeframes':<12} {'Win Rate Est.':<12}")
    print("-" * 100)

    for strategy in strategies:
        category = strategy['category']
        name = strategy['name'][:24]  # Truncate long names
        timeframes = "/".join(strategy['timeframes'])
        # Estimate win rate based on category (rough estimates)
        win_rate_est = {
            "Momentum": "45-55%",
            "Mean Reversion": "50-60%",
            "Trend Following": "40-50%",
            "Breakout": "35-45%",
            "Advanced": "50-65%"
        }.get(category, "40-50%")

        print(f"{strategy['id']:<3} {category:<15} {name:<25} {timeframes:<12} {win_rate_est:<12}")

    print("\n" + "=" * 100)

    # Print detailed descriptions for first 10 strategies
    print("\n📋 DETAILED DESCRIPTIONS (First 10 Strategies):")
    print("=" * 100)

    for i, strategy in enumerate(strategies[:10], 1):
        print(f"\n{i}. {strategy['name']} ({strategy['category']})")
        print(f"   Description: {strategy['description']}")
        print(f"   Indicators: {', '.join(strategy['indicators'])}")
        print(f"   Timeframes: {', '.join(strategy['timeframes'])}")
        print(f"   Entry: {', '.join(strategy['entry_conditions'])}")
        print(f"   Exit: {', '.join(strategy['exit_conditions'])}")

    print("\n💡 IMPLEMENTATION NOTES:")
    print("=" * 100)
    print("• Start with strategies 1-10 (Momentum) as they're generally reliable")
    print("• Test on multiple pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD")
    print("• Use consistent risk management: 1-2% per trade, 2:1 RR minimum")
    print("• Backtest on 6-12 months of data before live trading")
    print("• Combine with our winning Top Pairs Focused approach for best results")
    print("• Monitor drawdown and adjust position sizing accordingly")

if __name__ == "__main__":
    print_strategy_variants()