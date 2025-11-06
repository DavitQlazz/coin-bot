#!/usr/bin/env python3
"""
High Win-Rate Scalping Bot - Version 3
Focus: Quality over quantity with strict filters and ATR-based risk management

Improvements:
1. ATR-based dynamic stop loss (adapts to volatility)
2. Trend confirmation with multiple EMAs
3. Volatility regime filter (avoid choppy markets)
4. Support/resistance confluence
5. Risk-reward minimum 1:2
6. Momentum confirmation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

class HighWinRateScalpingBot:
    """Quality-focused scalping with ATR stops and calibrated filters"""
    
    def __init__(self, pair='EURUSD=X', atr_sl_multiplier=1.5, atr_tp_multiplier=3.0,
                 max_hold_minutes=45, min_win_rate_target=55,
                 base_required_score=4, relax_bars=200):
        """
        Args:
            pair: Forex pair to trade
            atr_sl_multiplier: ATR multiplier for stop loss (default 1.5)
            atr_tp_multiplier: ATR multiplier for take profit (default 3.0)
            max_hold_minutes: Max time to hold position
            min_win_rate_target: Target win rate %
        """
        self.pair = pair
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.max_hold_minutes = max_hold_minutes
        self.min_win_rate_target = min_win_rate_target
        self.initial_capital = 1000
        self.capital = self.initial_capital
        # Scoring/relaxation controls
        self.base_required_score = base_required_score
        self.relax_bars = relax_bars
        
        print(f"✅ High Win-Rate Scalping Bot initialized")
        print(f"   Pair: {self.pair}")
        print(f"   ATR SL: {self.atr_sl_multiplier}x ATR")
        print(f"   ATR TP: {self.atr_tp_multiplier}x ATR")
        print(f"   Risk-Reward: 1:{self.atr_tp_multiplier/self.atr_sl_multiplier:.1f}")
        print(f"   Max Hold Time: {self.max_hold_minutes} minutes")
        print(f"   Target Win Rate: {self.min_win_rate_target}%")
    print(f"   Strategy: ATR stops + graded scoring + VWAP confluence + early exits")
    
    def fetch_data(self, period='7d', interval='5m'):
        """Fetch forex data from Yahoo Finance"""
        print(f"\n📊 Fetching {interval} data for {self.pair}...")
        try:
            ticker = yf.Ticker(self.pair)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for {self.pair}")
            
            # Clean data
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            
            print(f"✅ Fetched {len(df)} candles")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            raise
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators with focus on quality signals"""
        
        # ATR (14) - for dynamic stops
        atr_indicator = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['ATR'] = atr_indicator.average_true_range()
        df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
        
        # Bollinger Bands (20, 2)
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # RSI (14)
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # Stochastic (14, 3, 3)
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                                  window=14, smooth_window=3)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Multiple EMAs for trend confirmation
        df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
        
        # Trend strength
        df['Trend_Aligned'] = (
            (df['EMA_9'] > df['EMA_21']) & 
            (df['EMA_21'] > df['EMA_50']) & 
            (df['EMA_50'] > df['EMA_200'])
        ).astype(int) - (
            (df['EMA_9'] < df['EMA_21']) & 
            (df['EMA_21'] < df['EMA_50']) & 
            (df['EMA_50'] < df['EMA_200'])
        ).astype(int)
        
        # MACD for momentum confirmation
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # ADX for trend strength
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        
        # Volume analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # VWAP (rolling window to approximate session; 20-period by default)
        try:
            vwap_ind = ta.volume.VolumeWeightedAveragePrice(
                high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20
            )
            df['VWAP'] = vwap_ind.volume_weighted_average_price()
        except Exception:
            # Fallback manual VWAP if ta API differs
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
            df['VWAP'] = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        df['VWAP_Dist_Pct'] = ((df['Close'] - df['VWAP']) / df['Close']).abs() * 100
        
        # Volatility regime (using ATR percentile)
        df['ATR_Percentile'] = df['ATR_Pct'].rolling(window=100).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
        )
        
        return df
    
    def generate_signal(self, df, index, relaxed=False):
        """
                Generate high-quality scalping signals with graded scoring.
        
                Scoring-based entry to avoid over-filtering:
                - Core features scored (BB extreme weighted 2x, plus RSI, Stoch, EMA bias, mean reversion vs EMA50,
                    MACD slope, ADX, Volume). Normalized thresholds with relaxed option.
                - Volatility regime and ADX sanity checks retained but slightly loosened.
        """
        row = df.iloc[index]
        
        # Check data validity
        if pd.isna(row['ATR']) or pd.isna(row['RSI']) or pd.isna(row['ADX']):
            return 'hold', 0, 0
        
        # Volatility filter - avoid extreme volatility (slightly wider band)
        if row['ATR_Percentile'] > 90 or row['ATR_Percentile'] < 10:
            return 'hold', 0, 0
        
        # Trend strength sanity: avoid very low or very high ADX
        adx_thr = 15 if not relaxed else 12
        adx_max = 30 if not relaxed else 35   # Lower ADX cap for tighter mean-reversion window
        if row['ADX'] < adx_thr or row['ADX'] > adx_max:
            return 'hold', 0, 0
        
        # Calculate ATR-based stops
        atr_value = row['ATR']
        sl_distance = atr_value * self.atr_sl_multiplier
        tp_distance = atr_value * self.atr_tp_multiplier
        
        sl_pct = (sl_distance / row['Close']) * 100
        tp_pct = (tp_distance / row['Close']) * 100
        
        # Thresholds (relaxed or base) - tuned for better TP hit rate
        bb_long_thr = 0.20 if not relaxed else 0.30   # Tighter BB for stronger MR signals
        bb_short_thr = 0.80 if not relaxed else 0.70
        rsi_long_thr = 35 if not relaxed else 40
        rsi_short_thr = 65 if not relaxed else 60
        stoch_long_thr = 30 if not relaxed else 40
        stoch_short_thr = 70 if not relaxed else 60
        vol_ratio_thr = 0.8 if not relaxed else 0.7
        vwap_dist_min = 0.01 if not relaxed else 0.005  # 1bp min distance to avoid noise flips

        prev = df.iloc[index-1] if index > 0 else row

        # LONG scoring
        bb_long = 1 if row['BB_Position'] <= bb_long_thr else 0
        rsi_long = 1 if row['RSI'] <= rsi_long_thr else 0
        stoch_long = 1 if row['Stoch_K'] <= stoch_long_thr else 0
        ema_bias_long = 1 if row['EMA_9'] > row['EMA_21'] else 0
        mean_rev_long = 1 if row['Close'] < row['EMA_50'] else 0
        macd_up = 1 if row['MACD_Hist'] >= prev['MACD_Hist'] else 0
        adx_ok = 1 if row['ADX'] >= adx_thr else 0
        vol_ok = 1 if row['Volume_Ratio'] >= vol_ratio_thr else 0
        vwap_long = 1 if (row['Close'] <= row['VWAP'] and row['VWAP_Dist_Pct'] >= vwap_dist_min) else 0
        long_score = (2*bb_long + rsi_long + stoch_long + ema_bias_long +
                      mean_rev_long + macd_up + adx_ok + vol_ok + vwap_long)

        # SHORT scoring
        bb_short = 1 if row['BB_Position'] >= bb_short_thr else 0
        rsi_short = 1 if row['RSI'] >= rsi_short_thr else 0
        stoch_short = 1 if row['Stoch_K'] >= stoch_short_thr else 0
        ema_bias_short = 1 if row['EMA_9'] < row['EMA_21'] else 0
        mean_rev_short = 1 if row['Close'] > row['EMA_50'] else 0
        macd_down = 1 if row['MACD_Hist'] <= prev['MACD_Hist'] else 0
        adx_ok_s = 1 if row['ADX'] >= adx_thr else 0
        vol_ok_s = 1 if row['Volume_Ratio'] >= vol_ratio_thr else 0
        vwap_short = 1 if (row['Close'] >= row['VWAP'] and row['VWAP_Dist_Pct'] >= vwap_dist_min) else 0
        short_score = (2*bb_short + rsi_short + stoch_short + ema_bias_short +
                       mean_rev_short + macd_down + adx_ok_s + vol_ok_s + vwap_short)

        required = self.base_required_score - (1 if relaxed else 0)

        if long_score >= required:
            return 'buy', sl_pct, tp_pct
        if short_score >= required:
            return 'sell', sl_pct, tp_pct
        return 'hold', 0, 0
    
    def backtest(self, period='7d', interval='5m'):
        """Backtest the high win-rate strategy"""
        
        print(f"\n{'='*70}")
        print(f"HIGH WIN-RATE SCALPING BACKTEST: {self.pair}")
        print(f"Period: {period} | Interval: {interval}")
        print(f"ATR-based stops: {self.atr_sl_multiplier}x SL, {self.atr_tp_multiplier}x TP")
        print(f"Max Hold: {self.max_hold_minutes} minutes")
        print(f"{'='*70}\n")
        
        # Fetch and prepare data
        df = self.fetch_data(period, interval)
        df = self.calculate_indicators(df)
        
        # Initialize tracking
        self.capital = self.initial_capital
        trades = []
        position = None
        last_trade_index = None
        cooldown_until = {'LONG': -1, 'SHORT': -1}
        
        # Backtest loop
        for i in range(100, len(df)):  # Start after indicators are valid
            current_time = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Check existing position
            if position:
                entry_time, entry_price, direction, sl_price, tp_price = position
                time_held = (current_time - entry_time).total_seconds() / 60
                atr_now = df['ATR'].iloc[i]
                macd_hist_now = df['MACD_Hist'].iloc[i]
                macd_hist_prev = df['MACD_Hist'].iloc[i-1]
                rsi_now = df['RSI'].iloc[i]
                vwap_now = df['VWAP'].iloc[i]
                
                # Calculate PnL
                if direction == 'LONG':
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    
                    # Check exits
                    # Breakeven after +1 ATR move
                    if current_price - entry_price >= atr_now and current_price > sl_price:
                        sl_price = max(sl_price, entry_price)
                    # VWAP touch early exit (reversion hit)
                    if entry_price < vwap_now <= current_price:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'VWAP', 'hold_minutes': time_held
                        })
                        print(f"🔶 VWAP EXIT (LONG): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                    # Momentum fade early exit
                    if macd_hist_now < macd_hist_prev and rsi_now >= 55 and time_held >= 5:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'EARLY', 'hold_minutes': time_held
                        })
                        print(f"⚠️ EARLY EXIT (LONG): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                    elif current_price <= sl_price:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'SL', 'hold_minutes': time_held
                        })
                        print(f"❌ STOP LOSS (LONG): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                        cooldown_until['LONG'] = i + 10
                    elif current_price >= tp_price:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TP', 'hold_minutes': time_held
                        })
                        print(f"✅ TAKE PROFIT (LONG): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                    elif time_held >= self.max_hold_minutes:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TIME', 'hold_minutes': time_held
                        })
                        print(f"⏰ TIME EXIT (LONG): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                
                else:  # SHORT
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    # Check exits
                    # Breakeven after +1 ATR move (in favor)
                    if entry_price - current_price >= atr_now and current_price < sl_price:
                        sl_price = min(sl_price, entry_price)
                    # VWAP touch early exit (reversion hit)
                    if entry_price > vwap_now >= current_price:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'VWAP', 'hold_minutes': time_held
                        })
                        print(f"🔶 VWAP EXIT (SHORT): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                    # Momentum fade early exit
                    if macd_hist_now > macd_hist_prev and rsi_now <= 45 and time_held >= 5:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'EARLY', 'hold_minutes': time_held
                        })
                        print(f"⚠️ EARLY EXIT (SHORT): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                    elif current_price >= sl_price:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'SL', 'hold_minutes': time_held
                        })
                        print(f"❌ STOP LOSS (SHORT): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                        cooldown_until['SHORT'] = i + 10
                    elif current_price <= tp_price:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TP', 'hold_minutes': time_held
                        })
                        print(f"✅ TAKE PROFIT (SHORT): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
                    elif time_held >= self.max_hold_minutes:
                        pnl = self.capital * (pnl_pct / 100)
                        self.capital += pnl
                        trades.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': current_price, 'pnl_pct': pnl_pct,
                            'exit_reason': 'TIME', 'hold_minutes': time_held
                        })
                        print(f"⏰ TIME EXIT (SHORT): {current_price:.5f} | PnL: {pnl_pct:+.2f}% | Held: {time_held:.0f}m | Capital: ${self.capital:.2f}")
                        position = None
            
            # Look for new entry if no position
            if not position:
                # Relax if no trade for a while
                relaxed = False
                if last_trade_index is not None and (i - last_trade_index) >= self.relax_bars:
                    relaxed = True
                signal, sl_pct, tp_pct = self.generate_signal(df, i, relaxed=relaxed)
                
                if signal == 'buy' and i >= cooldown_until['LONG']:
                    sl_price = current_price * (1 - sl_pct / 100)
                    tp_price = current_price * (1 + tp_pct / 100)
                    position = (current_time, current_price, 'LONG', sl_price, tp_price)
                    print(f"🟢 ENTER LONG: {current_price:.5f} | SL: {sl_pct:.2f}% | TP: {tp_pct:.2f}% | RR: 1:{tp_pct/sl_pct:.1f}")
                    last_trade_index = i
                
                elif signal == 'sell' and i >= cooldown_until['SHORT']:
                    sl_price = current_price * (1 + sl_pct / 100)
                    tp_price = current_price * (1 - tp_pct / 100)
                    position = (current_time, current_price, 'SHORT', sl_price, tp_price)
                    print(f"🔴 ENTER SHORT: {current_price:.5f} | SL: {sl_pct:.2f}% | TP: {tp_pct:.2f}% | RR: 1:{tp_pct/sl_pct:.1f}")
                    last_trade_index = i
        
        # Calculate metrics
        total_trades = len(trades)
        if total_trades == 0:
            print("\n⚠️  No trades generated")
            return {
                'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
                'roi': 0, 'profit_factor': 0, 'trades_per_day': 0,
                'final_capital': self.initial_capital
            }
        
        wins = sum(1 for t in trades if t['pnl_pct'] > 0)
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100
        
        winning_pnl = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
        losing_pnl = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
        
        avg_hold_time = sum(t['hold_minutes'] for t in trades) / total_trades
        
        days = (df.index[-1] - df.index[0]).total_seconds() / 86400
        trades_per_day = total_trades / days if days > 0 else 0
        
        # Exit reason breakdown
        tp_exits = sum(1 for t in trades if t['exit_reason'] == 'TP')
        sl_exits = sum(1 for t in trades if t['exit_reason'] == 'SL')
        time_exits = sum(1 for t in trades if t['exit_reason'] == 'TIME')
        early_exits = sum(1 for t in trades if t['exit_reason'] == 'EARLY')
        vwap_exits = sum(1 for t in trades if t['exit_reason'] == 'VWAP')
        
        # Print results
        print(f"\n{'='*70}")
        print(f"HIGH WIN-RATE SCALPING RESULTS: {self.pair}")
        print(f"{'='*70}")
        print(f"Total Trades:     {total_trades}")
        print(f"Wins:             {wins} ({win_rate:.1f}%)")
        print(f"Losses:           {losses}")
        print(f"Total Return:     ${self.capital - self.initial_capital:+.2f} ({((self.capital/self.initial_capital - 1) * 100):+.2f}%)")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Avg Hold Time:    {avg_hold_time:.1f} minutes")
        print(f"Trades/Day:       {trades_per_day:.1f}")
        print(f"Final Capital:    ${self.capital:.2f}")
        print(f"\nExit Breakdown:")
        print(f"  Take Profit:    {tp_exits} ({tp_exits/total_trades*100:.1f}%)")
        print(f"  Stop Loss:      {sl_exits} ({sl_exits/total_trades*100:.1f}%)")
        print(f"  Time Exit:      {time_exits} ({time_exits/total_trades*100:.1f}%)")
        print(f"  Early Exit:     {early_exits} ({early_exits/total_trades*100:.1f}%)")
        print(f"  VWAP Exit:      {vwap_exits} ({vwap_exits/total_trades*100:.1f}%)")
        print(f"{'='*70}\n")
        
        # Check targets
        target_met = "✅" if win_rate >= self.min_win_rate_target else "❌"
        print(f"🎯 Target Win Rate: {self.min_win_rate_target}%")
        print(f"📊 Achieved: {win_rate:.1f}% {target_met}")
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'roi': ((self.capital / self.initial_capital - 1) * 100),
            'profit_factor': profit_factor,
            'trades_per_day': trades_per_day,
            'final_capital': self.capital,
            'avg_hold_time': avg_hold_time,
            'tp_exits': tp_exits,
            'sl_exits': sl_exits,
            'time_exits': time_exits,
            'early_exits': early_exits,
            'vwap_exits': vwap_exits
        }


if __name__ == "__main__":
    print("Testing High Win-Rate Scalping Strategy...")
    print()

    tests = [
        ("EURUSD=X", "15m", 60),   # 1h hold for 15m
        ("GBPUSD=X", "15m", 60),
        ("USDJPY=X", "15m", 60),
        ("EURUSD=X", "30m", 90),   # 1.5h hold for 30m
        ("GBPUSD=X", "30m", 90),
        ("USDJPY=X", "30m", 90),
        ("EURUSD=X", "1h", 180),   # 3h hold for 1h
        ("GBPUSD=X", "1h", 180),
        ("USDJPY=X", "1h", 180),
    ]

    for pair, interval, hold_time in tests:
        print(f"\n>>> Running {pair} on {interval}...")
        bot = HighWinRateScalpingBot(
            pair=pair,
            atr_sl_multiplier=1.5,      # 1.5x ATR for stop loss
            atr_tp_multiplier=3.0,      # 3.0x ATR for take profit (1:2 RR)
            max_hold_minutes=hold_time, # Scaled hold time for timeframe
            min_win_rate_target=55,     # Target 55%+ win rate
            base_required_score=6,      # More selective scoring threshold; relax by 1 if no trades
            relax_bars=200              # If no trades for 200 bars, relax one notch
        )
        _ = bot.backtest(period='7d', interval=interval)
