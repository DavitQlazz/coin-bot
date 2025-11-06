#!/usr/bin/env python3
"""
Simple 4H Test - Test top 4 AUD pairs on 4h timeframe
"""

import subprocess
import json
from datetime import datetime

def test_pair_4h(pair):
    """Test a single pair on 4h using the main bot"""
    print(f"\nTesting {pair} on 4h...")
    cmd = f"""python3 -c "
import sys
sys.path.insert(0, '/workspaces/coin-bot')
from bot_forex_scalping_v3_highwr import HighWRScalpingBot

bot = HighWRScalpingBot('{pair}', atr_sl_multiplier=1.4, atr_tp_multiplier=2.4, max_hold_minutes=240)
result = bot.backtest(period='30d', interval='4h')
print('RESULT:', result)
"
"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        return result.stdout
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    pairs = ['AUDNZD=X', 'AUDCAD=X', 'AUDHKD=X', 'AUDEUR=X']
    print("\n" + "="*80)
    print(" "*20 + "4H TIMEFRAME TEST - TOP 4 AUD PAIRS")
    print("="*80)
    
    for idx, pair in enumerate(pairs, 1):
        print(f"\n[{idx}/{len(pairs)}] {pair}")
        test_pair_4h(pair)
