#!/usr/bin/env python3
"""
Compute CAGR, annualized Sharpe and rolling drawdown events for
results/gbpcad_s10_1h_long.json and save metrics + drawdown table.
"""
import json, os
import pandas as pd
from math import sqrt

FP = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'gbpcad_s10_1h_long.json')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

def main():
    with open(FP) as f:
        d = json.load(f)
    eq = d.get('equity', [])
    if not eq:
        print('No equity points found')
        return
    dates = pd.to_datetime([t for t,_ in eq], utc=True)
    vals = [v for _,v in eq]
    ser = pd.Series(vals, index=dates).sort_index()
    ser_daily = ser.resample('D').last().ffill()
    daily_ret = ser_daily.pct_change().dropna()

    start_val = ser_daily.iloc[0]
    end_val = ser_daily.iloc[-1]
    days = (ser_daily.index[-1] - ser_daily.index[0]).days
    years = days / 365.25 if days>0 else 0
    CAGR = (end_val/start_val)**(1/years)-1 if years>0 and start_val>0 else None

    sharpe = None
    if not daily_ret.empty and daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * sqrt(252)

    rolling_max = ser_daily.cummax()
    drawdown = (rolling_max - ser_daily) / rolling_max
    max_dd = drawdown.max()

    # detect drawdown events
    events = []
    in_dd = False
    start_idx = None
    for idx, dd in drawdown.items():
        if (not in_dd) and dd>0:
            in_dd = True
            start_idx = idx
        if in_dd and dd==0:
            end_idx = idx
            period = drawdown[start_idx:end_idx]
            trough_date = period.idxmax()
            trough_val = period.max()
            events.append({'start': start_idx.date().isoformat(), 'trough': trough_date.date().isoformat(), 'recovery': end_idx.date().isoformat(), 'drawdown_pct': round(trough_val*100,2)})
            in_dd = False
            start_idx = None
    if in_dd:
        period = drawdown[start_idx:]
        trough_date = period.idxmax()
        trough_val = period.max()
        events.append({'start': start_idx.date().isoformat(), 'trough': trough_date.date().isoformat(), 'recovery': None, 'drawdown_pct': round(trough_val*100,2)})

    metrics = {
        'start_date': ser_daily.index[0].date().isoformat(),
        'end_date': ser_daily.index[-1].date().isoformat(),
        'years': round(years,3),
        'start_value': round(float(start_val),2),
        'end_value': round(float(end_val),2),
        'CAGR': round(CAGR,4) if CAGR is not None else None,
        'annualized_sharpe': round(sharpe,4) if sharpe is not None else None,
        'max_drawdown_pct': round(max_dd*100,2),
        'trades': d.get('trades', None),
        'wins': d.get('wins', None),
        'win_rate_pct': round((d.get('wins',0)/d.get('trades',1))*100,2) if d.get('trades',0)>0 else None,
        'net': round(d.get('net',0),2)
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR,'gbpcad_s10_1h_long_metrics.json'),'w') as f:
        json.dump({'metrics':metrics,'events_count':len(events)},f,indent=2)
    if events:
        pd.DataFrame(events).to_csv(os.path.join(OUT_DIR,'gbpcad_s10_1h_long_drawdowns.csv'), index=False)

    print('Saved metrics and drawdown table to results/')
    print('CAGR:', metrics['CAGR'])
    print('Annualized Sharpe:', metrics['annualized_sharpe'])
    print('Max drawdown (%):', metrics['max_drawdown_pct'])
    print('Drawdown events found:', len(events))

if __name__ == '__main__':
    main()
