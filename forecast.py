"""Causal five-minute elevation forecasts. Run: python forecast.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

ROOT = Path(__file__).resolve().parent
HORIZONS = [12, 24, 36, 48, 60, 72]
TRAIN_END = pd.Timestamp('2024-10-04', tz='UTC')
VAL_END = pd.Timestamp('2025-10-04', tz='UTC')

def load():
    parts = []
    audit = []
    for path in sorted((ROOT / 'data/raw').glob('*.rdb')):
        d = pd.read_csv(path, sep='\t', comment='#', dtype=str).iloc[1:].copy()
        if not d.tz_cd.isin(['EDT', 'EST']).all():
            raise ValueError('Unknown time zone')
        d['time'] = pd.to_datetime(d.datetime + d.tz_cd.map({'EDT':'-04:00', 'EST':'-05:00'}), utc=True)
        d['level_ft'] = pd.to_numeric(d['301102_63160'], errors='coerce')
        d['rain_in'] = pd.to_numeric(d['89142_00045'], errors='coerce')
        audit.append({'file':path.name, 'rows':len(d)})
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)
    overlap = combined[combined.duplicated('time', keep=False)]
    conflicts = overlap.groupby('time')[['level_ft','rain_in']].nunique().gt(1).any(axis=1).sum()
    d = combined.drop_duplicates('time', keep='last').set_index('time').sort_index()
    grid = pd.date_range(d.index.min(), d.index.max(), freq='5min')
    d = d.reindex(grid)
    # Preserve missingness and USGS flags; never replace missing rain with zero.
    quality = {'sources':audit, 'unique_observations':int(combined.time.nunique()),
               'overlap_rows_removed':int(combined.time.duplicated().sum()),
               'conflicting_timestamps':int(conflicts), 'grid_rows':len(d),
               'missing_level':int(d.level_ft.isna().sum()),
               'missing_rain':int(d.rain_in.isna().sum()),
               'start_utc':str(d.index.min()),'end_utc':str(d.index.max())}
    return d, quality

def features(d):
    x = pd.DataFrame(index=d.index)
    x['level_now'] = d.level_ft
    for n in [1, 3, 6, 12, 24, 36, 72, 144, 288]:
        x[f'level_lag_{n}'] = d.level_ft.shift(n)
        x[f'level_change_{n}'] = d.level_ft - d.level_ft.shift(n)
    x['rain_now'] = d.rain_in
    for n in [3, 6, 12, 24, 36, 72, 144, 288]:
        # Only observations available at issue time; full-window min_periods.
        x[f'rain_sum_{n}'] = d.rain_in.rolling(n, min_periods=n).sum()
        x[f'rain_missing_fraction_{n}'] = d.rain_in.isna().rolling(n).mean()
    return x

def scores(y, pred, high):
    e = pred - y
    top = y >= high
    return {'mae_ft':float(mean_absolute_error(y,pred)),
            'rmse_ft':float(np.sqrt(mean_squared_error(y,pred))),
            'high_level_mae_ft':float(np.mean(np.abs(e[top]))) if top.any() else None,
            'high_level_bias_ft':float(np.mean(e[top])) if top.any() else None,
            'n':len(y),'high_level_n':int(top.sum())}

def run():
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    d, audit = load(); x = features(d)
    (out/'data_quality.json').write_text(json.dumps(audit,indent=2))
    high = float(d.loc[d.index < TRAIN_END,'level_ft'].quantile(.95))
    rows=[]; predictions=[]
    for steps in HORIZONS:
        target = d.level_ft.shift(-steps)
        # Entire future path must be observed, for event comparisons and gap safety.
        future_complete = d.level_ft.notna().rolling(steps+1).sum().shift(-steps).eq(steps+1)
        valid = d.level_ft.notna() & target.notna() & future_complete
        future_time = d.index + pd.Timedelta(minutes=5*steps)
        train = valid & (future_time < TRAIN_END)
        val = valid & (d.index >= TRAIN_END) & (future_time < VAL_END)
        test = valid & (d.index >= VAL_END)
        # Fit every 30 min; targets and input observations remain unsmoothed.
        train &= np.arange(len(d)) % 6 == 0
        weights = np.where(target[train] >= high, 4.0, 1.0)
        best=None; best_score=np.inf
        for leaves in [15,31]:
            model=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=leaves,
                     learning_rate=.08,l2_regularization=1,early_stopping=False,random_state=42)
            model.fit(x.loc[train],target[train]-d.level_ft[train],sample_weight=weights)
            p = d.level_ft[val].to_numpy()+model.predict(x.loc[val])
            e = np.abs(p-target[val].to_numpy())
            value=float(np.average(e,weights=np.where(target[val]>=high,4.,1.)))
            if value < best_score: best, best_score=model,value
        joblib.dump(best,out/f'model_{steps//12}h.joblib')
        for name,mask in [('validation',val),('test',test)]:
            y=target[mask].to_numpy(); now=d.level_ft[mask].to_numpy()
            pred=now+best.predict(x.loc[mask])
            for label,p in [('persistence',now),('gradient_boosting',pred)]:
                rows.append({'split':name,'horizon_hours':steps//12,'model':label,
                             **scores(y,p,high)})
            if name=='test':
                predictions.append(pd.DataFrame({'issue_time':d.index[mask],
                    'valid_time':future_time[mask],'horizon_hours':steps//12,
                    'observed_ft':y,'predicted_ft':pred,'persistence_ft':now}))
        print('Finished',steps//12,'hours',flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'test_predictions.csv.gz',index=False)
    (out/'configuration.json').write_text(json.dumps({'high_level_training_q95_ft':high,
        'training_end_exclusive':str(TRAIN_END),'validation_end_exclusive':str(VAL_END),
        'target':'water surface elevation ft NAVD88','forecast_hours':[1,2,3,4,5,6],
        'training_sampling_minutes':30,'peak_weight':4},indent=2))
    print(pd.DataFrame(rows).query("split=='test'").to_string(index=False))

if __name__ == '__main__':
    run()
