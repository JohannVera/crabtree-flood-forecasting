"""Issue latest forecasts from locally fitted models; append new RDB data first."""
import joblib
import pandas as pd
from forecast import ROOT, load, features, HORIZONS

if __name__=='__main__':
    d,_=load(); x=features(d)
    if pd.isna(d.level_ft.iloc[-1]):
        raise ValueError('Latest elevation is missing; no forecast issued')
    rows=[]
    for steps in HORIZONS:
        model=joblib.load(ROOT/'results'/f'model_{steps//12}h.joblib')
        rows.append({'issue_time_utc':str(d.index[-1]),
            'valid_time_utc':str(d.index[-1]+pd.Timedelta(minutes=steps*5)),
            'horizon_hours':steps//12,
            'elevation_ft_NAVD88':float(d.level_ft.iloc[-1]+model.predict(x.iloc[[-1]])[0])})
    print(pd.DataFrame(rows).to_string(index=False))
