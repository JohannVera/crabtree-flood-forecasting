"""Retrospective event diagnostics; these are not single-issue trajectories."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast import ROOT

def run():
    out=ROOT/'results'
    p=pd.read_csv(out/'test_predictions.csv.gz',parse_dates=['issue_time','valid_time'])
    threshold=json.loads((out/'configuration.json').read_text())['high_level_training_q95_ft']
    base=p[p.horizon_hours==1].set_index('valid_time').sort_index()
    above=base[base.observed_ft>=threshold]
    groups=above.index.to_series().diff().gt(pd.Timedelta(hours=6)).cumsum()
    events=[]
    for _,g in above.groupby(groups):
        peak=g.observed_ft.idxmax()
        start=g.index.min()-pd.Timedelta(hours=6);end=g.index.max()+pd.Timedelta(hours=6)
        events.append((start,end,peak,float(g.observed_ft.max())))
    rows=[];alerts=[]
    for horizon,q in p.groupby('horizon_hours'):
        q=q.set_index('valid_time').sort_index()
        obs=q.observed_ft>=threshold
        for name,col in [('gradient_boosting','predicted_ft'),('persistence','persistence_ft')]:
            alarm=q[col]>=threshold
            tp=int((alarm & obs).sum());fn=int((~alarm & obs).sum());fp=int((alarm & ~obs).sum())
            alerts.append({'horizon_hours':horizon,'model':name,'threshold_ft':threshold,
                'true_positive_points':tp,'missed_points':fn,'false_positive_points':fp,
                'recall':tp/(tp+fn) if tp+fn else np.nan,
                'precision':tp/(tp+fp) if tp+fp else np.nan})
            for eid,(start,end,peak,height) in enumerate(events,1):
                w=q.loc[start:end]
                # Exclude event windows crossing data gaps or truncated test boundaries.
                expected=int((end-start)/pd.Timedelta(minutes=5))+1
                if len(w)!=expected: continue
                predicted_peak=w[col].idxmax()
                rows.append({'event':eid,'horizon_hours':horizon,'model':name,
                    'observed_peak_time':str(peak),'observed_peak_ft':height,
                    'predicted_peak_ft':float(w[col].max()),
                    'peak_error_ft':float(w[col].max()-height),
                    'peak_timing_error_minutes':(predicted_peak-peak).total_seconds()/60,
                    'threshold_event_detected':bool((w[col]>=threshold).any())})
    pd.DataFrame(rows).to_csv(out/'event_metrics.csv',index=False)
    pd.DataFrame(alerts).to_csv(out/'threshold_metrics.csv',index=False)
    if events:
        start,end,peak,height=max(events,key=lambda e:e[3])
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for ax,h in zip(axes,[1,3,6]):
            q=p[p.horizon_hours==h].set_index('valid_time').loc[start:end]
            ax.plot(q.index,q.observed_ft,label='Observed',color='black')
            ax.plot(q.index,q.predicted_ft,label=f'Forecast issued {h} h earlier',color='#127b9b')
            ax.plot(q.index,q.persistence_ft,label='Persistence',color='#b47928',alpha=.7)
            ax.set_ylabel('Elevation (ft NAVD88)');ax.legend(fontsize=8);ax.grid(alpha=.2)
        axes[0].set_title('Largest test-period event: rolling forecasts aligned by valid time')
        axes[-1].set_xlabel('UTC');fig.autofmt_xdate();fig.tight_layout()
        fig.savefig(out/'largest_test_event.png',dpi=160);plt.close(fig)
    print('Candidate events:',len(events),'Complete event/model/horizon comparisons:',len(rows))

if __name__=='__main__': run()
