"""Optional corrected LSTM/GRU/BiLSTM comparison. Requires TensorFlow.
This module has been syntax-checked, but neural training was NOT run in v0.1.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from forecast import load, ROOT, TRAIN_END, VAL_END, HORIZONS

def run(kind='gru', epochs=30):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    d,_=load()
    channels=d[['level_ft','rain_in']].copy()
    channels['rise_5min_ft']=d.level_ft.diff()
    z=channels.to_numpy(dtype=np.float32)
    scaler=StandardScaler().fit(z[d.index<TRAIN_END])
    z=scaler.transform(z)
    target_scaler=StandardScaler().fit(d.loc[d.index<TRAIN_END,['level_ft']])
    levels=d.level_ft.to_numpy()
    lookback=72
    origins=np.arange(lookback-1,len(d)-max(HORIZONS),6)
    valid=[]
    for i in origins:
        # Do not join sequences across missing observations or input rain gaps.
        if np.isfinite(z[i-lookback+1:i+1]).all() and np.isfinite(levels[i:i+73]).all():
            valid.append(i)
    origins=np.asarray(valid)
    train=origins[d.index[origins+72]<TRAIN_END]
    val=origins[(d.index[origins]>=TRAIN_END)&(d.index[origins+72]<VAL_END)]
    test=origins[d.index[origins]>=VAL_END]
    cutoff=d.loc[d.index<TRAIN_END,'level_ft'].quantile(.95)
    def dataset(ids, weighted=False):
        def generate():
            for i in ids:
                y=target_scaler.transform(levels[i+np.array(HORIZONS)].reshape(-1,1)).ravel().astype('float32')
                w=np.float32(4 if weighted and (levels[i+np.array(HORIZONS)]>=cutoff).any() else 1)
                yield z[i-lookback+1:i+1].astype('float32'),y,w
        return tf.data.Dataset.from_generator(generate,output_signature=(
            tf.TensorSpec((lookback,3),tf.float32),tf.TensorSpec((6,),tf.float32),
            tf.TensorSpec((),tf.float32))).batch(64).prefetch(1)
    layer={'lstm':tf.keras.layers.LSTM,'gru':tf.keras.layers.GRU}.get(kind)
    if kind=='bilstm': recurrent=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
    elif layer: recurrent=layer(32)
    else: raise ValueError('Choose lstm, gru, bilstm')
    model=tf.keras.Sequential([tf.keras.Input((lookback,3)),recurrent,
                              tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(6)])
    model.compile(optimizer='adam',loss='mae')
    model.fit(dataset(train,True),validation_data=dataset(val,True),epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
    out=ROOT/'results';out.mkdir(exist_ok=True)
    model.save(out/f'{kind}.keras')
    import joblib,pandas as pd
    joblib.dump({'input':scaler,'target':target_scaler},out/f'{kind}_scalers.joblib')
    pred=model.predict(dataset(test).map(lambda x,y,w:x))
    pred=target_scaler.inverse_transform(pred.reshape(-1,1)).reshape(-1,6)
    rows=[]
    for k,h in enumerate(HORIZONS):
        rows.append(pd.DataFrame({'issue_time':d.index[test],'valid_time':d.index[test+h],
            'horizon_hours':h//12,'observed_ft':levels[test+h],
            'predicted_ft':pred[:,k],'persistence_ft':levels[test]}))
    pd.concat(rows).to_csv(out/f'{kind}_test_predictions.csv.gz',index=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--kind',default='gru')
    parser.add_argument('--epochs',type=int,default=30);args=parser.parse_args()
    run(args.kind,args.epochs)
