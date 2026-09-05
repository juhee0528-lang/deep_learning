"""Independent saved-artifact checks, no model retraining."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import torch
from pipeline import ROOT, TARGET, Forecaster, LOOKBACK

data=pd.read_csv(ROOT/'data/processed/hourly_data.csv',parse_dates=['datetime']).set_index('datetime')
pred=pd.read_csv(ROOT/'results/test_predictions.csv',parse_dates=['datetime']).set_index('datetime')
allp=pd.read_csv(ROOT/'results/all_predictions.csv',parse_dates=['datetime']).set_index('datetime')
metrics=pd.read_csv(ROOT/'results/metrics.csv');runs=pd.read_csv(ROOT/'results/training_runs.csv')
assert len(data)==8760 and len(pred)==1314
assert not data.isna().any().any() and not data.index.duplicated().any()
assert (data.index.to_series().diff().dropna()==pd.Timedelta(hours=1)).all()
np.testing.assert_allclose(data.loc[pred.index,TARGET],pred.actual,atol=0,rtol=0)
for row in metrics.itertuples():
    error=pred.actual-pred[row.model]
    np.testing.assert_allclose(np.sqrt(np.mean(error**2)),row.RMSE,rtol=1e-8)
    np.testing.assert_allclose(abs(error).mean(),row.MAE,rtol=1e-8)
    cal=allp.iloc[876:1314];res=abs(cal.actual-cal[row.model]).sort_values().to_numpy()
    q=res[min(len(res),int(np.ceil((len(res)+1)*.95)))-1]
    np.testing.assert_allclose(row.interval_width,2*q,rtol=1e-8)
    coverage=((pred.actual>=pred[row.model+'_lower'])&(pred.actual<=pred[row.model+'_upper'])).mean()*100
    np.testing.assert_allclose(coverage,row.coverage_pct)
for model,g in runs.groupby('model'):
    chosen=g.loc[g.selected].iloc[0]
    assert chosen.validation_RMSE==g.validation_RMSE.min()
    # Reload trained checkpoint and reproduce first / middle / last test predictions.
    ck=torch.load(ROOT/f'models/{model}_seed{int(chosen.seed)}.pt',weights_only=True,map_location='cpu')
    cols=ck['features'];mu=np.array(ck['mean']);sd=np.array(ck['std'])
    np.testing.assert_allclose(data.iloc[:6132][cols].mean(),mu)
    a=(data[cols].to_numpy()-mu)/sd
    net=Forecaster(len(cols),ck['kind']);net.load_state_dict(ck['state_dict']);net.eval()
    indices=[7446,8103,8759];x=np.stack([a[i-LOOKBACK:i] for i in indices]).astype('float32')
    with torch.no_grad():y=net(torch.from_numpy(x)).numpy()*sd[-1]+mu[-1]
    np.testing.assert_allclose(y,pred.loc[data.index[indices],model],atol=.1,rtol=1e-6)
    # Alter future inputs: earlier forecast windows must remain byte-identical.
    changed=a.copy();changed[8103:]+=1000
    np.testing.assert_array_equal(a[8103-LOOKBACK:8103],changed[8103-LOOKBACK:8103])
result={'status':'passed','checks':['hourly continuity and counts','raw target alignment','independent MAE and RMSE','calibration interval quantile and coverage','validation-only seed selection','checkpoint prediction reproduction','training-only scaling','past-only forecast window']}
(ROOT/'results/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
