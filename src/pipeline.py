"""Reproducible hourly demand experiment. Run from any working directory."""
from pathlib import Path
import json, hashlib, time, random, platform, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'power_demand_mwh'
LOOKBACK = 24
SEEDS = [42, 7, 123]
WEATHER = {'기온(°C)':'temperature_c','강수량(mm)':'rainfall_mm','풍속(m/s)':'wind_speed_ms',
           '습도(%)':'humidity_pct','일조(hr)':'sunshine_hr','일사(MJ/m2)':'solar_radiation_mj_m2'}
CALENDAR = ['hour_sin','hour_cos','day_sin','day_cos','month_sin','month_cos','is_weekend']
SPLITS = {'train':(0,6132),'validation':(6132,7008),'calibration':(7008,7446),'test':(7446,8760)}

def save_json(obj,path):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def read_csv(path):
    for enc in ['utf-8-sig','cp949']:
        try:return pd.read_csv(path,encoding=enc)
        except UnicodeDecodeError:pass
    raise ValueError(path)

def prepare():
    for name in ['results','figures','models','data/processed']: (ROOT/name).mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'data/raw').glob('*.csv'))
    power_path=next(p for p in files if '전력' in p.name)
    weather_path=next(p for p in files if 'OBS_' in p.name)
    p=read_csv(power_path); blank=int(p.isna().all(axis=1).sum()); p=p.dropna(how='all')
    p=p.melt(id_vars='날짜',var_name='hour',value_name=TARGET)
    hours=p.hour.str.replace('시','',regex=False).astype(int)
    assert hours.between(1,24).all()
    p['datetime']=pd.to_datetime(p['날짜'])+pd.to_timedelta(hours,unit='h')
    p=p.set_index('datetime')[[TARGET]].sort_index()
    assert len(p)==8760 and not p.index.duplicated().any() and p[TARGET].gt(0).all()
    assert (p.index.to_series().diff().dropna()==pd.Timedelta(hours=1)).all()
    w=read_csv(weather_path); w['일시']=pd.to_datetime(w['일시'])
    assert not w.duplicated(['지점','일시']).any()
    audits=[]; missing_rows=[]; data=p.copy()
    for city,g in w.groupby('지점명',sort=True):
        absent=p.index.difference(g['일시'])
        missing_rows.extend([{'city':city,'datetime':str(t)} for t in absent])
        g=g.set_index('일시').reindex(p.index)
        row_absent=~p.index.isin(w.loc[w.지점명.eq(city),'일시'])
        for raw,name in WEATHER.items():
            v=pd.to_numeric(g[raw],errors='raise')
            mask=v.isna()
            # Blank precipitation/sun values within an observed row: explicit zero assumption.
            # Entire missing rows and continuous variables: past observations only.
            if name in ['rainfall_mm','sunshine_hr','solar_radiation_mj_m2']:
                filled=v.mask(mask & ~row_absent,0).ffill()
                rule='observed-row blank: assumed zero; absent row: forward fill'
            else:
                filled=v.ffill();rule='forward fill only'
            if filled.isna().any():raise ValueError(f'No past value to fill {city} {raw}')
            data[f'{city}_{name}']=filled
            data[f'{city}_{name}_missing']=mask.astype(int)
            audits.append({'city':city,'variable':name,'observed_missing':int(mask.sum()-row_absent.sum()),
                           'absent_rows':int(row_absent.sum()),'total_missing':int(mask.sum()),'rule':rule})
    dt=data.index
    for name,values,period in [('hour',dt.hour,24),('day',dt.dayofweek,7),('month',dt.month-1,12)]:
        data[name+'_sin']=np.sin(2*np.pi*values/period);data[name+'_cos']=np.cos(2*np.pi*values/period)
    data['is_weekend']=(dt.dayofweek>=5).astype(int)
    assert not data.isna().any().any()
    data.to_csv(ROOT/'data/processed/hourly_data.csv',encoding='utf-8-sig')
    for name,(a,b) in SPLITS.items():data.iloc[a:b].to_csv(ROOT/f'data/processed/{name}.csv',encoding='utf-8-sig')
    pd.DataFrame(audits).to_csv(ROOT/'results/missing_audit.csv',index=False,encoding='utf-8-sig')
    meta={'rows':len(data),'columns':len(data.columns),'start':str(dt.min()),'end':str(dt.max()),
          'power_blank_rows_removed':blank,'missing_weather_rows':missing_rows,
          'splits':{k:{'rows':b-a,'start':str(dt[a]),'end':str(dt[b-1])} for k,(a,b) in SPLITS.items()},
          'sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    save_json(meta,ROOT/'results/data_audit.json')
    return data

def eda(data):
    # Model-directed EDA uses training data only. Full year is shown solely for coverage.
    train=data.iloc[:6132]; y=train[TARGET]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(name):
        plt.tight_layout();plt.savefig(ROOT/f'figures/{name}.png',dpi=160);plt.close()
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(data.index,data[TARGET]/1000,lw=.5,color='#184A52')
    for (name,(a,b)),color in zip(SPLITS.items(),['#dceee9','#d6e4f5','#fbe5b9','#eed7e3']):
        ax.axvspan(data.index[a],data.index[b-1],alpha=.4,color=color,label=name)
    ax.set(ylabel='Demand (1,000 MWh)',title='Data coverage and chronological split');ax.legend(ncol=4,fontsize=9);save('01_coverage')
    hourly=train.groupby(train.index.hour)[TARGET].agg(['mean','std'])
    weekday=train.groupby(train.index.dayofweek)[TARGET].agg(['mean','std'])
    monthly=train.groupby(train.index.month)[TARGET].mean()
    hourly.to_csv(ROOT/'results/eda_hourly.csv');weekday.to_csv(ROOT/'results/eda_weekday.csv');monthly.to_csv(ROOT/'results/eda_monthly.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(hourly.index,hourly['mean']/1000,color='#184A52',marker='o');axs[0].set(xlabel='Hour ending (00 = prior day 24)',ylabel='Mean demand (1,000 MWh)',title='Hourly pattern: training period')
    axs[1].bar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],weekday['mean']/1000,color='#599B87');axs[1].set(ylabel='Mean demand (1,000 MWh)',title='Weekday pattern: training period');save('02_patterns')
    temp=train[[c for c in train if c.endswith('_temperature_c')]].mean(axis=1)
    fig,ax=plt.subplots(figsize=(10,4));ax.scatter(temp,y/1000,s=3,alpha=.18,color='#184A52');ax.set(xlabel='Unweighted 7-city mean temperature (C)',ylabel='Demand (1,000 MWh)',title='Temperature and demand: training period');save('03_temperature')
    acf=pd.DataFrame({'lag_hours':range(1,169),'correlation':[y.autocorr(i) for i in range(1,169)]})
    acf.to_csv(ROOT/'results/eda_autocorrelation.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(acf.lag_hours,acf.correlation,color='#184A52');ax.set(xlabel='Lag (hours)',ylabel='Autocorrelation',title='Demand persistence: training period');save('04_autocorrelation')
    desc=train.describe().T;desc.to_csv(ROOT/'results/eda_descriptive.csv',encoding='utf-8-sig')
    q1,q3=y.quantile([.25,.75]); out=train[(y<q1-1.5*(q3-q1))|(y>q3+1.5*(q3-q1))][[TARGET]]
    out.to_csv(ROOT/'results/eda_outlier_candidates.csv')
    save_json({'training_rows':len(train),'peak_mean_hour':int(hourly['mean'].idxmax()),
               'minimum_mean_hour':int(hourly['mean'].idxmin()),'weekday_mean':float(y[train.index.dayofweek<5].mean()),
               'weekend_mean':float(y[train.index.dayofweek>=5].mean()),'acf_1':float(y.autocorr(1)),
               'acf_24':float(y.autocorr(24)),'acf_168':float(y.autocorr(168)),
               'temperature_pearson':float(temp.corr(y)),'iqr_outlier_candidates':len(out),
               'outlier_policy':'retain physical peaks; flag only, no automatic removal'},ROOT/'results/eda_summary.json')

class Forecaster(nn.Module):
    def __init__(self,dim,kind):
        super().__init__();self.kind=kind
        if kind=='LSTM':self.encoder=nn.LSTM(dim,32,num_layers=1,batch_first=True)
        else:
            self.projection=nn.Linear(dim,32)
            self.position=nn.Parameter(torch.zeros(1,LOOKBACK,32))
            layer=nn.TransformerEncoderLayer(32,4,64,dropout=.1,batch_first=True,activation='gelu')
            self.encoder=nn.TransformerEncoder(layer,1)
        self.head=nn.Sequential(nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x):
        if self.kind=='LSTM':z,_=self.encoder(x)
        else:z=self.encoder(self.projection(x)+self.position)
        # Residual target: learn change from latest observed demand.
        return x[:,-1,-1]+self.head(z[:,-1]).squeeze(-1)

def metric(y,p):
    e=y-p
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),
            'MAPE_pct':float(np.mean(np.abs(e/y))*100)}

def deep(data,kind,weather,seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    cols=([c for c in data if c not in CALENDAR+[TARGET]] if weather else [])+CALENDAR+[TARGET]
    train=data.iloc[:6132];mean=train[cols].mean().to_numpy();std=train[cols].std(ddof=0).to_numpy(copy=True);std[std<1e-8]=1
    a=(data[cols].to_numpy()-mean)/std
    X=np.stack([a[i-LOOKBACK:i] for i in range(LOOKBACK,len(a))]).astype('float32');y=a[LOOKBACK:,-1].astype('float32')
    model=Forecaster(len(cols),kind);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    criterion=nn.MSELoss();loader=DataLoader(TensorDataset(torch.from_numpy(X[:6132-LOOKBACK]),torch.from_numpy(y[:6132-LOOKBACK])),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed))
    vx=torch.from_numpy(X[6132-LOOKBACK:7008-LOOKBACK]);vy=torch.from_numpy(y[6132-LOOKBACK:7008-LOOKBACK])
    history=[];best=float('inf');stale=0;start=time.time();best_epoch=0
    for epoch in range(1,31):
        model.train();loss_sum=0;count=0
        for bx,by in loader:
            opt.zero_grad();loss=criterion(model(bx),by);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            loss_sum+=loss.item()*len(by);count+=len(by)
        model.eval()
        with torch.no_grad():val=criterion(model(vx),vy).item()
        history.append({'epoch':epoch,'train_mse_scaled':loss_sum/count,'validation_mse_scaled':val})
        if val<best-1e-6:
            best=val;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        if stale>=6:break
    model.load_state_dict(state);model.eval()
    with torch.no_grad():pred=np.concatenate([model(torch.from_numpy(X[i:i+256])).numpy() for i in range(0,len(X),256)])*std[-1]+mean[-1]
    name=kind+('_weather' if weather else '_calendar');tag=f'{name}_seed{seed}'
    torch.save({'state_dict':state,'kind':kind,'features':cols,'mean':mean.tolist(),'std':std.tolist(),'lookback':LOOKBACK,'seed':seed},ROOT/f'models/{tag}.pt')
    pd.DataFrame(history).to_csv(ROOT/f'results/history_{tag}.csv',index=False)
    return pred,{'model':name,'seed':seed,'best_epoch':best_epoch,'epochs_run':len(history),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.time()-start,'validation_RMSE':np.sqrt(best)*std[-1]}

def run():
    data=prepare();eda(data);y=data[TARGET].to_numpy();timestamps=data.index
    all_preds={};details=[];full=pd.DataFrame(index=timestamps[6132:]);full['actual']=y[6132:]
    for lag in [1,24,168]:all_preds[f'Naive_{lag}h']=data[TARGET].shift(lag).to_numpy()[6132:]
    print('Fitting SARIMA (2,1,2)(1,0,1,24)',flush=True)
    start=time.time()
    fitted=SARIMAX(y[:6132],order=(2,1,2),seasonal_order=(1,0,1,24),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=150)
    save_json({'order':[2,1,2],'seasonal_order':[1,0,1,24],'fit_details':fitted.mle_retvals,'seconds':time.time()-start},ROOT/'results/sarima_fit.json')
    save_json({'parameter_names':fitted.param_names,'parameters':fitted.params.tolist(),
               'reconstruction':'SARIMAX(training_target, order=(2,1,2), seasonal_order=(1,0,1,24), enforce_stationarity=False, enforce_invertibility=False).filter(parameters)'},ROOT/'models/sarima_parameters.json')
    # Fixed parameters, sequential state updates with past actuals. No refit on test.
    forecasts=[]
    for actual in y[6132:]:
        forecasts.append(float(np.asarray(fitted.forecast(1))[0]));fitted=fitted.extend([actual])
    all_preds['SARIMA']=np.array(forecasts)
    for kind,weather in [('LSTM',True),('LSTM',False),('Transformer',True)]:
        candidates=[]
        for seed in SEEDS:
            print('Training',kind,'weather=',weather,'seed=',seed,flush=True)
            pred,info=deep(data,kind,weather,seed);details.append(info)
            candidates.append((pred[6132-LOOKBACK:],info))
            full[f'{info["model"]}_seed{seed}']=pred[6132-LOOKBACK:]
        # Only validation chooses representative seed; test is not used for selection.
        chosen=min(candidates,key=lambda item:item[1]['validation_RMSE'])
        all_preds[chosen[1]['model']]=chosen[0]
        for _,info in candidates:info['selected']=info['seed']==chosen[1]['seed']
    pd.DataFrame(details).to_csv(ROOT/'results/training_runs.csv',index=False)
    records=[];slice_rows=[]
    ca=slice(7008-6132,7446-6132);te=slice(7446-6132,None)
    # Separate calibration, finite sample absolute residual quantile. Temporal dependence
    # means nominal 95% is a target, not a guaranteed time-series coverage statement.
    for name,p in all_preds.items():
        full[name]=p;errs=np.abs(y[7008:7446]-p[ca]);k=min(len(errs),int(np.ceil((len(errs)+1)*.95)));q=float(np.sort(errs)[k-1])
        full[name+'_lower']=p-q;full[name+'_upper']=p+q
        row={'model':name,**metric(y[7446:],p[te]),'coverage_pct':float(np.mean(np.abs(y[7446:]-p[te])<=q)*100),'interval_width':2*q,'validation_RMSE':metric(y[6132:7008],p[:876])['RMSE']}
        records.append(row)
        peak_threshold=np.quantile(y[:6132],.9)
        for group,mask in [('weekday',timestamps[7446:].dayofweek<5),('weekend',timestamps[7446:].dayofweek>=5),('high_demand',y[7446:]>=peak_threshold)]:
            slice_rows.append({'model':name,'group':group,'n':int(mask.sum()),**metric(y[7446:][mask],p[te][mask])})
    full.index.name='datetime';full.to_csv(ROOT/'results/all_predictions.csv')
    full.iloc[7446-6132:].to_csv(ROOT/'results/test_predictions.csv')
    metrics=pd.DataFrame(records);metrics.to_csv(ROOT/'results/metrics.csv',index=False)
    pd.DataFrame(slice_rows).to_csv(ROOT/'results/slice_metrics.csv',index=False)
    seedrows=[]
    for info in details:
        name=info['model'];z=full[f'{name}_seed{info["seed"]}'].to_numpy()[te]
        seedrows.append({'model':name,'seed':info['seed'],**metric(y[7446:],z)})
    pd.DataFrame(seedrows).to_csv(ROOT/'results/seed_metrics.csv',index=False)
    save_json({'lookback':LOOKBACK,'horizon':1,'seeds':SEEDS,'max_epochs':30,'early_stopping_patience':6,
               'batch_size':128,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'loss':'MSE',
               'gradient_clip':1.0,'selection':'minimum validation RMSE; calibration and test excluded',
               'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
               'observation_assumption':'observations through t-1 are available at the forecast origin; reporting latency not modeled'},ROOT/'results/experiment_config.json')
    result_plots(data,full,metrics)
    print(metrics.to_string(index=False),flush=True)

def result_plots(data,full,metrics):
    names=['Naive_1h','SARIMA','LSTM_weather','Transformer_weather']
    colors=['#999999','#184A52','#DB8546','#7268A6']
    t=full.iloc[-1314:];plot=t.iloc[:168]
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(plot.index,plot.actual/1000,color='black',label='Actual',lw=1.5)
    for name,color in zip(names,colors):ax.plot(plot.index,plot[name]/1000,label=name,color=color,lw=1,alpha=.8)
    ax.set(ylabel='Demand (1,000 MWh)',title='First test week: one-hour forecasts');ax.legend(fontsize=8,ncol=3);plt.tight_layout();plt.savefig(ROOT/'figures/05_predictions.png',dpi=160);plt.close()
    fig,ax=plt.subplots(figsize=(11,4));ax.barh(metrics.model,metrics.RMSE,color='#599B87');ax.set(xlabel='RMSE (MWh)',title='Test error: lower is better');plt.tight_layout();plt.savefig(ROOT/'figures/06_metrics.png',dpi=160);plt.close()
    fig,ax=plt.subplots(figsize=(11,4))
    for name,color in zip(['LSTM_weather','LSTM_calendar','Transformer_weather'],colors[1:]):
        runs=pd.read_csv(ROOT/'results/training_runs.csv');seed=int(runs.loc[runs.model.eq(name)&runs.selected,'seed'].iloc[0]);h=pd.read_csv(ROOT/f'results/history_{name}_seed{seed}.csv')
        ax.plot(h.epoch,h.validation_mse_scaled,label=name,color=color)
    ax.set(xlabel='Epoch',ylabel='Validation MSE (scaled)',title='Validation learning curves');ax.legend();plt.tight_layout();plt.savefig(ROOT/'figures/07_learning.png',dpi=160);plt.close()
    fig,ax=plt.subplots(figsize=(11,4));name='LSTM_weather'
    ax.plot(plot.index,plot.actual/1000,label='Actual',color='black');ax.plot(plot.index,plot[name]/1000,label='LSTM',color='#184A52');ax.fill_between(plot.index,plot[name+'_lower']/1000,plot[name+'_upper']/1000,alpha=.25,color='#599B87',label='Nominal 95% calibrated interval')
    ax.set(ylabel='Demand (1,000 MWh)',title='LSTM interval: independent calibration period');ax.legend(fontsize=9);plt.tight_layout();plt.savefig(ROOT/'figures/08_interval.png',dpi=160);plt.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:eda(prepare())
    else:run()
