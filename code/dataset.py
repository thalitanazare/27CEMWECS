"""D1 records, chronological partitions, training-only scaling and evaluation metrics."""
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly,welch
from common import ROOT
CACHE={}
def config():return json.loads((ROOT/'code/config.json').read_text(encoding='utf-8'))
def band_from_training(u,fs,cfg):
    f,p=welch(u,fs=fs,nperseg=cfg['nperseg'])
    mask=(p>=p.max()*10**(cfg['excitation_threshold_db']/10))&(f>0)&(f<0.4*fs)
    if np.count_nonzero(mask)<3: raise ValueError('Too few excited frequency bins')
    return mask

def channel(a,n):
    a=np.asarray(a)
    if a.ndim==1 and len(a)==n:return a
    if a.ndim==2 and a.shape[1]==n:return a[0] # D1 is first device in documented L0--L5 layouts.
    raise ValueError('Unrecognised D1/time channel shape '+str(a.shape))


def read(path,policy,cfg):
    key=(str(path),policy)
    if key in CACHE:return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in CACHE[key].items()}
    d=loadmat(path,simplify_cells=True);t=np.asarray(d['time']);n=len(t);tag={'UM':'UM','P':'CM_P','PI':'CM_PI'}[policy]
    if 'angularVel_'+tag not in d:return None
    fs=1/np.median(np.diff(t));ratio=int(round(fs/cfg['sample_rate_hz']))
    if not np.allclose(np.diff(t),1/fs) or not np.isclose(fs/ratio,cfg['sample_rate_hz']):raise ValueError('Invalid sample grid')
    keep=(t>=cfg['trim_start_s'])&(t<=t[-1]-cfg['trim_end_s']);edge=2*cfg['sample_rate_hz']
    raw={'exc':channel(d['excitationTorque_WE'],n),'tau':np.zeros(n) if policy=='UM' else channel(d['controlTorque_'+tag],n),'y':channel(d['angularVel_'+tag],n),'q':channel(d['angularPos_'+tag],n)}
    r={k:resample_poly(v[keep],1,ratio)[edge:-edge] for k,v in raw.items()}
    r.update({'layout':path.parent.name,'state':path.stem.split('_')[1],'policy':policy,'file':str(path.relative_to(ROOT)),'device':'D1'})
    if not all(np.isfinite(r[k]).all() for k in raw):raise ValueError('Nonfinite source record')
    CACHE[key]=r;return read(path,policy,cfg)


def sliced(r,a,b):return {k:v[a:b].copy() if isinstance(v,np.ndarray) else v for k,v in r.items()}


def prepare(layout,held=None,input_mode='aggregate',broadband=True):
    cfg=config()
    train=[];val=[];test=[]
    for state in cfg['states']:
        paths=sorted((ROOT/'data/raw'/layout).glob(f'*_{state}_*.mat'))
        if len(paths)!=2:raise ValueError('Expected two irregular realizations '+layout+' '+state)
        if state!=held:
            for policy in ['UM','P']:
                r=read(paths[0],policy,cfg)
                if r is None:raise ValueError('Missing required training policy')
                c=int(len(r['y'])*cfg['train_fraction']);g=cfg['guard_samples'];train.append(sliced(r,0,c-g));val.append(sliced(r,c+g,None))
        if held is None or state==held:
            for path,policy,kind in [(paths[0],'PI','policy'),(paths[1],'UM','wave_realisation')]:
                r=read(path,policy,cfg)
                if r is None:raise ValueError('Missing required test policy')
                r['test_kind']=('unseen_state_' if held else '')+kind;test.append(r)
    if broadband:
        for state in cfg['broadband_states']:
            paths=sorted((ROOT/'data/raw'/layout).glob(f'*_{state}_*.mat'))
            if len(paths)!=1:raise ValueError('Missing broadband record '+layout+' '+state)
            r=read(paths[0],'UM',cfg);c=int(len(r['y'])*cfg['train_fraction']);g=cfg['guard_samples'];train.append(sliced(r,0,c-g));val.append(sliced(r,c+g,None))
    def inputs(r):return (r['exc']+r['tau'])[:,None] if input_mode=='aggregate' else np.column_stack([r['exc'],r['tau']])
    raw_inputs=np.vstack([inputs(r) for r in train]);y=np.concatenate([r['y'] for r in train]);norm={'x_mean':raw_inputs.mean(0),'x_std':raw_inputs.std(0),'y_mean':float(y.mean()),'y_std':float(y.std())};norm['x_std']=np.maximum(norm['x_std'],1e-8)
    for group in [train,val,test]:
        for r in group:r['x']=(inputs(r)-norm['x_mean'])/norm['x_std'];r['y']=(r['y']-norm['y_mean'])/norm['y_std']
    mask=np.logical_or.reduce([band_from_training((r['exc']+r['tau'])[cfg['burn_samples']:],cfg['sample_rate_hz'],cfg) for r in train])
    audit={'layout':layout,'held_state':held,'input_mode':input_mode,'broadband':broadband,'normalisation':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in norm.items()},'mask_bins':np.flatnonzero(mask).tolist(),'training':[{'file':r['file'],'state':r['state'],'policy':r['policy'],'samples':len(r['y'])} for r in train],'validation':[{'file':r['file'],'state':r['state'],'policy':r['policy'],'samples':len(r['y'])} for r in val],'test_used_for_selection':False}
    return train,val,test,norm,mask,audit


def metrics(model,r,norm,mask):
    cfg=config();burn=cfg['burn_samples'];prediction=model.simulate(r['x'],r['y']);p=prediction[burn:];y=r['y'][burn:];tau=r['tau'][burn:]
    if not np.isfinite(p).all():return {'stable':False,'joint_error':None,'nrmse':None,'spectral_error':None,'energy_error':None}
    e=p-y;nr=float(np.linalg.norm(e)/max(np.linalg.norm(y),1e-12));n=cfg['nperseg'];w=np.hanning(n+1)[:-1];count=len(y)//n
    a=np.fft.rfft((e[:count*n].reshape(count,n)-e[:count*n].reshape(count,n).mean(1,keepdims=True))*w,axis=1)[:,mask]
    b=np.fft.rfft((y[:count*n].reshape(count,n)-y[:count*n].reshape(count,n).mean(1,keepdims=True))*w,axis=1)[:,mask]
    sp=float(np.linalg.norm(a)/max(np.linalg.norm(b),1e-12));physical=y*norm['y_std']+norm['y_mean'];power=-tau*physical;ep=-tau*e*norm['y_std'];meanabs=abs(float(power.mean()));rms=max(float(np.sqrt(np.mean(power**2))),1e-12)
    energy=float(abs(ep.mean())/meanabs) if meanabs/rms>=cfg['near_zero_energy_ratio'] else float(abs(ep.mean())/rms)
    if r['policy']=='UM':energy=None
    return {'stable':True,'joint_error':float(np.sqrt(nr*sp)),'nrmse':nr,'spectral_error':sp,'energy_error':energy,'relative_energy_error':float(abs(ep.mean())/meanabs) if meanabs/rms>=cfg['near_zero_energy_ratio'] else None,'normalised_mean_power_error':float(abs(ep.mean())/rms),'energy_denominator':'signed_measured_energy' if meanabs/rms>=cfg['near_zero_energy_ratio'] else 'duration_times_power_rms','measured_energy_j':float(power.sum()/cfg['sample_rate_hz']),'predicted_energy_j':float((power+ep).sum()/cfg['sample_rate_hz'])}


def score(model,records,norm,mask):
    vals=[metrics(model,r,norm,mask) for r in records]
    if any(not m['stable'] for m in vals):return float('inf')
    return float(np.mean([m['joint_error']+.25*(m['energy_error'] or 0) for m in vals]))
