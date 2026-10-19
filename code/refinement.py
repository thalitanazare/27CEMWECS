"""Recursive refinement used by the article's additional neural comparator."""
import copy,time
import numpy as np
from scipy.signal import get_window
from neural import NeuralModel
from experts import blank,design
def spectra(x,nperseg):
    windows=np.lib.stride_tricks.sliding_window_view(np.asarray(x),nperseg)[::nperseg//2]
    windows=windows-windows.mean(axis=1,keepdims=True)
    return np.fft.rfft(windows*get_window('hann',nperseg),axis=1)


def residual(y,pred,mask,nperseg):
    a=spectra(y,nperseg)[:,mask]; b=spectra(pred,nperseg)[:,mask]
    # A scalar normalization retains excitation weighting and avoids amplifying quiet bins.
    norm=max(np.linalg.norm(a),1e-12)
    z=(b-a).ravel()/norm
    return np.r_[z.real,z.imag]

def values(model,r,mask,norm,cfg):
    burn=cfg['burn_samples'];pred=model.simulate(r['u'],r['y'])[burn:];y=r['y'][burn:]
    if not model.linear_stable() or not np.isfinite(pred).all():return {'stable':False}
    error=pred-y;physical=y*norm['y_std']+norm['y_mean'];estimate=pred*norm['y_std']+norm['y_mean'];tau=r['tau'][burn:]
    nr=np.linalg.norm(error)/max(np.linalg.norm(y-y.mean()),1e-12)
    spec=np.linalg.norm(residual(y,pred,mask,cfg['nperseg']))
    one=model.predict(r['u'],r['y'])[burn-model.start:]
    measured_power=-tau*physical;pred_power=-tau*estimate
    power_scale=max(np.sqrt(np.mean(measured_power**2)),1e-6)
    power_error=np.sqrt(np.mean((pred_power-measured_power)**2))/power_scale if np.any(tau) else 0.
    energy=np.sum(measured_power)/cfg['sample_rate_hz'];pred_energy=np.sum(pred_power)/cfg['sample_rate_hz']
    return {'stable':True,'nrmse':float(nr),'spectral_error':float(spec),'one_step_nrmse':float(np.linalg.norm(y-one)/max(np.linalg.norm(y-y.mean()),1e-12)),
            'rmse_rad_s':float(np.sqrt(np.mean(error**2))*norm['y_std']),
            'nmape_percent':float(100*np.mean(abs(error))*norm['y_std']/max(abs(physical).max(),1e-12)),
            'power_nrmse':float(power_error),'measured_energy_j':float(energy),'predicted_energy_j':float(pred_energy),
            'relative_energy_error':float(abs(pred_energy-energy)/max(abs(energy),1e-6)) if np.any(tau) else None}

def validate(model,records,mask,norm,cfg,energy=False):
    terms=[]
    for r in records:
        m=values(model,r,mask,norm,cfg)
        if not m['stable']:return np.inf
        z=m['spectral_error']**2+cfg['broadband_weight']*m['nrmse']**2
        if energy:z+=cfg['power_weight']*m['power_nrmse']**2+cfg['energy_weight']*(m['relative_energy_error'] or 0.)**2
        terms.append(z)
    return float(np.sqrt(np.mean(terms)))


def torch_predict(model,u,y,parameters):
    """Measured history once; all subsequent lagged outputs are predicted."""
    import torch
    start=model.start;out=[y[:,k] for k in range(start)]
    neural=isinstance(model,NeuralModel)
    powers=None if neural else torch.as_tensor(model.powers,dtype=torch.int64)
    for k in range(start,u.shape[1]):
        z=torch.stack([out[k-i-1] for i in range(model.memory)]+[u[:,k-model.delay-i] for i in range(model.memory)],dim=1)
        if neural:
            value=torch.tanh(z@parameters[0]+parameters[1])@parameters[2]+parameters[3]
        else:value=torch.prod(z[:,None,:]**powers[None,:,:],dim=2)@parameters[0]
        out.append(value)
    return torch.stack(out,dim=1)[:,start:]


def common_loss(pred,target,tau,mask,norm,cfg,parameters,original,energy):
    import torch
    n=pred.shape[1];window=torch.hann_window(n,periodic=True,dtype=pred.dtype)
    a=torch.fft.rfft((pred-pred.mean(1,keepdim=True))*window,dim=1)[:,mask]
    b=torch.fft.rfft((target-target.mean(1,keepdim=True))*window,dim=1)[:,mask]
    # Per-window ratios and a common per-parameter relative anchor for both maps.
    loss=(abs(a-b).square().sum(1)/b.abs().square().sum(1).clamp(min=1e-12)).mean()
    err=pred-target
    loss+=cfg['broadband_weight']*(err.square().sum(1)/target.square().sum(1).clamp(min=1e-12)).mean()
    if energy:
        ep=-tau*err*norm['y_std'];p=-tau*(target*norm['y_std']+norm['y_mean'])
        scale=p.square().mean(1).clamp(min=1e-8)
        loss+=cfg['power_weight']*(ep.square().mean(1)/scale).mean()
        loss+=cfg['energy_weight']*(ep.mean(1).square()/scale).mean()
    anchor=torch.cat([((p-q)/q.abs().clamp(min=.1)).reshape(-1) for p,q in zip(parameters,original)]).square().mean()
    return loss+cfg['spectral_anchor']*anchor


def refine(model,train,val,mask,norm,cfg,rcfg,energy):
    import torch
    torch.set_num_threads(1)
    arrays=[model.w1,model.b1,model.w2,np.asarray(model.b2)] if isinstance(model,NeuralModel) else [model.coef]
    params=[torch.nn.Parameter(torch.tensor(a,dtype=torch.float64)) for a in arrays]
    original=[p.detach().clone() for p in params]
    optimizer=torch.optim.Adam(params,lr=rcfg['learning_rate'])
    rng=np.random.default_rng(rcfg['window_seed']);length=rcfg['rollout_samples'];start=model.start
    best=copy.deepcopy(model);bestscore=validate(best,val,mask,norm,cfg,energy);bestupdate=0
    history=[];checkpoints={};t0=time.perf_counter();stop=None
    def exported():
        a=[p.detach().numpy().copy() for p in params]
        if isinstance(model,NeuralModel):return NeuralModel(model.memory,model.delay,a[0],a[1],a[2],float(a[3]),copy.deepcopy(model.selection))
        m=copy.deepcopy(model);m.coef=a[0];return m
    for epoch in range(1,max(rcfg['gradient_updates'])+1):
        inputs=[];targets=[];torques=[];starts=[]
        for r in train:
            # Identical scored indices for every representation (max history <=8).
            c=int(rng.integers(8,len(r['u'])-length));starts.append(c)
            inputs.append(r['u'][c-start:c+length]);targets.append(r['y'][c-start:c+length]);torques.append(r['tau'][c:c+length])
        u=torch.tensor(np.asarray(inputs),dtype=torch.float64);y=torch.tensor(np.asarray(targets),dtype=torch.float64);tau=torch.tensor(np.asarray(torques),dtype=torch.float64)
        optimizer.zero_grad();pred=torch_predict(model,u,y,params)
        loss=common_loss(pred,y[:,start:],tau,torch.tensor(mask),norm,cfg,params,original,energy)
        if not torch.isfinite(loss) or torch.any(pred.abs()>1e6):stop='nonfinite_or_divergent_training_rollout';break
        loss.backward();torch.nn.utils.clip_grad_norm_(params,rcfg['gradient_clip']);optimizer.step()
        if epoch%rcfg['checkpoint_every']==0 or epoch in rcfg['gradient_updates']:
            candidate=exported();s=validate(candidate,val,mask,norm,cfg,energy)
            history.append({'update':epoch,'loss':float(loss.detach()),'validation_score':float(s) if np.isfinite(s) else None,'window_starts':starts})
            if s<bestscore:best,bestscore,bestupdate=candidate,s,epoch
        if epoch in rcfg['gradient_updates']:
            checkpoints[epoch]=(copy.deepcopy(best),{'selected_update':bestupdate,'validation_score':bestscore if np.isfinite(bestscore) else None,'refinement_seconds':time.perf_counter()-t0,'completed_updates':epoch,'stop':None})
    for budget in rcfg['gradient_updates']:
        if budget not in checkpoints:checkpoints[budget]=(copy.deepcopy(best),{'selected_update':bestupdate,'validation_score':bestscore if np.isfinite(bestscore) else None,'refinement_seconds':time.perf_counter()-t0,'completed_updates':epoch-1,'stop':stop})
    return checkpoints,history

def convert(net,train):
    lags=np.arange(1,net.memory+1);x,_=design(train,lags,lags);m=blank('neural_refined',4,lags,lags,x,copy.deepcopy(net.selection));m.w1=net.w1*m.scale[:,None];m.b1=net.b1+m.mean@net.w1;m.w2=net.w2.copy();m.coef=np.zeros(len(m.mean)+1);m.coef[0]=net.b2;return m
