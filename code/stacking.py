"""Constant simplex stacking with the adaptive model's selection objective."""
import numpy as np
from scipy.optimize import minimize
from dataset import config
from ensemble import fixed_mixture

def transforms(y, n=256):
    blocks=np.asarray(y)[:len(y)//n*n].reshape(-1,n)
    return np.fft.rfft((blocks-blocks.mean(1,keepdims=True))*np.hanning(n+1)[:-1],axis=1)


class MixtureObjective:
    """Exact mean J + .25 R_E, including record weights and energy fallback."""
    def __init__(self, predictions, records, norm, mask):
        self.records=[]
        cfg=config();burn=cfg['burn_samples']
        for pred,r in zip(predictions,records):
            p=pred[burn:];y=r['y'][burn:]
            if not np.isfinite(p).all():raise ValueError('Nonfinite calibration expert')
            residual=p-y[:,None]
            qt=residual.T@residual/max(float(y@y),1e-24)
            ey=transforms(y,cfg['nperseg'])[:,mask]
            ef=np.stack([transforms(residual[:,j],cfg['nperseg'])[:,mask].ravel() for j in range(p.shape[1])],axis=1)
            qf=(ef.conj().T@ef).real/max(float(np.vdot(ey,ey).real),1e-24)
            power=-r['tau'][burn:]*(y*norm['y_std']+norm['y_mean'])
            rms=max(float(np.sqrt(np.mean(power**2))),1e-12)
            denominator=abs(float(power.mean())) if abs(power.mean())/rms>=cfg['near_zero_energy_ratio'] else rms
            energy=(-r['tau'][burn:,None]*residual*norm['y_std']).mean(0)/denominator if r['policy']!='UM' else np.zeros(p.shape[1])
            self.records.append((qt,qf,energy))

    def value_gradient(self,w):
        value=0.;gradient=np.zeros(len(w))
        for qt,qf,energy in self.records:
            a=max(float(w@qt@w),1e-24);b=max(float(w@qf@w),1e-24)
            joint=(a*b)**.25;e=float(w@energy)
            value+=joint+.25*abs(e)
            gradient+=.5*joint*(qt@w/a+qf@w/b)+.25*np.sign(e)*energy
        return value/len(self.records),gradient/len(self.records)


def objective_stacking(template, records, mask, temporal_weights):
    predictions=[template.predictions(r['x'],r['y']) for r in records]
    objective=MixtureObjective(predictions,records,template.norm,mask)
    m=len(template.experts);identity=np.eye(m)
    best_vertex=min(identity,key=lambda w:objective.value_gradient(w)[0])
    starts=[np.ones(m)/m,np.asarray(temporal_weights),template.prior,best_vertex]
    candidates=[];audit=[]
    for index,start in enumerate(starts):
        start=np.maximum(start,0);start=start/start.sum()
        candidates.append(start)
        fitted=minimize(lambda w:objective.value_gradient(w),start,jac=True,method='SLSQP',
            bounds=[(0.,1.)]*m,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(m)}],
            options={'maxiter':400,'ftol':1e-10})
        feasible=np.isfinite(fitted.x).all() and np.min(fitted.x)>=-1e-7 and abs(fitted.x.sum()-1)<1e-6
        if feasible:
            w=np.maximum(fitted.x,0);candidates.append(w/w.sum())
        audit.append({'start':index,'success':bool(fitted.success),'feasible':bool(feasible),
                      'message':str(fitted.message),'iterations':int(fitted.nit)})
    weights=min(candidates,key=lambda w:objective.value_gradient(w)[0])
    model=fixed_mixture(template,weights,'objective_stacking')
    return model,{'objective':'mean sqrt(R_t R_f) + 0.25 applicable R_E',
                  'calibration_score':objective.value_gradient(weights)[0],
                  'weights':weights.tolist(),'starts':audit,'global_optimum_guaranteed':False,
                  'test_used_for_selection':False}
