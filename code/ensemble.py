"""Past-input context, forward risk labels and record-wise gate calibration."""
import json
import numpy as np
from numba import njit
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from catalogue import load
from common import save
from dataset import metrics
@njit(cache=True)
def context_statistics(x):
    # Causal fixed-window statistics with input autocorrelations as spectral moments.
    n=len(x);out=np.zeros((n,14));window=128
    for k in range(1,n):
        a=max(0,k-window);count=k-a
        for c in range(2):
            mean=np.mean(x[a:k,c]);variance=np.mean((x[a:k,c]-mean)**2);out[k,c*7]=mean;out[k,c*7+1]=np.log(variance+1e-8)
            for j,lag in enumerate([1,4,8,16,32]):
                if count>lag:out[k,c*7+j+2]=np.mean((x[a:k-lag,c]-mean)*(x[a+lag:k,c]-mean))/(variance+1e-8)
    return out


class ContextModel:
    family='spectral_context'
    def __init__(self,experts,expert_norms,input_modes,norm,regressor,temperature,shrinkage,prior,selection):
        self.experts=experts;self.expert_norms=expert_norms;self.input_modes=input_modes;self.norm=norm;self.regressor=regressor;self.temperature=temperature;self.shrinkage=shrinkage;self.prior=np.asarray(prior);self.selection=selection
    def predictions(self,x,y):
        physical=x*np.asarray(self.norm['x_std'])+np.asarray(self.norm['x_mean']);preds=[]
        for model,norm,mode in zip(self.experts,self.expert_norms,self.input_modes):
            u=physical.sum(1,keepdims=True) if mode=='aggregate' else physical;u=(u-np.asarray(norm['x_mean']))/np.asarray(norm['x_std']);initial=(y*self.norm['y_std']+self.norm['y_mean']-norm['y_mean'])/norm['y_std'];p=model.simulate(u,initial)
            preds.append((p*norm['y_std']+norm['y_mean']-self.norm['y_mean'])/self.norm['y_std'])
        return np.column_stack(preds)
    @property
    def parameter_count(self):return sum(e.parameter_count for e in self.experts)
    @property
    def stored_scalars(self):return sum(e.stored_scalars for e in self.experts)+sum(t.tree_.node_count*8 for t in self.regressor.estimators_)
    def simulate(self,x,y):
        preds=self.predictions(x,y)
        if self.shrinkage==1.:
            return preds@self.prior if np.isfinite(preds).all() else np.full(len(y),np.nan)
        context=context_statistics(x);risk=self.regressor.predict(context[:,self.selection.get('context_columns',list(range(14)))]);logw=-risk/self.temperature;logw-=logw.max(1,keepdims=True);w=np.exp(logw);w/=w.sum(1,keepdims=True);w=(1-self.shrinkage)*w+self.shrinkage*self.prior
        # Failed experts are not silently dropped: a divergent core invalidates the portfolio.
        if not np.isfinite(preds).all():return np.full(len(y),np.nan)
        return np.sum(w*preds,axis=1)
    def save(self,path):
        import pickle
        payload={'family':self.family,'selection':self.selection,'expert_files':[],'input_modes':self.input_modes,'expert_norms':self.expert_norms,'norm':self.norm,'temperature':self.temperature,'shrinkage':self.shrinkage,'prior':self.prior.tolist(),'regressor_file':path.with_suffix('.gate.pkl').name}
        for j,e in enumerate(self.experts):
            p=path.with_name(path.stem+f'_expert{j}.json');e.save(p);payload['expert_files'].append(p.name)
        path.with_suffix('.gate.pkl').write_bytes(pickle.dumps(self.regressor));save(path,payload)
    @classmethod
    def load(cls,path):
        import pickle
        d=json.loads(path.read_text());return cls([load(path.parent/p) for p in d['expert_files']],d['expert_norms'],d['input_modes'],d['norm'],pickle.loads((path.parent/d['regressor_file']).read_bytes()),d['temperature'],d['shrinkage'],d['prior'],d['selection'])


def serialnorm(norm):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in norm.items()}

def risk_labels(preds,r,norm,mask,starts):
 errors=(preds-r['y'][:,None])**2;scale=max(float(np.mean(r['y']**2)),1e-8)
 return np.asarray([errors[k:k+32].mean(0)/scale for k in starts])

def softmax(z):
    z=np.asarray(z);e=np.exp(z-z.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)


def fit_gate(experts,norms,modes,norm,val,mask,seed=11,columns=None):
    columns=list(range(14)) if columns is None else list(columns);norm=serialnorm(norm)
    reg=lambda:ExtraTreesRegressor(n_estimators=64,min_samples_leaf=24,random_state=seed,n_jobs=1)
    empty=ContextModel(experts,norms,modes,norm,reg(),.1,0,np.ones(len(experts))/len(experts),{})
    predictions=[empty.predictions(r['x'],r['y']) for r in val]
    contexts=[context_statistics(r['x'])[:,columns] for r in val]
    starts=[np.arange(128,len(r['y'])-32,8) for r in val]
    x=np.vstack([c[k] for c,k in zip(contexts,starts)])
    y=np.vstack([risk_labels(p,r,norm,mask,k) for p,r,k in zip(predictions,val,starts)])
    groups=np.concatenate([np.full(len(k),i) for i,k in enumerate(starts)])
    if not np.isfinite(y).all():raise ValueError('Undefined selected-expert calibration risk; no silent expert removal')
    folds=[]
    for i in range(len(val)):
        fitted=reg().fit(x[groups!=i],y[groups!=i]);risk=fitted.predict(contexts[i]);prior=softmax(-y[groups!=i].mean(0)/.1);folds.append((risk,prior))
    choices=[]
    for temperature in [.03,.1,.3]:
        for shrinkage in [0,.5,1.]:
            scores=[]
            for r,p,(risk,prior) in zip(val,predictions,folds):
                w=(1-shrinkage)*softmax(-risk/temperature)+shrinkage*prior;prediction=(w*p).sum(1)
                class Fixed:
                    def simulate(self,a,b):return prediction
                m=metrics(Fixed(),r,norm,mask)
                scores.append(m['joint_error']+.25*(m['energy_error'] or 0))
            choices.append({'temperature':temperature,'shrinkage':shrinkage,'crossvalidated_validation_score':float(np.mean(scores))})
    best=min(choices,key=lambda c:c['crossvalidated_validation_score']);forest=reg().fit(x,y);prior=softmax(-y.mean(0)/.1)
    selection={'variant':'conditional','cross_validation':choices,'context_columns':columns,'gate_seed':seed,'risk_target':'velocity','risk_horizon':32}
    model=ContextModel(experts,norms,modes,norm,forest,best['temperature'],best['shrinkage'],prior,selection)
    return model,predictions,best


def fixed_mixture(template,weights,variant):
    weights=np.asarray(weights);weights=weights/weights.sum()
    return ContextModel(template.experts,template.expert_norms,template.input_modes,template.norm,
                        template.regressor,template.temperature,1.,weights,{**template.selection,'variant':variant,'context_columns':template.selection.get('context_columns',list(range(14)))})


def global_controls(template,val):
    m=len(template.experts);g=np.zeros((m,m));b=np.zeros(m)
    for r in val:
        p=template.predictions(r['x'],r['y'])[80:];y=r['y'][80:];den=max(float(y@y),1e-12)
        g+=p.T@p/den/len(val);b+=p.T@y/den/len(val)
    fit=minimize(lambda w:float(w@g@w-2*b@w),np.ones(m)/m,jac=lambda w:2*(g@w-b),
                 method='SLSQP',bounds=[(0.,1.)]*m,constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(m)}],options={'maxiter':300,'ftol':1e-12})
    if not fit.success:raise RuntimeError('Global simplex stacking failed: '+fit.message)
    return {'uniform':fixed_mixture(template,np.ones(m),'uniform'),
            'global_prior':fixed_mixture(template,template.prior,'global_prior'),
            'global_stacking':fixed_mixture(template,np.maximum(fit.x,0),'global_stacking')}
