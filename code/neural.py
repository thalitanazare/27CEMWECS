"""Teacher-forced initialisation for the additional neural comparator."""
import json,warnings
from dataclasses import dataclass
import numpy as np
from numba import njit
from sklearn.neural_network import MLPRegressor
from polynomial import lags

@njit(cache=True)
def neural_step(z,w1,b1,w2,b2):
    value=b2
    for j in range(len(b1)):
        a=b1[j]
        for i in range(len(z)):a+=z[i]*w1[i,j]
        value+=np.tanh(a)*w2[j]
    return value


@njit(cache=True)
def neural_recurrence(u,y,memory,delay,w1,b1,w2,b2):
    start=max(memory,delay+memory-1);out=np.zeros(len(u));out[:start]=y[:start];z=np.zeros(2*memory)
    for k in range(start,len(u)):
        for i in range(memory):z[i]=out[k-i-1];z[memory+i]=u[k-delay-i]
        value=neural_step(z,w1,b1,w2,b2)
        if not np.isfinite(value) or abs(value)>1e6:out[k:]=np.nan;break
        out[k]=value
    return out


@dataclass
class NeuralModel:
    memory:int
    delay:int
    w1:np.ndarray
    b1:np.ndarray
    w2:np.ndarray
    b2:float
    selection:dict
    method:str='NeuralNARX'
    degree:int=0
    @property
    def start(self):return max(self.memory,self.delay+self.memory-1)
    @property
    def coef(self):return np.r_[self.w1.ravel(),self.b1,self.w2,self.b2]
    def linear_stable(self):return True # No linear-pole claim is made for NN models.
    def simulate(self,u,y):return neural_recurrence(u,y,self.memory,self.delay,self.w1,self.b1,self.w2,self.b2)
    def predict(self,u,y):return np.tanh(lags(u,y,self.memory,self.delay)@self.w1+self.b1)@self.w2+self.b2
    def save(self,path):
        path.write_text(json.dumps({'kind':'neural','memory':self.memory,'delay':self.delay,'w1':self.w1.tolist(),'b1':self.b1.tolist(),'w2':self.w2.tolist(),'b2':float(self.b2),'selection':self.selection},indent=2))
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text());d.pop('kind');
        for k in ['w1','b1','w2']:d[k]=np.array(d[k],dtype=float)
        return cls(**d)


def fit_teacher(train,width,seed,cfg):
    memory=cfg['neural_memory'];delay=cfg['neural_delay']
    xs=[lags(r['u'],r['y'],memory,delay) for r in train]
    ys=[r['y'][-len(x):] for r,x in zip(train,xs)]
    net=MLPRegressor(hidden_layer_sizes=(width,),activation='tanh',solver='lbfgs',alpha=1e-4,max_iter=cfg['neural_teacher_max_iter'],random_state=seed,tol=1e-7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');net.fit(np.vstack(xs),np.concatenate(ys))
    return NeuralModel(memory,delay,net.coefs_[0],net.intercepts_[0],net.coefs_[1].ravel(),float(net.intercepts_[1][0]),{'width':width,'seed':seed,'teacher_iterations':net.n_iter_,'warnings':[str(w.message) for w in caught]})
