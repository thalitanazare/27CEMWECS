"""Full polynomial NARMAX reference with FROLS and extended least squares."""
import itertools,json
import numpy as np
from numba import njit
from polynomial import frols
from dataset import score
LAG_SETS={'short':list(range(1,9)),'medium':list(range(1,9))+[10,12,16,20,24],'long':list(range(1,9))+[12,16,24,32,40]}
DICTIONARIES=[(lags,2,mode) for lags in LAG_SETS for mode in ['aggregate','separate']]+[('short',3,'aggregate')]
TERM_COUNTS=[5,10,15,20,25,30,35,40,50,60]
NOISE_LAGS=5
ELS_ITERATIONS=3

def regressors(lags,channels):
    """Regressor list as (source, lag); source 0 is the output, 1.. are input channels."""
    return [(0,l) for l in lags]+[(c+1,l) for c in range(channels) for l in lags]


def monomials(nreg,degree):
    """Exponent rows over the regressors for all monomials of total degree 0..degree."""
    rows=[]
    for d in range(degree+1):
        for combo in itertools.combinations_with_replacement(range(nreg),d):
            p=np.zeros(nreg,dtype=np.int64)
            for i in combo:p[i]+=1
            rows.append(p)
    return np.array(rows)


def lagged(r,regs):
    """Regressor matrix of one record, starting at the largest lag."""
    start=max(l for _,l in regs);n=len(r['y']);signals=[r['y']]+[r['x'][:,c] for c in range(r['x'].shape[1])]
    return np.column_stack([signals[s][start-l:n-l] for s,l in regs]),r['y'][start:],start


def design(R,powers):
    out=np.ones((len(R),len(powers)))
    for j,p in enumerate(powers):
        for i in np.flatnonzero(p):out[:,j]*=R[:,i]**p[i]
    return out


@njit(cache=True)
def recurrence(x,y,src,lag,powers,coef,start):
    n=len(y);out=np.zeros(n);out[:start]=y[:start];reg=np.zeros(len(src))
    for k in range(start,n):
        for i in range(len(src)):
            reg[i]=out[k-lag[i]] if src[i]==0 else x[k-lag[i],src[i]-1]
        value=0.
        for j in range(len(coef)):
            term=coef[j]
            for i in range(len(src)):
                p=powers[j,i]
                if p:term*=reg[i]**p
            value+=term
        # Numerical divergence guard, never a saturation of the trajectory.
        if not np.isfinite(value) or abs(value)>1e6:
            out[k:]=np.nan
            break
        out[k]=value
    return out


class NarmaxModel:
    def __init__(self,regs,powers,coef,selection):
        self.regs=[tuple(r) for r in regs];self.powers=np.asarray(powers,dtype=np.int64);self.coef=np.asarray(coef,float);self.selection=selection
        self.src=np.array([s for s,_ in self.regs],dtype=np.int64);self.lag=np.array([l for _,l in self.regs],dtype=np.int64)
        self.parameters=len(self.coef)

    def simulate(self,x,y):
        return recurrence(np.ascontiguousarray(x,dtype=float),np.asarray(y,float),self.src,self.lag,self.powers,self.coef,int(self.lag.max()))

    def terms(self):
        names=['y' if s==0 else ('u' if len(set(self.src))==2 else ['','tau_e','tau_u'][s]) for s in self.src]
        out=[]
        for p,c in zip(self.powers,self.coef):
            factors=[f"{names[i]}(k-{self.lag[i]})"+(f"^{p[i]}" if p[i]>1 else '') for i in np.flatnonzero(p)]
            out.append({'term':'*'.join(factors) or '1','coefficient':float(c)})
        return out

    def save(self,path):
        path.write_text(json.dumps({'regs':self.regs,'powers':self.powers.tolist(),'coef':self.coef.tolist(),'selection':self.selection},indent=1))


def els(blocks,ids,iterations=ELS_ITERATIONS,noise_lags=NOISE_LAGS):
    """Extended least squares: process terms ids plus linear MA terms of the one-step residual, per record."""
    theta=np.linalg.lstsq(np.vstack([P[:,ids] for P,_ in blocks]),np.concatenate([t for _,t in blocks]),rcond=None)[0]
    for _ in range(iterations):
        rows=[];targets=[]
        for P,t in blocks:
            e=t-P[:,ids]@theta[:len(ids)]
            E=np.column_stack([np.r_[np.zeros(l),e[:-l]] for l in range(1,noise_lags+1)])
            rows.append(np.column_stack([P[:,ids],E]));targets.append(t)
        theta=np.linalg.lstsq(np.vstack(rows),np.concatenate(targets),rcond=None)[0]
    return theta[:len(ids)],theta[len(ids):]


def fit_dictionary(train,val,norm,mask,lagname,degree):
    regs=regressors(LAG_SETS[lagname],train[0]['x'].shape[1]);powers=monomials(len(regs),degree)
    blocks=[]
    for r in train:
        R,t,_=lagged(r,regs);blocks.append((design(R,powers),t))
    phi=np.vstack([P for P,_ in blocks]);target=np.concatenate([t for _,t in blocks])
    ids,errs=frols(phi,target,max(TERM_COUNTS));del phi
    candidates=[]
    for n in TERM_COUNTS:
        chosen=ids[:n];coef,noise=els(blocks,chosen)
        model=NarmaxModel(regs,powers[chosen],coef,{'lags':lagname,'degree':degree,'terms':n,'ERR':errs[:n],'noise_coefficients':noise.tolist(),'dictionary_size':len(powers)})
        candidates.append((score(model,val,norm,mask),model))
    return candidates
