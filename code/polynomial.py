"""Lagged regressors and orthogonal term ranking."""
import numpy as np

def lags(u,y,memory,delay):
    start=max(memory,delay+memory-1)
    return np.column_stack([y[start-i-1:len(y)-i-1] for i in range(memory)]+
                           [u[start-delay-i:len(u)-delay-i] for i in range(memory)])


def frols(phi,y,max_terms):
    """Modified Gram-Schmidt with reorthogonalisation; ERR normalized by y'y."""
    remaining=phi.copy(); selected=[]; errs=[]; energy=float(y@y)
    for _ in range(min(max_terms,phi.shape[1])):
        norms=np.sum(remaining*remaining,axis=0)
        scores=(remaining.T@y)**2/np.maximum(norms,1e-24)/max(energy,1e-24)
        scores[norms<1e-20]=-np.inf
        if selected:scores[selected]=-np.inf
        j=int(np.argmax(scores))
        if not np.isfinite(scores[j]):break
        selected.append(j);errs.append(float(scores[j]));q=remaining[:,j]/np.sqrt(norms[j])
        for _ in range(2): remaining-=q[:,None]*(q@remaining)[None,:]
    return selected,errs
