"""The eight independently simulated model families used in the article."""
import json,warnings
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.linalg import solve
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
import torch

@njit(cache=True)
def feature_vector(x,out,k,ylags,ulags):
    z=np.empty(len(ylags)+len(ulags)*x.shape[1]);j=0
    for lag in ylags:z[j]=out[k-lag];j+=1
    for c in range(x.shape[1]):
        for lag in ulags:z[j]=x[k-lag,c];j+=1
    return z


@njit(cache=True)
def tree_values(z,program):
    result=np.empty(program.shape[0]);nodes=np.empty(program.shape[1])
    for g in range(program.shape[0]):
        root=0.
        for j in range(program.shape[1]):
            op=int(program[g,j,0]);a=int(program[g,j,1]);b=int(program[g,j,2])
            if op<0:break
            if op==0:nodes[j]=z[a]
            elif op==1:nodes[j]=nodes[a]+nodes[b]
            elif op==2:nodes[j]=nodes[a]-nodes[b]
            elif op==3:nodes[j]=nodes[a]*nodes[b]
            elif op==4:nodes[j]=np.tanh(nodes[a])
            else:nodes[j]=program[g,j,2]
            root=nodes[j]
        result[g]=root
    return result


@njit(cache=True)
def value(kind,z,coef,centers,alpha,projection,pairs,program,w1,b1,w2,conv,memory):
    if kind==4:return np.tanh(z@w1+b1)@w2+coef[0]
    if kind==5:
        hidden=np.zeros((memory,len(b1)));channels=len(z)//memory
        for pos in range(memory):
            for h in range(len(b1)):
                a=b1[h]
                for c in range(channels):
                    for lag in range(conv.shape[0]):
                        if pos+lag<memory:a+=z[c*memory+pos+lag]*conv[lag,c,h]
                hidden[pos,h]=np.tanh(a)
        return hidden.reshape(memory*len(b1))@w2+coef[0]
    if kind==2:
        a=z@projection;q=alpha[0]+a@alpha[1:1+len(a)]
        for j in range(len(pairs)):q+=alpha[1+len(a)+j]*a[pairs[j,0]]*a[pairs[j,1]]
        return q
    q=coef[0]+z@coef[1:]
    if kind==1:
        for j in range(len(centers)):q+=np.exp(-.5*np.sum((z-centers[j])**2))*alpha[j]
    elif kind==3:q+=tree_values(z,program)@alpha
    return q


@njit(cache=True)
def rollout(kind,x,y,ylags,ulags,mean,scale,coef,centers,alpha,projection,pairs,program,w1,b1,w2,conv,memory):
    start=max(np.max(ulags),np.max(ylags) if len(ylags) else 0);out=np.zeros(len(y));out[:start]=y[:start]
    for k in range(start,len(y)):
        z=(feature_vector(x,out,k,ylags,ulags)-mean)/scale
        out[k]=value(kind,z,coef,centers,alpha,projection,pairs,program,w1,b1,w2,conv,memory)
        if not np.isfinite(out[k]) or abs(out[k])>1e6:out[k:]=np.nan;break
    return out


@dataclass
class FeatureModel:
    family:str
    kind:int
    ylags:np.ndarray
    ulags:np.ndarray
    mean:np.ndarray
    scale:np.ndarray
    coef:np.ndarray
    centers:np.ndarray
    alpha:np.ndarray
    projection:np.ndarray
    pairs:np.ndarray
    program:np.ndarray
    w1:np.ndarray
    b1:np.ndarray
    w2:np.ndarray
    conv:np.ndarray
    memory:int
    selection:dict
    @property
    def start(self):return int(max(max(self.ulags),max(self.ylags) if len(self.ylags) else 0))
    @property
    def parameter_count(self):
        if self.kind==4:return int(self.w1.size+self.b1.size+self.w2.size+1)
        if self.kind==5:return int(self.conv.size+self.b1.size+self.w2.size+1)
        if self.kind==2:return int(self.alpha.size)
        return int(self.coef.size+self.alpha.size)
    @property
    def stored_scalars(self):return int(sum(getattr(self,k).size for k in ['mean','scale','coef','centers','alpha','projection','pairs','program','w1','b1','w2','conv']))
    def simulate(self,x,y):return rollout(self.kind,np.ascontiguousarray(x),np.ascontiguousarray(y),self.ylags,self.ulags,self.mean,self.scale,self.coef,self.centers,self.alpha,self.projection,self.pairs,self.program,self.w1,self.b1,self.w2,self.conv,self.memory)
    def save(self,path):path.write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(self).items()}))
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text())
        for k in ['ylags','ulags','mean','scale','coef','centers','alpha','projection','pairs','program','w1','b1','w2','conv']:d[k]=np.asarray(d[k],dtype=np.int64 if k in ['ylags','ulags','pairs'] else np.float64)
        # JSON loses empty-array trailing dimensions.
        d['centers']=d['centers'].reshape(-1,len(d['mean']));d['projection']=d['projection'].reshape(len(d['mean']),-1);d['pairs']=d['pairs'].reshape(-1,2);d['program']=d['program'].reshape(-1,d['selection'].get('program_nodes',1),3);d['w1']=d['w1'].reshape(len(d['mean']),-1);d['conv']=d['conv'].reshape(d['selection'].get('conv_shape',[0,0,0]))
        return cls(**d)


def design(records,ylags,ulags):
    xs=[];ys=[];start=int(max(max(ulags),max(ylags) if len(ylags) else 0))
    for r in records:
        n=len(r['y']);xs.append(np.column_stack([r['y'][start-l:n-l] for l in ylags]+[r['x'][start-l:n-l,c] for c in range(r['x'].shape[1]) for l in ulags]));ys.append(r['y'][start:])
    return np.vstack(xs),np.concatenate(ys)


def blank(family,kind,ylags,ulags,x,selection):
    d=x.shape[1];mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    return FeatureModel(family,kind,np.asarray(ylags,dtype=np.int64),np.asarray(ulags,dtype=np.int64),mean,scale,np.zeros(d+1),np.empty((0,d)),np.empty(0),np.empty((d,0)),np.empty((0,2),dtype=np.int64),np.empty((0,1,3)),np.empty((d,0)),np.empty(0),np.empty(0),np.empty((0,0,0)),len(ulags),selection)


def ridge(x,y,penalty):
    a=np.column_stack([np.ones(len(x)),x]);p=np.eye(a.shape[1])*penalty;p[0,0]=1e-10
    return np.linalg.solve(a.T@a/len(a)+p,a.T@y/len(a))


def random_tree(rng,d,depth):
    if depth<=0 or rng.random()<.3:return (0,int(rng.integers(d)))
    op=int(rng.choice([1,2,3,4]));return (op,random_tree(rng,d,depth-1)) if op==4 else (op,random_tree(rng,d,depth-1),random_tree(rng,d,depth-1))


def pack_trees(trees):
    programs=[]
    for tree in trees:
        nodes=[]
        def visit(t):
            if t[0]==0:nodes.append([0,t[1],0]);return len(nodes)-1
            a=visit(t[1]);b=visit(t[2]) if len(t)==3 else 0;nodes.append([t[0],a,b]);return len(nodes)-1
        visit(tree);programs.append(nodes)
    size=max(map(len,programs));result=np.full((len(trees),size,3),-1.,dtype=float)
    for i,p in enumerate(programs):result[i,:len(p)]=p
    return result


def eval_trees(x,program):return np.asarray([tree_values(z,program) for z in x])


def fit_feature(family,settings,train,seed):
    rng=np.random.default_rng(seed);settings=copy_settings={**settings,'seed':seed}
    if family in ['arx','frols','symbolic']:lags=np.arange(1,settings.get('memory',4)+1);ylags=lags;ulags=lags
    elif family=='volterra':ylags=np.array([],dtype=int);ulags=np.arange(1,settings['memory']+1)
    elif family=='fir':ylags=np.array([],dtype=int);ulags=np.arange(1,settings['memory']+1)
    elif family=='neural':
        lags=np.arange(1,settings['memory']+1) if settings['architecture'] in ['legacy','tcn'] else np.array([1,2,4,8,16,32]);ylags=lags;ulags=lags
    else:ylags=np.array([1,2,4,8,16,32]);ulags=ylags.copy()
    x,y=design(train,ylags,ulags);kind={'arx':0,'fir':0,'frols':3,'symbolic':3,'kernel':1,'volterra':2,'neural':4}[family];model=blank(family,kind,ylags,ulags,x,copy_settings);z=(x-model.mean)/model.scale
    penalty=settings.get('ridge',.01)
    if family in ['arx','fir']:model.coef=ridge(z,y,penalty)
    elif family=='kernel':
        # Sparse GP with a fixed ridge-ARX mean and a DTC RBF residual posterior.
        model.coef=ridge(z,y,.01);res=y-(np.column_stack([np.ones(len(z)),z])@model.coef)
        length=settings['lengthscale']*np.sqrt(z.shape[1]);model.scale*=length;z/=length;model.coef[1:]*=length
        centers=z[rng.choice(len(z),min(settings['centers'],len(z)),replace=False)];k=np.exp(-.5*np.sum((z[:,None]-centers[None])**2,axis=2));kmm=np.exp(-.5*np.sum((centers[:,None]-centers[None])**2,axis=2))+np.eye(len(centers))*1e-6
        model.centers=centers;model.alpha=np.linalg.solve(k.T@k/len(k)+penalty*kmm,k.T@res/len(k));model.selection['gp_approximation']='DTC posterior mean, data-fitted ARX mean; variance not used for ranking'
    elif family=='volterra':
        # PCA of causal input histories; quadratic expansion is a low-rank Volterra kernel.
        cov=z.T@z/len(z);eig,v=np.linalg.eigh(cov);ids=np.argsort(eig)[::-1][:settings['components']];model.projection=v[:,ids];a=z@model.projection
        model.pairs=np.asarray([(i,j) for i in range(a.shape[1]) for j in range(i,a.shape[1])],dtype=np.int64);phi=np.column_stack([a]+[a[:,i]*a[:,j] for i,j in model.pairs]);model.alpha=ridge(phi,y,penalty);model.coef=np.zeros(len(model.mean)+1)
    elif family in ['symbolic','frols']:
        model.coef=ridge(z,y,.01);res=y-np.column_stack([np.ones(len(z)),z])@model.coef
        sample=np.sort(rng.choice(len(z),min(len(z),1600),replace=False));zz=z[sample];rr=res[sample]
        if family=='frols':
            from polynomial import frols
            trees=[(3,(0,i),(0,j)) for i in range(z.shape[1]) for j in range(i,z.shape[1])];program=pack_trees(trees);phi=eval_trees(zz,program);ids,_=frols(phi,rr,settings.get('genes',10));chosen=[trees[i] for i in ids]
        else:
            population=[[random_tree(rng,z.shape[1],settings['depth']) for _ in range(settings['genes'])] for _ in range(24)];best=None
            for generation in range(8):
                ranked=[]
                for trees in population:
                    p=pack_trees(trees);phi=eval_trees(zz,p);coef=ridge(phi,rr,.01);err=rr-np.column_stack([np.ones(len(phi)),phi])@coef;fitness=float(np.mean(err**2)+1e-6*np.sum(p[:,:,0]>=0));ranked.append((fitness,trees))
                ranked.sort(key=lambda a:a[0]);best=ranked[0][1];elite=[a[1] for a in ranked[:6]];population=copy_population(elite)
                while len(population)<24:
                    a=elite[int(rng.integers(len(elite)))];b=elite[int(rng.integers(len(elite)))];child=[a[j] if rng.random()<.5 else b[j] for j in range(len(a))]
                    child[int(rng.integers(len(child)))]=random_tree(rng,z.shape[1],settings['depth']);population.append(child)
            chosen=best;model.selection['evolution']={'population':24,'generations':8,'fitness':'training temporal residual error plus node count','crossover':'gene-wise','mutation':'random subtree replacement at gene root'}
        model.program=pack_trees(chosen);model.selection['program_nodes']=model.program.shape[1];phi=eval_trees(z,model.program);c=ridge(phi,res,.01);model.coef[0]+=c[0];model.alpha=c[1:]
    elif family=='neural':
        if settings['architecture']!='tcn':
            net=MLPRegressor(hidden_layer_sizes=(settings.get('width',16),),activation='tanh',solver='lbfgs',alpha=penalty,max_iter=350,random_state=seed,tol=1e-7)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');net.fit(z,y)
            model.w1=net.coefs_[0];model.b1=net.intercepts_[0];model.w2=net.coefs_[1].ravel();model.coef=np.zeros(len(model.mean)+1);model.coef[0]=net.intercepts_[1][0];model.selection['warnings']=[str(w.message) for w in caught]
        else:fit_tcn(model,z,y,settings,seed)
    return model


def copy_population(elite):return [list(a) for a in elite]


def fit_tcn(model,z,y,settings,seed):
    torch.set_num_threads(1);torch.manual_seed(seed);m=settings['memory'];channels=z.shape[1]//m;h=8
    inputs=torch.tensor(z.reshape(len(z),channels,m).copy(),dtype=torch.float32);targets=torch.tensor(y,dtype=torch.float32)
    # right-padding over lag coordinates preserves causal access to older samples.
    conv=torch.nn.Conv1d(channels,h,3);readout=torch.nn.Linear(h*m,1);optimizer=torch.optim.Adam(list(conv.parameters())+list(readout.parameters()),lr=.003,weight_decay=settings['ridge']);rng=np.random.default_rng(seed)
    for _ in range(200):
        ids=rng.integers(len(z),size=256);optimizer.zero_grad();hidden=torch.tanh(conv(torch.nn.functional.pad(inputs[ids],(0,2))));p=readout(hidden.reshape(len(ids),-1))[:,0];loss=(p-targets[ids]).square().mean();loss.backward();optimizer.step()
    model.kind=5;model.conv=conv.weight.detach().numpy().transpose(2,1,0).astype(float).copy();model.b1=conv.bias.detach().numpy().astype(float).copy();model.w2=readout.weight.detach().numpy().reshape(h,m).T.reshape(-1).astype(float).copy();model.coef=np.zeros(len(model.mean)+1);model.coef[0]=float(readout.bias.detach()[0]);model.selection['conv_shape']=list(model.conv.shape);model.selection['teacher_updates']=200


@njit(cache=True)
def reservoir_rollout(x,y,win,w,leak,coef,memory,teacher):
    n=len(y);out=np.zeros(n);out[:memory]=y[:memory];state=np.zeros(len(w));features=np.empty(1+memory*(1+x.shape[1])+len(w));z=np.empty(2+x.shape[1]);stored=np.empty((n-memory,len(features)))
    for k in range(memory,n):
        z[0]=1.;z[1]=y[k-1] if teacher else out[k-1];z[2:]=x[k-1];state=(1-leak)*state+leak*np.tanh(w@state+win@z)
        features[0]=1.;j=1
        for lag in range(1,memory+1):features[j]=y[k-lag] if teacher else out[k-lag];j+=1
        for c in range(x.shape[1]):
            for lag in range(1,memory+1):features[j]=x[k-lag,c];j+=1
        features[j:]=state;out[k]=features@coef;stored[k-memory]=features
        if not teacher and (not np.isfinite(out[k]) or abs(out[k])>1e6):out[k:]=np.nan;break
    return out,stored


@dataclass
class ReservoirModel:
    win:np.ndarray
    w:np.ndarray
    leak:np.ndarray
    coef:np.ndarray
    memory:int
    selection:dict
    family:str='reservoir'
    @property
    def parameter_count(self):return self.coef.size
    @property
    def stored_scalars(self):return self.win.size+self.w.size+self.leak.size+self.coef.size
    def simulate(self,x,y):return reservoir_rollout(x,y,self.win,self.w,self.leak,self.coef,self.memory,False)[0]
    def save(self,path):path.write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(self).items()}))
    @classmethod
    def load(cls,path):
        d=json.loads(path.read_text())
        for k in ['win','w','leak','coef']:d[k]=np.array(d[k])
        return cls(**d)


def fit_reservoir(settings,train,seed):
    rng=np.random.default_rng(seed);n=settings['units'];w=rng.normal(size=(n,n));w[rng.random(w.shape)>.15]=0;radius=max(abs(np.linalg.eigvals(w)));w*=settings['radius']/max(radius,1e-8);win=rng.normal(size=(n,2+train[0]['x'].shape[1]))*.2
    leak=np.geomspace(.1,1,n) if settings['leak']=='multiscale' else np.full(n,settings['leak']);m=4;coef=np.zeros(1+m*(1+train[0]['x'].shape[1])+n);xx=[];yy=[]
    for r in train:
        _,phi=reservoir_rollout(r['x'],r['y'],win,w,leak,coef,m,True);xx.append(phi[80:]);yy.append(r['y'][m+80:])
    x=np.vstack(xx);y=np.concatenate(yy);p=np.eye(len(coef))*.01;p[0,0]=1e-10;coef=np.linalg.solve(x.T@x/len(x)+p,x.T@y/len(x))
    return ReservoirModel(win,w,leak,coef,m,{**settings,'seed':seed,'reservoir_weights_fixed':True,'ridge':.01})
