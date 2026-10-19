"""Candidate grid and calibration-only configuration selection. Final code October 2026."""

import hashlib,json
from collections import defaultdict
import numpy as np
from dataset import config
from experts import fit_feature,fit_reservoir,FeatureModel,ReservoirModel
def catalogue():
    rows=[]
    def add(family,choices):
        for mode in config()['input_modes']:
            for c in choices:rows.append({'family':family,'input_mode':mode,**c})
    add('arx',[{'memory':m,'ridge':r} for m in [4,12,24] for r in [.0001,.01]])
    add('fir',[{'memory':m,'ridge':r} for m in [20,40,80] for r in [.0001,.01]])
    add('frols',[{'memory':m,'genes':g} for m in [4,8] for g in [6,10,16]])
    add('kernel',[{'centers':n,'lengthscale':l,'ridge':.01} for n in [32,64] for l in [.5,1,2]])
    add('reservoir',[{'units':n,'leak':leak,'radius':.9} for n in [32,64] for leak in [.2,.7,'multiscale']])
    add('symbolic',[{'memory':4,'genes':g,'depth':d} for g in [3,6] for d in [2,3,4]])
    add('neural',[{'architecture':a,'memory':m,'ridge':r} for a,m in [('legacy',4),('multiscale',6),('tcn',12)] for r in [.0001,.01]])
    add('volterra',[{'memory':m,'components':c,'ridge':.01} for m in [20,40,80] for c in [8,16]])
    for r in rows:r['id']=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()[:12]
    return rows


def fit(family,settings,train,seed):return fit_reservoir(settings,train,seed) if family=='reservoir' else fit_feature(family,settings,train,seed)

def load(path):
 d=json.loads(path.read_text(encoding='utf-8'))
 return (ReservoirModel if d['family']=='reservoir' else FeatureModel).load(path)

def selected(rows,family,layout,legacy=False):
    cfg=config();groups=defaultdict(list)
    for r in rows:
        c=r['configuration']
        if c['family']!=family or r['layout']!=layout:continue
        if legacy and c.get('architecture')!='legacy':continue
        groups[c['id']].append(r)
    valid={k:v for k,v in groups.items() if len(v)>=(1 if family in ['arx','fir','frols','volterra'] else len(cfg['seeds']))}
    if not valid:return []
    def score(rs):return float(np.mean([r.get('validation_score') if r.get('validation_score') is not None else np.inf for r in rs]))
    return min(valid.values(),key=score)
