"""Measured-input replay of the released OCEAN model on native layout L0."""
import numpy as np
from scipy.io import loadmat
from scipy.signal import cont2discrete,resample_poly
from common import ROOT
from dataset import config,channel
class Ocean:
    def __init__(self,fs):
        d=loadmat(ROOT/'data/reference/OCEAN_L0.mat',simplify_cells=True)
        self.a=d['A'];self.b=d['B'].reshape(-1,1);self.c=d['C'].reshape(1,-1);self.fs=fs
        self.ad,self.bd,self.cd,_,_=cont2discrete((self.a,self.b,self.c,np.zeros((1,1))),1/fs)

class OceanReplay:
    """Simulates OCEAN at the raw sampling rate over the complete record, then applies the measurement
    trimming, anti-alias resampling and edge removal of dataset.read to the prediction."""
    def __init__(self,record,norm):
        self.record=record;self.norm=norm

    def simulate(self,x,y):
        cfg=config();d=loadmat(ROOT/self.record['file'],simplify_cells=True);t=np.asarray(d['time']);n=len(t)
        tag={'UM':'UM','P':'CM_P','PI':'CM_PI'}[self.record['policy']];fs=1/np.median(np.diff(t))
        u=channel(d['excitationTorque_WE'],n)+(0 if self.record['policy']=='UM' else channel(d['controlTorque_'+tag],n))
        ocean=Ocean(fs);ad,bd,cd=ocean.ad,ocean.bd.ravel(),ocean.cd.ravel();state=np.zeros(len(bd));v=np.zeros(n)
        for k in range(n):
            v[k]=cd@state;state=ad@state+bd*u[k]
        keep=(t>=cfg['trim_start_s'])&(t<=t[-1]-cfg['trim_end_s']);edge=2*cfg['sample_rate_hz']
        v=resample_poly(v[keep],1,int(round(fs/cfg['sample_rate_hz'])))[edge:-edge]
        if len(v)!=len(self.record['y']):raise ValueError('Replay/record length mismatch')
        return (v-self.norm['y_mean'])/self.norm['y_std']
