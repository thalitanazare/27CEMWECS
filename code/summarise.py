"""Summarise paired metrics using the six shared wave-condition clusters."""
import argparse
import csv
from pathlib import Path
import numpy as np
from common import ROOT
from diagnostics import clustered
from run import write_csv


def summarise(paths):
    indexed={}
    for path in paths:
        with path.open(newline='',encoding='utf-8-sig') as handle:
            for r in csv.DictReader(handle):
                key=(r['scope'],r['layout'],r['state'],r['policy'],r['variant'])
                if key in indexed:raise ValueError('Duplicate record: '+str(key))
                indexed[key]=r
    output=[]
    for scope in sorted({k[0] for k in indexed}):
        for phase,layouts in [('development',['L0','L1','L2']),('holdout',['L3','L4','L5'])]:
            variants=sorted({k[4] for k in indexed if k[0]==scope and k[1] in layouts}-{'ensemble'})
            for variant in variants:
                values=[];failed_ensemble=failed_reference=0
                for key,r in sorted(indexed.items()):
                    if key[0]!=scope or key[1] not in layouts or key[4]!='ensemble':continue
                    alt=indexed.get((*key[:4],variant))
                    if alt is None:continue
                    failed_ensemble+=r['stable'].lower()!='true'
                    failed_reference+=alt['stable'].lower()!='true'
                    if r['stable'].lower()!='true' or alt['stable'].lower()!='true':continue
                    a,b=float(r['joint_error']),float(alt['joint_error'])
                    if not np.isfinite([a,b]).all() or b<=0:continue
                    values.append({'state':r['state'],'policy':r['policy'],'reduction_percent':100*(1-a/b)})
                if values:
                    output.append({'scope':scope,'phase':phase,'comparison':variant,
                         'wins':sum(r['reduction_percent']>0 for r in values),
                         'ensemble_failures':failed_ensemble,'reference_failures':failed_reference,**clustered(values)})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,action='append',help='One or more metrics.csv files. Defaults to reference/metrics.csv.')
    p.add_argument('--output',type=Path,default=ROOT/'results/summary.csv')
    args=p.parse_args()
    rows=summarise(args.input or [ROOT/'reference/metrics.csv'])
    if not rows:raise RuntimeError('No finite paired comparisons.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_csv(args.output,rows)
    for r in rows:
        if r['scope']=='matched':
            print(f"{r['phase']:11s} {r['comparison']:20s} {r['median_reduction_percent']:6.1f}% "
                  f"({r['wins']}/{r['records']}) p={r['signflip_p_two_sided']:.3f}")
    print('Saved',args.output)

if __name__=='__main__':main()