"""Fit and evaluate only the experiments retained in the article.

Examples:
    python code/run.py --layout L3
    python code/run.py --layout L3 --exclude ISS1
    python code/run.py --layout L0 --ocean
"""
import argparse
import csv
import json
import traceback
from pathlib import Path
import numpy as np
from common import ROOT, save
from dataset import prepare, score, metrics, config
from catalogue import catalogue, fit, load, selected
from ensemble import ContextModel, serialnorm, fit_gate, global_controls, fixed_mixture, context_statistics
from stacking import objective_stacking
from diagnostics import diagnostics

FAMILIES = ['arx','fir','frols','kernel','reservoir','symbolic','neural','volterra']
DETERMINISTIC = {'arx','fir','frols','volterra'}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_candidates(layout, held, folder):
    rows = []
    for c in catalogue():
        for seed in ([11] if c['family'] in DETERMINISTIC else config()['seeds']):
            path = folder / 'candidates' / f"{c['id']}_{seed}.json"
            if path.exists():
                rows.append(json.loads(path.read_text(encoding='utf-8')))
                continue
            row = {'layout':layout, 'configuration':c, 'seed':seed, 'validation_score':None}
            try:
                train, val, _, norm, mask, audit = prepare(layout, held, c['input_mode'], broadband=True)
                model = fit(c['family'], c, train, seed)
                loss = score(model, val, norm, mask)
                target = path.with_name(path.stem + '_model.json')
                target.parent.mkdir(parents=True, exist_ok=True)
                model.save(target)
                row.update(model_file=str(target.relative_to(ROOT)),
                           validation_score=float(loss) if np.isfinite(loss) else None)
                save(folder / ('partition_' + c['input_mode'] + '.json'), audit)
            except Exception as exc:
                row['error'] = repr(exc)
                row['traceback'] = traceback.format_exc()
            save(path, row)
            rows.append(row)
            print(layout, held or 'matched', c['family'], seed, row['validation_score'], flush=True)
    return rows


def refined_candidates(layout, held, folder):
    """Fifteen comparator fits, outside the heterogeneous expert bank."""
    from neural import fit_teacher
    from refinement import refine, convert
    cfg = json.loads((ROOT/'code/neural_config.json').read_text())
    rcfg = json.loads((ROOT/'code/refinement_config.json').read_text())
    train,val,_,norm,mask,_ = prepare(layout,held,'aggregate',broadband=True)
    tr = [{**r,'u':r['x'][:,0]} for r in train]
    va = [{**r,'u':r['x'][:,0]} for r in val]
    rows = []
    for seed in config()['seeds']:
        paths = [folder/'refined'/f'{o}_{b}_{seed}.json' for o,b in
                 [('temporal',0),('spectral',25),('spectral',50),('power_spectral',25),('power_spectral',50)]]
        if all(p.exists() for p in paths):
            rows.extend(json.loads(p.read_text()) for p in paths)
            continue
        net = fit_teacher(tr,16,seed,cfg)
        fits = [('temporal',0,net)]
        for objective in ['spectral','power_spectral']:
            checkpoints,_ = refine(net,tr,va,mask,norm,cfg,rcfg,objective=='power_spectral')
            fits.extend((objective,b,m) for b,(m,_) in checkpoints.items())
        for objective,budget,net in fits:
            c = {'family':'neural_refined','input_mode':'aggregate','objective':objective,
                 'budget':budget,'id':f'neural_{objective}_{budget}'}
            path = folder/'refined'/f'{objective}_{budget}_{seed}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            model = convert(net,train)
            modelpath = path.with_name(path.stem+'_model.json')
            model.save(modelpath)
            loss = score(model,val,norm,mask)
            row = {'layout':layout,'configuration':c,'seed':seed,
                   'validation_score':float(loss) if np.isfinite(loss) else None,
                   'model_file':str(modelpath.relative_to(ROOT))}
            save(path,row)
            rows.append(row)
    return rows


def assemble(choices, layout, held):
    experts, norms, modes = [], [], []
    for choice in choices:
        for row in choice:
            mode = row['configuration']['input_mode']
            _,_,_,norm,_,_ = prepare(layout,held,mode,broadband=True)
            experts.append(load(ROOT/row['model_file']))
            norms.append(serialnorm(norm))
            modes.append(mode)
    return experts,norms,modes


def mean_metrics(values):
    stable = all(v['stable'] for v in values)
    return {'stable':stable, **{k:float(np.mean([v[k] for v in values]))
            if stable and all(v.get(k) is not None for v in values) else None
            for k in ['nrmse','spectral_error','joint_error','energy_error']}}


def evaluate(layout, held, narmax=False, ablations=False, ocean=False):
    folder = ROOT/'results'/layout/(held or 'matched')
    folder.mkdir(parents=True,exist_ok=True)
    settings = {'layout':layout,'excluded_state':held,'protocol':config(),
                'test_used_for_selection':False}
    save(folder/'configuration.json',settings)
    rows = train_candidates(layout,held,folder)
    choices = {f:selected(rows,f,layout) for f in FAMILIES}
    if any(not rs or any(r['validation_score'] is None for r in rs) for rs in choices.values()):
        raise RuntimeError('A family has no complete finite calibration-selected configuration.')
    experts,norms,modes = assemble(choices.values(),layout,held)
    _,val,test,norm,mask,audit = prepare(layout,held,'separate',broadband=True)
    save(folder/'partition.json',audit)
    main,_,gate_choice = fit_gate(experts,norms,modes,norm,val,mask)
    main.selection['expert_descriptions'] = [
        {'family':r['configuration']['family'],'seed':r['seed'],'configuration':r['configuration']}
        for rs in choices.values() for r in rs]
    controls = global_controls(main,val)
    joint,stack_audit = objective_stacking(main,val,mask,controls['global_stacking'].prior)
    models = {'ensemble':main,'uniform':controls['uniform'],'prior':controls['global_prior'],
              'temporal_stacking':controls['global_stacking'],'joint_stacking':joint}
    refined = refined_candidates(layout,held,folder)
    refined_choice = selected(refined,'neural_refined',layout)
    baseline_choices = [choices[f] for f in ['arx','fir','frols']]
    baseline_choices += [selected(rows,'neural',layout,True),refined_choice]
    baseline_choices = [rs for rs in baseline_choices if rs and all(r['validation_score'] is not None for r in rs)]
    baseline = min(baseline_choices,key=lambda rs:np.mean([r['validation_score'] for r in rs]))
    save(folder/'selection.json',{'gate':gate_choice,'stacking':stack_audit,
         'experts':main.selection['expert_descriptions'],'baseline':baseline[0]['configuration'],
         'test_used_for_selection':False})
    main.save(folder/'ensemble.json')
    output, ranks = [], []
    def emit(variant,record,value):
        output.append({'scope':'excluded_'+held if held else 'matched','layout':layout,
                       'state':record['state'],'policy':record['policy'],'variant':variant,**value})
    for variant,model in models.items():
        for r in test:emit(variant,r,metrics(model,r,norm,mask))
    for family,rs in list(choices.items()) + [('neural_refined',refined_choice),('baseline',baseline)]:
        ee,nn,mm = assemble([rs],layout,held)
        for r in test:
            values = [metrics(ContextModel([e],[n],[m],serialnorm(norm),main.regressor,.1,1.,np.ones(1),{}),r,norm,mask)
                      for e,n,m in zip(ee,nn,mm)]
            value=mean_metrics(values)
            if family=='baseline' and value['stable']:
                value['joint_error']=float(np.sqrt(value['nrmse']*value['spectral_error']))
            emit(family,r,value)
    for r in test:
        predictions=main.predictions(r['x'],r['y'])
        risk=main.regressor.predict(context_statistics(r['x']))
        ranks.append(diagnostics(risk,predictions,r,norm,mask,main))
    write_csv(folder/'risk_ranks.csv',ranks)
    if ablations:
        # Remove a family or a context channel, then recalibrate using the same records.
        variants = [('excitation_only',list(range(7)),list(range(len(experts)))),
                    ('pto_only',list(range(7,14)),list(range(len(experts))))]
        families=[d['family'] for d in main.selection['expert_descriptions']]
        variants += [('without_'+f,list(range(14)),[j for j,v in enumerate(families) if v!=f]) for f in FAMILIES]
        for name,columns,kept in variants:
            model,_,_ = fit_gate([experts[j] for j in kept],[norms[j] for j in kept],
                                [modes[j] for j in kept],norm,val,mask,columns=columns)
            for r in test:emit(name,r,metrics(model,r,norm,mask))
    if narmax:
        from narmax import DICTIONARIES, fit_dictionary
        best=None
        for lagname,degree,mode in DICTIONARIES:
            tr,va,_,nn,ma,_=prepare(layout,held,mode,broadband=True)
            for loss,model in fit_dictionary(tr,va,nn,ma,lagname,degree):
                if np.isfinite(loss) and (best is None or loss<best[0]):best=(loss,model,mode)
        if best is None:raise RuntimeError('No finite NARMAX calibration fit.')
        _,model,mode=best
        _,_,tt,nn,ma,_=prepare(layout,held,mode,broadband=True)
        model.save(folder/'narmax.json')
        for r in tt:emit('narmax',r,metrics(model,r,nn,ma))
    if ocean:
        if layout!='L0' or held:raise ValueError('OCEAN is restricted to matched-state L0.')
        from ocean import OceanReplay
        for r in test:emit('ocean',r,metrics(OceanReplay(r,norm),r,norm,mask))
    write_csv(folder/'metrics.csv',output)
    print('Results:',folder/'metrics.csv',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout',choices=['L0','L1','L2','L3','L4','L5'],action='append',required=True)
    p.add_argument('--exclude',choices=['ISS1','ISS2','ISS3'])
    p.add_argument('--narmax',action='store_true')
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--ocean',action='store_true')
    args=p.parse_args()
    if args.ocean and (args.layout!=['L0'] or args.exclude):p.error('--ocean requires only --layout L0 and no exclusion')
    for layout in args.layout:evaluate(layout,args.exclude,args.narmax,args.ablations,args.ocean)

if __name__=='__main__':main()