import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from catalogue import catalogue,fit,load
from dataset import metrics
from ensemble import context_statistics,risk_labels,ContextModel,fit_gate
from stacking import MixtureObjective,objective_stacking
from diagnostics import clustered
from summarise import summarise
from common import ROOT


def records(count=3,n=512):
    rng=np.random.default_rng(4)
    rows=[]
    for i in range(count):
        x=rng.normal(size=(n,2));y=np.zeros(n)
        for k in range(1,n):y[k]=.3*y[k-1]+.15*x[k-1,0]-.08*x[k-1,1]
        rows.append({'x':x,'y':y,'tau':x[:,1],'layout':'L0','state':'ISS'+str(i+1),
                     'policy':'P','test_kind':'synthetic'})
    return rows


def test_context_and_risk_timing():
    x=records(1)[0]['x'];changed=x.copy();changed[200:]+=100
    a,b=context_statistics(x),context_statistics(changed)
    np.testing.assert_array_equal(a[:201],b[:201])
    np.testing.assert_allclose(a[200,0],x[72:200,0].mean())
    r=records(1)[0];p=np.column_stack([r['y']+1,r['y']+2])
    expected=np.array([[1,4]])/np.mean(r['y']**2)
    np.testing.assert_allclose(risk_labels(p,r,None,None,[128]),expected)


@pytest.mark.parametrize('family',['arx','fir','frols','kernel','reservoir','symbolic','neural','volterra'])
def test_family_fit_serialisation_and_recursive_history(family,tmp_path):
    c=next(c for c in catalogue() if c['family']==family and c['input_mode']=='separate')
    model=fit(family,c,records(),11)
    r=records(1)[0];a=model.simulate(r['x'],r['y'])
    assert np.isfinite(a).all()
    path=tmp_path/'model.json';model.save(path)
    np.testing.assert_allclose(load(path).simulate(r['x'],r['y']),a)
    changed=r['y'].copy();changed[80:]+=100
    np.testing.assert_allclose(model.simulate(r['x'],changed),a)


def test_gate_and_same_objective_stacking():
    rows=records();experts=[fit('arx',{'memory':4,'ridge':r},rows,11) for r in [.0001,.01]]
    norm={'x_mean':[0.,0.],'x_std':[1.,1.],'y_mean':0.,'y_std':1.}
    mask=np.ones(129,dtype=bool);mask[0]=False
    model,_,_=fit_gate(experts,[norm,norm],['separate','separate'],norm,rows,mask)
    prediction=model.simulate(rows[0]['x'],rows[0]['y'])
    assert np.isfinite(prediction).all()
    objective=MixtureObjective([model.predictions(r['x'],r['y']) for r in rows],rows,norm,mask)
    w=np.array([.4,.6]);_,gradient=objective.value_gradient(w)
    eps=1e-6
    numeric=np.array([(objective.value_gradient(w+np.eye(2)[j]*eps)[0]-objective.value_gradient(w-np.eye(2)[j]*eps)[0])/(2*eps) for j in range(2)])
    np.testing.assert_allclose(gradient,numeric,rtol=1e-4,atol=1e-6)
    constant,audit=objective_stacking(model,rows,mask,np.ones(2)/2)
    assert np.min(constant.prior)>=0
    np.testing.assert_allclose(constant.prior.sum(),1)
    assert audit['calibration_score']<=objective.value_gradient(np.ones(2)/2)[0]+1e-9


def test_cluster_statistics_and_reported_reductions():
    rows=[{'state':s,'policy':p,'reduction_percent':10.} for s in ['ISS1','ISS2','ISS3'] for p in ['UM','PI']]
    result=clustered(rows,draws=100)
    assert result['clusters']==6
    assert result['signflip_p_two_sided']==2/64
    result=summarise([ROOT/'reference/metrics.csv'])
    values={r['comparison']:r for r in result if r['scope']=='matched' and r['phase']=='holdout'}
    assert values['baseline']['median_reduction_percent']==pytest.approx(24.2,abs=.1)
    assert values['volterra']['median_reduction_percent']==pytest.approx(17.4,abs=.1)
    assert values['baseline']['records']==18
    assert values['baseline']['wins']==14
@pytest.mark.parametrize('architecture',['multiscale','tcn'])
def test_other_neural_architectures(architecture,tmp_path):
    c=next(c for c in catalogue() if c['family']=='neural' and c['architecture']==architecture and c['input_mode']=='separate')
    model=fit('neural',c,records(),11);r=records(1)[0]
    prediction=model.simulate(r['x'],r['y'])
    assert np.isfinite(prediction).all()
    p=tmp_path/'network.json';model.save(p)
    np.testing.assert_allclose(load(p).simulate(r['x'],r['y']),prediction)


def test_narmax_estimation_and_neural_refinement(monkeypatch):
    import narmax
    from neural import fit_teacher
    from refinement import refine,convert
    import json
    rows=records();norm={'x_mean':[0.,0.],'x_std':[1.,1.],'y_mean':0.,'y_std':1.}
    mask=np.ones(129,dtype=bool);mask[0]=False
    monkeypatch.setattr(narmax,'TERM_COUNTS',[5,10])
    candidates=narmax.fit_dictionary(rows,rows,norm,mask,'short',2)
    assert any(np.isfinite(loss) for loss,_ in candidates)
    model=min(candidates,key=lambda a:a[0])[1]
    changed=rows[0]['y'].copy();changed[80:]+=20
    np.testing.assert_allclose(model.simulate(rows[0]['x'],changed),model.simulate(rows[0]['x'],rows[0]['y']))
    scalar=[{**r,'u':r['x'][:,0]} for r in rows]
    cfg=json.loads((ROOT/'code/neural_config.json').read_text())
    cfg['neural_teacher_max_iter']=30
    rcfg=json.loads((ROOT/'code/refinement_config.json').read_text())
    rcfg.update(gradient_updates=[1,2],checkpoint_every=1)
    net=fit_teacher(scalar,16,11,cfg)
    for energy in [False,True]:
        checkpoints,_=refine(net,scalar,scalar,mask,norm,cfg,rcfg,energy)
        refined=checkpoints[2][0]
        converted=convert(refined,[{**r,'x':r['x'][:,:1]} for r in rows])
        np.testing.assert_allclose(converted.simulate(rows[0]['x'][:,:1],rows[0]['y']),
                                   refined.simulate(scalar[0]['u'],scalar[0]['y']),rtol=1e-8,atol=1e-10)
