from __future__ import annotations
import itertools, json, math, subprocess, sys
from pathlib import Path
import pytest
import torch
from reference.routing import ordered_log_prob, sample_ordered_topk, actor_loss
from reference.support import Support
from tools.compile_matrix import compile_all, validate, canonical
from tools.dispatch import fair_order, external_blockers, capability_blockers, check_result
ROOT=Path(__file__).resolve().parents[1]

def test_pl_probability_normalizes():
    x=torch.tensor([.3,-.2,1.1,.0],dtype=torch.float64)
    p=sum(ordered_log_prob(x,torch.tensor(o)).exp() for o in itertools.permutations(range(4),2))
    assert abs(p.item()-1)<1e-12

def test_pl_matches_naive():
    x=torch.tensor([4.,-3.,2.,.4],dtype=torch.float64,requires_grad=True)
    o=torch.tensor([2,0,3]);remaining=list(range(4));v=x.sum()*0
    for i in o.tolist():
        v=v+x[i]-torch.logsumexp(x[remaining],0);remaining.remove(i)
    actual=ordered_log_prob(x,o)
    assert torch.allclose(v,actual,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(v,x,retain_graph=True)[0],torch.autograd.grad(actual,x)[0])

def test_pl_extreme_stable():
    x=torch.tensor([1000.,-1000.,500.,499.],requires_grad=True)
    p=ordered_log_prob(x,torch.tensor([0,2,3]))
    assert torch.isfinite(p)
    p.backward();assert torch.isfinite(x.grad).all()

def test_pl_rejects_duplicates():
    with pytest.raises(ValueError):ordered_log_prob(torch.zeros(3),torch.tensor([1,1]))

def test_pl_rejects_invalid():
    with pytest.raises(ValueError):ordered_log_prob(torch.zeros(3),torch.tensor([3]))

def test_sample_hard_unique():
    g=torch.Generator().manual_seed(5)
    idx,logp=sample_ordered_topk(torch.arange(10.).requires_grad_(),4,g)
    assert len(idx)==len(idx.unique())==4 and torch.isfinite(logp)

def test_sample_zero_full():
    for k in [0,5]:
        x=torch.ones(5,requires_grad=True);idx,p=sample_ordered_topk(x,k)
        assert len(idx)==k and p.item()==0
        p.backward();assert torch.equal(x.grad,torch.zeros_like(x))

def test_actor_sign_and_detach():
    z=torch.tensor(0.,requires_grad=True);r=torch.tensor(2.,requires_grad=True);b=torch.tensor(1.,requires_grad=True)
    loss=actor_loss(r,b,torch.nn.functional.logsigmoid(z));loss.backward()
    assert z.grad>0  # gradient descent lowers probability of costly action
    assert r.grad is None and b.grad is None

def test_support_gap_not_filled():
    s=Support(((1.,2.),(10.,11.)))
    assert s.contains(1.5) and not s.contains(5.)
    assert s.observed_duration==2 and s.distance(5.)==3

def test_support_empty():
    s=Support(())
    assert not s.contains(1) and math.isinf(s.distance(1))

def test_support_rejects_overlap():
    with pytest.raises(ValueError):Support(((1.,3.),(2.,4.)))

def test_support_rejects_roi():
    with pytest.raises(ValueError):Support(((1.,2.),),(.8,0,.2,1))

def test_manifest_deterministic():
    a,_=compile_all();b,_=compile_all()
    assert canonical(a)==canonical(b)
    validate(a)

def test_manifest_no_training_gate():
    jobs,_=compile_all()
    assert all(j['depends_on']==[] and j['epochs']==60 for j in jobs if j['kind']=='train')
    assert not any('metric_gate' in j for j in jobs)

def test_manifest_three_seeds_per_configuration():
    jobs,_=compile_all();groups={}
    for j in jobs:
        if j['kind']=='train':groups.setdefault(canonical([j['model'],j['dataset']]),set()).add(j['seed'])
    assert all(x=={0,1,2} for x in groups.values())

def test_evaluation_only_own_checkpoint():
    jobs,_=compile_all()
    for j in jobs:
        if j['kind']!='train':assert j['depends_on']==[j['source_train_id']]

def test_fair_order_covers_all():
    jobs,_=compile_all();ordered=fair_order(jobs)
    assert {x['job_id'] for x in ordered}=={x['job_id'] for x in jobs}
    assert len(ordered)==len(jobs)

def test_missing_assets_fail_closed():
    j={'external_requirements':['dataset:x','checkpoint:b','legacy_duca_manifest'],'capabilities':['route_A']}
    assert len(external_blockers(j,{}))==3
    assert capability_blockers(j,{})==['route_A']

def test_mock_scientific_result_rejected(tmp_path):
    p=tmp_path/'result.json'
    p.write_text(json.dumps(dict(job_id='x',status='completed',is_mock=True,source_commit='test',
              resolved_config_sha256='test',split_sha256='test')))
    with pytest.raises(ValueError):check_result({'job_id':'x','kind':'evaluate'},tmp_path)

def test_default_dispatch_never_launches():
    result=subprocess.run([sys.executable,str(ROOT/'tools/dispatch.py'),'--manifest',str(ROOT/'manifests/experiments.jsonl'),
        '--bindings',str(ROOT/'configs/bindings.example.json')],text=True,capture_output=True,check=True)
    lines=[json.loads(x) for x in result.stdout.splitlines()]
    assert lines[0]['mode']=='plan_only' and lines[-1]['launched']==0
