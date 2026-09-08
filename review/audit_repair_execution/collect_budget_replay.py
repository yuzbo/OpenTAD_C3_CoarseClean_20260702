"""Collect read-only replay summaries and, when complete, raw window records."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from deploy_corrected import command

HERE=Path(__file__).resolve().parent
REMOTE=r'''
from collections import Counter
import json,math
from pathlib import Path
root=Path(ROOT);out=root/'results'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def tail(p):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4500));return f.read().decode(errors='replace')
def pct(values,q):
 v=sorted(values);i=(len(v)-1)*q;l=int(i);r=min(l+1,len(v)-1);return v[l]+(v[r]-v[l])*(i-l)
header=read(out/'header.json');receipt=read(out/'receipt.json');rows=[]
if header:
 for case in header['cases']:
  p=out/(case['job_id']+'.epoch_%03d.windows.jsonl'%case['completed_epochs'])
  data=[]
  if p.exists():
   for line in p.open():
    try:data.append(json.loads(line))
    except ValueError:pass
  run=Path(case['checkpoint_path']).parent.parent
  metric_path=run/'intermediate_eval'/('epoch_%03d'%case['completed_epochs'])/'metrics.json'
  metric=read(metric_path)
  if metric and metric.get('status')=='completed_validation':
   if not (metric['source_train_id']==case['job_id'] and metric['completed_epochs']==case['completed_epochs']
    and metric['provenance']==case['provenance'] and metric['weights']=='ema'
    and metric['subset']=='validation' and metric['windows']==792 and sorted(metric['videos'])==header['videos']):
    raise ValueError('accuracy is not paired to the replayed epoch/split')
  else:metric=None
  best=read(run/'best.json')
  if best and best['completed_epochs']==case['completed_epochs']:
   if best['checkpoint_identity']['checkpoint_sha256']!=case['checkpoint_sha256']:
    raise ValueError('best checkpoint differs from the replayed immutable checkpoint')
  result=dict(job_id=case['job_id'],completed_epochs=case['completed_epochs'],controller=case['controller'],
   metrics=metric['metrics'] if metric else None,validation_receipt=metric,validation_receipt_path=str(metric_path),
   windows=len(data),path=str(p),checkpoint_sha256=case['checkpoint_sha256'])
  if data:
   h=Counter(r['requested_budget'] for r in data);n=len(data)
   m=[r['heavy_ratio_padded_full'] for r in data]
   v=[r['heavy_ratio_valid_full'] for r in data if r['heavy_ratio_valid_full'] is not None]
   dynamic=case['config']['budget_mode']=='dynamic'
   result.update(videos=len({r['video_id'] for r in data}),budget_hist=dict(h),budget_head_used=dynamic,
    requested_budget_mean=sum(r['requested_budget'] for r in data)/n,
    full_budget_fraction=h.get(1.,0)/n,
    probability_expected_budget_mean=sum(sum(q*p for q,p in zip(r['menu'],r['budget_probabilities'])) for r in data)/n if dynamic else None,
    heavy_padded_mean=sum(m)/n,heavy_padded_p50=pct(m,.5),heavy_padded_p95=pct(m,.95),
    heavy_valid_mean=sum(v)/len(v) if v else None,
    heavy_valid_weighted=sum(r['heavy_macs'] for r in data)/sum(r['valid_full_heavy_macs'] for r in data),
    full_valid_heavy_fraction=sum(abs(x-1)<1e-8 for x in v)/len(v) if v else None,
    mean_budget_probabilities=[sum(r['budget_probabilities'][i] for r in data)/n for i in range(len(data[0]['menu']))] if dynamic else None,
    menu=data[0]['menu'])
  if receipt and INCLUDE_ROWS:result['raw_rows']=data
  rows.append(result)
print(json.dumps(dict(header=header,receipt=receipt,rows=rows,stdout_tail=tail(root/'srun.out'),
 stderr_tail=tail(root/'srun.err'),launcher_tail=tail(root/'launcher.log'))))
'''
def collect(cluster,target,launch,include):
    code=REMOTE.replace("ROOT",repr(launch["root"])).replace("INCLUDE_ROWS",repr(include))
    result=subprocess.run(command(target,"python3 -"),input=code.encode(),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(cluster+": "+result.stderr.decode(errors="replace"))
    return cluster,json.loads(result.stdout)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--complete",action="store_true");args=parser.parse_args()
    launches=json.loads((HERE/"dynamic_budget_replay_launch.json").read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(collect,c,t,launches[c],args.complete) for c,t in [("N16R4","source"),("A100","destination")]]
        results=dict(f.result() for f in futures)
    results["observed_at_utc"]=datetime.now(timezone.utc).isoformat()
    (HERE/"dynamic_budget_replay_summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    for cluster in ("N16R4","A100"):
        host=results[cluster]
        for row in host["rows"]:
            print(json.dumps(dict(cluster=cluster,**{k:v for k,v in row.items() if k!="raw_rows"}),ensure_ascii=False))
        if not host["rows"] or not host["receipt"]:
            print(json.dumps(dict(cluster=cluster,stdout_tail=host["stdout_tail"],stderr_tail=host["stderr_tail"],launcher_tail=host["launcher_tail"]),ensure_ascii=False))
