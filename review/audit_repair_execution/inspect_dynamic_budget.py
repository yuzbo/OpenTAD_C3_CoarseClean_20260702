"""Read existing run records; no inference, training or scheduler mutation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from deploy_corrected import command

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import json, time
from collections import Counter
from pathlib import Path
root=Path(ROOT)
def read(p):
 return json.loads(p.read_text()) if p.is_file() else None
rows=[]
for jid in JOBS:
 run=root/'runs'/jid
 job=read(run/'job.json'); protocol=read(run/'resolved_protocol.json')
 row=dict(job_id=jid,job=job,protocol=protocol,source=read(run/'source_commits.json'),
  progress=read(run/'progress.json'),best=read(run/'best.json'),trace_epochs={},
  validation=[],selection_exports=[])
 for p in sorted((run/'intermediate_eval').glob('epoch_*/metrics.json')):
  m=read(p)
  row['validation'].append({k:m.get(k) for k in ['status','completed_epochs','checkpoint_identity','weights','metrics','windows','seconds','provenance']})
 trace=run/'cost_trace.jsonl'
 row['trace_path']=str(trace);row['trace_bytes']=trace.stat().st_size if trace.is_file() else None
 if trace.is_file():
  for line in trace.open():
   try: r=json.loads(line)
   except ValueError: continue
   e=str(r['epoch']+1)
   summary=row['trace_epochs'].setdefault(e,dict(records=0,samples=0,budget_hist={},sum_qkv=0,sum_qkv_squared=0,rows_by_layer={},trace_native_sizes={},probe_records=0,first_step_records=0,per_record=[]))
   qkv=[x['qkv_tokens'] for x in r['layers']]
   budgets=r['requested_budget'];summary['records']+=1;summary['samples']+=len(budgets)
   for q in budgets:summary['budget_hist'][str(q)]=summary['budget_hist'].get(str(q),0)+1
   summary['sum_qkv']+=sum(qkv);summary['sum_qkv_squared']+=sum(x*x for x in qkv)
   summary['probe_records']+=int(r.get('acquisition') is not None)
   summary['first_step_records']+=int(r['step']==0)
   for x in r['layers']:
    k=str(x['layer']);summary['rows_by_layer'][k]=summary['rows_by_layer'].get(k,0)+1
    k=str(x['native_tokens']);summary['trace_native_sizes'][k]=summary['trace_native_sizes'].get(k,0)+1
   summary['per_record'].append(dict(step=r['step'],microbatch=r['microbatch'],budgets=budgets,
    selected_native_members=r['selected_native_members'],sum_qkv=sum(qkv),sum_qkv_squared=sum(x*x for x in qkv),
    video_ids=r['video_ids'],acquisition=r.get('acquisition')))
 # Small manifests only. Never read full predictions, videos or checkpoints here.
 for base in [run/'analysis',run/'selection_export',run/'selection_exports']:
  if base.exists():
   for p in base.rglob('export_receipt.json'):row['selection_exports'].append(dict(path=str(p),receipt=read(p)))
 rows.append(row)
print(json.dumps(dict(observed_at_unix=time.time(),rows=rows)))
'''

def collect(cluster, target, delivery):
    binding=json.loads((delivery/f"bindings.primary.{cluster}.json").read_text())
    code=REMOTE.replace("ROOT",repr(binding["work_root"])).replace("JOBS",repr(binding["assigned_training_ids"]))
    result=subprocess.run(command(target,"python3 -"),input=code.encode(),capture_output=True,timeout=90)
    if result.returncode:
        raise RuntimeError(cluster+": "+result.stderr.decode(errors="replace"))
    return cluster,json.loads(result.stdout)

if __name__=="__main__":
    delivery=Path(json.loads((HERE/"audit_deployment.latest.json").read_text())["local_delivery"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(collect,c,t,delivery) for c,t in [("N16R4","source"),("A100","destination")]]
        data=dict(f.result() for f in futures)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out=HERE/f"dynamic_budget_observation_{stamp}.json"
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    for cluster,host in data.items():
        for row in host["rows"]:
            last=list(row["trace_epochs"])[-1] if row["trace_epochs"] else None
            print(json.dumps(dict(cluster=cluster,job_id=row["job_id"],progress=row["progress"],best=row["best"],
              protocol=row["protocol"],trace_bytes=row["trace_bytes"],trace_epochs=len(row["trace_epochs"]),
              latest_trace_epoch=last,latest_trace_summary={k:v for k,v in row["trace_epochs"].get(last,{}).items() if k!="per_record"},
              selection_exports=row["selection_exports"]),ensure_ascii=False),flush=True)
    print("Saved "+str(out))
