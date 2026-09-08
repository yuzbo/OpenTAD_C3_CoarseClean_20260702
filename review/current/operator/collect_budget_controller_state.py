"""Read selected saved controller buffers on CPU; no model forward or mutation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, shlex, subprocess
from deploy_corrected import command

HERE=Path(__file__).resolve().parent
OUT=HERE/'failure_analysis_20260908'
PY=r'''
import torch,json,inspect
from pathlib import Path
torch.set_num_threads(1)
mmap_args={'mmap':True} if 'mmap' in inspect.signature(torch.load).parameters else {}
root=Path(ROOT);out=[]
for jid,epochs in REQUESTS:
 for epoch in epochs:
  p=root/'runs/b70ae056/runs'/jid/'checkpoint'/('epoch_'+str(epoch-1)+'.pth')
  if not p.exists():out.append(dict(job_id=jid,epoch_completed=epoch,error='missing'));continue
  d=torch.load(p,map_location='cpu',weights_only=False,**mmap_args)
  row=dict(job_id=jid,epoch_completed=int(d['epoch'])+1,path=str(p),successful_updates=d.get('successful_optimizer_updates',d.get('updates')),controller={})
  for key in ['state_dict','state_dict_ema']:
   s=d[key];row['controller'][key]={n:float(v) for n,v in s.items() if n.endswith(('.dual','.cost_ema'))}
   # Magnitude only; these are not parameter gradient or feature-use measurements.
   row[key+'_scout_norms']={n:dict(norm=float(v.float().norm()),mean=float(v.float().mean())) for n,v in s.items() if n in ['geosparse.scout.gain.2.weight','geosparse.scout.gain.2.bias','geosparse.scout.budget.weight','geosparse.scout.budget.bias']}
  out.append(row);del d
print(json.dumps(out))
'''
TARGETS={
 'source':('/data/run01/sczc063/yuzibo/geosparse_official_20260908/audit_repair_b70ae056',[('tr-0afe8e4fc09b',[40,60]),('tr-624db235c913',[50])]),
 'destination':('/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/corrected_20260908/audit_repair_b70ae056',[('tr-1d7d835dcbe2',[20]),('tr-5789d4417b4b',[20])])}
def collect(item):
 target,(root,requests)=item
 bootstrap="import json,subprocess,sys;from pathlib import Path;b=json.loads(Path("+repr(root+'/control/bindings.json')+").read_text());r=subprocess.run([b['entrypoint'][0],'-'],input=sys.stdin.buffer.read(),capture_output=True);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)"
 script=PY.replace('ROOT',repr(root)).replace('REQUESTS',repr(requests))
 try:
  r=subprocess.run(command(target,'python3 -c '+shlex.quote(bootstrap)),input=script.encode(),capture_output=True,timeout=85)
 except subprocess.TimeoutExpired:
  print(json.dumps(dict(target=target,error='read-only controller collection timed out after85s')));return
 if r.returncode:raise RuntimeError(target+': '+r.stderr.decode(errors='replace')[-2000:])
 data=json.loads(r.stdout);OUT.mkdir(exist_ok=True)
 (OUT/(target+'_controller_state.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
 print(json.dumps(dict(target=target,states=[{k:v for k,v in row.items() if k in ['job_id','epoch_completed','controller','error']} for row in data])))
if __name__=='__main__':
 import sys
 selected={k:v for k,v in TARGETS.items() if len(sys.argv)==1 or k in sys.argv[1:]}
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collect,selected.items()))
