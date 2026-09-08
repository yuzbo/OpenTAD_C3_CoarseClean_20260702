"""Read the unique C20 audit; no submissions or training changes."""
from pathlib import Path
import json,subprocess
from deploy_corrected import command
from submit_c20_budget_review import STAGE

HERE=Path(__file__).resolve().parent
SCRIPT=r'''
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,subprocess
stage=Path(STAGE);sub=json.loads((stage/'submission.json').read_text());sid=sub['slurm_id'];out=stage/'output'
r=subprocess.run(['squeue','-h','-j',sid,'-o','%i|%T|%R|%M|%N'],capture_output=True,text=True,timeout=20)
d=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),submission=sub,slurm=r.stdout.strip(),slurm_error=r.stderr.strip(),files=[p.name for p in out.glob('*')] if out.exists() else [])
for kind in ['out','err']:
 p=stage/('slurm-'+sid+'.'+kind);d['last_'+kind]=p.read_text(errors='replace')[-1600:] if p.exists() else None
p=out/'receipt.json'
if p.exists():d['receipt']=json.loads(p.read_text())
p=out/'tr-5789d4417b4b.epoch_020.windows.jsonl'
if p.exists():
 rows=[json.loads(s) for s in p.read_text().splitlines()];d['observed_rows']=len(rows);d['videos']=len({r['video_id'] for r in rows});d['requested_histogram']=dict(Counter(str(r['requested_budget']) for r in rows))
 if rows:d['eligible_heavy_mean']=sum(r['heavy_ratio_valid_full'] for r in rows)/len(rows);d['all_fine_fraction']=sum(r['requested_budget']==1 for r in rows)/len(rows)
print(json.dumps(d))
'''
if __name__=='__main__':
 try:r=subprocess.run(command('destination','python3 -'),input=SCRIPT.replace('STAGE',repr(STAGE)).encode(),capture_output=True,timeout=45)
 except subprocess.TimeoutExpired:raise RuntimeError('Read-only C20 observation timed out; job state unknown') from None
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:])
 data=json.loads(r.stdout);out=HERE/'c20_budget_review_20260908';out.mkdir(exist_ok=True)
 (out/'observation.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in data.items() if k!='receipt'},ensure_ascii=True))
