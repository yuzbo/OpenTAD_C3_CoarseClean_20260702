"""Keep existing Slurm IDs; start each dynamic B/C method before its fixed run."""
import json
from pathlib import Path
from deploy_corrected import ssh

HERE = Path(__file__).resolve().parent
SCRIPT = r'''
from pathlib import Path
import json,subprocess,time
root=Path('/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/corrected_20260908/tooling_902fa05b')
record=root/'control'/'primary_start_priority.json'
def show(sid):
 r=subprocess.run(['scontrol','show','job','-o',sid],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout
operations=[]
for fixed,dynamic,jid in [('244973','244989','tr-be89cd6cb2ca'),('244975','244991','tr-a45686a40afa')]:
 before=show(fixed);other=show(dynamic)
 assert 'JobName='+jid+' ' in before
 assert 'UserId=pxyai_0057(' in before and 'UserId=pxyai_0057(' in other
 if 'JobState=PENDING ' not in before:
  operations.append(dict(job_id=jid,slurm_id=fixed,status='ALREADY_STARTED_UNCHANGED',before=before));continue
 assert not (root/'runs/902fa05b/runs'/jid/'progress.json').exists(), 'retain any training already performed'
 command=['scontrol','update','JobId='+fixed,'Dependency=after:'+dynamic]
 r=subprocess.run(command,capture_output=True,text=True)
 item=dict(job_id=jid,slurm_id=fixed,primary_dynamic_slurm_id=dynamic,command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,before=before,after=show(fixed))
 operations.append(item)
 record.write_text(json.dumps(dict(observed_at=time.time(),operations=operations),indent=2))
 if r.returncode:raise RuntimeError('priority update rejected; exact response preserved: '+r.stderr)
print(json.dumps(dict(operations=operations,source_and_training_protocol_unchanged=True)))
'''


if __name__ == '__main__':
    result = json.loads(ssh('destination', 'python3 -', SCRIPT.encode()))
    (HERE / 'primary_start_priority.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps([dict(job_id=row['job_id'], slurm_id=row['slurm_id'], dynamic_first=row.get('primary_dynamic_slurm_id'), returncode=row.get('returncode')) for row in result['operations']]))
