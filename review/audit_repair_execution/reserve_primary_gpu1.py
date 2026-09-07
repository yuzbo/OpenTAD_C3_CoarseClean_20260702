"""Return the two verified GPU1 slots to the two primary A methods.

This changes only the supplementary resource allowance and the current owned
allocation's end time after its first full checkpoint has been saved.
"""
import json
from pathlib import Path, PurePosixPath

from deploy_corrected import ssh

HERE = Path(__file__).resolve().parent
CODE = r'''
from pathlib import Path
import json,math,os,re,subprocess,time
ctl=Path(CONTROL);b=json.loads((ctl/'bindings.json').read_text());root=Path(b['work_root'])
assert b['assigned_training_ids']==['tr-786cf750fa74'] and b['cluster']=='N16R4'
assert str(root).startswith('/data/run01/sczc063/yuzibo/geosparse_official_20260908/audit_repair_b70ae056_secondary/')
path=ctl/'gpu1_nodes.json';nodes=json.loads(path.read_text())
record_path=ctl/'primary_slot_reservation.json'
record=json.loads(record_path.read_text()) if record_path.exists() else dict(
    created_at=time.time(),previous_node_allowance=nodes,
    primary_training_ids=['tr-0afe8e4fc09b','tr-624db235c913'],
    reason='Two verified GPU1 slots are required by the two primary A methods. Supplementary A uniform resumes its own checkpoint when a primary releases a slot or another verified spare slot is available.')
nodes['nodes']=[]
temporary=path.with_suffix('.json.new');temporary.write_text(json.dumps(nodes,indent=2));temporary.replace(path)
q=json.loads((root/'slurm_state.json').read_text())['jobs']['tr-786cf750fa74']
run=root/'runs/tr-786cf750fa74';p=run/'progress.json';ckpt=run/'checkpoint/last.pth'
progress=json.loads(p.read_text()) if p.exists() else {}
record.update(observed_at=time.time(),queue=q,progress=progress,checkpoint_path=str(ckpt))
if record.get('shortened_allocation'):
    record['status']='ALREADY_RESERVED'
elif not ckpt.is_file() or not progress.get('completed_epochs'):
    record['status']='WAITING_FOR_FIRST_CHECKPOINT'
else:
    sid=q['slurm_id'];assert sid=='1278034', 'Inspect changed allocation before modifying it'
    job=subprocess.check_output(['scontrol','show','job','-o',sid],text=True)
    assert 'JobName=tr-786cf750fa74 ' in job and re.search(r'UserId=\S+\('+str(os.getuid())+r'\) ',job)
    assert 'Command='+str(run/'run.sbatch')+' ' in job
    assert 'JobState=RUNNING ' in job,job
    hours,minutes,seconds=map(int,re.search(r'RunTime=(\d+:\d+:\d+)',job)[1].split(':'))
    limit=math.ceil((3600*hours+60*minutes+seconds)/60)+2
    assert limit<60, 'Existing allocation is already near its end; no shortening needed'
    result=subprocess.run(['scontrol','update','JobId='+sid,'TimeMin='+str(limit),'TimeLimit='+str(limit)],capture_output=True,text=True)
    record.update(status='ALLOCATION_END_SHORTENED' if result.returncode==0 else 'UPDATE_FAILED',
                  before=job,command_result=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),
                  checkpoint_bytes=ckpt.stat().st_size)
    if result.returncode==0:
        record['shortened_allocation']=sid
        record['after']=subprocess.check_output(['scontrol','show','job','-o',sid],text=True)
record_path.write_text(json.dumps(record,indent=2))
print(json.dumps(record))
'''


if __name__ == '__main__':
    binding = json.loads((HERE/'audit_repair_b70ae056/bindings.secondary.N16R4.json').read_text())
    control = str(PurePosixPath(binding['work_root']).parent.parent/'control')
    result = json.loads(ssh('source', 'python3 -', CODE.replace('CONTROL', repr(control)).encode()))
    (HERE/'audit_primary_slot_reservation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
