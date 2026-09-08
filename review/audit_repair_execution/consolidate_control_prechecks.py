"""Group four unstarted A100 GPU prechecks to fit the actual ten-job limit."""
import json
from pathlib import Path, PurePosixPath
import subprocess

from activate_submitted_controls import activate
from deploy_corrected import command

HERE = Path(__file__).resolve().parent
CODE = r'''
import fcntl,json,os,re,shlex,signal,socket,subprocess,sys,time
from pathlib import Path
binding_path=Path(BINDING)
ctl=binding_path.parent
b=json.loads(binding_path.read_text())
assert b['cluster']=='A100' and b['primary_binding_paths']
co=json.loads((ctl/'coordinator.json').read_text())
rows=[]
if co['host']==socket.gethostname():
    record_path=ctl/'grouped_prechecks_amendment.json'
    if record_path.exists():
        raise RuntimeError('existing consolidation record: inspect it; do not repeat cancellation/submission')
    expected={'tr-ee184df8b88b','tr-e2084b928c6c','tr-3d90bb0b7c1d','tr-373b13f87219'}
    assert set(b['assigned_training_ids'])==expected
    root=Path(b['work_root'])
    state=json.loads((root/'slurm_state.json').read_text())
    assert state['source_commit']==b['source_commit']
    submissions_path=ctl/'precheck_submissions.json'
    submissions=json.loads(submissions_path.read_text())
    jobs={j['job_id']:j for j in (json.loads(line) for line in (ctl/'experiments.jsonl').read_text().splitlines() if line.strip())}
    details={}
    old_ids=[]
    for jid in b['assigned_training_ids']:
        assert jobs[jid]['seed']==0 and jobs[jid]['kind']=='train'
        assert jobs[jid]['model']['budget_mode']!='dynamic'
        sid=str(submissions[jid]['slurm_id']);old_ids.append(sid)
        text=subprocess.check_output(['scontrol','show','job',sid,'-o'],text=True)
        assert re.search(r'\bJobState=PENDING\b',text),text
        assert re.search(r'\bUserId=\S+\('+str(os.getuid())+r'\)',text),text
        out=ctl.parent/'prechecks'/jid
        assert 'Command='+str(out/'run.sbatch') in text,text
        assert not (out/'gpu_precheck.json').exists(), 'inspect existing precheck result before consolidation'
        details[jid]=dict(old_slurm_id=sid,slurm_detail=text,output=str(out))
    sys.path.insert(0,str(ctl))
    from audit_queue import primary_state
    primary_ids,waiting=primary_state(b)
    assert not waiting,waiting
    first=ctl.parent/'prechecks'/b['assigned_training_ids'][0]/'run.sbatch'
    original=first.read_text()
    assert original.rfind('exec ')>=0
    prefix=original[:original.rfind('exec ')]
    assert '#SBATCH --time=00:30:00' in prefix
    prefix=prefix.replace('#SBATCH --time=00:30:00','#SBATCH --time=02:00:00')
    prefix=re.sub(r'^#SBATCH --dependency=.*$', '#SBATCH --dependency=after:'+':'.join(primary_ids),prefix,flags=re.M)
    script=prefix+'status=0\n'
    commands={}
    for jid in b['assigned_training_ids']:
        out=Path(details[jid]['output'])
        argv=[b['entrypoint'][0],'-m','geosparse_ext.gpu_precheck','--manifest',str(ctl/'experiments.jsonl'),'--bindings',str(binding_path),'--output',str(out),'--train-ids',jid,'--certify']
        original_command=shlex.split((out/'run.sbatch').read_text().splitlines()[-1])
        assert original_command==['exec']+argv,(jid,original_command,argv)
        stdout=str(out/'grouped-precheck.out');stderr=str(out/'grouped-precheck.err')
        commands[jid]=dict(command=argv,stdout=stdout,stderr=stderr)
        script+='if '+shlex.join(argv)+' > '+shlex.quote(stdout)+' 2> '+shlex.quote(stderr)+'; then\n  :\nelse\n  status=1\nfi\n'
    script+='exit "$status"\n'
    batch_path=ctl/'grouped_prechecks.sbatch'
    batch_path.write_text(script)
    subprocess.run(['bash','-n',str(batch_path)],check=True)
    pid=co['pid'];proc=Path('/proc')/str(pid)
    cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    assert proc.stat().st_uid==os.getuid() and str(ctl/'audit_queue.py') in cmd and str(binding_path) in cmd,cmd
    record=dict(status='PREPARED',source_commit=b['source_commit'],coordinator=co,primary_ids=primary_ids,old_prechecks=details,commands=commands,script=str(batch_path),observed_at=time.time(),reason='A100 association MaxSubmitJobs=10: combine four unstarted correctness checks; no model/training change')
    def save_record():
        record_path.write_text(json.dumps(record,indent=2))
    save_record()
    os.kill(pid,signal.SIGTERM)
    lock=open(ctl/'audit_supervisor.lock','a+')
    for _ in range(50):
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(.1)
    else:
        raise RuntimeError('old supplementary coordinator did not release its lock')
    # These exact owned prechecks were all pending; preserve their records.
    (ctl/'precheck_submissions.before_grouping.json').write_text(json.dumps(submissions,indent=2))
    record['status']='RETIRING_UNSTARTED_PRECHECKS';save_record()
    subprocess.run(['scontrol','hold',','.join(old_ids)],check=True)
    subprocess.run(['scancel',*old_ids],check=True)
    pending=subprocess.check_output(['squeue','-h','--me','-o','%i'],text=True).split()
    assert not set(old_ids)&set(pending),'old prechecks not yet removed; inspect before proceeding'
    argv=['sbatch','--parsable','--job-name=gs-audit-check-controls',f'--output={ctl}/grouped-prechecks-%j.out',f'--error={ctl}/grouped-prechecks-%j.err',str(batch_path)]
    record.update(status='BATCH_SUBMITTING',submission_command=argv);save_record()
    result=subprocess.run(argv,capture_output=True,text=True)
    record.update(submission_returncode=result.returncode,submission_stdout=result.stdout,submission_stderr=result.stderr);save_record()
    if result.returncode:
        record['status']='SUBMISSION_REJECTED';save_record()
        raise RuntimeError(result.stderr)
    sid=result.stdout.strip().split(';')[0]
    assert sid.isdigit(),'ambiguous submission: inspect record, do not resubmit'
    for jid in b['assigned_training_ids']:
        submissions[jid]=dict(slurm_id=sid,source_commit=b['source_commit'],submitted_at=time.time(),primary_start_dependencies=primary_ids,grouped_precheck=True,replaced_pending_slurm_id=details[jid]['old_slurm_id'],**commands[jid])
    submissions_path.write_text(json.dumps(submissions,indent=2))
    record.update(status='SUBMITTED',slurm_id=sid,updated_at=time.time());save_record()
    rows.append(record)
print(json.dumps(rows))
'''


if __name__ == '__main__':
    delivery=Path(json.loads((HERE/'audit_deployment.latest.json').read_text())['local_delivery'])
    binding=json.loads((delivery/'bindings.secondary.A100.json').read_text())
    control=str(PurePosixPath(binding['work_root']).parent.parent/'control')
    for _ in range(24):
        response=subprocess.run(command('destination','python3 -'),input=CODE.replace('BINDING',repr(control+'/bindings.json')).encode(),capture_output=True,timeout=120)
        if response.returncode:
            print(response.stdout.decode(errors='replace'),flush=True)
            raise RuntimeError(response.stderr.decode(errors='replace'))
        rows=json.loads(response.stdout)
        if rows:
            (HERE/'audit_grouped_control_prechecks.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(rows,indent=2),flush=True)
            print(json.dumps(activate('A100','destination',delivery),indent=2),flush=True)
            break
    else:
        raise RuntimeError('recorded A100 coordinator login host was not reached')
