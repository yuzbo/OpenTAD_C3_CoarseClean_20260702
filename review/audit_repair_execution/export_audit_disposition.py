"""Join registered jobs, current implementation refusal and observed execution.

This is an execution disposition, not another independent semantic audit.
"""
import csv
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
SOURCE=Path('E:/DeskTop/TAD/GeoSparse_Audit_Fixes_20260908')
sys.path.insert(0,str(SOURCE))
from geosparse_ext.matrix import compile_all,validate
from geosparse_ext.protocol import implementation_blockers


def main():
    jobs,refs=compile_all();validate(jobs);by_id={j['job_id']:j for j in jobs}
    snapshot=json.loads((HERE/'audit_status.json').read_text())
    latest=json.loads((HERE/'audit_deployment.latest.json').read_text())
    delivery=Path(latest['local_delivery'])
    inverse={v:k for k,v in json.loads((delivery/'old_to_repaired_job_ids.json').read_text()).items()}
    observed={}
    for host in snapshot['states'].values():
        live={r[0]:r for line in host['slurm'].splitlines() if len(r:=line.split('|'))>=3}
        for jid,run in host['runs'].items():
            q=run['queue'] or {};status=q.get('status','PENDING')
            if q.get('slurm_id') in live:status=live[q['slurm_id']][2]
            if host.get('observation_error'):status='OBSERVATION_UNAVAILABLE'
            observed[jid]=dict(cluster=host['cluster'],execution_status=status,slurm_id=q.get('slurm_id',''),
                completed_epochs=(run['progress'] or {}).get('completed_epochs',0),
                observed_at_utc=host.get('observed_at_utc',snapshot['observed_at_utc']),observation_error=host.get('observation_error') or '',
                gpu_precheck=(run['precheck'] or {}).get('routes',{}).get(jid,{}).get('status','NOT_RUN'))
    rows=[]
    for job in jobs:
        parent=job if job['kind']=='train' else by_id[job['source_train_id']]
        reasons=implementation_blockers(parent['model'])
        if parent['dataset']!='thumos14':reasons.append(parent['dataset']+' dataset protocol adapter is not implemented')
        if job['kind']=='diagnostic':reasons.append(job['suite']+' diagnostic implementation is pending')
        run=observed.get(parent['job_id'],{})
        status='BLOCKED_IMPLEMENTATION' if reasons else 'DEFERRED_SEED' if job['seed']!=0 else 'DEFERRED_PRIORITY'
        if not reasons and job['seed']==0 and run:
            status=run['execution_status'] if job['kind']=='train' or run.get('observation_error') else 'WAITING_OWN_COMPLETED_TRAIN_AND_SELECTION'
        row=dict(job_id=job['job_id'],old_job_id=inverse[job['job_id']],kind=job['kind'],training_id=parent['job_id'],
            label=job['label'],route=job['route'],dataset=job['dataset'],seed=job['seed'],
            source_commit=snapshot['source_commit'],static_status='BLOCKED' if reasons else 'ACCEPTED_NOT_COMPLETION',
            execution_status=status,blocking_reasons='; '.join(reasons),cluster=run.get('cluster',''),
            train_slurm_id=run.get('slurm_id',''),completed_train_epochs=run.get('completed_epochs',0),
            production_gpu_precheck=run.get('gpu_precheck','NOT_RUN'),
            artifact_contract=json.dumps(parent['artifact_contract'],sort_keys=True),
            observed_at_utc=run.get('observed_at_utc',snapshot['observed_at_utc']),observation_error=run.get('observation_error',''),model=json.dumps(parent['model'],sort_keys=True))
        rows.append(row)
    configs=[row for row in rows if row['kind']=='train' and row['seed']==0]
    assert len(rows)==1545 and len(configs)==204
    for name,data in [('job_disposition_1545.csv',rows),('configuration_disposition_204.csv',configs)]:
        with (delivery/name).open('w',encoding='utf-8-sig',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    from collections import Counter
    summary=dict(source_commit=snapshot['source_commit'],observed_at_utc=snapshot['observed_at_utc'],
        configurations=len(configs),jobs=len(rows),static_configurations=dict(Counter(r['static_status'] for r in configs)),
        job_execution_states=dict(Counter(r['execution_status'] for r in rows)),
        scope='registry and implementation refusal plus observed seed0 execution; not an independent semantic audit or evidence of final performance')
    (delivery/'disposition_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
