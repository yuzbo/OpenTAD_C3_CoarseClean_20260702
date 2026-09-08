"""Activate held formal control submissions on the two secondary coordinators."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import subprocess
from deploy_corrected import command, put
from activate_dependency_refresh import CODE as PREVIOUS_ACTIVATION

HERE=Path(__file__).resolve().parent
REVISION='backfill-1h-held-controls-v3-release-query-retry'
CODE=PREVIOUS_ACTIVATION.replace('before_dependency_refresh_', 'before_control_submission_').replace(
    'dependency_refresh_amendment.json', 'control_submission_amendment.json').replace(
    "str(ctl/'audit_queue.py')],check=True)",
    "str(ctl/'audit_queue.py'),str(ctl/'held_controls.py')],check=True)")


def activate(cluster,target,delivery):
    b=json.loads((delivery/f'bindings.secondary.{cluster}.json').read_text())
    control=str(PurePosixPath(b['work_root']).parent.parent/'control')
    for name in ('held_controls.py','audit_queue.py'):
        put(target,control+'/'+name,(HERE/name).read_bytes())
    for _ in range(24):
        code=CODE.replace('BINDING',repr(control+'/bindings.json')).replace('REVISION',repr(REVISION))
        response=subprocess.run(command(target,'python3 -'),input=code.encode(),capture_output=True,timeout=120)
        if response.returncode:
            raise RuntimeError(response.stderr.decode(errors='replace'))
        rows=json.loads(response.stdout)
        if rows:
            (HERE/f'audit_submitted_controls_activation.{cluster}.json').write_text(json.dumps(rows,indent=2))
            return cluster,rows
    raise RuntimeError('recorded coordinator host was not reached: '+cluster)


if __name__=='__main__':
    delivery=Path(json.loads((HERE/'audit_deployment.latest.json').read_text())['local_delivery'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        result=dict(pool.map(lambda args:activate(*args,delivery),[('N16R4','source'),('A100','destination')]))
    print(json.dumps(result,indent=2))
