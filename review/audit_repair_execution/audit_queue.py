"""Coordinate repaired seed0 methods and deferred controls on actual Slurm resources.

Readiness uses source-specific GPU prechecks and artifact/resource dependencies.
Primary methods never depend on a score; controls wait for primary deployment.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def primary_state(bindings):
    """Only deployment/start state is used; no accuracy or loss is inspected."""
    started, waiting = [], []
    for path in bindings['primary_binding_paths']:
        primary = load(path)
        state_path = Path(primary['work_root']) / 'slurm_state.json'
        states = load(state_path)['jobs'] if state_path.exists() else {}
        for jid in primary['assigned_training_ids']:
            row = states.get(jid, {'status': 'PENDING'})
            if row.get('slurm_id') and row['status'] == 'SUBMITTED':
                started.append(str(row['slurm_id']))
            elif row['status'] != 'DONE':
                waiting.append(jid)
    return started, waiting


def scheduled_script(script, primary_ids, nice, resumable=False):
    directives = ['#SBATCH --nice=' + str(nice)]
    if resumable:
        # Keep the 24h maximum but accept earlier backfill allocations >=1h.
        # Training saves full state each epoch and resumes the same 60-epoch job.
        directives.append('#SBATCH --time-min=01:00:00')
    if primary_ids:
        # Slurm "after" becomes satisfied when the primary job starts. This
        # is not afterok and does not wait for a result, checkpoint, or mAP.
        directives.append('#SBATCH --dependency=after:' + ':'.join(primary_ids))
    return script.replace('#!/usr/bin/env bash\n', '#!/usr/bin/env bash\n' + '\n'.join(directives) + '\n', 1)


def refresh_primary_start_dependencies(bindings):
    """Keep pending controls behind the current primary resume allocations."""
    if not bindings['primary_binding_paths']:
        return {}
    primary_ids, waiting = primary_state(bindings)
    if waiting or not primary_ids:
        return {}
    primary_ids = sorted(set(primary_ids), key=int)
    root = Path(bindings['work_root'])
    control = root.parent.parent / 'control'
    prechecks_path = control / 'precheck_submissions.json'
    prechecks = load(prechecks_path) if prechecks_path.exists() else {}
    state_path = root / 'slurm_state.json'
    states = load(state_path)['jobs'] if state_path.exists() else {}
    owned = {str(row['slurm_id']) for row in list(prechecks.values()) + list(states.values())
             if row.get('slurm_id')}
    pending = set(subprocess.check_output(
        ['squeue', '-h', '--me', '--states=PENDING', '-o', '%i'], text=True).split())
    record_path = control / 'primary_start_dependencies.json'
    records = load(record_path) if record_path.exists() else {}
    updates = {}
    for sid in sorted(owned & pending, key=int):
        if records.get(sid, {}).get('primary_ids') == primary_ids:
            continue
        result = subprocess.run(
            ['scontrol', 'update', 'JobId=' + sid, 'Dependency=after:' + ':'.join(primary_ids)],
            capture_output=True, text=True)
        updates[sid] = dict(primary_ids=primary_ids, updated_at=time.time(),
                            status='UPDATED' if result.returncode == 0 else 'UPDATE_FAILED',
                            stderr=result.stderr.strip())
        if result.returncode == 0:
            records[sid] = updates[sid]
    if updates:
        save(record_path, records)
    return updates


def precheck(args, bindings, queue):
    root = Path(bindings['work_root']).parent.parent
    control = root / 'control'
    submissions_path = control / 'precheck_submissions.json'
    submissions = load(submissions_path) if submissions_path.exists() else {}
    jobs = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_id = {row['job_id']: row for row in jobs}
    primary_ids, waiting = primary_state(bindings)
    if waiting:
        return dict(status='WAITING_PRIMARY_ALLOCATION', jobs=waiting, prechecks=submissions)
    if __import__('shutil').disk_usage(root).free < 20 * 1024**3:
        return dict(status='BLOCKED_STORAGE', prechecks=submissions)
    nodes = []
    if bindings['cluster'] == 'N16R4':
        nodes = queue.free_gpu1_nodes(subprocess.check_output(['scontrol', 'show', 'nodes', '-d'], text=True))
    for jid in bindings['assigned_training_ids']:
        _, missing = queue.capabilities_for(by_id[jid], bindings)
        if not missing or jid in submissions:
            continue
        if bindings['cluster'] == 'N16R4' and not nodes:
            break
        out = root / 'prechecks' / jid
        out.mkdir(parents=True, exist_ok=True)
        script = queue.worker_script(out / 'unused_job.json', out, args.bindings, bindings)
        script = script[:script.rfind('exec ')]
        script = script.replace('#SBATCH --time=24:00:00', '#SBATCH --time=00:30:00')
        script = scheduled_script(script, primary_ids, bindings['slurm_nice'])
        if by_id[jid]['model']['budget_mode'] == 'dynamic':
            witness = [bindings['entrypoint'][0], str(control / 'full_method_witness.py'),
                       '--manifest', str(args.manifest), '--bindings', str(args.bindings),
                       '--job-id', jid, '--output', str(out / 'dynamic_policy_check.json')]
            script += shlex.join(witness) + '\n'
        argv = [bindings['entrypoint'][0], '-m', 'geosparse_ext.gpu_precheck', '--manifest', str(args.manifest), '--bindings', str(args.bindings), '--output', str(out), '--train-ids', jid, '--certify']
        script += 'exec ' + shlex.join(argv) + '\n'
        path = out / 'run.sbatch'
        path.write_text(script)
        argv = ['sbatch', '--parsable', '--job-name=gs-audit-check-' + jid, '--output=' + str(out / 'slurm-%j.out'), '--error=' + str(out / 'slurm-%j.err')]
        if nodes:
            argv.append('--nodelist=' + nodes.pop(0))
        argv.append(str(path))
        response = subprocess.run(argv, capture_output=True, text=True)
        if response.returncode:
            save(out / 'submission_blocked.json', dict(stderr=response.stderr, stdout=response.stdout, command=argv, observed_at=time.time()))
            break
        sid = response.stdout.strip().split(';')[0]
        if not sid.isdigit():
            raise RuntimeError('ambiguous precheck submission; inspect Slurm before retrying: ' + response.stdout)
        submissions[jid] = dict(slurm_id=sid, source_commit=bindings['source_commit'], submitted_at=time.time(), primary_start_dependencies=primary_ids)
        save(submissions_path, submissions)
    return dict(status='DEPLOYED', prechecks=submissions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--bindings', required=True, type=Path)
    args = parser.parse_args()
    bindings = load(args.bindings)
    sys.path.insert(0, bindings['repo_root'])
    from geosparse_ext import slurm_queue as queue
    import fcntl
    root = Path(bindings['work_root'])
    root.mkdir(parents=True, exist_ok=True)
    control = root.parent.parent / 'control'
    original_nodes = queue.free_gpu1_nodes
    if bindings['cluster'] == 'N16R4':
        def verified_nodes(text):
            # Slurm can choose GPU4 even when GPU1 is free, depending on CPU
            # affinity. Prefer only nodes with an actual GPU1 allocation witness;
            # the worker still rejects any later allocation of another device.
            allowed = load(control / 'gpu1_nodes.json')['nodes']
            available = set(original_nodes(text))
            return [node for node in allowed if node in available]
        queue.free_gpu1_nodes = verified_nodes
    lock = open(control / 'audit_supervisor.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original_worker = queue.worker_script
    original_readiness = queue.readiness
    def worker(job_file, output, bindings_file, current):
        primary_ids, waiting = primary_state(current)
        if waiting:
            raise RuntimeError('primary submission changed; keep supplementary work queued')
        return scheduled_script(original_worker(job_file, output, bindings_file, current), primary_ids, current['slurm_nice'], resumable=load(job_file)['kind']=='train')
    def readiness(job, by_id, current, directory, states):
        status, reasons = original_readiness(job, by_id, current, directory, states)
        if status == 'QUEUED_RESOURCE':
            _, waiting = primary_state(current)
            if waiting:
                return 'WAITING_PRIMARY_ALLOCATION', waiting
        return status, reasons
    while True:
        # Production queue is run for one scheduling pass. The supervisor
        # coordinates prechecks too, including after a storage block clears.
        dependency_updates = refresh_primary_start_dependencies(bindings)
        queue.worker_script = original_worker
        status = precheck(args, bindings, queue)
        queue.worker_script = worker
        queue.readiness = readiness
        sys.argv = [sys.argv[0], '--manifest', str(args.manifest), '--bindings', str(args.bindings), '--execute']
        queue.main()
        state = load(root / 'slurm_state.json')
        save(control / 'audit_progress.json', dict(updated_at=time.time(), pid=os.getpid(), host=socket.gethostname(), precheck_status=status, primary_dependency_updates=dependency_updates, states={jid:state['jobs'][jid]['status'] for jid in bindings['assigned_training_ids']}))
        time.sleep(60)


if __name__ == '__main__':
    main()
