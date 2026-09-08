"""Submit assigned controls now; release their held Slurm jobs when ready."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time


@contextmanager
def state_file(args, bindings, queue):
    root = Path(bindings['work_root'])
    with (root / 'slurm_queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / 'slurm_state.json'
        state = queue.load_json(path)
        if state['source_commit'] != bindings['source_commit'] or state['manifest_sha256'] != queue.file_id(args.manifest):
            raise ValueError('held submission source/manifest changed')
        if any(row['status'] == 'SUBMITTING' for row in state['jobs'].values()):
            raise RuntimeError('ambiguous earlier submission; reconcile Slurm before retrying')
        yield root, path, state


def jobs_for(args):
    return {row['job_id']: row for row in
            (json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip())}


def submit_controls(args, bindings, queue, schedule):
    """Only the current secondary assignment is eligible; no new variants."""
    if not bindings['primary_binding_paths']:
        return
    if queue.source_commit(Path(bindings['repo_root'])) != bindings['source_commit']:
        raise ValueError('execution snapshot differs from binding')
    jobs = jobs_for(args)
    with state_file(args, bindings, queue) as (root, path, state):
        for jid in bindings['assigned_training_ids']:
            item, job = state['jobs'][jid], jobs[jid]
            if item['status'] in {'DONE', 'FAILED', 'SUBMITTED', 'SUBMITTING', 'HELD_SUBMITTED', 'BLOCKED_EXECUTION', 'SELECTION_PENDING'}:
                continue
            if job['kind'] != 'train' or job['seed'] != 0:
                raise ValueError('only assigned seed0 control training may be pre-submitted')
            implementation, assets = queue.job_blockers(job, bindings)
            if implementation or assets:
                continue
            if shutil.disk_usage(root).free < 20 * 1024**3:
                break
            output = root / 'runs' / jid
            output.mkdir(parents=True, exist_ok=True)
            if (output / 'result.json').exists() or (output / 'failure.json').exists():
                raise RuntimeError('existing terminal receipt must be reconciled: ' + jid)
            caps, _ = queue.capabilities_for(job, bindings)
            job_path, binding_path = output / 'job.json', output / 'bindings.json'
            queue.save_json(job_path, dict(job, dependency_outputs={}, capability_receipts=caps))
            queue.save_json(binding_path, bindings)
            script = output / 'run.sbatch'
            script.write_text(schedule(queue.worker_script(job_path, output, binding_path, bindings)))
            attempt = len(item['attempts']) + 1
            item.update(status='SUBMITTING', reason='submitting held control; do not duplicate')
            queue.save_json(path, state)
            command = ['sbatch', '--parsable', '--hold', '--job-name=' + jid,
                       f'--output={output}/attempt{attempt:02d}-%j.out',
                       f'--error={output}/attempt{attempt:02d}-%j.err', str(script)]
            response = subprocess.run(command, capture_output=True, text=True)
            if response.returncode:
                item.update(status='QUEUED_RESOURCE' if 'AssocMaxSubmitJobLimit' in response.stderr else 'FAILED',
                            reason='sbatch rejected held submission: ' + response.stderr)
                queue.save_json(path, state)
                break
            sid = response.stdout.strip().split(';')[0]
            if not sid.isdigit():
                raise RuntimeError('ambiguous sbatch response; inspect SUBMITTING state before retry')
            item['attempts'].append(dict(slurm_id=sid, submitted_at=time.time(), node='held-unassigned'))
            item.update(status='HELD_SUBMITTED', slurm_id=sid, slurm_state='PENDING',
                        held_reason='waiting for own GPU precheck and permitted resources',
                        reason='formal training submitted with Slurm user hold')
            queue.save_json(path, state)


def release_controls(args, bindings, queue, readiness, primary_state):
    if not bindings['primary_binding_paths']:
        return
    jobs = jobs_for(args)
    with state_file(args, bindings, queue) as (root, path, state):
        held = [jid for jid in bindings['assigned_training_ids'] if state['jobs'][jid]['status'] == 'HELD_SUBMITTED']
        if not held:
            return
        live = dict(line.split('|', 1) for line in subprocess.check_output(
            ['squeue', '-h', '--me', '-o', '%i|%T'], text=True).splitlines() if '|' in line)
        active = sum(row['status'] == 'SUBMITTED' for row in state['jobs'].values())
        primary_ids, waiting = primary_state(bindings)
        nodes = None
        for jid in held:
            item, job = state['jobs'][jid], jobs[jid]
            sid = str(item['slurm_id'])
            if live.get(sid) != 'PENDING':
                # Delegate actual cancellation/completion/accounting to the original reconciler.
                item.update(status='SUBMITTED', reason='held Slurm job changed state; reconcile actual status')
                continue
            ready, reasons = readiness(job, jobs, bindings, root, state['jobs'])
            if ready != 'QUEUED_RESOURCE':
                item['held_reason'] = ready + ': ' + '; '.join(map(str, reasons))
                continue
            if waiting:
                item['held_reason'] = 'waiting for current primary allocation: ' + ', '.join(waiting)
                continue
            if active >= bindings['max_concurrent_jobs']:
                item['held_reason'] = 'waiting for supplementary concurrency slot'
                continue
            if shutil.disk_usage(root).free < 20 * 1024**3:
                item['held_reason'] = 'waiting for 20 GiB free disk'
                continue
            if bindings['cluster'] == 'N16R4':
                if nodes is None:
                    nodes = queue.free_gpu1_nodes(subprocess.check_output(['scontrol', 'show', 'nodes', '-d'], text=True))
                if not nodes:
                    item['held_reason'] = 'waiting for permitted physical GPU1; primary node reservation remains active'
                    continue
                node = nodes.pop(0)
            else:
                node = 'slurm-selected'
            caps, missing = queue.capabilities_for(job, bindings)
            if missing:
                item['held_reason'] = 'GPU capability changed before release'
                continue
            output = root / 'runs' / jid
            queue.save_json(output / 'job.json', dict(job, dependency_outputs={}, capability_receipts=caps))
            updates = ['scontrol', 'update', 'JobId=' + sid, 'Dependency=' + ('after:' + ':'.join(primary_ids) if primary_ids else '')]
            if node != 'slurm-selected':
                updates.append('ReqNodeList=' + node)
            response = subprocess.run(updates, capture_output=True, text=True)
            if response.returncode:
                item['held_reason'] = 'Slurm prerequisite update failed: ' + response.stderr
                continue
            response = subprocess.run(['scontrol', 'release', sid], capture_output=True, text=True)
            if response.returncode:
                item['held_reason'] = 'Slurm release failed: ' + response.stderr
                continue
            item.update(status='SUBMITTED', slurm_state='PENDING', reason=[], held_reason='', released_at=time.time())
            item['attempts'][-1].update(node=node, released_at=item['released_at'])
            active += 1
            queue.save_json(path, state)
        queue.save_json(path, state)
