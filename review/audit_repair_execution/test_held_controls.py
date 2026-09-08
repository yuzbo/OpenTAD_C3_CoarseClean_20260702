"""Focused scheduler tests with fake Slurm; no model or scientific results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if os.name == 'nt':
    # Windows replay tests cover scheduling logic; Linux also exercises real flock.
    sys.modules['fcntl'] = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
import held_controls as controls


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


class HeldControlsTest(unittest.TestCase):
    def setUp(self):
        fixtures = Path(__file__).resolve().parent / 'held_control_unit_fixtures'
        fixtures.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='is_mock_', dir=fixtures))
        write(self.root / 'IS_MOCK.json', {'is_mock': True, 'scope': 'Slurm scheduling unit test'})
        self.jobs = [dict(job_id='tr-a', kind='train', seed=0, depends_on=[], capabilities=['native']),
                     dict(job_id='tr-b', kind='train', seed=0, depends_on=[], capabilities=['native'])]
        self.args = SimpleNamespace(manifest=self.root / 'manifest.jsonl', bindings=self.root / 'bindings.json')
        self.args.manifest.write_text('\n'.join(json.dumps(job) for job in self.jobs))
        self.b = dict(work_root=str(self.root), repo_root=str(self.root), source_commit='M',
            primary_binding_paths=['primary.json'], assigned_training_ids=['tr-a', 'tr-b'],
            cluster='A100', max_concurrent_jobs=1)
        write(self.args.bindings, self.b)
        self.state_path = self.root / 'slurm_state.json'
        write(self.state_path, dict(source_commit='M', manifest_sha256='manifest', jobs={
            j['job_id']:dict(status='BLOCKED_CAPABILITY', attempts=[]) for j in self.jobs}))
        self.ready = False
        self.nodes = []
        self.q = SimpleNamespace(load_json=read, save_json=write, file_id=lambda p: 'manifest',
            source_commit=lambda p: 'M', job_blockers=lambda j,b: ([],[]),
            capabilities_for=self.caps, worker_script=lambda *a: '#!/usr/bin/env bash\nexec actual_entry\n',
            free_gpu1_nodes=lambda text: list(self.nodes))
        self.commands = []
        self.live = {}
        self.next_id = 100
        self.fail_release = False
        self.malformed = False
        self.run_patch = patch.object(controls.subprocess, 'run', side_effect=self.run_command)
        self.output_patch = patch.object(controls.subprocess, 'check_output', side_effect=self.output_command)
        self.disk_patch = patch.object(controls.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*1024**3))
        self.run_patch.start(); self.output_patch.start(); self.disk_patch.start()
        self.addCleanup(self.run_patch.stop); self.addCleanup(self.output_patch.stop); self.addCleanup(self.disk_patch.stop)

    def caps(self, job, bindings):
        return ({'native': {'ready': True, 'commit': 'M', 'supported_variants': [job['job_id']] }}, []) if self.ready else ({}, ['native'])

    def readiness(self, job, *args):
        return ('QUEUED_RESOURCE', []) if self.ready else ('BLOCKED_CAPABILITY', ['native'])

    def run_command(self, args, **kwargs):
        self.commands.append(args)
        if args[0] == 'sbatch':
            assert '--hold' in args
            if self.malformed:
                return subprocess.CompletedProcess(args, 0, 'unknown response', '')
            self.next_id += 1
            self.live[str(self.next_id)] = 'PENDING'
            return subprocess.CompletedProcess(args, 0, str(self.next_id)+'\n', '')
        if args[:2] == ['scontrol', 'release'] and self.fail_release:
            return subprocess.CompletedProcess(args, 1, '', 'release refused')
        return subprocess.CompletedProcess(args, 0, '', '')

    def output_command(self, args, **kwargs):
        if args[0] == 'squeue':
            return '\n'.join(k+'|'+v for k,v in self.live.items())
        return 'real node parser is covered separately'

    def submit(self):
        controls.submit_controls(self.args, self.b, self.q, lambda script: script)

    def release(self, waiting=None):
        controls.release_controls(self.args, self.b, self.q, self.readiness,
                                  lambda b: (['10','11'], waiting or []))

    def test_submit_all_once_and_preserve_existing_attempt(self):
        state=read(self.state_path)
        state['jobs']['tr-a']['attempts']=[dict(slurm_id='old', status='RESUME_PENDING')]
        write(self.state_path,state)
        self.submit(); self.submit()
        self.assertEqual(sum(c[0]=='sbatch' for c in self.commands),2)
        state=read(self.state_path)
        self.assertEqual({r['status'] for r in state['jobs'].values()},{'HELD_SUBMITTED'})
        self.assertEqual(state['jobs']['tr-a']['attempts'][0]['slurm_id'],'old')
        self.assertEqual(len(state['jobs']['tr-a']['attempts']),2)
        self.assertEqual(read(self.root/'runs/tr-b/job.json')['capability_receipts'],{})

    def test_missing_capability_and_primary_wait_do_not_release(self):
        self.submit(); self.release()
        self.assertFalse(any(c[:2]==['scontrol','release'] for c in self.commands))
        self.ready=True; self.release(['primary-not-submitted'])
        self.assertFalse(any(c[:2]==['scontrol','release'] for c in self.commands))

    def test_release_injects_real_receipts_and_respects_concurrency(self):
        self.submit(); self.ready=True; self.release()
        state=read(self.state_path)
        self.assertEqual(state['jobs']['tr-a']['status'],'SUBMITTED')
        self.assertEqual(state['jobs']['tr-b']['status'],'HELD_SUBMITTED')
        self.assertTrue(read(self.root/'runs/tr-a/job.json')['capability_receipts']['native']['ready'])
        self.assertIn(['scontrol','update','JobId=101','Dependency=after:10:11'],self.commands)
        self.assertEqual(sum(c[:2]==['scontrol','release'] for c in self.commands),1)
        self.submit()
        self.assertEqual(sum(c[0]=='sbatch' for c in self.commands),2)

    def test_n16_reservation_blocks_then_sets_verified_node(self):
        self.b['cluster']='N16R4'; self.submit(); self.ready=True; self.release()
        self.assertFalse(any(c[:2]==['scontrol','release'] for c in self.commands))
        self.nodes=['g0087']; self.release()
        self.assertIn(['scontrol','update','JobId=101','Dependency=after:10:11','ReqNodeList=g0087'],self.commands)
        self.assertEqual(read(self.state_path)['jobs']['tr-a']['attempts'][-1]['node'],'g0087')

    def test_release_failure_keeps_held(self):
        self.submit(); self.ready=True; self.fail_release=True; self.release()
        self.assertEqual({r['status'] for r in read(self.state_path)['jobs'].values()},{'HELD_SUBMITTED'})

    def test_missing_slurm_job_delegates_reconciliation_without_resubmit(self):
        self.submit(); self.live={}; self.release(); self.submit()
        self.assertEqual(sum(c[0]=='sbatch' for c in self.commands),2)
        self.assertEqual({r['status'] for r in read(self.state_path)['jobs'].values()},{'SUBMITTED'})

    def test_ambiguous_submission_blocks_retry(self):
        self.malformed=True
        with self.assertRaisesRegex(RuntimeError,'ambiguous sbatch'): self.submit()
        with self.assertRaisesRegex(RuntimeError,'ambiguous earlier'): self.submit()
        self.assertEqual(sum(c[0]=='sbatch' for c in self.commands),1)

    def test_low_storage_does_not_submit(self):
        with patch.object(controls.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)):
            self.submit()
        self.assertFalse(self.commands)


if __name__=='__main__':
    unittest.main(verbosity=2)
