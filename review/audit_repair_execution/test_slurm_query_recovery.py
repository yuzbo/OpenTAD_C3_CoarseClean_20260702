"""Replay Slurm query failures without submitting or running any model."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import audit_queue


class QueryRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='geosparse_mock_query_')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'runs' / 'M'
        self.root.mkdir(parents=True)
        self.control = self.base / 'control'
        self.control.mkdir()
        self.bindings = dict(work_root=str(self.root), primary_binding_paths=['unused'])
        (self.control / 'precheck_submissions.json').write_text(json.dumps({
            'first': {'slurm_id': '100'}, 'second': {'slurm_id': '100'}}))
        (self.root / 'slurm_state.json').write_text(json.dumps({
            'jobs': {'tr-a': {'slurm_id': '101', 'status': 'HELD_SUBMITTED'}}}))
        self.record = self.control / 'primary_start_dependencies.json'
        self.record.write_text(json.dumps({'100': {'primary_ids': ['9']}}))
        self.primary_patch = patch.object(audit_queue, 'primary_state', return_value=(['11', '10'], []))
        self.primary_patch.start()
        self.addCleanup(self.primary_patch.stop)

    def test_failed_read_preserves_records_and_makes_no_updates(self):
        before = self.record.read_bytes()
        error = subprocess.CalledProcessError(1, ['squeue'], stderr='Socket timed out')
        with patch.object(audit_queue.subprocess, 'check_output', side_effect=error), \
             patch.object(audit_queue.subprocess, 'run') as update:
            result = audit_queue.refresh_primary_start_dependencies(self.bindings)
        self.assertIn('Socket timed out', result['slurm_query_error']['detail'])
        self.assertEqual(self.record.read_bytes(), before)
        update.assert_not_called()

    def test_timeout_is_bounded_and_deferred(self):
        error = subprocess.TimeoutExpired(['squeue'], 60, stderr=b'controller busy')
        with patch.object(audit_queue.subprocess, 'check_output', side_effect=error) as query, \
             patch.object(audit_queue.subprocess, 'run') as update:
            result = audit_queue.refresh_primary_start_dependencies(self.bindings)
        self.assertEqual(query.call_args.kwargs['timeout'], 60)
        self.assertEqual(result['slurm_query_error']['detail'], 'controller busy')
        update.assert_not_called()

    def test_recovery_updates_only_owned_pending_jobs_once(self):
        error = subprocess.CalledProcessError(1, ['squeue'], stderr='Socket timed out')
        ok = subprocess.CompletedProcess(['scontrol'], 0, '', '')
        with patch.object(audit_queue.subprocess, 'check_output', side_effect=[error, '100\n101\n999\n', '100\n101\n999\n']), \
             patch.object(audit_queue.subprocess, 'run', return_value=ok) as update:
            self.assertIn('slurm_query_error', audit_queue.refresh_primary_start_dependencies(self.bindings))
            recovered = audit_queue.refresh_primary_start_dependencies(self.bindings)
            self.assertEqual(set(recovered), {'100', '101'})
            self.assertEqual(audit_queue.refresh_primary_start_dependencies(self.bindings), {})
        self.assertEqual(update.call_count, 2)
        self.assertEqual({c.args[0][2] for c in update.call_args_list}, {'JobId=100', 'JobId=101'})
        self.assertEqual(json.loads(self.record.read_text())['100']['primary_ids'], ['10', '11'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
