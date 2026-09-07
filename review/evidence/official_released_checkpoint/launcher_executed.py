"""Run untouched upstream entry points and retain full-split mAP-best externally."""
import argparse
import json
import inspect
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    args = parser.parse_args()
    binding = json.loads(Path(args.bindings).read_text())
    repo = Path(binding['repo'])
    output = Path(binding['outputs'][args.mode])
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    from mmengine.config import Config
    from opentad.datasets import build_dataset
    from opentad.models.detectors.actionformer import ActionFormer
    source_path = inspect.getfile(ActionFormer)
    assert Path(source_path).resolve().is_relative_to(repo.resolve())
    config = repo / 'configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py'
    cfg = Config.fromfile(str(config))
    overrides = {}
    for subset in ('train', 'val', 'test'):
        for key, value in [('ann_file', binding['annotations']), ('class_map', binding['class_map']),
                           ('data_path', binding['train_videos'] if subset == 'train' else binding['test_videos'])]:
            overrides[f'dataset.{subset}.{key}'] = value
    overrides.update({'evaluation.ground_truth_filename': binding['annotations'],
                      'model.backbone.custom.pretrain': binding['recognition_checkpoint'],
                      'work_dir': str(output), 'post_processing.save_dict': True})
    if args.mode == 'train':
        overrides.update({'workflow.val_start_epoch': 0, 'workflow.val_eval_interval': 5,
                          'workflow.checkpoint_interval': 5})
    cfg.merge_from_dict(overrides)
    annotation = json.loads(Path(binding['annotations']).read_text())['database']
    names = {part: sorted(k for k, v in annotation.items() if v['subset'] == part)
             for part in ('training', 'validation')}
    assert len(names['training']) == 200 and len(names['validation']) == 211
    assert cfg.scheduler.max_epoch == 100 and cfg.scheduler.warmup_epoch == 5
    assert cfg.workflow.end_epoch == 60 and cfg.solver.train.batch_size == 2
    assert cfg.model.type == 'ActionFormer' and cfg.model.projection.max_seq_len == 768
    datasets = {part: build_dataset(cfg.dataset[part]) for part in ('train', 'val', 'test')}
    assert len(datasets['train']) == 200 and len(datasets['test']) == 792
    for part in names:
        video_root = Path(binding['train_videos'] if part == 'training' else binding['test_videos'])
        missing = [name for name in names[part] if not (video_root / (name + '.mp4')).is_file()]
        assert not missing, missing
    cfg.dump(str(output / 'resolved_config.py'))
    save(output / 'preflight.json', dict(status='PASS', upstream_commit=binding['upstream_commit'],
         source_path=source_path, train_videos=names['training'], test_videos=names['validation'],
         dataset_lengths={k: len(v) for k, v in datasets.items()}, global_train_batch=2,
         schedule_epochs=100, training_epochs=60, changed_fields=overrides,
         status_scope='assets and official configuration; no performance claim'))
    if os.environ.get('PRECHECK_ONLY') == '1':
        return
    assert os.environ.get('SLURM_JOB_ID'), 'Execution requires an allocated compute node'
    world = 2 if args.mode == 'train' else 1
    argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}',
            str(repo / 'tools' / ('train.py' if args.mode == 'train' else 'test.py')), str(output / 'resolved_config.py'),
            '--seed', '0', '--id', '0']
    if args.mode == 'evaluate':
        argv += ['--checkpoint', binding['official_checkpoint']]
    # DictAction strips whitespace inside values. The resolved Config file
    # preserves the actual dataset paths (which contain spaces) exactly.
    shutil.copyfile(__file__, output / 'launcher_source.py')
    run = output / f'gpu{world}_id0'
    save(output / 'launch.json', dict(command=argv, source_commit=binding['upstream_commit'],
         mode=args.mode, seed=0, slurm_id=os.environ['SLURM_JOB_ID'], started_at=time.time(),
         checkpoint_origin='recognition pretraining' if args.mode == 'train' else 'official released TAD EMA'))
    epoch = None
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=str(repo))
    process = subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (output / 'console.log').open('a', buffering=1) as log:
        for line in process.stdout:
            log.write(line)
            if 'INFO:' in line or 'Error' in line or 'Traceback' in line:
                print(line, end='', flush=True)
            match = re.search(r'\[Train\]: Epoch (\d+) started', line)
            if match:
                epoch = int(match[1])
                save(output / 'progress.json', dict(status='TRAINING', active_epoch=epoch,
                     completed_epochs=epoch, slurm_id=os.environ['SLURM_JOB_ID']))
            if 'Average-mAP:' in line:
                from opentad.evaluations import build_evaluator
                prediction = run / 'result_detection.json'
                values = json.loads(prediction.read_text())
                assert set(values['results']) == set(names['validation'])
                evaluator = build_evaluator(dict(prediction_filename=values, **cfg.evaluation))
                metrics = {key: float(value) for key, value in evaluator.evaluate().items()}
                folder = output / ('released_checkpoint' if epoch is None else f'epoch_{epoch:03d}')
                folder.mkdir(exist_ok=True)
                shutil.copyfile(prediction, folder / 'predictions.json')
                row = dict(metrics=metrics, checkpoint_epoch=epoch, evaluated_videos=211,
                           inference_windows=792, weights='ema', seed=0, is_mock=False,
                           source_commit=binding['upstream_commit'], mode=args.mode)
                save(folder / 'metrics.json', row)
                if epoch is not None:
                    best_file = output / 'best.json'
                    previous = json.loads(best_file.read_text()) if best_file.exists() else None
                    if previous is None or metrics['average_mAP'] > previous['metrics']['average_mAP']:
                        checkpoint = run / 'checkpoint' / f'epoch_{epoch}.pth'
                        assert checkpoint.is_file()
                        shutil.copyfile(checkpoint, output / 'best.pth')
                        save(best_file, dict(row, checkpoint_path=str(output / 'best.pth')))
                else:
                    save(output / 'metrics.json', row)
    code = process.wait()
    completed = code == 0 and (output / ('best.json' if args.mode == 'train' else 'metrics.json')).exists()
    save(output / 'result.json', dict(status='COMPLETED' if completed else 'FAILED', exit_code=code,
         source_commit=binding['upstream_commit'], is_mock=False, mode=args.mode,
         completed_epochs=60 if completed and args.mode == 'train' else None,
         ended_at=time.time(), seed=0))
    if not completed:
        raise RuntimeError(f'Official {args.mode} did not complete; exit={code}; inspect console.log')


if __name__ == '__main__':
    main()
