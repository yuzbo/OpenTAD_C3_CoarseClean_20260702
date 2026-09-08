"""One real window: verify an A Heavy-MAC cap against the frozen GPU executor.

Run as a file. This is an executor precheck, not a trained constrained model,
full evaluation, detector-loss test, or latency measurement.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def probe_executor(geo, video):
    import torch
    from geosparse_ext.geometry import native_layout, detector_validity
    from geosparse_ext.sparse import heavy_macs
    from geosparse_research.heavy_cap import allocate_heavy_prefix

    if geo.training or geo.route != 'A' or geo.config['selector'] not in {'hybrid', 'pg_only'}:
        raise ValueError('precheck requires eval-mode A with the learned selector')
    if video.frames_hi.shape[0] != 1:
        raise ValueError('executor precheck accepts exactly one window')
    encoder = geo.encoder
    width, layers = encoder.source.embed_dims, len(encoder.source.blocks)
    if encoder.active_layers != tuple(range(layers)) or encoder.prefix_depth:
        raise ValueError('precheck requires the default complete Heavy depth')
    original = copy.deepcopy(geo.config)
    valid = native_layout(video).valid
    expected_mask = detector_validity(video.valid_frames, geo.query_length)
    cases = []
    try:
        with torch.no_grad():
            # Independent source-plan full reference, retaining the learned selector
            # so padded positions are not re-enabled by M's selector=none limit.
            geo.config.update(budget_mode='fixed', budget=1.0)
            reference, plan, _, _, _ = geo(video)
            full_trace = copy.deepcopy(encoder.trace)
            full_cost = heavy_macs(full_trace, width)
            if not torch.equal(plan.selected_native.reshape_as(valid), valid):
                raise AssertionError('source full plan does not cover exactly eligible native positions')
            geo.config.clear()
            geo.config.update(original)
            for fraction in (0.0, 0.5, 1.0):
                capped, budget = allocate_heavy_prefix(plan, valid, 'A', width=width,
                                                      layers=layers, max_fraction=fraction)
                state, executed, _, _, _ = geo(video, forced_plan=capped)
                trace = copy.deepcopy(encoder.trace)
                actual = heavy_macs(trace, width)
                row = budget['rows'][0]
                if actual != row['allocated_heavy_macs'] or actual > row['cap_macs']:
                    raise AssertionError('actual QKV/MLP execution differs from the cap allocation')
                if full_cost != row['full_valid_heavy_macs']:
                    raise AssertionError('independent source full trace differs from the denominator')
                if not torch.equal(executed.selected_native, capped.selected_native):
                    raise AssertionError('forced cap plan was not consumed')
                if not torch.equal(state.valid, expected_mask) or state.features.shape != reference.features.shape:
                    raise AssertionError('cap changed the dense detector grid or frame-valid mask')
                if not torch.isfinite(state.features).all():
                    raise AssertionError('nonfinite detector-input features')
                if fraction == 1.0:
                    torch.testing.assert_close(reference.features, state.features, atol=0, rtol=0)
                cases.append(dict(cap=fraction, allocation=budget, actual_heavy_macs=actual,
                                  trace=trace, feature_shape=list(state.features.shape),
                                  feature_dtype=str(state.features.dtype), valid_positions=int(state.valid.sum()),
                                  full_reference_equal=(True if fraction == 1.0 else None)))
    finally:
        geo.config.clear()
        geo.config.update(original)
    return dict(status='PASS', route='A', width=width, layers=layers, cases=cases,
                source_full_heavy_macs=full_cost, source_full_trace=full_trace,
                uses_torch_dispatch=False, is_scientific_result=False,
                scope='GeoSparse executor through detector-input features and mask; no head loss, mAP or latency')


def run(args):
    training = args.training_run.resolve()
    bindings = json.loads((training / 'bindings.json').read_text())
    model_root = Path(bindings['repo_root']).resolve()
    research_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(model_root))
    sys.path.insert(1, str(research_root))
    import torch
    from mmengine.config import Config
    import geosparse_ext.detector as model_module
    import geosparse_research.heavy_cap as cap_module
    from geosparse_ext.data import video_batch
    from geosparse_ext.prediction_export import training_source, evaluation_dataset
    from geosparse_ext.records import load_json, save_json, source_commit, content_id, file_id
    from geosparse_ext.runtime import require_gpu, seed_all, loader, load_trained_model

    bindings, completion, provenance, training_job = training_source(training)
    if (source_commit(model_root) != completion['source_commit']
            or Path(model_module.__file__).resolve().parents[1] != model_root
            or Path(cap_module.__file__).resolve().parents[1] != research_root):
        raise ValueError('model or measurement imports escaped their recorded source')
    if training_job['route'] != 'A':
        raise ValueError('this first production precheck is registered for A only')
    require_gpu()
    seed_all(training_job['seed'])
    cfg = Config.fromfile(str(training / 'resolved_opentad.py'))
    resolved = load_json(training / 'resolved_config.json')
    if (content_id(resolved) != provenance['resolved_config_sha256']
            or json.loads(json.dumps(cfg.to_dict())) != resolved['opentad']
            or file_id(cfg.model.recognition_checkpoint) != provenance['weights_sha256']):
        raise ValueError('typed config or initialization differs from this training')
    split = load_json(Path(bindings['protocol_root']) / training_job['dataset'] / 'split.json')
    if content_id(split) != provenance['split_sha256'] or len(split['training']) != 200 or len(split['validation']) != 211:
        raise ValueError('expected recorded full200/full211 protocol')
    dataset, _ = evaluation_dataset(cfg, 'validation')
    data = loader(dataset, 1, bindings['runtime']['num_workers'], training_job['seed'])
    if len(data.dataset) != 792:
        raise ValueError('expected the complete official 792-window pipeline')
    args.output.mkdir(parents=True, exist_ok=False)
    identity = dict(model_source=completion['source_commit'], measurement_source=source_commit(research_root),
                    source_train_id=training_job['job_id'], training_provenance=provenance,
                    checkpoint_identity=completion['checkpoint_identity'], seed=training_job['seed'],
                    precision='float32_without_autocast', measured_windows=1,
                    selection='first window in official test loader order', is_scientific_result=False)
    save_json(args.output / 'measurement.json', identity)
    try:
        job = dict(training_job, kind='evaluate', depends_on=[training_job['job_id']],
                   dependency_outputs={training_job['job_id']: str(training)})
        model = load_trained_model(job, cfg, args.output, provenance)
        batch = next(iter(data))
        video = video_batch(batch['inputs'], batch['masks'], batch['metas'], 'cuda')
        if tuple(video.frames_hi.shape) != (1, 3, 768, 160, 160):
            raise ValueError('registered precheck requires the real 768x160 input')
        result = probe_executor(model.geosparse, video)
        torch.cuda.synchronize()
        result.update(identity, device=torch.cuda.get_device_name(0),
                      video_id=str(video.video_id[0]), window_id=str(video.window_id[0]),
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        save_json(args.output / 'result.json', result)
        print(json.dumps(dict(status=result['status'], source_train_id=training_job['job_id'],
                              measured_windows=1, is_scientific_result=False)), flush=True)
    except Exception as error:
        save_json(args.output / 'failure.json', dict(identity, status='FAILED',
                  error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
