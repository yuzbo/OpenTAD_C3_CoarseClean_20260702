"""Real-batch dynamic-policy gradient witness; no scientific metrics or training run."""
import argparse
import gc
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--bindings', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    # This script lives outside the immutable model repository; resolve its
    # imports from the actual bound snapshot instead of the script directory.
    b = json.loads(args.bindings.read_text())
    sys.path.insert(0, b['repo_root'])
    from geosparse_ext.records import load_json, save_json, source_commit, prepare_split
    from geosparse_ext.protocol import resolve_opentad_config
    from geosparse_ext.runtime import require_gpu, seed_all, loader, optimizer_for
    from opentad.models.builder import build_detector
    assert source_commit(b['repo_root']) == b['source_commit']
    require_gpu()
    job = next(j for j in map(json.loads, args.manifest.read_text().splitlines()) if j['job_id'] == args.job_id)
    assert job['kind'] == 'train' and job['seed'] == 0
    assert job['model']['budget_mode'] == 'dynamic' and job['model']['estimator'] == 'pg_acquisition'
    split_dir = Path(b['protocol_root']) / 'thumos14'
    prepare_split(b['annotations'], split_dir)
    cfg = resolve_opentad_config(job, b, split_dir / 'annotations.json')
    seed_all(0)
    batch = next(iter(loader(cfg.dataset.train, b['runtime']['microbatch'], 0, 0)))
    model = build_detector(cfg.model).cuda().train()
    optimizer = optimizer_for(model, cfg.optimizer)
    # Exercise the registered post-warmup curriculum. No model/config field is
    # changed, and this model is discarded after the correctness witness.
    model.epoch = 21
    budgets, report = [], None
    for attempt in range(8):
        seed_all(attempt)
        model.minibatch = 32
        model.pending_cost.clear()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.cuda.amp.autocast(enabled=cfg.solver.amp):
            losses = model(**batch, return_loss=True)
        plan = model.latest_route_plan
        budgets.append(plan.requested_budget.detach().cpu().tolist())
        if not bool(plan.learned_sample.any()) or model.latest_acquisition is None:
            del losses, plan
            gc.collect()
            continue
        assert {'actor_loss', 'critic_loss', 'acquisition_loss'} <= losses.keys()
        assert torch.isfinite(losses['cost'])
        losses['cost'].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        budget_grad = model.geosparse.scout.budget.weight.grad
        gain_grad = model.geosparse.scout.gain[-1].weight.grad
        assert budget_grad is not None and float(budget_grad.float().norm()) > 0
        assert gain_grad is not None and float(gain_grad.float().norm()) > 0
        old_weight = model.geosparse.scout.budget.weight.detach().clone()
        before = dict(dual=float(model.geosparse.dual), cost_ema=float(model.geosparse.cost_ema))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.solver.clip_grad_norm)
        optimizer.step()
        model.after_optimizer_step()
        assert not model.pending_cost
        assert not torch.equal(old_weight, model.geosparse.scout.budget.weight.detach())
        assert torch.isfinite(model.geosparse.dual) and torch.isfinite(model.geosparse.cost_ema)
        report = dict(status='PASS', job_id=job['job_id'], source_commit=b['source_commit'],
            scope='correctness witness only; discarded model, zero formal training epochs, no scientific metrics',
            is_mock=False, actual_input_shape=list(batch['inputs'].shape),
            budget_mode=job['model']['budget_mode'], curriculum_epoch=21, probe_minibatch=32,
            learned_samples=plan.learned_sample.detach().cpu().tolist(), sampled_budgets=budgets,
            budget_gradient_norm=float(budget_grad.float().norm()), gain_gradient_norm=float(gain_grad.float().norm()),
            acquisition=model.latest_acquisition, optimizer_updated_budget_head=True,
            controller_before=before, controller_after=dict(dual=float(model.geosparse.dual), cost_ema=float(model.geosparse.cost_ema)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        break
    assert report is not None, 'No learned dynamic action plus acquisition was exercised'
    save_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
