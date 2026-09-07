"""Describe the actual entry validator; acceptance is NOT a readiness/result claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geosparse_ext.entry import job_blockers
from geosparse_ext.matrix import compile_all, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'review/manifests/experiments.corrected.full.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'review')
    args = parser.parse_args()
    jobs = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    validate(jobs)
    compiled, _ = compile_all()
    assert jobs == compiled, 'Review matrix must match the pinned source compiler exactly'
    by_id = {job['job_id']: job for job in jobs}
    rows = []
    for job in jobs:
        resolved = job if job['kind'] == 'train' else dict(job, model=by_id[job['source_train_id']]['model'])
        problems, _ = job_blockers(resolved, {})
        rows.append(dict(job_id=job['job_id'], kind=job['kind'], route=job['route'], dataset=job['dataset'],
                         seed=job['seed'], label=job['label'], families=job['families'],
                         static_entry_accepts=not problems, implementation_blockers=problems,
                         readiness='NOT_ASSESSED', result='NOT_ASSESSED'))
    count = Counter(row['kind'] for row in rows)
    accepted = Counter(row['kind'] for row in rows if row['static_entry_accepts'])
    unique = [row for row in rows if row['kind'] == 'train' and row['seed'] == 0]
    args.output.mkdir(parents=True, exist_ok=True)
    payload = dict(model_source_commit='902fa05b5c64452ff1c94b82cabce801943d3484',
                   scope='Actual static entry checks only; assets, GPU receipts, semantic completeness and results are NOT inferred.',
                   job_counts=dict(count), static_entry_accepted_counts=dict(accepted),
                   unique_dataset_configurations=len(unique), jobs=rows)
    (args.output / 'IMPLEMENTATION_CATALOG.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 逐配置实现登记', '',
             '依据实际 `entry.job_blockers` 生成。通过静态参数检查不等于所有字段生效、GPU 就绪、实验完成或科学结论成立。',
             '表中只展示 seed 0；全部种子、子任务及其拒绝原因见同目录 JSON。真实运行状态另见 `STATUS.zh.md`。', '',
             f'共 {len(unique)} 个配置与数据集组合，{count["train"]} 个训练注册项；静态入口接受其中 {accepted["train"]} 个训练注册项。', '',
             '|seed 0 训练 ID|路线|数据集|配置名|静态检查|明确拒绝原因|', '|---|---|---|---|---|---|']
    for row in unique:
        reason = '; '.join(row['implementation_blockers']).replace('|', '\\|') or '仍需语义检查、真实资产及本配置 GPU 凭证'
        lines.append(f'|{row["job_id"]}|{row["route"]}|{row["dataset"]}|{row["label"]}|{"接受参数" if row["static_entry_accepts"] else "阻塞"}|{reason}|')
    (args.output / 'IMPLEMENTATION_CATALOG.zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({key: payload[key] for key in ['job_counts', 'static_entry_accepted_counts', 'unique_dataset_configurations']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
