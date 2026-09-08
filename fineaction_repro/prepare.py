"""Index existing FineAction media and create a native-frame annotation view."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def inspect_video(item):
    name, path = item
    try:
        import decord
        source = Path(path)
        reader = decord.VideoReader(str(source), num_threads=1)
        frames = len(reader)
        if frames <= 0:
            raise ValueError('no decoded frames')
        # Endpoints catch unusable decoder/index combinations before training.
        shape = list(reader.get_batch([0, frames-1]).shape)
        return dict(video=name, path=str(source), status='PASS', frame=frames,
                    decoder_fps=float(reader.get_avg_fps()), decoded_shape=shape,
                    bytes=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns)
    except Exception as error:
        return dict(video=name, path=str(path), status='FAILED', error=repr(error))


def native_annotation(annotation, records):
    result = deepcopy(annotation)
    required = {k for k,v in annotation['database'].items()
                if v['subset'] in ('training', 'validation')}
    if set(records) != required or any(r['status'] != 'PASS' for r in records.values()):
        raise ValueError('complete successful indexing of train and validation is required')
    for name in required:
        result['database'][name]['frame'] = int(records[name]['frame'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', required=True)
    p.add_argument('--videos', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    assets, videos, output = map(Path, (args.assets, args.videos, args.output))
    annotation = json.loads((assets/'annotations/annotations_gt.json').read_text())
    class_map = (assets/'annotations/category_idx.txt').read_text().splitlines()
    db = annotation['database']
    counts = {s:sum(v['subset']==s for v in db.values()) for s in ('training','validation','testing')}
    assert counts == dict(training=8440, validation=4174, testing=4118), counts
    assert len(class_map)==106 and len(set(class_map))==106
    assert {a['label'] for v in db.values() for a in v['annotations']} <= set(class_map)
    required = {k:v for k,v in db.items() if v['subset'] in ('training','validation')}
    paths = {}
    for name in required:
        matches = [videos/(name+suffix) for suffix in ('.mp4','.webm') if (videos/(name+suffix)).is_file()]
        if len(matches)!=1:
            raise ValueError(f'{name}: expected one mp4/webm source; found {matches}')
        paths[name] = matches[0]
    output.mkdir(parents=True,exist_ok=True)
    journal = output/'video_index.jsonl'
    indexed = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line); name=row['video']
            if name in paths and row['status']=='PASS':
                stat=paths[name].stat()
                if row['path']==str(paths[name]) and row['bytes']==stat.st_size and row['mtime_ns']==stat.st_mtime_ns:
                    indexed[name]=row
    save(output/'progress.json',dict(status='INDEXING',complete=len(indexed),total=len(required)))
    todo = [(name,str(path)) for name,path in paths.items() if name not in indexed]
    with journal.open('a',buffering=1) as stream, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(inspect_video,item) for item in todo]
        for future in as_completed(futures):
            row = future.result(); indexed[row['video']] = row
            stream.write(json.dumps(row)+'\n')
            if len(indexed)%100==0 or row['status']!='PASS':
                status=dict(status='INDEXING',complete=len(indexed),total=len(required),
                            failures=sum(r['status']!='PASS' for r in indexed.values()))
                save(output/'progress.json',status);print(json.dumps(status),flush=True)
    failures = [row for row in indexed.values() if row['status']!='PASS']
    if failures:
        save(output/'receipt.json',dict(status='FAILED',failures=failures,counts=counts))
        raise RuntimeError(f'{len(failures)} media indexing failures; training is blocked')
    native = native_annotation(annotation,indexed)
    save(output/'annotations/native_metadata.json',native)
    # Original GT remains the evaluator's source, including the original durations.
    (output/'annotations/annotations_gt.json').write_bytes((assets/'annotations/annotations_gt.json').read_bytes())
    (output/'annotations/category_idx.txt').write_bytes((assets/'annotations/category_idx.txt').read_bytes())
    links=output/'videos';links.mkdir(exist_ok=True)
    for name,path in paths.items():
        link=links/(name+'.mp4')
        if link.is_symlink():
            assert link.resolve()==path.resolve()
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(path.resolve())
    receipt=dict(status='PASS',counts=counts,indexed_labelled_videos=len(indexed),
                 frame_count_differences=sum(row['frame']!=required[name]['frame_num'] for name,row in indexed.items()),
                 original_ground_truth=str(output/'annotations/annotations_gt.json'),
                 native_metadata=str(output/'annotations/native_metadata.json'),
                 completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 scope='all labelled videos indexed and endpoints decoded; not all frames decoded or model tested')
    save(output/'receipt.json',receipt);save(output/'progress.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
