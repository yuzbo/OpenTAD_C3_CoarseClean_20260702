"""The two clusters start independent writers of the shared protocol records."""
import json
import multiprocessing


def _write_at_barrier(path, barrier, results, value):
    from geosparse_ext import records
    replace = records.os.replace
    def simultaneous_replace(source, target):
        barrier.wait(timeout=20)
        replace(source, target)
    records.os.replace = simultaneous_replace
    try:
        records.save_json(path, {"writer": value, "videos": list(range(200))})
        results.put(None)
    except Exception as exc:
        results.put(repr(exc))


def test_independent_protocol_writers_keep_complete_records(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier, results = context.Barrier(2), context.Queue()
    path = tmp_path / "split.json"
    processes = [context.Process(target=_write_at_barrier, args=(path, barrier, results, i)) for i in range(2)]
    for process in processes:
        process.start()
    errors = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert errors == [None, None]
    record = json.loads(path.read_text())
    assert record["writer"] in {0, 1} and record["videos"] == list(range(200))
    assert not list(tmp_path.glob("*.tmp"))
