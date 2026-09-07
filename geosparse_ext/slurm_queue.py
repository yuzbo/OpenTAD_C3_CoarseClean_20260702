"""Independent physical GPU1 allocations; all manifest jobs remain registered.

The coordinator runs on the login node and only submits/observes owned Slurm
jobs. Model execution is always inside a Slurm allocation and a clean snapshot.
"""
import argparse
from collections import Counter, OrderedDict, deque
import fcntl
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from .entry import job_blockers
from .records import load_json, save_json, file_id, source_commit


def fair_order(jobs):
    groups = OrderedDict()
    for job in jobs:
        groups.setdefault((job["family"], job["route"], job["kind"]), deque()).append(job)
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.popleft())
    return ordered


def free_gpu1_nodes(text):
    nodes = []
    for block in text.split("NodeName=")[1:]:
        match = re.search(r"GresUsed=.*?IDX:([0-9,-]+)\)", block)
        if not match or "State=MIXED " not in block:
            continue
        used = set()
        for item in match[1].split(","):
            bounds = item.split("-")
            used.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        # With GPU0 occupied, GPU1 is the first available GRES index. The
        # worker still verifies the actual allocation before creating CUDA.
        allocated = int(re.search(r"CPUAlloc=(\d+)", block)[1])
        total = int(re.search(r"CPUTot=(\d+)", block)[1])
        if 0 in used and 1 not in used and total - allocated >= 8:
            nodes.append(block.split()[0])
    return nodes


def capabilities_for(job, bindings):
    receipts, missing = {}, []
    variant = job.get("source_train_id", job["job_id"])
    for name in job["capabilities"]:
        path = Path(bindings["capability_dir"]) / f"{name}.json"
        if not path.is_file():
            missing.append(name)
            continue
        cap = load_json(path)
        if (not cap.get("ready") or cap.get("commit") != bindings["source_commit"]
                or variant not in cap.get("supported_variants", [])
                or not Path(cap.get("test_receipt", "/missing")).is_file()):
            missing.append(name)
        else:
            receipts[name] = cap
    return receipts, missing


def readiness(job, jobs_by_id, bindings, root, states):
    if job["seed"] not in bindings.get("active_seeds", [0, 1, 2]):
        return "DEFERRED_SEED", ["single-seed feasibility phase: repeated seeds are registered but deferred"]
    if ("assigned_training_ids" in bindings
            and job.get("source_train_id", job["job_id"]) not in bindings["assigned_training_ids"]):
        return "DEFERRED_HOST", ["training and its dependent artifacts are assigned to the other cluster"]
    resolved = job if job["kind"] == "train" else dict(job, model=jobs_by_id[job["source_train_id"]]["model"])
    implementation, assets = job_blockers(resolved, bindings)
    if implementation:
        return "BLOCKED_IMPLEMENTATION", implementation
    if assets:
        return "BLOCKED_EXTERNAL_ASSET", assets
    _, missing = capabilities_for(job, bindings)
    if missing:
        return "BLOCKED_CAPABILITY", missing
    if any(states.get(dep, {}).get("status") != "DONE" for dep in job["depends_on"]):
        return "BLOCKED_ARTIFACT", job["depends_on"]
    return "QUEUED_RESOURCE", []


def worker_script(job_file, output, bindings_file, bindings):
    q = shlex.quote
    if bindings.get("cluster", "N16R4") == "A100":
        return "\n".join([
            "#!/usr/bin/env bash", "#SBATCH --partition=a100x", "#SBATCH --account=pxyai", "#SBATCH --qos=normal",
            "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=12", "#SBATCH --gres=gpu:a100:1",
            "#SBATCH --mem-per-gpu=120G",
            "#SBATCH --time=24:00:00", "source /etc/profile", "set -eo pipefail", "module load CUDA/11.8",
            "source " + q(str(Path(bindings["entrypoint"][0]).parent / "activate")),
            "set -u", "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1", "cd " + q(bindings["repo_root"]),
            "exec " + " ".join(q(value) for value in [bindings["entrypoint"][0], "-m", "geosparse_ext.entry",
                                "--job", str(job_file), "--output", str(output), "--bindings", str(bindings_file)]), ""])
    return "\n".join([
        "#!/usr/bin/env bash", "#SBATCH --partition=gpu", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8", "#SBATCH --gres=gpu:1", "#SBATCH --time=24:00:00",
        "source /etc/profile", "set -euo pipefail", "module load cuda/11.8", "module load miniforge3/24.11",
        "source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate",
        '[[ "${SLURM_JOB_GPUS:-}" == "1" && "${CUDA_VISIBLE_DEVICES:-}" == "0" ]] || exit 78',
        "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2",
        "cd " + q(bindings["repo_root"]),
        "exec " + " ".join(q(value) for value in [bindings["entrypoint"][0], "-m", "geosparse_ext.entry",
                            "--job", str(job_file), "--output", str(output), "--bindings", str(bindings_file)]), ""])


def reconcile(job, state, root):
    if state.get("status") != "SUBMITTED":
        return
    slurm_id = state["slurm_id"]
    query = subprocess.run(["squeue", "-h", "-j", slurm_id, "-o", "%i|%T"], capture_output=True, text=True)
    if any(line.split("|")[0] == slurm_id for line in query.stdout.splitlines()):
        state["slurm_state"] = query.stdout.strip().split("|")[-1]
        return
    output = root / "runs" / job["job_id"]
    if (output / "result.json").is_file():
        result = load_json(output / "result.json")
        recorded = load_json(output / "source_commits.json") if (output / "source_commits.json").is_file() else {}
        if (result.get("job_id") != job["job_id"] or result.get("status") != "completed" or result.get("is_mock") is not False
                or not recorded or any(result.get(key) != value for key, value in recorded.items())
                or (job["kind"] == "train" and (result.get("completed_epochs") != 60 or not Path(result.get("checkpoint_path", "/missing")).is_file()))):
            state.update(status="FAILED", reason="invalid completion receipt")
        else:
            state.update(status="DONE", result=str(output / "result.json"))
    elif (output / "failure.json").is_file():
        state.update(status="FAILED", reason=load_json(output / "failure.json").get("reason"))
    elif (output / "blocked.json").is_file():
        state.update(status="BLOCKED_EXECUTION", reason=load_json(output / "blocked.json"))
    else:
        query = subprocess.run(["sacct", "-X", "-n", "-P", "-j", slurm_id, "--format=JobIDRaw,State,ExitCode"], capture_output=True, text=True)
        rows = [line.split("|") for line in query.stdout.splitlines() if line.startswith(slurm_id + "|")]
        if not rows:
            return  # Slurm accounting has not published the completed job yet.
        terminal, exit_code = rows[0][1:3]
        if terminal in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}:
            return
        if terminal == "TIMEOUT" and (output / "checkpoint" / "last.pth").is_file():
            state.update(status="RESUME_PENDING", reason="own 24-hour allocation timed out; resume own checkpoint")
        elif exit_code == "78:0":
            state.update(status="RESOURCE_RETRY", reason="allocation did not map physical GPU1 to container GPU0")
        else:
            state.update(status="FAILED", reason=f"Slurm {terminal}, exit {exit_code}; no successful result receipt")
    if state["status"] != "SUBMITTED" and state.get("attempts"):
        state["attempts"][-1].update(status=state["status"], finished_at=time.time(), reason=state.get("reason"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    bindings = load_json(args.bindings)
    active_seeds = bindings.get("active_seeds", [0, 1, 2])
    if not active_seeds or len(set(active_seeds)) != len(active_seeds) or set(active_seeds) - {0, 1, 2}:
        raise ValueError("active_seeds must be a non-empty subset of the registered seeds 0/1/2")
    max_concurrent = bindings.get("max_concurrent_jobs", 1)
    if not isinstance(max_concurrent, int) or max_concurrent < 1:
        raise ValueError("max_concurrent_jobs must be a positive integer")
    if bindings.get("cluster", "N16R4") not in {"N16R4", "A100"}:
        raise ValueError("this coordinator supports the verified N16R4 and A100 clusters")
    root = Path(bindings["work_root"])
    root.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_id = {j["job_id"]: j for j in jobs}
    if len(by_id) != len(jobs) or any("metric_gate" in j or (j["kind"] == "train" and j["depends_on"]) for j in jobs):
        raise ValueError("duplicate IDs or scientific result gates")
    if any(set(j["depends_on"]) - by_id.keys() for j in jobs):
        raise ValueError("unknown artifact dependency")
    if "assigned_training_ids" in bindings and set(bindings["assigned_training_ids"]) - {j["job_id"] for j in jobs if j["kind"] == "train"}:
        raise ValueError("host assignment contains unknown training IDs")
    if source_commit(Path(bindings["repo_root"])) != bindings["source_commit"]:
        raise ValueError("bound execution snapshot is not clean or differs")
    path = root / "slurm_state.json"
    state = load_json(path) if path.exists() else dict(manifest_sha256=file_id(args.manifest), source_commit=bindings["source_commit"], cursor=0, jobs={})
    if state["manifest_sha256"] != file_id(args.manifest) or state["source_commit"] != bindings["source_commit"]:
        raise ValueError("source/manifest changed; use a versioned work root")
    state["active_seeds"] = active_seeds
    state["max_concurrent_jobs"] = max_concurrent
    lock = open(root / "slurm_queue.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any(item["status"] == "SUBMITTING" for item in state["jobs"].values()):
        raise RuntimeError("previous sbatch response was not recorded: reconcile the saved script/Slurm job before restarting; no duplicate submitted")
    ordered, last_counts = fair_order(jobs), None
    priority = {job_id: index for index, job_id in enumerate(bindings.get("priority_training_ids", []))}
    while True:
        for job in jobs:
            item = state["jobs"].setdefault(job["job_id"], dict(status="PENDING", attempts=[]))
            if args.execute:
                reconcile(job, item, root)
            if item["status"] in {"DONE", "FAILED", "SUBMITTED", "SUBMITTING", "BLOCKED_EXECUTION"}:
                continue
            status, reason = readiness(job, by_id, bindings, root, state["jobs"])
            item.update(status=status, reason=reason)
        active = sum(item["status"] in {"SUBMITTED", "SUBMITTING"} for item in state["jobs"].values())
        ready = [i for i in list(range(state["cursor"], len(ordered))) + list(range(state["cursor"]))
                 if state["jobs"][ordered[i]["job_id"]]["status"] == "QUEUED_RESOURCE"]
        ready.sort(key=lambda i: priority.get(ordered[i].get("source_train_id", ordered[i]["job_id"]), len(priority)))
        free_bytes = shutil.disk_usage(root).free
        if ready and free_bytes < 20 * 1024**3:
            for index in ready:
                state["jobs"][ordered[index]["job_id"]].update(reason=f"storage wait: {free_bytes / 1024**3:.1f} GiB free; 20 GiB required before starting another run")
        if args.execute and ready and active < max_concurrent and free_bytes >= 20 * 1024**3:
            nodes = (["slurm-selected"] if bindings.get("cluster") == "A100" else
                     free_gpu1_nodes(subprocess.check_output(["scontrol", "show", "nodes", "-d"], text=True)))
            if nodes:
                index = ready[0]
                job = ordered[index]
                item = state["jobs"][job["job_id"]]
                output = root / "runs" / job["job_id"]
                output.mkdir(parents=True, exist_ok=True)
                job_file, local_bindings = output / "job.json", output / "bindings.json"
                if (output / "result.json").exists():
                    raise RuntimeError(f"{job['job_id']} already has a completion receipt; reconcile it before any new attempt")
                caps, missing = capabilities_for(job, bindings)
                if missing:
                    raise RuntimeError("capability changed during submission")
                save_json(job_file, dict(job, dependency_outputs={dep: str(root / "runs" / dep) for dep in job["depends_on"]}, capability_receipts=caps))
                save_json(local_bindings, bindings)
                script = output / "run.sbatch"
                script.write_text(worker_script(job_file, output, local_bindings, bindings))
                attempt = len(item["attempts"]) + 1
                item.update(status="SUBMITTING", reason="submission in progress; do not duplicate on restart")
                save_json(path, state)
                command = ["sbatch", "--parsable", "--job-name=" + job["job_id"],
                           f"--output={output}/attempt{attempt:02d}-%j.out", f"--error={output}/attempt{attempt:02d}-%j.err", str(script)]
                if bindings.get("cluster", "N16R4") == "N16R4":
                    command.insert(2, "--nodelist=" + nodes[0])
                submission = subprocess.run(command, capture_output=True, text=True)
                if submission.returncode:
                    resource_wait = "AssocMaxSubmitJobLimit" in submission.stderr
                    item.update(status="QUEUED_RESOURCE" if resource_wait else "FAILED",
                                reason="sbatch rejected submission: " + submission.stderr)
                else:
                    slurm_id = submission.stdout.strip().split(";")[0]
                    if not slurm_id.isdigit():
                        raise RuntimeError("ambiguous sbatch response; reconcile SUBMITTING state before retry")
                    item["attempts"].append(dict(slurm_id=slurm_id, submitted_at=time.time(), node=nodes[0]))
                    item.update(status="SUBMITTED", slurm_id=slurm_id, slurm_state="PENDING", reason=[])
                    state["cursor"] = (index + 1) % len(ordered)
                    print(json.dumps(dict(submitted=job["job_id"], slurm_id=slurm_id, node=nodes[0])), flush=True)
        save_json(path, state)
        counts = dict(Counter(item["status"] for item in state["jobs"].values()))
        if counts != last_counts:
            print(json.dumps(dict(counts=counts, mode="execute" if args.execute else "plan_only", free_disk_gib=round(shutil.disk_usage(root).free / 1024**3, 1))), flush=True)
            last_counts = counts
        if not args.watch or not args.execute or all(item["status"] in {"DONE", "FAILED", "DEFERRED_SEED", "DEFERRED_HOST"} for item in state["jobs"].values()):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
