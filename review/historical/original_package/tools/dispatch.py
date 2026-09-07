#!/usr/bin/env python3
"""A local multi-GPU artifact-DAG dispatcher, NOT an agent spawning service.

Plans safely by default. Real execution requires --execute and user-owned assets.
Runs all ready families fairly; it never reads model accuracy to unblock work.
Benchmark jobs reserve every configured local slot to avoid co-run interference.
"""
from __future__ import annotations
import argparse, collections, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(tmp,path)

def load_jobs(path):
    jobs=[json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    ids={x["job_id"] for x in jobs}
    if len(ids)!=len(jobs):raise ValueError("duplicate job IDs")
    for j in jobs:
        if j["kind"]=="train" and j["depends_on"]:raise ValueError("training result gate forbidden")
        if "metric_gate" in j:raise ValueError("accuracy gates forbidden")
        if any(d not in ids for d in j["depends_on"]):raise ValueError("unresolved dependency")
    return jobs

def external_blockers(job, bindings):
    missing=[]
    for req in job.get("external_requirements",[]):
        if req.startswith("dataset:"):
            p=bindings.get("datasets",{}).get(req.split(":",1)[1])
        elif req.startswith("checkpoint:"):
            p=bindings.get("checkpoints",{}).get(req.split(":",1)[1])
        else:p=bindings.get(req)
        if not p or not Path(p).exists():missing.append(req)
    return missing

def capability_blockers(job, bindings):
    directory=Path(bindings.get("capability_dir","/does-not-exist"))
    bad=[]
    for cap in job.get("capabilities",[]):
        p=directory/(cap+".json")
        try:
            obj=load_json(p)
            if not (obj.get("ready") is True and obj.get("commit") and obj.get("test_receipt")):
                bad.append(cap)
        except (OSError,ValueError):bad.append(cap)
    return bad

def check_result(job, output_dir):
    p=Path(output_dir)/"result.json"
    result=load_json(p)
    required=["job_id","status","is_mock","source_commit","resolved_config_sha256","split_sha256"]
    if any(k not in result for k in required):raise ValueError("incomplete result receipt")
    if result["job_id"]!=job["job_id"] or result["status"]!="completed" or result["is_mock"] is not False:
        raise ValueError("invalid result status or mock result")
    if job["kind"]=="train":
        if result.get("completed_epochs")!=job["epochs"]:raise ValueError("training epochs incomplete")
        ckpt=result.get("checkpoint_path")
        if not ckpt or not Path(ckpt).is_file():raise ValueError("checkpoint missing")
    return result

def fair_order(jobs):
    buckets=collections.OrderedDict()
    for j in jobs:buckets.setdefault((j["family"],j["route"],j["kind"]),collections.deque()).append(j)
    out=[]
    while any(buckets.values()):
        for q in buckets.values():
            if q:out.append(q.popleft())
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",required=True,type=Path)
    ap.add_argument("--bindings",required=True,type=Path)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--watch",action="store_true",help="Keep checking readiness, no accuracy gate.")
    ap.add_argument("--poll-seconds",type=float,default=5)
    ap.add_argument("--retry-failed",action="store_true")
    ap.add_argument("--retry-blocked",action="store_true",help="Retry adapter exit-78 jobs after implementation changes.")
    args=ap.parse_args()
    jobs=load_jobs(args.manifest);bind=load_json(args.bindings)
    print(json.dumps({"mode":"execute" if args.execute else "plan_only", "jobs":len(jobs),
        "kinds":dict(collections.Counter(j["kind"] for j in jobs)),
        "training_has_experiment_dependencies":False},ensure_ascii=False))
    if not args.execute:
        reasons=collections.Counter()
        for j in jobs:
            if external_blockers(j,bind):reasons["external_assets_missing"]+=1
            if capability_blockers(j,bind):reasons["implementation_not_ready"]+=1
        print(json.dumps({"readiness":dict(reasons),"launched":0},ensure_ascii=False));return 0
    repo=Path(bind["repo_root"]).resolve()
    if not repo.is_dir():raise ValueError("repo_root not found; bind real assets first")
    root=Path(bind["work_root"]).resolve();root.mkdir(parents=True,exist_ok=True)
    slots=bind.get("slots",[])
    if not slots:raise ValueError("explicit resource slots required")
    devices=[s["cuda_visible_devices"] for s in slots]
    flat=[x.strip() for d in devices for x in d.split(",")]
    if len(flat)!=len(set(flat)):raise ValueError("GPU slots overlap")
    if any(s.get("host","local")!="local" for s in slots):
        raise ValueError("This dispatcher is local; use a cluster adapter, not fake remote slots")
    lock=open(root/"dispatcher.lock","a+")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("another dispatcher owns this run root")
    manifest_sha=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    state_path=root/"state.json"
    state=load_json(state_path) if state_path.exists() else {"manifest_sha256":manifest_sha,"jobs":{}}
    if state["manifest_sha256"]!=manifest_sha:raise ValueError("manifest changed: use versioned work_root")
    for entry in state["jobs"].values():
        if entry.get("status")=="RUNNING":
            raise RuntimeError("orphan RUNNING job: reconcile owned PID/checkpoint before resuming; do not duplicate")
        if args.retry_failed and entry.get("status")=="FAILED":entry.update(status="PENDING")
        if args.retry_blocked and entry.get("status")=="BLOCKED_IMPLEMENTATION":entry.update(status="PENDING")
    ordered=fair_order(jobs)
    active={};last_signature=None;stopping=False;fair_cursor=0;completed_since_bench=0
    def stop(signum, frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        while True:
            # Refresh only readiness/asset paths. Entry points & protocol must be pinned by adapters.
            new_bind=load_json(args.bindings)
            if new_bind.get("slots")!=bind.get("slots") or new_bind.get("repo_root")!=bind.get("repo_root"):
                raise ValueError("cannot mutate resources/repository during dispatch")
            bind=new_bind
            for jid,item in list(active.items()):
                rc=item["process"].poll()
                if rc is None:continue
                item["log"].close()
                e=state["jobs"][jid]
                try:
                    if rc==78:
                        e.update(status="BLOCKED_IMPLEMENTATION",reason="adapter exit 78",finished_at=time.time())
                    elif rc!=0:raise RuntimeError(f"worker exited {rc}")
                    else:
                        check_result(item["job"],item["output"])
                        e.update(status="DONE",finished_at=time.time())
                except Exception as exc:e.update(status="FAILED",reason=str(exc),finished_at=time.time())
                del active[jid]
                if item["job"]["kind"] != "benchmark":completed_since_bench += 1
            if stopping:break
            used={i for x in active.values() for i in x["reserved"]}
            launched=False
            ready_bench = any(
                j.get("exclusive") and state["jobs"].get(j["job_id"],{}).get("status") not in {"DONE","RUNNING","FAILED","BLOCKED_IMPLEMENTATION"}
                and all(state["jobs"].get(d,{}).get("status")=="DONE" for d in j["depends_on"])
                and not external_blockers(j,bind) and not capability_blockers(j,bind)
                for j in ordered)
            draining = ready_bench and (completed_since_bench >= max(1,len(slots)) or not active)
            indices=list(range(fair_cursor,len(ordered)))+list(range(0,fair_cursor))
            for idx in indices:
                job=ordered[idx]
                jid=job["job_id"];e=state["jobs"].setdefault(jid,{"status":"PENDING"})
                if e["status"] in {"DONE","RUNNING","FAILED","BLOCKED_IMPLEMENTATION"}:continue
                deps=job.get("depends_on",[])
                if any(state["jobs"].get(d,{}).get("status")!="DONE" for d in deps):
                    e.update(status="BLOCKED_ARTIFACT",reason=deps);continue
                ext=external_blockers(job,bind)
                if ext:e.update(status="BLOCKED_EXTERNAL_ASSET",reason=ext);continue
                caps=capability_blockers(job,bind)
                if caps:e.update(status="BLOCKED_CAPABILITY",reason=caps);continue
                if draining and not job.get("exclusive"):
                    e.update(status="QUEUED_RESOURCE",reason="benchmark isolation; not a scientific gate");continue
                free=[i for i,s in enumerate(slots) if i not in used and s["class"]==job["slot_class"]]
                if not free or (job.get("exclusive") and active):
                    e.update(status="QUEUED_RESOURCE");continue
                slot_i=free[0];reserved=set(range(len(slots))) if job.get("exclusive") else {slot_i}
                output=root/"runs"/jid;output.mkdir(parents=True,exist_ok=True)
                job_file=output/"job.json"
                cap_receipts={cap:load_json(Path(bind["capability_dir"])/(cap+".json")) for cap in job["capabilities"]}
                atomic_json(job_file,dict(job,dependency_outputs={d:str(root/"runs"/d) for d in deps},
                    capability_receipts=cap_receipts,
                    launch_bindings_sha256=hashlib.sha256(args.bindings.read_bytes()).hexdigest()))
                fmt={"job_file":str(job_file),"output_dir":str(output),"bindings_file":str(args.bindings.resolve())}
                command=[s.format(**fmt) for s in bind["entrypoint"]]
                env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=slots[slot_i]["cuda_visible_devices"]
                env["GEOSPARSE_JOB_ID"]=jid
                log=open(output/"worker.log","a",buffering=1)
                try:proc=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                except Exception as exc:
                    log.close();e.update(status="FAILED",reason=str(exc));continue
                e.update(status="RUNNING",pid=proc.pid,started_at=time.time(),slot=slots[slot_i]["id"],
                         command=command,manifest_sha256=manifest_sha)
                active[jid]=dict(process=proc,log=log,job=job,output=output,reserved=reserved)
                used|=reserved;launched=True;fair_cursor=(idx+1)%len(ordered)
                if job.get("exclusive"):completed_since_bench=0
            atomic_json(state_path,state)
            counts=dict(collections.Counter(e["status"] for e in state["jobs"].values()))
            signature=json.dumps(counts,sort_keys=True)
            if signature!=last_signature:print(signature,flush=True);last_signature=signature
            if not active:
                if all(state["jobs"].get(j["job_id"],{}).get("status")=="DONE" for j in jobs):break
                if not args.watch:break
            time.sleep(max(.1,args.poll_seconds))
    finally:
        for jid,item in active.items():
            proc=item["process"]
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            item["log"].close()
            state["jobs"][jid].update(status="FAILED",reason="dispatcher interrupted; resume from own checkpoint")
        atomic_json(state_path,state)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    unfinished=sum(state["jobs"].get(j["job_id"],{}).get("status")!="DONE" for j in jobs)
    return 2 if unfinished else 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:print(f"ERROR: {exc}",file=sys.stderr);sys.exit(1)
