#!/usr/bin/env python3
"""Compile a finite preregistered experiment DAG. No training is performed.

Only artifact dependencies are admitted: every training job is independent of
all other training results. Reused identical configurations are deduplicated.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2]
PRIMARY = ["thumos14", "activitynet13"]
FAMILIES = {
 "F00": "稠密极限与架构性能税",
 "F01": "三路线主结果及固定/动态预算曲线",
 "F02": "选择策略与DUCA相关基线",
 "F03": "静态深度和后缀条件执行",
 "F04": "选择原子与编码粒度",
 "F05": "时间几何、支持范围与receiver",
 "F06": "局部配额、动态预算与成本目标",
 "F07": "Scout可见性、分辨率和容量",
 "F08": "训练估计器、探索和反事实成本",
 "F09": "时间空间因子与原图ROI",
 "F10": "混合尺度与表示交互",
 "F11": "一次/两次获取及上下文范围",
 "F12": "第三数据集/更大Backbone迁移",
 "F13": "检测网格与检测头稳健性",
}

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
def digest(x, n=12):
    return hashlib.sha256(canonical(x).encode()).hexdigest()[:n]

def base(route, **kw):
    d = dict(route=route, axis="ST", selector="hybrid", budget_mode="fixed",
             budget=0.5, temporal_atom_tubelets=1, spatial_group=2,
             geometry="native_support", receiver="support_attention",
             scout_resolution=112, scout_width=128, scout_temporal_stride=1,
             exploration="default", estimator="pg_acquisition",
             probe_every=32, backbone="videomae_b", head="actionformer",
             query_length=768, heavy_context="parent16",
             tia_scope="source_native", cost_target="traced_macs",
             roi_mode="none", source_resolution=160,
             quota="global_zero_allowed", warmup_epochs=6, evidence_slots=4,
             receiver_layers=2, fusion_variant="residual", coarse_variant="mean_shared",
             route_refresh="once", rounds=1, allow_global_zero=True,
             detail_probe_fraction=0.0, actionness_aux=False)
    d.update(kw)
    if d["selector"] == "none":
        d["estimator"] = "none"
        d["probe_every"] = 0
    return d


def compile_all():
    train_by_sig = {}
    refs = defaultdict(list)
    def add(family, label, cfg, datasets=PRIMARY, seeds=SEEDS):
        for dataset in datasets:
            for seed in seeds:
                signature = dict(model=cfg, dataset=dataset, seed=seed,
                                 protocol_version="geosparse-audit-repair-20260908", epochs=60)
                sig = digest(signature)
                jid = "tr-" + sig
                if sig not in train_by_sig:
                    route = cfg["route"]
                    caps = ["common", "route_"+route]
                    if cfg["selector"] in {"hybrid","pg_only","threshold","gradient","st"}:
                        caps += ["router"]
                    if cfg.get("cost_target")=="hardware_lut":caps += ["hardware_lut"]
                    if cfg.get("roi_mode")!="none":caps += ["roi"]
                    if cfg.get("rounds",1)>1:caps += ["acquisition_round2"]
                    req = ["dataset:"+dataset, "checkpoint:"+cfg["backbone"]]
                    if route == "DUCA": req.append("legacy_duca_manifest")
                    train_by_sig[sig] = dict(
                      job_id=jid, kind="train", family=family, families=[family],
                      label=label, route=route, dataset=dataset, seed=seed,
                      model=cfg, epochs=60, checkpoints=list(range(4,60,5)),
                      artifact_contract=dict(version="audit-repair-20260908", checkpoint_indexing="zero_based",
                                             saved_completed_epochs=list(range(5,61,5)),
                                             diagnostic_completed_epochs=[5,20,40,60]),
                      depends_on=[], capabilities=sorted(set(caps)),
                      external_requirements=req, slot_class="gpu", exclusive=False,
                      protocol_version="geosparse-audit-repair-20260908", is_mock=False)
                else:
                    job = train_by_sig[sig]
                    if family not in job["families"]: job["families"].append(family)
                refs[family].append(jid)

    # F00: dense limit and representation taxes. Router disabled for all-full.
    # This dense arm shares GeoSparse's amended preprocessing, split and head
    # grid. The untouched upstream AdaTAD baseline is tracked separately.
    for route, label in [("DENSE","geosparse_dense_control"),("A","A_full"),
                         ("B","B_full"),("C","C_all_fine"),("COARSE","cheap_only")]:
        add("F00", label, base(route, selector="none", estimator="none",
            budget=0.0 if route=="COARSE" else 1.0, probe_every=0))

    # F01: primary routes; all are registered at once, not promoted by results.
    for route in ["A","B"]:
        for axis in ["T","ST"]:
            for r in [.25,.5,.75]:
                add("F01",f"{route}_{axis}_fixed_{r}",base(route,axis=axis,budget=r))
        for r in [.25,.5,.75]:
            add("F01",f"{route}_ST_dynamic_{r}",base(route,budget=r,budget_mode="dynamic"))
    for r in [.25,.5,.75]:
        add("F01",f"C_fixed_{r}",base("C",budget=r,
            selector="none" if r==.25 else "hybrid",
            estimator="none" if r==.25 else "pg_acquisition",
            probe_every=0 if r==.25 else 32))
    for r in [.5,.75]:
        add("F01",f"C_dynamic_{r}",base("C",budget=r,budget_mode="dynamic"))

    # F02: route selection, not claimed to reproduce papers unless assets match.
    for route in ["A","B"]:
        for policy in ["uniform","random","motion","actionness","uncertainty","cdf_native","pg_only"]:
            add("F02",f"{route}_{policy}",base(route,axis="T",selector=policy,
                estimator="pg" if policy=="pg_only" else "task_aux_or_none",probe_every=0),["thumos14"])
    for r in [.25,.5,.75]:
        add("F02",f"legacy_DUCA_{r}",base("DUCA",axis="T",selector="legacy",budget=r,
            estimator="legacy",probe_every=0))
        for policy in ["uniform","hybrid"]:
            add("F02",f"wholeclip_{policy}_{r}",base("B",axis="T",selector=policy,
                budget=r,temporal_atom_tubelets=8,
                estimator="pg_acquisition" if policy=="hybrid" else "none",
                probe_every=32 if policy=="hybrid" else 0),["thumos14"])
    for policy in ["uniform","random"]:
        add("F02",f"C_{policy}",base("C",selector=policy,estimator="none",probe_every=0),["thumos14"])

    # F03: fair teacher-free depth baselines, distinct from official PBD.
    for depth in [4,6,8,10]:
        add("F03",f"static_depth_{depth}",base("DEPTH",selector="static",active_depth=depth,
            estimator="none",probe_every=0),PRIMARY)
    add("F03","pbd_inspired_tf",base("DEPTH",selector="pbd_inspired_tf",active_depth=8,
        estimator="current_model_ablation",probe_every=0),PRIMARY)
    for prefix in [0,2,4]:
        add("F03",f"BCR_prefix{prefix}",base("BCR",axis="T",prefix_depth=prefix),["thumos14"])

    # F04: no fixed local quota; group size may naturally change effective count.
    for route in ["A","B"]:
        for p in [1,2,4,8]:
            add("F04",f"{route}_atom{p}",base(route,axis="T",temporal_atom_tubelets=p),["thumos14"])
    for group in [1,2,7]:
        add("F04",f"A_spatialgroup{group}",base("A",spatial_group=group),["thumos14"])

    # F05: matched training and inference distributions plus deliberately bad controls.
    for receiver in ["rank_interp","physical_interp","concat_scatter","timestamp_attention","support_attention"]:
        add("F05",f"B_{receiver}",base("B",receiver=receiver),["thumos14"])
    for variant in ["no_position","native_position_only","physical_bias","rank_tia_negative_control"]:
        add("F05",f"A_{variant}",base("A",geometry=variant),["thumos14"])
    for variant in ["centroid_only","support_no_scale","native_support"]:
        add("F05",f"C_{variant}",base("C",geometry=variant),["thumos14"])

    # F06: dynamic budget controller, fixed quota and hardware-cost alternatives.
    for route in ["A","B"]:
        for option in ["per_clip_equal_quota","global_nonzero_per_clip","global_zero_allowed"]:
            add("F06",f"{route}_{option}",base(route,axis="T",quota=option),["thumos14"])
        for selector in ["threshold","pg_only"]:
            add("F06",f"{route}_dynamic_{selector}",base(route,selector=selector,
                budget_mode="dynamic",estimator="pg_acquisition" if selector=="threshold" else "pg"),["thumos14"])
        add("F06",f"{route}_hardware_cost",base(route,budget_mode="dynamic",cost_target="hardware_lut"),["thumos14"])
        add("F06",f"{route}_budget_nozero",base(route,budget_mode="dynamic",allow_global_zero=False),["thumos14"])

    # F07: one-factor-at-a-time; no unbounded Cartesian product.
    for route in ["A","B"]:
        for res in [80,112,160]:
            add("F07",f"{route}_scout_res{res}",base(route,scout_resolution=res),["thumos14"])
        for width in [64,128,256]:
            add("F07",f"{route}_scout_width{width}",base(route,scout_width=width),["thumos14"])
        for stride in [1,2,4]:
            add("F07",f"{route}_scout_tstride{stride}",base(route,scout_temporal_stride=stride),["thumos14"])
        add("F07",f"{route}_detail_probe",base(route,detail_probe_fraction=.03125),["thumos14"])
        add("F07",f"{route}_actionness_aux",base(route,actionness_aux=True),["thumos14"])

    # F08: all train for 60 epochs, not early-screen/promote.
    for route in ["A","B"]:
        for est in ["pg","pg_acquisition","retention_gradient","zero_gate_probe","straight_through"]:
            add("F08",f"{route}_est_{est}",base(route,estimator=est,
                probe_every=0 if est=="pg" else 32),["thumos14"])
        for exploration in ["none","constant_0.1","default","constant_0.5"]:
            add("F08",f"{route}_explore_{exploration}",base(route,exploration=exploration),["thumos14"])
        for interval in [8,32,128]:
            add("F08",f"{route}_probe_every{interval}",base(route,probe_every=interval),["thumos14"])
        add("F08",f"{route}_warmup0",base(route,warmup_epochs=0),["thumos14"])

    # F09: timestamp-preserving spatial acquisition; token counts not equal-cost claims.
    for rt,rs in [(1.0,224),(.5,224),(1.0,112),(.5,112)]:
        add("F09",f"B_factor_T{rt}_S{rs}",base("B",axis="T",budget=rt,
            source_resolution=rs,selector="hybrid" if rt<1 else "none",
            estimator="pg_acquisition" if rt<1 else "none"),["thumos14"])
    for size in [112,160,224]:
        for nroi in [1,2]:
            add("F09",f"B_roi{nroi}_size{size}",base("B",axis="T",roi_mode="source_crop",
                roi_size=size,roi_count=nroi,roi_trajectory="tubelet_constant"),["thumos14"])
    for option in ["fullframe_fallback","smooth_trajectory","resize_lowres_negative_control"]:
        add("F09",f"B_roi_{option}",base("B",axis="T",roi_mode="source_crop",roi_size=160,
            roi_count=1,roi_option=option),["thumos14"])

    # F10: compression/fusion simplicity and mixed-scale support.
    for slots in [1,4,8]:
        add("F10",f"B_slots{slots}",base("B",evidence_slots=slots),["thumos14"])
    for layers in [1,2,4]:
        add("F10",f"B_receiverlayers{layers}",base("B",receiver_layers=layers),["thumos14"])
    for variant in ["no_null","feature_l2","coarse_overwrite"]:
        add("F10",f"B_{variant}",base("B",fusion_variant=variant),["thumos14"])
    for variant in ["mean_shared","learned_projection","no_scale_embedding","coarse_update_no_tia"]:
        add("F10",f"C_{variant}",base("C",coarse_variant=variant),["thumos14"])

    # F11: sequential acquisition is a comparison, never a dependency of other routes.
    for proposal in ["random","cheap_boundary_single","cheap_boundary_pair"]:
        add("F11",f"B_2round_{proposal}",base("B",rounds=2,second_round_proposal=proposal,
            budget_includes_both_rounds=True),["thumos14"])
    for variant in ["parent16","local4_tubelets","cross_parent_selected"]:
        add("F11",f"B_context_{variant}",base("B",heavy_context=variant),["thumos14"])
    add("F11","A_refresh_each_layer",base("A",route_refresh="each_layer"),["thumos14"])

    # F12: external data/assets may be unavailable: mark blocked, never silently skip.
    for route in ["DENSE","A","B","C"]:
        cfg = base(route, selector="none" if route=="DENSE" else "hybrid",
                   budget=1.0 if route=="DENSE" else .5)
        add("F12",f"{route}_fineaction",cfg,["fineaction"])
        add("F12",f"{route}_large",dict(cfg,backbone="videomae_l"),["thumos14"])

    # F13: dense control for alternate head plus both grid sizes.
    for route in ["DENSE","A","B"]:
        for head in ["actionformer","tridet"]:
            add("F13",f"{route}_{head}",base(route,head=head,
                selector="none" if route=="DENSE" else "hybrid",
                budget=1.0 if route=="DENSE" else .5),["thumos14"])
    for route in ["A","B"]:
        for q in [384,768]:
            add("F13",f"{route}_query{q}",base(route,query_length=q),["thumos14"])

    trains = list(train_by_sig.values())
    jobs = list(trains)
    def child(parent, kind, suite, **extra):
        cid = f"{kind[:2]}-{digest([parent['job_id'],kind,suite,extra])}"
        j = {k:parent[k] for k in ["route","dataset","seed","family","families","capabilities","external_requirements","protocol_version"]}
        j.update(job_id=cid,kind=kind,label=suite,depends_on=[parent["job_id"]],
                 source_train_id=parent["job_id"],suite=suite,slot_class="gpu",
                 exclusive=(kind=="benchmark"),is_mock=False)
        j["capabilities"] = sorted(set(j["capabilities"] +
            [{"evaluate":"evaluation","benchmark":"benchmark","diagnostic":"diagnostics"}[kind]]))
        j.update(extra)
        jobs.append(j)
        return j
    for tr in trains:
        child(tr,"evaluate","official_and_all_risk_slices",checkpoint_selection="best_full_validation_average_mAP")
        if tr["seed"] == 0:
            child(tr,"benchmark","isolated_end_to_end",batches=[1,8,32],
                  warmup=50,repeats=200,video_subset_size=100,
                  modes=["device_model","decoded_tensor_to_output","encoded_video_to_output"],
                  implementation_pairs=["reference","optimized"])
        cfg=tr["model"]
        if cfg==base(tr["route"]) and tr["route"] in ["A","B","C"] and tr["dataset"]=="thumos14":
            for suite in ["D01_finite_menu_and_utility","D02_support_gap_geometry", 
                          "D04_time_space_interaction","D05_mask_content_switch","D06_cache_validity"]:
                child(tr,"diagnostic",suite,checkpoint_epochs=[4,19,39,59],
                      diagnostic_split="internal_diagnostic",finite_menu_atoms=8,
                      max_windows=128,oracle_label="finite_menu_loss_best")
        if cfg["budget_mode"]=="dynamic" and cfg["route"] in ["A","B","C"]:
            child(tr,"diagnostic","D03_budget_histogram_shuffle",checkpoint_epochs=[59],
                  diagnostic_split="internal_diagnostic")
    return jobs, refs


def validate(jobs):
    ids={x["job_id"] for x in jobs}
    if len(ids)!=len(jobs): raise ValueError("duplicate job_id")
    if any(x["kind"]=="train" and x["depends_on"] for x in jobs):
        raise ValueError("training may not wait for another experiment")
    for j in jobs:
        if any(d not in ids for d in j["depends_on"]): raise ValueError("missing dependency")
        if j["kind"]=="train" and j["epochs"]!=60: raise ValueError("unregistered epoch length")
        if "metric_gate" in j: raise ValueError("metric gates prohibited")
        if j["kind"] == "diagnostic":
            parent = next(p for p in jobs if p["job_id"] == j["source_train_id"])
            if not set(j["checkpoint_epochs"]) <= set(parent["checkpoints"]):
                raise ValueError("diagnostic requests a checkpoint outside its training artifact contract")
    visiting,done=set(),set()
    byid={x["job_id"]:x for x in jobs}
    def visit(i):
        if i in visiting: raise ValueError("dependency cycle")
        if i in done:return
        visiting.add(i)
        for d in byid[i]["depends_on"]:visit(d)
        visiting.remove(i);done.add(i)
    for i in ids:visit(i)


def matrix_summary(jobs, refs):
    content="".join(canonical(j)+"\n" for j in jobs)
    count=Counter(j["kind"] for j in jobs)
    return {"protocol_version":"geosparse-audit-repair-20260908","artifact_contract_version":"audit-repair-20260908","counts":dict(count),
             "total_jobs":len(jobs),"training_epochs_total":60*count["train"],
             "train_jobs_have_experiment_dependencies":False,
             "matrix_sha256":hashlib.sha256(content.encode()).hexdigest(),
             "families":{f:{"name":FAMILIES[f],"unique_train_jobs":len(set(refs[f]))} for f in FAMILIES},
             "training_by_dataset":dict(Counter(j["dataset"] for j in jobs if j["kind"]=="train")),
             "training_by_route":dict(Counter(j["route"] for j in jobs if j["kind"]=="train"))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"manifests")
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    jobs,refs=compile_all();validate(jobs)
    content="".join(canonical(j)+"\n" for j in jobs)
    (a.out/"experiments.jsonl").write_text(content,encoding="utf-8")
    count=Counter(j["kind"] for j in jobs)
    summary=matrix_summary(jobs,refs)
    (a.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 完整预注册实验矩阵", "", "这是设计清单，不是运行结果。所有train为60 epochs；所有训练配置seed=0,1,2。", "",
           "|实验族|问题|关联训练任务（跨族可复用）|","|---|---|---:|"]
    for f in FAMILIES:lines.append(f"|{f}|{FAMILIES[f]}|{len(set(refs[f]))}|")
    lines += ["",f"去重后：{count['train']} training，{count['evaluate']} evaluation，{count['benchmark']} hardware benchmark，{count['diagnostic']} diagnostic；总计 {len(jobs)} jobs。",
              "", "每个benchmark覆盖batch=1/8/32；同卡测量必须隔离，不与训练同时计时。所有GPU任务均由资源队列调度；同时注册不等于无限GPU。",
              "", "FineAction、VideoMAE-L和legacy DUCA资源未就绪时保留BLOCKED_EXTERNAL_ASSET，不删除矩阵、不阻塞其他路线。",
              "", "## 全部训练任务", "", "|ID|族|route|dataset|seed|配置|", "|---|---|---|---|---:|---|"]
    for j in jobs:
        if j["kind"]=="train":lines.append(f"|{j['job_id']}|{','.join(j['families'])}|{j['route']}|{j['dataset']}|{j['seed']}|{j['label']}|")
    (a.out/"MATRIX.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
