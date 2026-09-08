"""Refresh repaired runs and official baseline, retaining old audit trajectories."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import html
import json
from pathlib import Path, PurePosixPath
import subprocess
from deploy_corrected import command
from refresh_corrected_status import PROBE as BASE_PROBE

HERE = Path(__file__).resolve().parent
VIS = HERE.parent / 'visualizations'
PROBE = BASE_PROBE.replace("secondary_progress=read(control/'secondary_progress.json')", "audit_progress=read(control/'audit_progress.json')")
PROBE = PROBE.replace(
    "logs=sorted(p.glob('attempt*.out'),key=lambda q:q.stat().st_mtime)",
    "sid=str((rows[jid]['queue'] or {}).get('slurm_id',''))\n"
    "  logs=sorted(p.glob('attempt*-'+sid+'.out'),key=lambda q:q.stat().st_mtime) if sid else []",
)
PROBE = PROBE.replace('print(json.dumps(dict(cluster=', '''
accounting=subprocess.run(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobIDRaw,State,ExitCode,NodeList'],capture_output=True,text=True).stdout if ids else ''
for jid,row in rows.items():
 p=root.parent.parent/'prechecks'/jid
 logs=sorted(p.glob('slurm-*.err'),key=lambda q:q.stat().st_mtime)
 if logs:
  with logs[-1].open('rb') as stream:
   stream.seek(max(0,logs[-1].stat().st_size-3000));row['precheck_stderr_tail']=stream.read().decode(errors='replace')
print(json.dumps(dict(gpu1_node_policy=read(control/'gpu1_nodes.json'),primary_slot_reservation=read(control/'primary_slot_reservation.json'),accounting=accounting,cluster=''')

OLD_LABELS = {'tr-c43d3e9cad58':'A · 固定50%', 'tr-be89cd6cb2ca':'B · 固定50%', 'tr-a45686a40afa':'C · 固定50%',
              'tr-32754eef72d1':'A · 完整动态预算', 'tr-8a922c4e6ca7':'B · 完整动态预算', 'tr-72ba800e6f08':'C · 完整动态预算',
              'tr-c2d93c4c3979':'补充 · B-full', 'tr-bacdbb058eca':'补充 · GeoSparse dense control',
              'tr-f90ea0ae2ce4':'补充 · A uniform', 'tr-6f26bfd4ff96':'补充 · B uniform', 'tr-067daa13036b':'补充 · C uniform'}


def main():
    latest = json.loads((HERE/'audit_deployment.latest.json').read_text())
    delivery = Path(latest['local_delivery'])
    mapping = json.loads((delivery/'old_to_repaired_job_ids.json').read_text())
    labels = {mapping[k]:v for k,v in OLD_LABELS.items()}
    previous_path = HERE/'audit_status.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {'states':{}}
    def probe(item):
        phase, cluster, target = item
        key = phase+'.'+cluster
        b = json.loads((delivery/f'bindings.{phase}.{cluster}.json').read_text())
        binding = str(PurePosixPath(b['work_root']).parent.parent/'control/bindings.json')
        try:
            response = subprocess.run(command(target,'python3 -'),
                input=PROBE.replace('BINDING',repr(binding)).encode(),capture_output=True,timeout=120)
            if response.returncode:
                raise RuntimeError(response.stderr.decode(errors='replace').strip())
            result = json.loads(response.stdout)
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            if key not in previous['states']:
                raise  # No previous observation exists to show for this host.
            result = dict(previous['states'][key])
            result['observed_at_utc'] = result.get('observed_at_utc',previous['observed_at_utc'])
            result['query_attempt_at_utc'] = datetime.now(timezone.utc).isoformat()
            result['observation_error'] = str(exc)
            return key,result
        result['phase'] = phase
        result['work_root'] = b['work_root']
        result['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
        result['query_attempt_at_utc'] = result['observed_at_utc']
        result['observation_error'] = None
        return key,result
    targets = [(phase,cluster,target) for phase in ['primary','secondary'] for cluster,target in [('N16R4','source'),('A100','destination')]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = dict(pool.map(probe, targets))
    now = datetime.now(timezone.utc)
    stamp = now.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    errors={key:host['observation_error'] for key,host in states.items() if host.get('observation_error')}
    report = dict(observed_at_utc=now.isoformat(),source_commit=latest['source_commit'],states=states,query_errors=errors)
    (HERE/'audit_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for key, host in states.items():
        live = {r[0]:r for line in host['slurm'].splitlines() if len(r:=line.split('|'))>=3}
        accounting = {r[0]:r for line in host['accounting'].splitlines() if len(r:=line.split('|'))>=3}
        for jid, data in host['runs'].items():
            q=data['queue'] or {};sid=q.get('slurm_id');p=data['progress'] or {};best=data['best']
            if host.get('observation_error'):
                recorded=datetime.fromisoformat(host['observed_at_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
                detail=f"暂不可查询；上次成功查询 {recorded} 北京时间；当时记录{p.get('completed_epochs',0)}/60 epochs"
                score=f"{best['score']*100:.4f}%（上次记录的阶段best）" if best else '上次未记录修订后性能结果'
                rows.append([host['cluster'],labels[jid],jid,'OBSERVATION_UNAVAILABLE',detail,score,str(sid or '')])
                continue
            state=q.get('status','PENDING');detail=f"{p.get('completed_epochs',0)}/60 epochs"
            if sid and state in {'SUBMITTED','SUBMITTING'}:
                state=live[str(sid)][2] if str(sid) in live else accounting.get(str(sid),['',state])[1]
                if str(sid) not in live and accounting.get(str(sid),['','',''])[2]=='78:0':
                    state='RESOURCE_RETRY';detail='非GPU1分配已拒绝，等待合法资源；未执行模型'
                if state=='RUNNING' and data.get('last_update'):
                    step=data['last_update'];detail+=f"；第{step['epoch']+1}轮 / 批次{step['step']+1}"
                if str(sid) in live and state=='PENDING':
                    if p.get('completed_epochs',0):detail+='；检查点已保留，等待续训'
                    detail+='；'+live[str(sid)][-1]
            elif not sid and data.get('precheck_submission'):
                pre_id=str(data['precheck_submission']['slurm_id'])
                pre_state=live[pre_id][2] if pre_id in live else accounting.get(pre_id,['','UNKNOWN'])[1]
                passed=(data.get('precheck') or {}).get('routes',{}).get(jid,{}).get('status')=='PASS'
                state='READY' if passed else 'PRECHECK_'+pre_state
                detail=('GPU预检通过，等待训练分配' if passed else f'GPU预检 {pre_id} / {pre_state}')
                if pre_id in live and pre_state=='PENDING':detail+='；'+live[pre_id][-1]
            elif (host.get('audit_progress') or {}).get('precheck_status',{}).get('status')=='WAITING_PRIMARY_ALLOCATION':
                state='WAITING_PRIMARY_ALLOCATION';detail='已部署，等待本服务器主方法取得训练分配'
            if (state in {'QUEUED_RESOURCE','WAITING_PRIMARY_ALLOCATION'} and host['cluster']=='N16R4' and host['phase']=='secondary'
                    and host.get('primary_slot_reservation') and not (host.get('gpu1_node_policy') or {}).get('nodes')):
                state='WAITING_PRIMARY_RESOURCES';detail=f"{p.get('completed_epochs',0)}/60 epochs；保留自身检查点，GPU1优先供两项A主方法"
            if state=='RUNNING' and data.get('validation_progress'):
                val=data['validation_progress'];state='VALIDATING';detail+=f"；全量验证{val['batch']+1}/{val['total_batches']}批次"
            if data.get('failure'):detail+='；'+str(data['failure'].get('reason','查看失败凭证'))[:200]
            score=f"{best['score']*100:.4f}%（阶段best）" if best else '尚无修订后性能结果'
            rows.append([host['cluster'],labels[jid],jid,state,detail,score,str(sid or '')])
    official_host=states['primary.A100']
    live={r[0]:r for line in official_host['slurm'].splitlines() if len(r:=line.split('|'))>=3}
    for mode,data in official_host['official'].items():
        sid=str((data['submission'] or {}).get('slurm_id',''));result=data['result'] or {};p=data['progress'] or {}
        state=live[sid][2] if sid in live else result.get('status','UNKNOWN')
        detail=f"{result.get('completed_epochs') or p.get('completed_epochs',0)}/60 epochs" if mode=='train' else '发布权重，全211视频/792窗口推理'
        metric=data['best'] or data['metrics'];score=f"{metric['metrics']['average_mAP']*100:.4f}%" if metric else '尚无完成结果'
        if official_host.get('observation_error'):
            state='OBSERVATION_UNAVAILABLE'
            detail='暂不可查询；上次成功查询 '+official_host['observed_at_utc']+'；当时记录：'+detail
            score+='（上次记录）'
        rows.append(['A100','官方原版独立训练' if mode=='train' else '官方发布权重复测','official_'+mode,state,detail,score,sid])
    table='<table><tr>'+''.join('<th>'+x+'</th>' for x in ['服务器','实验','训练ID','实际状态','进度','精度','训练/推理Slurm ID'])+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoSparse · 审计修订后实验进度</title><style>body{font:15px/1.7 system-ui;max-width:1450px;margin:36px auto;padding:0 24px;color:#192a40;background:#f6f8fc}section{background:white;padding:22px;border-radius:12px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #dce3ec}td:nth-child(3){font:12px monospace}.note{border-left:5px solid #b86c26}a{color:#185bb8}</style>'''
    page+=f"<h1>GeoSparse · 审计修订后实验进度</h1><p>刷新完成：{stamp} 北京时间 · 源码 {latest['source_commit'][:12]}</p>"
    if errors:
        page+='<section class="note"><b>部分远端查询失败</b><p>连接失败不等于训练失败；相关行仅显示上次成功查询的记录。</p>'+''.join('<p>'+html.escape(key+': '+error)+'</p>' for key,error in errors.items())+'</section>'
    page+='<section><b>主方法优先</b><p>A/B/C固定50%与动态预算，共六项完整方法，只推进seed0。4090侧分配A，A100侧分配B/C。各配置通过自身GPU预检即排队训练，不以其他路线的mAP或Oracle为条件。补充控制使用较低优先级，在主方法部署后排队。</p></section>'
    page+='<section><b>训练和选点协议</b><p>全200视频训练，全211测试视频/792窗口验证；768帧、160px、global batch2、warm-up5、cosine100、训练60轮。GeoSparse每5轮全量EMA验证并选best；官方保持42/44/…/60轮验证。另报告固定60轮和共同50/60轮best。选点机会不同；统一dense control不称官方训练等价。</p></section>'
    page+='<section>'+table+'</section>'
    page+='<section class="note"><b>旧运行保留为审计轨迹</b><p>902fa05的A固定50%停于完成10轮，动态A停于完成4轮。真实生产数据回放发现有效125帧被扩大为126检测位置，因此新协议从识别预训练初始化，不把旧权重升级为修正目标下的60轮结果。B/C此前未开始训练；官方原版任务未停止。旧成本归一化和未配对计时不进入新主表。</p></section>'
    page+='<section><b>验证与结果边界</b><p>124项CPU回归已通过；生产GPU预检及训练进度以本表和原始凭证为准。71.1387948970%属于官方发布EMA权重复测，不能借作GeoSparse精度。1545项是登记总量，117项诊断仍未实现；没有完整配对的硬件加速结论。</p><p><a href="../../official_adatad_audit/AUDIT_REPAIR_STATUS.zh.md">审计修复与运行记录</a> · <a href="../../official_adatad_audit/audit_status.json">本次原始状态</a></p></section></html>'
    (VIS/'current').mkdir(exist_ok=True)
    (VIS/'current/index.html').write_text(page,encoding='utf-8')
    (VIS/'CURRENT_STATUS.zh.md').write_text('刷新完成：'+stamp+' 北京时间。\n\n|服务器|实验|训练ID|状态|进度|精度|Slurm ID|\n|---|---|---|---|---|---|---|\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows),encoding='utf-8')
    print(json.dumps(dict(observed_at=stamp,query_errors=errors,rows=rows),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
