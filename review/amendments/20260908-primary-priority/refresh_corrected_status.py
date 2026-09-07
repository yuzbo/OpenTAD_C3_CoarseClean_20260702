"""Refresh only corrected runs; preserve the withdrawn protocol's old figure book."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VIS = HERE.parent / 'visualizations'
sys.path.insert(0, str(HERE.parent / 'deployment_a100'))
from transfer_assets import command

PROBE = r'''
import json,re,subprocess,shutil
from pathlib import Path
b=json.loads(Path(BINDING).read_text())
root=Path(b['work_root']); control=root.parent.parent/'control'
def read(p):
 return json.loads(p.read_text()) if p.exists() else None
state=read(root/'slurm_state.json') or {'jobs':{}}
pre=read(control/'precheck_submissions.json') or {}
rows={}
for jid in b['assigned_training_ids']:
 p=root/'runs'/jid
 rows[jid]=dict(queue=state['jobs'].get(jid),progress=read(p/'progress.json'),best=read(p/'best.json'),result=read(p/'result.json'),failure=read(p/'failure.json'),precheck_submission=pre.get(jid),precheck=read(root.parent.parent/'prechecks'/jid/'gpu_precheck.json'),dynamic_policy_check=read(root.parent.parent/'prechecks'/jid/'dynamic_policy_check.json'))
 if (p/'train.log').exists():
  with (p/'train.log').open('rb') as stream:
   stream.seek(max(0,(p/'train.log').stat().st_size-65536));lines=stream.read().splitlines()
  for line in reversed(lines):
   try:row=json.loads(line)
   except (ValueError,UnicodeDecodeError):continue
   if 'epoch' in row and 'step' in row:
    rows[jid]['last_update']={key:row[key] for key in ['epoch','step','successful_update','gradient_status','seconds','lr']};break
 completed=(rows[jid]['progress'] or {}).get('completed_epochs',0)
 if completed and completed%5==0 and not (p/'intermediate_eval'/('epoch_'+str(completed).zfill(3))/'metrics.json').exists():
  logs=sorted(p.glob('attempt*.out'),key=lambda q:q.stat().st_mtime)
  if logs:
   with logs[-1].open('rb') as stream:
    stream.seek(max(0,logs[-1].stat().st_size-32768));tail=stream.read().splitlines()
   for line in reversed(tail):
    try:event=json.loads(line)
    except (ValueError,UnicodeDecodeError):continue
    if event.get('event')=='validation_progress' and event.get('completed_epochs')==completed:
     rows[jid]['validation_progress']=event;break
official={}
if b['cluster']=='A100':
 base=Path('/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/official_adatad/runs')
 for mode in ['train','evaluate']:
  p=base/('official_b_'+mode+'_seed0')
  official[mode]={name:read(p/(name+'.json')) for name in ['submission','progress','best','result','metrics']}
  if mode=='evaluate' and (p/'console.log').exists():
   with (p/'console.log').open('rb') as stream:
    stream.seek(max(0,(p/'console.log').stat().st_size-65536));tail=stream.read()
   matches=re.findall(rb'(\d+)/(\d+)\s*\[',tail)
   if matches:
    done,total=map(int,matches[-1]);official[mode]['inference_progress']=dict(completed_batches=done,total_batches=total)
ids=[str(v['slurm_id']) for v in pre.values()]
ids += [str(v['queue']['slurm_id']) for v in rows.values() if v['queue'] and v['queue'].get('slurm_id')]
ids += [str(v['submission']['slurm_id']) for v in official.values() if v['submission']]
slurm=subprocess.run(['squeue','-h','-j',','.join(ids),'-o','%i|%j|%T|%M|%N|%R'],text=True,capture_output=True).stdout if ids else ''
print(json.dumps(dict(cluster=b['cluster'],source_commit=b['source_commit'],runs=rows,official=official,slurm=slurm,coordinator=read(control/'coordinator.json'),free_disk_gib=shutil.disk_usage(root).free/1024**3,secondary_progress=read(control/'secondary_progress.json'))))
'''


def main():
    def probe(cluster, target, filename):
        b = json.loads((HERE / filename).read_text())
        control = str(Path(b['work_root']).parent.parent).replace('\\', '/') + '/control/bindings.json'
        r = subprocess.run(command(target, 'python3 -'), input=PROBE.replace('BINDING', repr(control)).encode(), capture_output=True)
        if r.returncode:
            raise RuntimeError(r.stderr.decode())
        result = json.loads(r.stdout)
        for row in result['runs'].values():
            row['source_commit'] = result['source_commit']
            row['work_root'] = b['work_root']
            row['secondary_precheck_status'] = (result.get('secondary_progress') or {}).get('precheck_status')
        return cluster, result
    targets = [('N16R4','source','bindings.corrected.N16R4.json'),('A100','destination','bindings.corrected.A100.json')]
    for phase in ['dynamic', 'secondary']:
        for cluster, target in [('N16R4','source'),('A100','destination')]:
            filename='bindings.'+phase+'.'+cluster+'.json'
            if (HERE/filename).exists():
                targets.append((cluster,target,filename))
    if (HERE / 'bindings.retained.A100.json').exists():
        retained = json.loads((HERE / 'bindings.retained.A100.json').read_text())
        current = json.loads((HERE / 'bindings.corrected.A100.json').read_text())
        if retained['work_root'] != current['work_root']:
            targets.append(('A100','destination','bindings.retained.A100.json'))
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for cluster, result in pool.map(lambda args: probe(*args), targets):
            if cluster not in results:
                results[cluster] = result
                result['coordinators'] = [result['coordinator']]
            else:
                existing = results[cluster]
                assert not (set(existing['runs']) & set(result['runs'])), 'duplicate host assignments'
                existing['runs'].update(result['runs'])
                existing['slurm'] = '\n'.join(dict.fromkeys((existing['slurm']+'\n'+result['slurm']).splitlines()))
                existing['coordinators'].append(result['coordinator'])
    now = datetime.now(timezone.utc)
    results['observed_at_utc'] = now.isoformat()
    (HERE / 'corrected_status.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    stamp = now.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    legacy = VIS / 'withdrawn_180video_20260908'
    if not legacy.exists() and (VIS / 'current').exists():
        shutil.copytree(VIS / 'current', legacy)
    labels = {'tr-c43d3e9cad58':'A · 固定50%','tr-be89cd6cb2ca':'B · 固定50%','tr-a45686a40afa':'C · 固定50%',
              'tr-32754eef72d1':'A · 完整动态预算','tr-8a922c4e6ca7':'B · 完整动态预算','tr-72ba800e6f08':'C · 完整动态预算',
              'tr-c2d93c4c3979':'补充 · B-full','tr-bacdbb058eca':'补充 · 同结构稠密对照',
              'tr-f90ea0ae2ce4':'补充 · A uniform','tr-6f26bfd4ff96':'补充 · B uniform','tr-067daa13036b':'补充 · C uniform'}
    rows=[]
    for cluster in ['N16R4','A100']:
        current = {line.split('|')[0]: line.split('|') for line in results[cluster]['slurm'].splitlines()}
        for jid, data in results[cluster]['runs'].items():
            q=data['queue'] or {}; sid=q.get('slurm_id')
            state=current.get(str(sid), ['', '', q.get('status','WAITING_DEPLOYMENT')])[2]
            progress=data['progress'] or {}; best=data['best']
            detail=f"{progress.get('completed_epochs',0)}/60 epochs"
            if data.get('last_update'):
                last=data['last_update'];detail+=f"；第 {last['epoch']+1} 轮 / 第 {last['step']+1} 批次"
            if data.get('validation_progress'):
                val=data['validation_progress'];state='VALIDATING'
                detail=f"已训练{progress.get('completed_epochs',0)}/60 epochs；全量验证 {val['batch']+1}/{val['total_batches']} 批次"
            if not sid and data['precheck_submission']:
                check_id=data['precheck_submission']['slurm_id']
                detail='正确性预检 '+check_id+' / '+current.get(str(check_id),['','','已退出，见凭证'])[2]
                passed=((data.get('precheck') or {}).get('routes') or {}).get(jid,{}).get('status')=='PASS'
                if passed:
                    detail='GPU预检已通过；等待训练资源'
                    if (data.get('dynamic_policy_check') or {}).get('status')=='PASS':
                        detail='动态策略与GPU预检均通过；等待训练资源'
            reason=q.get('reason')
            if isinstance(reason,str) and reason.startswith('storage wait:'):
                state='BLOCKED_STORAGE'
                detail=f"共享盘仅余{results[cluster]['free_disk_gib']:.1f} GiB；新训练至少需要20 GiB"
            elif state=='WAITING_PRIMARY_ALLOCATION':
                detail='已部署；优先等待本服务器主方法取得训练分配'
            elif state=='BLOCKED_CAPABILITY' and (data.get('secondary_precheck_status') or {}).get('status')=='WAITING_PRIMARY_ALLOCATION':
                detail='已部署；主方法取得训练分配后执行本配置预检'
            elif state=='PENDING' and str(sid) in current:
                why=current[str(sid)][-1]
                detail+='；'+{'(Priority)':'集群优先级排队','(Dependency)':'等待主方法开始训练'}.get(why,why)
            score=f"{best['score']*100:.4f}%" if best else '尚无正式验证结果'
            rows.append([cluster,labels[jid],state,detail,score,str(sid or '')])
        for mode, data in results[cluster]['official'].items():
            sid=(data['submission'] or {}).get('slurm_id'); result=data['result'] or {}
            state=current.get(str(sid),['','',result.get('status','PENDING')])[2]
            progress=data['progress'] or {}
            detail=f"{result.get('completed_epochs') or progress.get('completed_epochs',0)}/60 epochs" if mode=='train' else '官方公开检查点，全 211 视频复测'
            if mode=='evaluate' and data.get('inference_progress'):
                p=data['inference_progress'];detail+=f"；{p['completed_batches']}/{p['total_batches']} 推理批次"
            metric=data['best'] or data['metrics']
            score=f"{metric['metrics']['average_mAP']*100:.4f}%" if metric else '尚无完整结果'
            rows.append([cluster,'官方 AdaTAD 从头训练' if mode=='train' else '官方 AdaTAD 权重复测',state,detail,score,str(sid or '')])
    body=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoSparse · 修正协议的实验进度</title><style>body{font:16px/1.7 system-ui;margin:40px auto;max-width:1300px;padding:0 24px;color:#192a40;background:#f6f8fc}h1{font-size:30px}section{background:white;padding:24px;border-radius:14px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #dce3ec}.note{border-left:5px solid #c57428}a{color:#185bb8}</style>'''
    page+=f'<h1>GeoSparse · 官方基线与全量数据</h1><p>实际查询时间：{stamp} 北京时间</p>'
    page+='<section><b>当前协议</b><p>训练：全部 200 视频。每次正式验证与最终测试：全部 211 测试视频、792 窗口。官方默认 768 帧 / 160px / 768 检测网格；global batch 2；warm-up 5；余弦周期 100；训练结束于 60 epochs。原版 AdaTAD 保持官方节奏：完成第 42、44、…、60 epochs 后验证；GeoSparse 路线及统一训练器对照每 5 epochs 验证以便尽早观察。当前仅 seed 0，按本配置完整验证的 EMA best 选结果。</p></section>'
    if (HERE/'FULL_METHODS_FIRST.zh.md').exists():
        page+='<section><b>当前优先级：完整方法先行，后续实验并行排队</b><p>A/B/C各保留固定50%与完整动态预算版本，共六项seed0主实验；官方AdaTAD基线继续。主方法部署后，补充队列登记B-full、同结构稠密对照和A/B/C uniform。补充任务使用更低Slurm优先级及主方法开始依赖，不等待训练完成或mAP达标。固定B/C保留原作业ID，等待各自动态版本先开始。全部仍为seed0、60epochs，不恢复旧协议或重复训练。<a href="../../official_adatad_audit/FULL_METHODS_FIRST.zh.md">查看本轮执行修订</a>。</p></section>'
    free=results['N16R4']['free_disk_gib']
    if free<20:
        page+=f'<section class="note"><b>4090侧存储阻塞</b><p>实际共享盘剩余{free:.1f} GiB，新训练要求至少20 GiB。已有训练继续；A动态与后续任务等待存储恢复。不会通过降低输入、训练轮数或删除数据来绕过此限制。</p></section>'
    page+='<section><table><tr><th>服务器</th><th>实验</th><th>实际状态</th><th>进度</th><th>已测性能</th><th>训练/测试 Slurm ID</th></tr>'+body+'</table></section>'
    page+='<section class="note"><b>旧协议已撤销</b><p>180 视频、224 输入、384 检测网格、余弦周期 60 的旧任务已停止；原检查点和日志封存，不纳入正式主表。旧图册只供审计：<a href="../withdrawn_180video_20260908/index.html">查看旧协议历史图册</a>。</p></section>'
    measured = results['A100']['official'].get('evaluate', {})
    result = measured.get('result') or {}
    baseline_note = '71.14% 是官方作者报告，完整复测结果尚未产生。'
    if result.get('status') == 'COMPLETED' and measured.get('metrics'):
        score = measured['metrics']['metrics']['average_mAP'] * 100
        baseline_note = f'官方发布权重的全量复测已完成：{score:.4f}%，保留两位小数为 {score:.2f}%；作者报告为 71.14%。这是发布权重复测，独立 seed 0 训练另行记录。<a href="../../official_adatad_audit/official_b_full_test_20260908/VERIFICATION.zh.md">查看逐阈值核对和完整预测</a>。'
    page+='<section><b>来源与可视化</b><p>模型基于固定官方 OpenTAD commit 346d09d1，原始 opentad 源码与配置保持不变。'+baseline_note+'修正协议的训练曲线、真实选择分布和性能—计算量图随实际产物生成，不填入旧协议数据。</p><p><a href="../../official_adatad_audit/PROTOCOL_CORRECTION.zh.md">协议修订和撤销记录</a> · <a href="../../official_adatad_audit/corrected_status.json">本次原始状态记录</a></p></section></html>'
    (VIS/'current').mkdir(exist_ok=True)
    (VIS/'current/index.html').write_text(page,encoding='utf-8')
    report='观察时间：'+stamp+' 北京时间。\n\n| 服务器 | 实验 | 状态 | 进度 | 已测性能 | Slurm ID |\n|---|---|---|---|---|---|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)
    (VIS/'CURRENT_STATUS.zh.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(observed_at=stamp,rows=rows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
