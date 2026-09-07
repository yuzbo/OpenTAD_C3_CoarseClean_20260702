"""Self-contained viewer of real source frames, native selections, supports, GT and predictions."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np


def native_polygons(transform, side):
    inverse = np.linalg.inv(np.asarray(transform, dtype=float))
    result = []
    for y in range(side):
        for x in range(side):
            corners = np.array([[x, y, 1], [x + 1, y, 1], [x + 1, y + 1, 1], [x, y + 1, 1]], dtype=float)
            corners[:, :2] /= side
            original = corners @ inverse.T
            result.append((original[:, :2] / original[:, 2:]).tolist())
    return result


def build(source, output):
    source, output = Path(source), Path(output)
    read = lambda name: json.loads((source / name).read_text(encoding="utf-8"))
    receipt = read("export_receipt.json")
    if receipt.get("status") != "completed_selection_export" or receipt.get("is_mock") is not False:
        raise ValueError("viewer requires a completed real selection export")
    windows = [json.loads(line) for line in (source / "windows.jsonl").read_text().splitlines() if line.strip()]
    for row in windows:
        row["polygons"] = native_polygons(row["spatial_transform"], row["native_shape"][1])
        row["images"] = {str(entry["tubelet"]): ["data:image/jpeg;base64," + base64.b64encode((source / path).read_bytes()).decode() for path in entry["paths"]] for entry in row.get("thumbnails", [])}
        # All numerical/scout/evidence arrays remain available in plans/*.npz;
        # the viewer embeds only fields used by its display.
        row.pop("layers", None)
    payload = dict(receipt=receipt, windows=windows, annotations=read("annotations.json"), predictions=read("predictions.json")["results"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(PAGE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")
    return len(windows)


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoSparse 原生选择查看器</title>
<style>body{font:15px system-ui;color:#223;background:#f3f6f9;margin:0}main{max-width:1240px;margin:28px auto;padding:0 20px}h1{font-size:29px}.box{background:white;border-radius:12px;padding:20px;margin:18px 0}p{line-height:1.7}select,input,button{padding:8px;margin:5px;font:inherit}select{max-width:92%}.frames{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}.frame{position:relative;background:#edf1f5}.frame img{width:100%;display:block}.frame svg{position:absolute;inset:0;width:100%;height:100%}svg.timeline{width:100%;height:200px;background:white}#status,#detail{white-space:pre-wrap;overflow-wrap:anywhere;font:13px ui-monospace;line-height:1.8}button{cursor:pointer}#native{width:55%}.legend{color:#456}#grid{width:240px;height:240px;image-rendering:pixelated}a{color:#0072b2}</style>
<main><h1>GeoSparse · 选中了哪些信息</h1><div class="box"><p id="summary"></p><p class="legend">两幅图是一个原生 tubelet 的真实两帧。橙色为选中区域；路线 C 的蓝色区域仍执行粗粒度 Heavy。A/B 未选区域有低成本状态。原图上的选中位置由真实裁剪/翻转矩阵反变换得到。</p>
<label>视频 <select id="video"></select></label><label>窗口 <select id="window"></select></label><br>
<label>原生 tubelet <input type="range" id="native" min="0" value="0"><span id="nativeLabel"></span></label><label><input type="checkbox" id="overlay" checked>显示空间选择</label><br>
<label>已导出的双帧 <select id="pictured"></select></label><div id="status"></div></div>
<div class="box"><div class="frames"><div class="frame" id="frame0"></div><div class="frame" id="frame1"></div></div><p id="frameNote"></p></div>
<div class="box"><h2>原始物理时间上的选择与定位</h2><svg class="timeline" id="timeline" viewBox="0 0 1100 200"></svg><p class="legend">绿色：GT；紫色：分数最高的至多 20 个预测（仅改变显示）；橙色：实际观测区间，分离的支持不会连成完整观测。黑线为当前原生 tubelet 的参考时刻。图中 selection 高度表示该时间片的空间选择比例。</p></div>
<div class="box"><h2>这个时间片的空间 mask 与窗口计算量</h2><canvas id="grid"></canvas><pre id="detail"></pre><p>窗口计算量包含所有 parent clips。计数为实际执行的 Conv/Linear/Matmul MACs；不包含归一化、softmax、激活、插值和预处理。此导出经过插桩，不能用来测延迟。Scout 的逐原子 gain、actionness、采样顺序和 B 的实际 evidence 支持保存在 NPZ 中，可供诊断绘图。</p></div></main>
<script>const D=__PAYLOAD__;
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let current, bits;
const videos=[...new Set(D.windows.map(w=>w.video_id))].sort();
videos.forEach(v=>$('video').add(new Option(v,v)));
$('summary').textContent=`模型来源 ${D.receipt.model_source_commit.slice(0,8)} · ${D.receipt.source_train_id} · seed ${D.receipt.seed} · checkpoint 完成 ${D.receipt.selected_checkpoint_epoch+1} epochs · ${D.receipt.is_final_checkpoint?'最终 best':'中间 checkpoint，非主表结果'} · ${D.receipt.full_split?'全量 split':'明确子集'}`;
function video(){const rows=D.windows.filter(w=>w.video_id===$('video').value);$('window').replaceChildren();rows.forEach(w=>$('window').add(new Option(`${w.window_index}: ${w.window_id}`,w.window_index)));windowChange();}
function windowChange(){current=D.windows.find(w=>w.window_index===+$('window').value);bits=Uint8Array.from(atob(current.selected_bits),c=>c.charCodeAt(0));$('native').max=current.native_shape[0]-1;$('native').value=0;$('pictured').replaceChildren(new Option('选择已导出的真实双帧',''));Object.keys(current.images).forEach(t=>$('pictured').add(new Option(`tubelet ${t}`,t)));draw();}
function selected(t,i){const k=t*current.native_shape[1]**2+i;return (bits[k>>3]>>(k%8))&1;}
function draw(){const t=+$('native').value,w=current,side=w.native_shape[1], imgs=w.images[t];$('nativeLabel').textContent=`${t} / ${w.native_shape[0]-1}`;
const valid=w.valid_tubelets[t];$('status').textContent=`source frames: ${w.source_frame_id[t].join(', ')} | actual intervals: ${JSON.stringify(w.support_intervals_s[t])} s | valid: ${valid}`;
const polygons=$('overlay').checked?w.polygons.map((p,i)=>`<polygon points="${p.map(v=>v.join(',')).join(' ')}" fill="${selected(t,i)?'#E69F00':w.route==='C'?'#0072B2':'none'}" fill-opacity=".36" stroke="#fff" stroke-width=".0015"/>`).join(''):'';
for(let i=0;i<2;i++)$('frame'+i).innerHTML=imgs?`<img src="${imgs[i]}" alt="actual source frame"><svg viewBox="0 0 1 1" preserveAspectRatio="none">${polygons}</svg>`:'<p style="padding:24px">该位置未导出缩略图。精确的 mask 和支持仍可查看。</p>';
$('frameNote').textContent=imgs?'显示实际解码帧，覆盖只表示执行选择，不是动作区域标注。':'可从“已导出的双帧”选择有图像的位置；不以邻近帧冒充当前帧。';
const g=$('grid');g.width=side;g.height=side;const ctx=g.getContext('2d');for(let y=0;y<side;y++)for(let x=0;x<side;x++){ctx.fillStyle=selected(t,y*side+x)?'#E69F00':w.route==='C'?'#0072B2':'#e4e9ee';ctx.fillRect(x,y,1,1);}
$('detail').textContent=`Route ${w.route} · ${w.selected_semantics}\n请求预算: ${w.requested_budget}\n实际选择 native members: ${(100*w.selected_native_ratio).toFixed(2)}%\n至少有一个选中位置的时间片: ${(100*w.selected_temporal_ratio).toFixed(2)}%\nHeavy GMACs: ${(w.heavy_macs/1e9).toFixed(3)}; 相对 full: ${(100*w.heavy_mac_ratio).toFixed(2)}%\n模型已计数 GMACs: ${(w.model_macs_counted/1e9).toFixed(3)}; 支持算子计数完整: ${w.mac_count_complete}\n零 Heavy 的 parent clips: ${(100*w.zero_heavy_parent_fraction).toFixed(2)}%\n组件 MACs: ${JSON.stringify(w.mac_components)}\n逐原子/证据数组: ${w.plan_path}`;
const intervals=w.support_intervals_s.flatMap((pair,i)=>pair.filter((_,j)=>w.support_valid[i][j])), start=Math.min(...intervals.map(s=>s[0])),end=Math.max(...intervals.map(s=>s[1]));const X=s=>90+980*(s-start)/(end-start),parts=[];
const rect=(a,b,y,h,color,title)=>{if(b<=start||a>=end)return;parts.push(`<rect x="${X(Math.max(start,a))}" y="${y}" width="${Math.max(.5,X(Math.min(end,b))-X(Math.max(start,a)))}" height="${h}" fill="${color}"><title>${esc(title)}</title></rect>`);};
parts.push('<text x="5" y="36">GT</text><text x="5" y="73">Pred.</text><text x="5" y="137">Selection</text>');
(D.annotations[w.video_id].annotations||[]).forEach(a=>rect(...a.segment,20,20,'#009E73',a.label+' '+a.segment));
[...(D.predictions[w.video_id]||[])].sort((a,b)=>b.score-a.score).slice(0,20).forEach((a,i)=>rect(...a.segment,54+(i%2)*10,9,'#CC79A7',a.label+' '+a.score+' '+a.segment));
w.support_intervals_s.forEach((pair,i)=>{if(w.temporal_selected_fraction[i]>0)pair.forEach((s,j)=>{if(w.support_valid[i][j])rect(...s,150-50*w.temporal_selected_fraction[i],50*w.temporal_selected_fraction[i],'#E69F00','tubelet '+i);});});
parts.push(`<line x1="${X(w.nominal_time_s[t])}" x2="${X(w.nominal_time_s[t])}" y1="12" y2="155" stroke="#222"/>`);
for(let i=0;i<6;i++){const time=start+(end-start)*i/5;parts.push(`<text x="${X(time)}" y="182" text-anchor="middle">${time.toFixed(2)} s</text>`);}
$('timeline').innerHTML=parts.join('');}
$('video').onchange=video;$('window').onchange=windowChange;$('native').oninput=draw;$('overlay').onchange=draw;$('pictured').onchange=()=>{if($('pictured').value!==''){$('native').value=$('pictured').value;draw();}};video();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Rendered {build(args.selection_export, args.output)} real windows")


if __name__ == "__main__":
    main()
