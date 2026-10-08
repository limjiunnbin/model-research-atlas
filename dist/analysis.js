import {costEstimate, parallelEstimate, checkContract, mechanism, dot} from './analysis-math.mjs';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const cache = new Map();
const load = async path => {if (!cache.has(path)) {const r = await fetch(path); if (!r.ok) throw Error('分析资料读取失败'); cache.set(path, await r.json());} return cache.get(path);};
const fmt = x => x == null ? '未知' : Math.abs(x) < 1e-8 && x !== 0 ? x.toExponential(3) : Number.isInteger(x) ? x.toLocaleString('en-US') : x.toLocaleString('en-US', {maximumFractionDigits: 6});
const bytes = x => x === 0 ? '0 B' : x >= 1e9 ? (x / 1e9).toFixed(3) + ' GB' : x >= 1e6 ? (x / 1e6).toFixed(3) + ' MB' : x >= 1e3 ? (x / 1e3).toFixed(3) + ' kB' : fmt(x) + ' B';
const flops = x => x >= 1e12 ? (x / 1e12).toFixed(4) + ' T' : x >= 1e9 ? (x / 1e9).toFixed(4) + ' G' : fmt(x);
const table = (headers, rows, cls = '') => `<div class="table-wrap ${cls}"><table><thead><tr>${headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const matrixTable = (name, matrix) => `<h3>${esc(name)}</h3>` + table(['行', ...matrix[0].map((_, i)=>'维度 '+i)], matrix.map((r,i)=>[i,...r.map(fmt)]));
const refs = list => `<details class="research-refs"><summary>固定来源（${list.length}）</summary>${list.map(r=>`<p><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.symbol || r.path)} ↗</a><small>${esc(r.revision)} · L${r.line}–${r.end}</small></p>`).join('')}</details>`;
const head = (title, description) => `<div class="page-head"><div class="eyebrow">STATIC ANALYSIS / SOURCE SNAPSHOT</div><h1>${esc(title)}</h1><p class="intro">${esc(description)}</p></div>`;
const downloads = () => `<div class="link-row"><a class="button" href="assets/deepseek/DeepSeek-静态分析.md">分析说明</a><a class="button" href="assets/deepseek/DeepSeek-静态分析.xlsx" download>分析工作簿</a><a class="button" href="assets/deepseek/DeepSeek-静态分析.json" download>完整分析 JSON</a></div>`;
const num = (id, label, value, min, max, step = 1) => `<label>${label}<input id="${id}" type="number" required min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
const select = (id, label, options, value) => `<label>${label}<select id="${id}">${options.map(([v,n])=>`<option value="${esc(v)}" ${String(v)===String(value)?'selected':''}>${esc(n)}</option>`).join('')}</select></label>`;
const metrics = rows => `<div class="analysis-metrics">${rows.map(([label, value, note])=>`<article><span>${esc(label)}</span><b>${esc(value)}</b>${note?`<small>${esc(note)}</small>`:''}</article>`).join('')}</div>`;
const assumptions = list => `<details class="panel"><summary>公式、假设与证据边界</summary><ul>${list.map(x=>`<li>${esc(x)}</li>`).join('')}</ul></details>`;
const downloadJSON = (name, value) => {const url = URL.createObjectURL(new Blob([JSON.stringify(value,null,2)+'\n'], {type:'application/json'})), a = document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url), 1000);};

export async function analysisMarkup(family, page, detail) {
  const d = await load(family.analysisPath);
  if (page === 'cost') {
    const model = d.costModels.find(m=>m.id===detail) || d.costModels.find(m=>m.id==='v4-flash');
    return `<section id="analysis-atlas" data-page="cost" data-path="${esc(family.analysisPath)}" data-model="${model.id}">${head('静态计算与存储成本', '调整场景，查看主干矩阵收缩、attention、缓存与声明张量的理论规模。结果保留精度、参考路径和统计范围。')}${downloads()}<div class="notice">覆盖 21 个 checkpoint 的主干；MTP/DSpark 参数和协议独立。未测时间、吞吐和峰值显存保持未知。</div><form id="cost-controls" class="filters analysis-controls">${select('cost-model','版本',d.costModels.map(m=>[m.id,m.name]),model.id)}${select('cost-phase','阶段',[['prefill','Prefill'],['decode','Decode（单 token）'],['verify','Target verify（多 query）']],'prefill')}${num('cost-batch','Batch',1,1,1024)}${num('cost-query','新增 query token',512,1,model.context)}${num('cost-history','此前 history token',0,0,model.context)}${select('cost-logits','输出 logits',[['last','仅最后 query'],['all','全部 query']],model.defaultLogits)}${select('cost-path','MLA 参考数学路径',[['latent','潜变量 / 吸收式'],['expanded','展开 K/V']],'latent')}${select('cost-cache-bytes','主 KV / V4 index 精度假设',[[1,'1 B/元素'],[2,'2 B/元素'],[4,'4 B/元素']],2)}${select('cost-act-bytes','投影激活精度假设',[[1,'1 B/元素'],[2,'2 B/元素'],[4,'4 B/元素']],2)}${select('cost-weight-bytes','统一逻辑权重精度假设',[[0.5,'0.5 B/元素'],[1,'1 B/元素'],[2,'2 B/元素'],[4,'4 B/元素']],2)}${select('cost-layer','统计范围',[['all','全部主干']], 'all')}</form><div id="cost-result" aria-live="polite"></div><button id="cost-export">下载当前场景 JSON</button>${assumptions([...d.costFormulas.map(x=>x.name+'：'+x.formula),...d.limits])}</section>`;
  }
  if (page === 'contracts') return `<section id="analysis-atlas" data-page="contracts" data-path="${esc(family.analysisPath)}">${head('CANN 算子约束手册','逐项核对固定 tag 的 API 文档、参数表、补充条件与入口 guard。保留平台、格式、量化和版本上下文。')}${downloads()}<div class="notice">${d.counts.cannFamilies} 个源码家族 · ${d.counts.cannClaims.toLocaleString()} 条表行与条件。自动检查仅覆盖已编码必要条件，未发现矛盾不证明运行支持。</div><div class="filters analysis-controls">${select('contract-operator','算子家族',d.cannContracts.map(x=>[x.id,x.id]),'rms-norm')}${select('contract-document','API 文档',[['all','全部文档']],'all')}${select('contract-category','条件类别',[['all','全部类别'],['shape','Shape / 维度'],['dtype','数据类型'],['layout','布局 / 连续性'],['quantization','量化 / scale'],['platform','平台'],['condition','补充条件'],['parameter','参数 / 原型']],'all')}<label>查找参数或条件<input id="contract-search" type="search" placeholder="groupSize、ND、950、gamma…"></label></div><div id="contract-results"></div><section class="panel"><h2>已编码必要条件的核对</h2><p>RMSNorm / SwiGLU 核对已提取的类型、格式和形状条件；Matmul 只检查无转置二维子集。其他家族展示完整文档记录，自动判定仍为未核对。</p><form id="contract-check-controls" class="filters analysis-controls"><label>x shape<input id="contract-x" value="2,128" placeholder="2,128"></label><label>gamma shape<input id="contract-gamma" value="128" placeholder="128"></label><label>weight shape（二维）<input id="contract-w" value="128,256" placeholder="128,256"></label>${num('contract-dim','SwiGLU dim',-1,-8,7)}${select('contract-dtype','输入 x 类型',[['FLOAT16','FLOAT16'],['BFLOAT16','BFLOAT16'],['FLOAT32','FLOAT32'],['INT8','INT8']],'BFLOAT16')}${select('contract-layout','格式',[['ND','ND'],['NZ','NZ']],'ND')}</form><div id="contract-check-result" aria-live="polite"></div></section></section>`;
  if (page === 'parallel') return `<section id="analysis-atlas" data-page="parallel" data-path="${esc(family.analysisPath)}">${head('并行、通信与 P/D 分离','按固定框架源码说明进程组与数据流，用明确的均分、路由和 collective 假设估算逻辑容量与发送量。')}${downloads()}<div class="notice">不生成通信耗时或性能排名。参数均分、路由分布和 ring 公式是分析假设，实际 loader、backend 与链路算法需分别核对。</div><form id="parallel-controls" class="filters analysis-controls">${select('parallel-model','版本',d.costModels.map(m=>[m.id,m.name]),'v4-flash')}${num('parallel-tp','Tensor parallel（TP）',2,1,128)}${num('parallel-dp','Data parallel（DP）',4,1,128)}${num('parallel-batch','每 DP replica 的 batch',1,1,1024)}${num('parallel-query','每请求 query token',1,1,1e6)}${num('parallel-history','此前 history token',4095,0,1e6)}${num('parallel-redundant','冗余专家总数',0,0,4096)}${select('parallel-policy','非专家权重容量假设',[['replicated','复制非专家权重'],['uniform-tp','二维权重理想均分到 TP']],'uniform-tp')}${select('parallel-remote','远端路由比例',[['uniform','均匀路由假设'],[0,'0%'],[0.5,'50%'],[1,'100%']],'uniform')}${num('parallel-allreduces','每层 TP AllReduce 次数假设',1,0,8)}${select('parallel-weight','统一逻辑权重精度',[[0.5,'0.5 B/元素'],[1,'1 B/元素'],[2,'2 B/元素'],[4,'4 B/元素']],2)}${select('parallel-activation','通信向量精度',[[1,'1 B/元素'],[2,'2 B/元素'],[4,'4 B/元素']],2)}</form><div id="parallel-result" aria-live="polite"></div><button id="parallel-export">下载当前并行假设 JSON</button>${d.parallel.topics.map(t=>`<section class="panel"><h2>${esc(t.title)}</h2><p>${esc(t.summary)}</p>${refs(t.refs)}</section>`).join('')}${assumptions([...d.parallel.formulas.map(x=>x.name+'：'+x.formula),...d.parallel.limits])}</section>`;
  if (page === 'mechanisms') {
    const m = d.mechanisms.find(m=>m.id===detail) || d.mechanisms[0];
    return `<section id="analysis-atlas" data-page="mechanisms" data-path="${esc(family.analysisPath)}" data-mechanism="${m.id}">${head('机制交互演示', '用公开公式与小规模教学数值，逐步观察中间结果。可修改输入、前后步进和核对固定来源。')}<div class="link-row mechanism-tabs">${d.mechanisms.map(x=>`<a class="button ${x.id===m.id?'primary':''}" href="#/family/deepseek/mechanisms/${x.id}">${esc(x.title)}</a>`).join('')}</div><section class="panel"><h2>${esc(m.title)}</h2><p class="notice">${esc(m.limit)}</p><div id="demo-controls" class="filters"></div><div id="demo-flow"></div><div class="demo-toolbar"><button id="demo-prev">← 上一步</button><span id="demo-step-label" aria-live="polite"></span><button id="demo-next">下一步 →</button><button id="demo-reset">重置</button></div><div id="demo-result" aria-live="polite"></div>${refs(m.references)}</section></section>`;
  }
  throw Error('Unknown analysis page');
}

export function mountAnalysis() {
  const root = document.querySelector('#analysis-atlas'); if (!root) return;
  const d = cache.get(root.dataset.path), $ = id => document.getElementById(id), value = id => $(id).value.trim()===''?NaN:Number($(id).value);
  root.querySelectorAll('form').forEach(form=>form.onsubmit=e=>e.preventDefault());
  if (root.dataset.page === 'cost') {
    let current;
    function setModel() {
      const model = d.costModels.find(m=>m.id===$('cost-model').value);
      $('cost-layer').innerHTML = '<option value="all">全部主干</option>'+model.groups.flatMap(g=>g.nodes.map(n=>`<option value="${n}">${n} · ${g.type} / ${g.ffn}</option>`)).join('');
      $('cost-path').disabled = ['v4','gqa'].includes(model.kind);
      if ($('cost-phase').value !== 'verify') $('cost-logits').value = model.defaultLogits;
      $('cost-query').max = $('cost-history').max = model.context;
      render();
    }
    function render() {
      const model = d.costModels.find(m=>m.id===$('cost-model').value);
      current = costEstimate(model,{batch:value('cost-batch'),query:value('cost-query'),history:value('cost-history'),phase:$('cost-phase').value,
        path:$('cost-path').value,logits:$('cost-logits').value,cacheBytes:value('cost-cache-bytes'),activationBytes:value('cost-act-bytes'),weightBytes:value('cost-weight-bytes'),layer:$('cost-layer').value});
      $('cost-export').disabled = !current.valid;
      if (!current.valid) {$('cost-result').innerHTML='<div class="analysis-error" role="alert">'+current.errors.map(esc).join('；')+'</div>'; return;}
      $('cost-result').innerHTML = `<p class="small">${esc(model.name)} · ${fmt(current.totalTokens)} 个可见 token · 固定 revision ${esc(model.revision)}。GB 为十进制单位。</p>`+current.warnings.map(w=>`<p class="notice">${esc(w)}</p>`).join('')+metrics([
        ['参考收缩 / 槽位上界',flops(current.contractionFlops),'矩阵乘加按 2 次计；其余算术未纳入'],['有效位置收缩量',flops(current.usefulContractionFlops),'有效因果/稀疏位置的理论值'],
        ['主 KV 容量',bytes(current.mainKvBytes),'V4 窗口预留容量与压缩前缀'],['Indexer cache',bytes(current.indexCacheBytes),'DSA FP8 key/F32 scale；V4 按所选精度'],
        ['FP32 压缩状态',bytes(current.compressionStateBytes),'两个状态及 overlap 单独计数'],['缓存与状态合计',bytes(current.sequenceCacheBytes),'不包含 workspace / allocator / RoPE 表'],
        ['完整主干逻辑权重容量',bytes(current.mainLogicalWeightBytes),'所选统一精度，不含 scale/副本'],['实际发布载荷',bytes(current.publishedWeightBytes),'全 checkpoint header，包含辅助模块和 metadata'],
      ])+`<section class="panel"><h2>计算分项</h2>${table(['分项','矩阵收缩次数'],[['层内投影',flops(current.projectionFlops)],['Head / 流折叠',flops(current.headFlops+current.headControlFlops)],['Attention 参考/槽位上界',flops(current.coreFlops)],['Indexer 完整候选打分',flops(current.indexFlops)]])}<p>投影输入与输出逻辑字节相加：<b>${bytes(current.projectionIOBytes)}</b>。重复消费重复计，未考虑融合或复用。</p></section><section class="panel"><h2>按层型核对</h2>${table(['层型','ratio','层数','投影','Attention 上界','主 KV','Index cache','FP32 状态'],current.rows.map(r=>[esc(r.type),r.ratio,r.layers,flops(r.projectionFlops),flops(r.coreFlops),bytes(r.mainKvBytes),bytes(r.indexCacheBytes),bytes(r.compressionStateBytes)]))}</section><section class="panel"><h2>声明张量的逻辑大小</h2>${table(['张量','最大单层 / 所选输出'],Object.entries(current.declaredTensors).map(([k,v])=>[esc({queryBytes:'Query 投影',residualBytes:'残差流',scoreBytes:'Score / 稀疏槽位（F32）',expertIntermediateBytes:'Gate/up 专家中间量',logitsBytes:'Logits（F32）'}[k]),bytes(v)]))}<p class="small">这些逻辑张量不必同时分配；融合实现可不显式生成 score，不能将它们相加当作峰值显存。</p><a href="#/family/deepseek/model/${model.id}">查看本版结构、矩阵和来源 →</a></section>`;
    }
    $('cost-controls').oninput=render;
    $('cost-model').onchange=setModel;
    $('cost-phase').onchange=()=>{const phase=$('cost-phase').value;$('cost-query').value=phase==='decode'?1:phase==='verify'?5:512;$('cost-history').value=phase==='prefill'?0:4095;if(phase==='verify')$('cost-logits').value='all';render();};
    $('cost-export').onclick=()=>downloadJSON('DeepSeek-cost-scenario.json',{scope:d.scope,limits:d.limits,result:current}); setModel();
  } else if (root.dataset.page === 'contracts') {
    let page=0;
    const chosen=()=>d.cannContracts.find(x=>x.id===$('contract-operator').value);
    function check() {
      const parse=id=>{const raw=$(id).value.trim();if(!raw)return null;const parts=raw.split(/[,×x\s]+/);if(parts.some(x=>x===''))return null;const v=parts.map(Number);return v.length&&v.every(n=>Number.isSafeInteger(n)&&n>=0)?v:null;};
      const contract=chosen(), shapes=['contract-x','contract-gamma','contract-w'].map(parse);
      if (shapes.some(x=>!x)) {$('contract-check-result').innerHTML='<div class="analysis-error" role="alert">shape 需要非负整数，使用逗号分隔；不能把格式错误的输入当作检查通过。</div>';return;}
      const result=checkContract(contract,{x:shapes[0],gamma:shapes[1],w:shapes[2],dim:value('contract-dim'),dtype:$('contract-dtype').value,layout:$('contract-layout').value});
      $('contract-check-result').innerHTML=`<div class="notice ${result.status==='contradiction'?'analysis-error':''}" data-status="${result.status}">${esc(result.message)}${!contract.rules.length?' 此家族尚未编码自动谓词，请按上方原文条件逐项核对。':''}</div>`+table(['必要条件','核对结果','证据'],result.results.map(r=>[esc(r.label),esc({contradiction:'矛盾',unknown:'输入不足 / 子集未核对',necessary_condition_met:'满足这一必要条件'}[r.status]),`<a target="_blank" rel="noopener" href="${esc(r.reference.url)}">固定范围 ↗</a>`]));
    }
    function render() {
      const c=chosen(), doc=$('contract-document').value, category=$('contract-category').value, q=$('contract-search').value.toLowerCase().trim();
      const claims=c.claims.filter(r=>(doc==='all'||r.apiDocument===doc)&&(category==='all'||r.categories.includes(category))&&(!q||JSON.stringify(r).toLowerCase().includes(q)));
      const pages=Math.max(1,Math.ceil(claims.length/40));page=Math.min(page,pages-1);
      $('contract-results').innerHTML=`<section class="panel"><h2>${esc(c.directory)}</h2><p>${esc(c.frameworkContext)}</p><p class="small">${esc(c.tag)} @ ${esc(c.revision)} · ${c.apis.map(esc).join(', ')}</p>${assumptions(c.limits)}<details><summary>文档中的产品声明（按 API 文档分别保存）</summary>${table(['文档','产品','所选文档声明','来源'],c.productSupport.map(p=>[esc(p.apiDocument),esc(p.product),esc(p.status),`<a href="${esc(p.reference.url)}" target="_blank" rel="noopener">固定行 ↗</a>`]))}</details><p class="count">${claims.length} 条匹配 · 第 ${page+1}/${pages} 页</p>${claims.length?table(['文档 / 上下文','参数 / 类别','原文表行或条件','来源'],claims.slice(page*40,page*40+40).map(r=>[esc(r.apiDocument)+`<small>${esc(r.context||'API 入口')}</small>`,esc(r.parameter)+`<small>${r.categories.map(esc).join(' / ')}</small>`,esc(r.statement)+`${r.tableMapping==='raw_row_with_context'?'<small>合并单元格保留原始列及上下文，未自动映射成单一参数契约。</small>':''}`,`<a href="${esc(r.reference.url)}" target="_blank" rel="noopener">L${r.reference.line}–${r.reference.end} ↗</a>`]),'contract-table'):'<div class="empty">没有匹配条件，请调整筛选。</div>'}<div class="link-row"><button id="contract-prev" ${page===0?'disabled':''}>上一页</button><button id="contract-next" ${page+1===pages?'disabled':''}>下一页</button></div></section>`;
      $('contract-prev').onclick=()=>{page--;render();};$('contract-next').onclick=()=>{page++;render();};check();
    }
    function choose(){const c=chosen();const documents=[...new Set(c.claims.map(x=>x.apiDocument))];$('contract-document').innerHTML='<option value="all">全部文档</option>'+documents.map(x=>`<option>${esc(x)}</option>`).join('');page=0;render();}
    $('contract-operator').onchange=choose;
    for(const id of ['contract-document','contract-category','contract-search'])$(id).oninput=()=>{page=0;render();};
    $('contract-check-controls').oninput=check;choose();
  } else if (root.dataset.page === 'parallel') {
    let current;
    function render(){
      const model=d.costModels.find(m=>m.id===$('parallel-model').value),remote=$('parallel-remote').value;
      current=parallelEstimate(model,{tp:value('parallel-tp'),dp:value('parallel-dp'),batch:value('parallel-batch'),query:value('parallel-query'),history:value('parallel-history'),weightBytes:value('parallel-weight'),activationBytes:value('parallel-activation'),cacheBytes:2,redundantExperts:value('parallel-redundant'),remoteFraction:remote==='uniform'?remote:+remote,allReduces:value('parallel-allreduces'),policy:$('parallel-policy').value});
      $('parallel-export').disabled=!current.valid;
      if(!current.valid){$('parallel-result').innerHTML='<div class="analysis-error" role="alert">'+current.errors.map(esc).join('；')+'</div>';return;}
      const {tp,dp}=current.input;
      $('parallel-result').innerHTML=`<p class="small">${esc(model.name)} · EP=${current.ep} · TP×DP=${tp}×${dp} · ${fmt(current.globalTokens)} 个全局 unique query token · 远端路由假设 ${(current.remoteFraction*100).toFixed(2)}%。</p>`+metrics([
        ['每 rank 逻辑权重容量',bytes(current.logicalWeightBytesPerRank),'明确的复制 / 均分及统一精度假设'],['每 MoE 层 dispatch 发送',bytes(current.dispatchSendBytesPerMoeLayer),'全局一向 payload，不计接收副本'],['每 MoE 层 combine 发送',bytes(current.combineSendBytesPerMoeLayer),'返回同宽 hidden 向量假设'],['全部 MoE 层逻辑发送',bytes(current.moeSendBytes),'dispatch + combine；不含 metadata/padding'],['均匀假设的每 rank MoE 发送',bytes(current.uniformMoeSendBytesPerRank),'均值不代表最忙 rank'],['每 rank TP ring 发送',bytes(current.tpRingSendBytesPerRank),'按指定 collective 次数和标准 ring 公式'],['全部 TP 组 ring 发送',bytes(current.tpRingSendBytes),'含全部 DP×TP rank 的发送'],['每 replica P/D 主 KV',bytes(current.pdMainKvBytesPerReplica),'有效记录；主 KV 精度暂按 2 B 假设'],
      ])+`<section class="panel"><h2>逻辑权重的三类份额</h2>${table(['类别','每 rank 元素'],[['复制的向量 / 位置表',fmt(current.replicatedElementsPerRank)],['非专家二维权重',fmt(current.nonExpertLinearElementsPerRank)],['路由专家与冗余副本',fmt(current.expertElementsPerRank)]])}<p>Indexer cache 另为每 replica ${bytes(current.indexCacheBytesPerReplica)}，是否迁移或重算依 connector 与布局协议。</p></section><section class="panel"><h2>进程组示意</h2><p>按 TP×DP 展示；只画前 64 个 rank。expert 布局为本页的均分假设。</p><div class="rank-grid">${Array.from({length:Math.min(current.ranks,64)},(_,rank)=>`<div><b>Rank ${rank}</b><span>DP ${Math.floor(rank/tp)} · TP ${rank%tp}</span><small>${current.ep>1?'EP '+rank:'无 EP 分片'}</small></div>`).join('')}</div></section>`;
    }
    $('parallel-controls').oninput=render;$('parallel-export').onclick=()=>downloadJSON('DeepSeek-parallel-scenario.json',{scope:d.scope,limits:d.parallel.limits,result:current});render();
  } else if (root.dataset.page === 'mechanisms') mountDemo(root.dataset.mechanism);
}

let crossSelected;
export async function comparisonMarkup(path) {
  const d=await load(path);
  if(!crossSelected)crossSelected=new Set([d.models.find(m=>m.key==='deepseek:v4-flash'),d.models.find(m=>m.key==='kimi:k3'),d.models.find(m=>m.familyId==='glm'&&m.id.includes('5.3'))].filter(Boolean).map(m=>m.key));
  return `<main id="content" class="cross-comparison" data-path="${esc(path)}"><a class="back" href="#/">← 全部研究专题</a>${head('跨家族统一比较','选择 DeepSeek、Kimi 与 GLM 的已有条目，核对结构、参数、存储和来源。保留每版的统计范围及读取深度。')}<div class="notice">${d.models.length} 个既有条目，包含未核对的历史节点。未知值不补零；没有统一条件下的速度或能力排名。</div><div class="filters">${select('cross-family','按家族筛选',[['all','全部家族'],...d.families.map(f=>[f.id,f.name])],'all')}<label>搜索版本<input id="cross-search" type="search" placeholder="V4、K3、GLM、历史…"></label><button id="cross-reset">恢复默认选择</button></div><details><summary>选择比较条目（最多 6 个）</summary><div id="cross-models" class="cross-models"></div></details><div id="cross-result" aria-live="polite"></div>${assumptions(d.limits)}<div class="link-row"><a class="button" href="assets/deepseek/跨家族比较.json" download>下载比较数据</a><a class="button" href="assets/deepseek/跨家族比较.csv" download>下载比较 CSV</a></div></main>`;
}
export function mountComparison() {
  const root=document.querySelector('.cross-comparison');if(!root)return;
  const d=cache.get(root.dataset.path),$=id=>document.getElementById(id);
  const names={layers:'主干层数',hidden:'残差宽度',attention:'注意力结构',heads:'Query heads',experts:'路由专家总数',topK:'每 token 选择专家',shared:'共享专家等价数量',expertWidth:'专家中间宽度',denseWidth:'Dense 中间宽度',context:'配置上下文上限',vocab:'词表',parameters:'逻辑参数（各版口径）',payload:'权重有效载荷',quantization:'精度披露 / 配置',mtpParameters:'MTP 参考参数',dsparkStages:'DSpark stages'};
  const evidence={official:'官方披露',derived:'计算推导',interpretation:'解释判断',unknown:'未知'};
  function choices(){const q=$('cross-search').value.trim().toLowerCase(),f=$('cross-family').value;const models=d.models.filter(m=>(f==='all'||m.familyId===f)&&(!q||(m.name+' '+m.family+' '+m.summary).toLowerCase().includes(q)));$('cross-models').innerHTML=models.length?models.map(m=>`<label><input type="checkbox" data-cross="${esc(m.key)}" ${crossSelected.has(m.key)?'checked':''}><span>${esc(m.family)} · ${esc(m.name)}<small>${esc(m.branch)}</small></span></label>`).join(''):'<div class="empty">没有匹配条目。</div>';}
  function results(){const models=d.models.filter(m=>crossSelected.has(m.key));$('cross-result').innerHTML=models.length?`<p class="count">已选择 ${models.length} 个条目。横向滚动核对，来源和口径保留在各列。</p>`+table(['维度',...models.map(m=>m.family+' / '+m.name)],d.dimensions.map(key=>[esc(names[key]),...models.map(m=>{const fact=m.facts[key],v=fact?.value;return `<span>${v==null?'未核对 / 未收录':esc(key==='payload'&&typeof v==='number'?bytes(v):typeof v==='number'?fmt(v):v)}</span><span class="evidence ${fact?.evidence||'unknown'}">${esc(evidence[fact?.evidence||'unknown'])}</span>${fact?.note?`<small>${esc(fact.note)}</small>`:''}${fact?.source?`<small><a href="${esc(fact.source)}" target="_blank" rel="noopener">事实来源 ↗</a></small>`:''}`;})]).concat([
    ['Cache 表示与范围',...models.map(m=>m.cache?`<details class="comparison-cache"><summary>查看本版 cache 公式</summary>${Object.entries(m.cache).map(([k,v])=>`<p><b>${esc(k)}</b>：${esc(typeof v==='string'?v:JSON.stringify(v))}</p>`).join('')}</details>`:'未核对 / 未收录')],['固定 revision',...models.map(m=>m.revision?`<code>${esc(m.revision)}</code>`:'未知')],['核验范围',...models.map(m=>esc(m.evidenceDepth))],['参数统计范围',...models.map(m=>esc(typeof m.parameterScope==='string'?m.parameterScope:JSON.stringify(m.parameterScope)))],['结构与研究入口',...models.map(m=>`<a href="${esc(m.detailPath)}">查看本版结构 →</a>`)]
  ]),'cross-table'):'<div class="empty">请添加比较条目。</div>';}
  root.addEventListener('change',e=>{const key=e.target.dataset.cross;if(!key)return;if(e.target.checked&&crossSelected.size>=6){e.target.checked=false;$('cross-result').insertAdjacentHTML('afterbegin','<p role="alert">最多比较 6 个条目，请先移除一个。</p>');return;}e.target.checked?crossSelected.add(key):crossSelected.delete(key);results();});
  $('cross-search').oninput=choices;$('cross-family').onchange=choices;$('cross-reset').onclick=()=>{crossSelected=new Set(['deepseek:v4-flash','kimi:k3',d.models.find(m=>m.familyId==='glm'&&m.id.includes('5.3'))?.key].filter(Boolean));$('cross-search').value='';$('cross-family').value='all';choices();results();};choices();results();
}

function mountDemo(kind) {
  const $=id=>document.getElementById(id);let step=0;
  const controls={mla:num('demo-rank','教学 latent rank',3,2,4),dsa:num('demo-topk','选择 top-k',3,1,8)+num('demo-visible','可见历史长度',6,1,8),compression:num('demo-tokens','教学 token 数',11,1,24)+select('demo-ratio','压缩 ratio',[[4,'4（含 overlap）'],[8,'8（教学非 overlap）'],[128,'128（HCA 分块示意）']],4),mhc:num('demo-iterations','Sinkhorn 迭代次数',20,1,40),dspark:num('demo-block','教学 block size',5,2,6)+num('demo-threshold','前缀生存概率阈值',0.6,0.1,0.95,0.05)};
  $('demo-controls').innerHTML=controls[kind];
  function render(){
    const input=id=>+$(id).value;
    const options=kind==='mla'?{rank:input('demo-rank')}:kind==='dsa'?{topK:input('demo-topk'),visible:input('demo-visible')}:kind==='compression'?{tokens:input('demo-tokens'),ratio:input('demo-ratio')}:kind==='mhc'?{iterations:input('demo-iterations')}:{block:input('demo-block'),threshold:input('demo-threshold')};
    const invalid=[...$('demo-controls').querySelectorAll('input')].some(el=>!el.checkValidity());
    if(invalid){$('demo-result').innerHTML='<div class="analysis-error" role="alert">请使用控件允许范围内的教学输入。</div>';return;}
    const r=mechanism(kind,options);$('demo-flow').innerHTML='<ol class="demo-flow">'+r.steps.map((name,i)=>`<li><button data-demo-step="${i}" class="${i===step?'active':''}" ${i===step?'aria-current="step"':''}><span>${i+1}</span>${esc(name)}</button></li>`).join('')+'</ol>';
    $('demo-step-label').textContent=`${step+1}/5 · ${r.steps[step]}`;$('demo-prev').disabled=step===0;$('demo-next').disabled=step===4;
    let body='';
    if(kind==='mla'){
      body=step===0?matrixTable('潜变量 C',r.C)+matrixTable('W_K',r.Wk)+matrixTable('W_V',r.Wv):step===1?matrixTable('展开 K',r.K)+matrixTable('展开 V',r.V):step===2?table(['历史位置','展开打分','吸收打分'],r.expandedScores.map((v,i)=>[i,fmt(v),fmt(r.latentScores[i])])):step===3?table(['历史位置','softmax 概率'],r.probability.map((v,i)=>[i,fmt(v)])):metrics([['两条输出的最大绝对差',fmt(r.maxError),'JavaScript double 的教学代数检查'],['展开缓存元素',r.expandedCacheElements,'本教学单 head；含 RoPE'],['latent 缓存元素',r.latentCacheElements,'R + RoPE；真实模型维度另见计算器']])+table(['维度','展开输出','吸收输出'],r.outExpanded.map((v,i)=>[i,fmt(v),fmt(r.outLatent[i])]));
    }else if(kind==='dsa'){
      body=table(['历史位置','可见','教学 key','Indexer 分数','选中','独立 attention 分数','最终权重'],r.keys.map((k,i)=>[i,i<r.visible?'是':'未来 / mask',k.map(fmt).join(', '),step>=1?fmt(r.indexScores[i]):'—',step>=2?(r.selected.includes(i)?'✓':'—'):'—',step>=3?(r.selected.includes(i)?fmt(r.targetScores[i]):'−∞'):'—',step>=4?fmt(r.probability[i]):'—']));
      if(step===4)body+=`<p>教学上下文聚合：[${[0,1].map(j=>fmt(dot(r.probability,r.keys.map(k=>k[j])))).join(', ')}]。选择分数与 attention 分数使用不同教学 query。</p>`;
    }else if(kind==='compression'){
      if(step===0||step===1)body=table(['Token','教学向量','块号','已完成块'],r.values.map((v,i)=>[i,v.map(fmt).join(', '),Math.floor(i/r.ratio),i<r.remainderStart?'是':'否，留在状态']));
      if(step===2)body=r.compressed.length?table(['压缩记录','来源位置（overlap 保留）','教学 gating 权重','池化向量'],r.compressed.map(x=>[x.block,x.positions.join(', '),x.weights.map(fmt).join(', '),x.value.map(fmt).join(', ')])):'<div class="empty">没有完整块，不生成压缩记录。</div>';
      if(step===3)body=metrics([['完整压缩记录',r.compressed.length,'floor(token/ratio)'],['尾部 token',r.remainder.length,'保留到后续输入，不丢弃']])+ (r.remainder.length?table(['原始位置','状态中的教学向量'],r.remainder.map((v,i)=>[r.remainderStart+i,v.map(fmt).join(', ')])):'<p>本次没有尾部。</p>');
      if(step===4)body=metrics([['窗口位置',r.windowPositions.join(', '),'教学窗口长度为 8'],['压缩前缀记录',r.compressed.length,'与窗口并存，位置可能重叠'],['未满块状态',r.remainder.length,'不提前生成压缩记录']])+`<p>ratio=4 使用前一完整块与当前块的 overlap。ratio=128 的 HCA 只展示分块；教学没有 learned indexer、位置变换或 QAT。</p>`;
    }else if(kind==='mhc'){
      body=step===0?matrixTable('残差 residual[stream,hidden]',r.residual):step===1?table(['流','指定 pre','残差'],r.pre.map((v,i)=>[i,fmt(v),r.residual[i].map(fmt).join(', ')]))+`<p>折叠输入：[${r.input.map(fmt).join(', ')}]；指定 pre 不要求和为 1。</p>`:step===2?`<p>教学子层 tanh(x + bias)：[${r.sublayer.map(fmt).join(', ')}]。</p>`:step===3?matrixTable('comb[i,j]（行 i → 输出流 j）',r.comb)+table(['流','行和','列和'],r.rowSums.map((v,i)=>[i,fmt(v),fmt(r.columnSums[i])])):matrixTable('post_j × sublayer + Σ_i comb[i,j] × residual_i',r.output);
    }else if(kind==='dspark'){
      if(step===0)body=matrixTable('一次并行产生的教学 base logits（5-token 词表）',r.baseLogits);
      if(step===1)body=table(['块内位置','顺序选择的 token','Markov 修正后的草稿分布'],r.drafts.map((t,i)=>[i,t,r.probabilities[i].map(fmt).join(', ')]));
      if(step===2)body=metrics([['预测接受长度 ΣS_k',fmt(r.expectedAcceptedPrediction),'教学 confidence，非实际接受比例'],['选择验证前缀',r.verifyLength,'阈值示例至少验证 1 个位置']])+table(['位置','confidence 示例','前缀生存概率 S_k','送 target verify'],r.confidences.map((v,i)=>[i,fmt(v),fmt(r.prefixSurvival[i]),i<r.verifyLength?'是':'否']));
      if(step===3)body=table(['验证位置','草稿 token','draft q(token)','target p(token)','min(1,p/q)','教学抽样 u','结果'],r.verification.map(v=>[v.position,v.token,fmt(v.draft[v.token]),fmt(v.target[v.token]),fmt(v.alpha),fmt(v.draw),v.accepted?'接受':'拒绝，后续停止']));
      if(step===4)body=metrics([['接受的草稿前缀',r.accepted,'仅已验证且未拒绝的连续前缀'],['提交 token 序列',r.committed.join(', '),r.rejection?'末尾来自归一化 max(p−q,0) 校正分布':'末尾为教学 target bonus'],['丢弃的草稿数量',r.drafts.length-r.accepted,'拒绝位置及未验证后缀不当作历史']])+(r.rejection?table(['词表 token','校正分布'],r.rejection.correctionDistribution.map((v,i)=>[i,fmt(v)])):'<p>本次验证前缀全部接受，再由目标分布生成 bonus token。</p>');
    }
    $('demo-result').innerHTML=`<div class="demo-stage" data-step="${step}" data-kind="${kind}"><h3>${esc(r.steps[step])}</h3>${body}</div>`;
    document.querySelectorAll('[data-demo-step]').forEach(button=>button.onclick=()=>{step=+button.dataset.demoStep;render();});
  }
  $('demo-prev').onclick=()=>{step=Math.max(0,step-1);render();};$('demo-next').onclick=()=>{step=Math.min(4,step+1);render();};$('demo-reset').onclick=()=>{step=0;$('demo-controls').innerHTML=controls[kind];render();};$('demo-controls').oninput=render;render();
}
