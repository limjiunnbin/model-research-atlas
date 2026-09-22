const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cache=new Map();
export async function research(family,page,topic='all',modelId=''){
 if(!family.hardwarePath)return '<h1>实现与硬件研究尚未收录</h1>';
 if(!cache.has(family.hardwarePath)){const r=await fetch(family.hardwarePath);if(!r.ok)throw Error('实现研究资料读取失败');cache.set(family.hardwarePath,await r.json())}
 const d=cache.get(family.hardwarePath),model=family.models.find(m=>m.id===modelId),base=`#/family/${family.id}`,url=(p,t='all')=>`${base}/${p}/${t}${model?'/'+model.id:''}`;
 const refs=rs=>`<details class="research-refs"><summary>核对来源与固定版本（${rs.length}）</summary><ul>${rs.map(r=>{const s=d.sources.find(s=>s.id===r.id);return `<li><a target="_blank" rel="noopener" href="${esc(s.url+(r.line?'#L'+r.line:''))}">${esc(s.title)}${r.line?' · L'+r.line:''} ↗</a><small>${esc(s.revision)} · 访问 ${esc(s.accessed)}</small></li>`}).join('')}</ul></details>`;
 const related=ids=>`<div class="link-row small">适用条目：${ids.map(id=>family.models.find(m=>m.id===id)).filter(Boolean).map(m=>`<a href="${base}/model/${m.id}">${esc(m.name)}</a>`).join('')}</div>`;
 const title={implementation:'实现与算子',hardware:'硬件与部署',ascend:'Ascend 优化研究'}[page];
 let h=`<div class="page-head"><div class="eyebrow">IMPLEMENTATION RESEARCH / ${esc(d.updated)}</div><h1>${title}</h1><p class="intro">${esc(d.scope)}</p></div><div class="section-nav">${[['implementation','实现与算子'],['hardware','硬件与部署'],['ascend','Ascend 优化']].map(([p,n])=>`<a class="${page===p?'active':''}" href="${url(p)}">${n}</a>`).join('')}</div>`;
 if(model)h+=`<div class="notice">当前版本：<a href="${base}/model/${model.id}">${esc(model.name)}</a> · ${esc(d.versionNotes.find(x=>x.models.includes(model.id))?.text||'本条目没有单独的代码核验。')} <a href="${base}/${page}">查看全家族</a></div>`;
 h+=`<div class="research-legend"><span>源码核验：看到对应路径</span><span>部署文档：项目给出配方</span><span>厂商实验：来源自行测量</span><span>本站硬件实测：未进行</span></div>`;
 if(page==='implementation'){
  const ms=d.modules.filter(m=>!model||m.models.includes(model.id)),selected=ms.find(m=>m.id===topic);
  h+=`<div class="section-nav"><a class="${!selected?'active':''}" href="${url(page)}">全部专题</a>${ms.map(m=>`<a class="${selected?.id===m.id?'active':''}" href="${url(page,m.id)}">${esc(m.title.split('：')[0])}</a>`).join('')}</div>`;
  h+=(selected?[selected]:ms).map(m=>`<section class="panel research-card"><div class="eyebrow">${esc(m.evidence)}</div><h2>${esc(m.title)}</h2><ol class="compute-flow">${m.flow.map(x=>`<li>${esc(x)}</li>`).join('')}</ol><p>${esc(m.meaning)}</p><dl class="research-pairs"><dt>源码对应</dt><dd>${esc(m.code)}</dd><dt>硬件含义</dt><dd>${esc(m.hardware)}</dd><dt>结论边界</dt><dd>${esc(m.limit)}</dd></dl>${related(m.models)}${refs(m.refs)}</section>`).join('')||'<div class="empty">该版本尚无实现专题；不会借用其他版本的结果。</div>';
 }else if(page==='hardware'){
 h+=`<div class="notice">先确认具体代际，再确认检查点格式、实际内核和部署栈。下面是支持证据比较，不是速度或性价比排名。${model?'硬件表保留全家族条目；每行明确其实际覆盖的模型，不能视为当前版本全覆盖。':''}</div><div class="table-wrap"><table class="hardware-table"><thead><tr><th>硬件代际</th><th>精度与内核</th><th>模型支持证据</th><th>软件快照</th><th>边界与出处</th></tr></thead><tbody>${d.platforms.map(p=>`<tr><th scope="row">${esc(p.name)}</th><td>${esc(p.precision)}</td><td>${esc(p.support)}</td><td>${esc(p.stack)}</td><td>${esc(p.boundary)}${refs(p.refs)}</td></tr>`).join('')}</tbody></table></div><h2>公平比较的最低条件</h2><ol class="protocol">${d.protocol.map(p=>`<li>${esc(p)}</li>`).join('')}</ol>`;
 }else{
 h+=`<div class="notice">优先级表示建议验证顺序，不是预计收益排名。当前没有证据支持“Ascend 完全不支持 K3”；明确问题是版本一致性，其他性能项需先测量。</div>`;
 h+=d.optimizations.map(o=>`<section class="panel research-card"><div class="priority-row"><span class="pill">${o.priority}</span><span class="small">${esc(o.status)}</span></div><h2>${esc(o.title)}</h2><p class="small">范围：${esc(o.scope)}</p><dl class="research-pairs">${[['已观察','observation'],['建议动作','proposal'],['验证指标','metric'],['风险 / 约束','risk']].map(([label,k])=>`<dt>${label}</dt><dd>${esc(o[k])}</dd>`).join('')}</dl>${refs(o.refs)}</section>`).join('');
 h+=`<h2>统一实验协议</h2><ol class="protocol">${d.protocol.map(p=>`<li>${esc(p)}</li>`).join('')}</ol>`;
 }
 h+=`<section class="panel"><h2>保存与复核</h2><div class="link-row"><a class="button primary" href="${esc(d.downloads.report)}" download>下载中文研究报告</a><a class="button" href="${esc(d.downloads.sources)}" download>下载 ${d.sources.length} 条来源索引</a><a href="${esc(family.hardwarePath)}" download>研究数据 JSON</a></div><p class="small">源码来源固定提交；厂商滚动文档记录访问日。代码文件 SHA-256 在来源 CSV 中供复核，未将第三方源码重新分发。</p><details class="research-refs"><summary>展开全部来源</summary><ul>${d.sources.map(s=>`<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title)}</a><small>${esc(s.kind)} · ${esc(s.revision)} · ${esc(s.accessed)}</small></li>`).join('')}</ul></details></section>`;
 return h;
}
