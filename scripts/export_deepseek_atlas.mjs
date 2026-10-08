// Author the independent research workbook and CSV exports with the bundled Artifact Tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
const require = createRequire(path.join(process.cwd(),'artifact-loader.cjs'));
const { Workbook, SpreadsheetFile } = await import(require.resolve('@oai/artifact-tool'));
const root=fileURLToPath(new URL('../',import.meta.url));
const data=path.join(root,'data/families/deepseek');const out=path.join(root,'dist/assets/deepseek');
const load=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const family=await load(path.join(data,'family.json'));const compute=await load(path.join(data,'compute.json'));
const hardware=await load(path.join(data,'hardware.json'));const evidence=await load(path.join(data,'research/site-proofs.json'));
const speculation=await load(path.join(data,'research/speculation.json'));const tokenizer=await load(path.join(data,'research/r1-full-tokenizer-audit.json'));
const wb=Workbook.create();
const text=v=>v==null?'unknown':Array.isArray(v)?v.join(' × '):String(v);
const shapeHeaders=['model_id','model','component','group','source_index_0based','display_layer','type','phase','path','step_id','operation','input_shape','output_shape','dtype','logical_weight_reference_parameters','stored_weight_evidence','source_ids','reference_functions','source_urls','math_relation','duration','scope'];
const shapeRows=[];const templateRows=[];const matrixRows=[];const summaryRows=[];
for(const model of family.models){
 const arch=await load(path.join(root,model.architecturePath));
 summaryRows.push([model.name,model.revision,model.facts.layers.value,model.facts.attention.value,model.facts.hidden.value,model.facts.parameters.value,arch.mtpLogicalParameters??'unknown',arch.dsparkLogicalParameters??0,arch.dspark?.stages??0,arch.dspark?.blockSize??'不适用',model.facts.quantization.value,model.auditPath?'完整本 checkpoint 文件头':'配置+源码推导',model.baseModel||'未声明权重底座']);
 const seen=new Set();
 for(const n of arch.nodes){
  const key=JSON.stringify([n.group,n.type,n.ffn,n.routing,n.compressRatio,n.modules.map(m=>[m.id,m.matrices.map(x=>[x.tensor_template,x.logical_shape,x.count,x.role])])]);
  if(seen.has(key))continue;seen.add(key);
  const matches=arch.nodes.filter(p=>JSON.stringify([p.group,p.type,p.ffn,p.routing,p.compressRatio,p.modules.map(m=>[m.id,m.matrices.map(x=>[x.tensor_template,x.logical_shape,x.count,x.role])])])===key);
  const layers=matches.map(x=>x.number||x.id).join(',');
  for(const mod of n.modules)for(const m of mod.matrices){
   const p=evidence.proofs[m.proof];
   matrixRows.push([model.name,layers,n.type,mod.title,m.tensor_template,text(m.logical_shape),m.logical_parameters_each,m.count,m.logical_parameters_each*m.count,text(m.stored_shape),text(m.stored_dtype),m.payloadBytes,m.role,m.evidence,p?.symbol||'文件头/共享口径',p?.url||m.source]);
  }
 }
 const cm=compute.models.find(x=>x.id===model.id);
 for(const node of cm.nodes){
  for(const s of compute.templates[node.template].steps){
   const proofs=s.reference.proofs.map(id=>compute.proofs[id]);
   const row=[model.id,model.name,node.id,node.group,node.sourceIndex,node.layer,node.type,s.phase,s.path,s.id,s.title,s.input,s.output,s.dtype,s.parameters,cm.weightEvidence,proofs.map(p=>p.source_id).join(';'),proofs.map(p=>p.symbol).join(';'),proofs.map(p=>p.url).join(';'),s.relation,null,s.note];
   shapeRows.push(row);
  }
 }
}
for(const t of Object.values(compute.templates))for(const s of t.steps){
 const model=family.models.find(m=>m.id===t.modelId);const p=s.reference.proofs.map(id=>compute.proofs[id]);
 templateRows.push([model.name,t.id,t.layerType,t.ffn||'global',s.phase,s.path,s.title,s.input,s.output,s.parameters,s.relation,p.map(x=>x.symbol).join(';'),p.map(x=>x.url).join(';'),s.note]);
}
const sourceRows=Object.values(evidence.sources).map(s=>[s.id,s.title||s.modelId||s.repo,s.revision,s.path||'metadata',s.kind,s.sha256,s.readScope,s.url]);
const gapRows=hardware.optimizations.map(o=>[o.priority,o.title,o.status,o.scope,o.observation,o.proposal,o.metric,o.risk]);
const specRows=[];
for(const m of speculation.models){for(const [k,v] of Object.entries(m.config))specRows.push([m.name,'checkpoint 配置',k,text(typeof v==='object'?JSON.stringify(v):v),m.referenceProofs.map(id=>evidence.proofs[id].url).join(';'),'MTP namespace 与 DSpark 实际 stage 分开；服务 token 数另记']);for(const c of m.conflicts)specRows.push([m.name,'配置/服务口径',c.field,JSON.stringify(c),'固定 config/inference/header/model card',c.resolution]);}
for(const p of speculation.protocol)specRows.push(['MTP / DSpark','服务协议',p.phase,p.operation,p.proofs.map(id=>evidence.proofs[id].url).join(';'),p.limit||'静态协议；没有设备执行验证']);
const tokenRows=tokenizer.records.map(r=>[r.model,r.baseModel,r.distillVocabularyEntries,r.distillMerges,r.addedTokens,r.embeddingRows,r.comparison?.commonTokenIdsChanged??'gated',r.comparison?.differentMergePositions??'gated',r.comparison?(r.comparison.addedTokensIdentical?'是':'否'):'gated',JSON.stringify(r.specialTokens),r.gaps.length?'原底座 gated，公开 Distill 已核对':'完整 vocab/merge/config ID 已核对']);
gapRows.push(['说明','本次验收范围','静态研究与网站','结构/来源/shape/导出/网站','当前没有硬件，设备实验已移出本次范围','未来需要时单独发起设备研究','本次按静态与网站校验验收','未测 duration 等保持空值，不等于 0 或测试通过']);
const definitions=[
 {name:'版本概览',title:'DeepSeek 模型与算子研究',headers:['模型','固定 revision','主干层','注意力层型','hidden','主干逻辑参数','MTP 参考参数','DSpark 独立参数','DSpark stages','DSpark block','精度披露/配置','存储核验','底座'],rows:summaryRows,widths:[42,46,12,34,12,24,24,24,18,18,68,32,44]},
 {name:'矩阵模板',title:'层与模块矩阵模板',headers:['模型','覆盖层/组件','层型','模块','张量模板','逻辑 shape','单份逻辑参数','份数','总逻辑参数','存储 shape','存储 dtype','单份 payload bytes','角色','依据','参考符号','来源'],rows:matrixRows,widths:[42,42,16,40,62,26,23,12,24,26,18,24,24,18,46,100]},
 {name:'算子模板',title:'计算模板：全层 CSV 通过映射展开',headers:['模型','模板 ID','层型','FFN','阶段','分支','步骤','输入 shape','输出 shape','引用参数','数学关系','参考函数','来源','条件/缺口'],rows:templateRows,widths:[42,34,16,16,16,30,58,80,70,24,96,64,100,100]},
 {name:'来源',title:'固定来源与读取范围',headers:['source_id','来源','revision','path','类型','SHA-256','已读取范围','URL'],rows:sourceRows,widths:[58,60,48,62,26,70,95,110]},
 {name:'实验与缺口',title:'可选后续研究与证据边界',headers:['优先级','任务','状态','范围','观察','方案','指标','风险'],rows:gapRows,widths:[12,48,44,55,95,95,75,95]},
 {name:'推测解码',title:'MTP / DSpark：checkpoint 与协议',headers:['模型/方法','类别','项目/阶段','配置/关系','来源','边界'],rows:specRows,widths:[44,26,40,100,110,105]},
 {name:'Tokenizer核对',title:'完整 tokenizer.json / BPE 静态核对',headers:['蒸馏模型','原底座','vocab entries','merge entries','added tokens','embedding rows','共同 token ID 变化','merge 差异位置','added tokens一致','特殊 token ID','范围/缺口'],rows:tokenRows,widths:[44,48,20,20,18,22,24,24,24,110,72]},
];
const renderSheets=process.env.ATLAS_RENDER_SHEETS?new Set(process.env.ATLAS_RENDER_SHEETS.split(',')):null;
if(renderSheets)for(const name of renderSheets)if(!definitions.some(d=>d.name===name))throw Error('Unknown preview sheet: '+name);
await fs.mkdir(out,{recursive:true});
for(const def of definitions){
 const sheet=wb.worksheets.add(def.name);sheet.showGridLines=false;
 sheet.getRange('A2').values=[[def.title]];sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true};
 const grid=[def.headers,...def.rows];const range=sheet.getRangeByIndexes(4,0,grid.length,def.headers.length);range.values=grid;
 range.format.font={name:'Arial',size:10};range.format.verticalAlignment='center';range.format.wrapText=true;
 range.format.rowHeight=48;
 for(let i=0;i<def.widths.length;i++)sheet.getRangeByIndexes(4,i,grid.length,1).format.columnWidth=def.widths[i];
 const header=sheet.getRangeByIndexes(4,0,1,def.headers.length);header.format={fill:'#20344A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:34,wrapText:true};
 sheet.freezePanes.freezeRows(5);sheet.freezePanes.freezeColumns(1);
 for(let c=0;c<def.headers.length;c++){
  if(def.rows.some(row=>typeof row[c]==='number'))sheet.getRangeByIndexes(5,c,def.rows.length,1).format.numberFormat='#,##0';
 }
 range.format.autofitRows();
}
// CSV is a serialization of verified Artifact Tool range values; it has no formatting dependency.
const csvBook=Workbook.create();const csvSheet=csvBook.worksheets.add('全层数据');
csvSheet.getRangeByIndexes(0,0,shapeRows.length+1,shapeHeaders.length).values=[shapeHeaders,...shapeRows];
wb.recalculate();csvBook.recalculate();
const csvValues=csvSheet.getRangeByIndexes(0,0,shapeRows.length+1,shapeHeaders.length).values;
if(csvValues.length!==shapeRows.length+1||csvValues[0].join('|')!==shapeHeaders.join('|'))throw Error('CSV range mismatch');
for(let i=1;i<csvValues.length;i++)if(JSON.stringify(csvValues[i])!==JSON.stringify(shapeRows[i-1]))throw Error('CSV source values changed at row '+i);
const quote=v=>{const s=v==null?'':String(v);return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s};
const serialize=rows=>'\uFEFF'+rows.map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n';
await fs.writeFile(path.join(out,'DeepSeek-all-layer-shapes.csv'),serialize(csvValues));
const perModel=[];
for(const model of family.models){
 const rows=csvValues.slice(1).filter(r=>r[0]===model.id);const file='per-model/'+model.id+'-layers.csv';
 await fs.mkdir(path.join(out,'per-model'),{recursive:true});await fs.writeFile(path.join(out,file),serialize([shapeHeaders,...rows]));perModel.push({modelId:model.id,path:'assets/deepseek/'+file,rows:rows.length});
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:10},maxChars:1500});console.log(errors.ndjson);
const inspect=await wb.inspect({kind:'table',range:'版本概览!A5:F8',include:'values,formulas',tableMaxRows:4,tableMaxCols:6,maxChars:1600});console.log(inspect.ndjson);
for(const def of definitions){
 const sheet=wb.worksheets.getItem(def.name);const actual=sheet.getRangeByIndexes(5,0,def.rows.length,def.headers.length).values;
 if(JSON.stringify(actual)!==JSON.stringify(def.rows))throw Error('Workbook values changed: '+def.name);
 if(!renderSheets||renderSheets.has(def.name)){
  const noteRow=def.name==='算子模板'?def.rows.findIndex(row=>row[13]?.includes('本版文件头核对 I8'))+6:null;
  const sourceRow=def.name==='来源'?def.rows.findIndex(row=>row[0]?.startsWith('cann-'))+6:null;
  const ranges=def.name==='版本概览'?[['','A1:F9'],['-auxiliary','G5:J26'],['-precision','K5:M26']]:def.name==='算子模板'&&noteRow>=6?[['','A1:F9'],['-storage-note',`N${noteRow}:N${noteRow+2}`]]:def.name==='来源'&&sourceRow>=6?[['','A1:F9'],['-cann',`A${sourceRow}:D${sourceRow+3}`],['-cann-links',`G${sourceRow}:H${sourceRow+3}`]]:[['','A1:F9']];
  for(const [suffix,range]of ranges){
   const preview=await wb.render({sheetName:def.name,range,scale:1.2,format:'png'});await fs.writeFile(path.join(process.cwd(),'atlas-'+def.name+suffix+'.png'),new Uint8Array(await preview.arrayBuffer()));
  }
 }
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'DeepSeek-模型与算子.xlsx'));
await fs.rm(path.join(out,'DeepSeek-模型与算子.xlsx.inspect.ndjson'),{force:true});
await fs.writeFile(path.join(data,'research/export-manifest.json'),JSON.stringify({schemaVersion:1,snapshot:'2026-10-01',computeSha256:createHash('sha256').update(await fs.readFile(path.join(data,'compute.json'))).digest('hex'),expandedRows:shapeRows.length,templateRows:templateRows.length,matrixTemplateRows:matrixRows.length,perModel,workbook:{path:'assets/deepseek/DeepSeek-模型与算子.xlsx',sheets:definitions.map(d=>({name:d.name,rows:d.rows.length,columns:d.headers.length}))},scope:'独立研究工作簿；模板/参数/源码证据与设备实测分开，全层 CSV 使用相同模板展开；未知性能字段为空'},null,2)+'\n');
console.log(JSON.stringify({expandedRows:shapeRows.length,templateRows:templateRows.length,matrixTemplateRows:matrixRows.length,models:perModel.length}));
