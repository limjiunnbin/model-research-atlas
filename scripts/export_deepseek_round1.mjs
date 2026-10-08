// Run with bundled Node from a writable directory whose node_modules links to the bundled runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';

const require = createRequire(path.join(process.cwd(), 'artifact-loader.cjs'));
const { Workbook } = await import(require.resolve('@oai/artifact-tool'));
const root = fileURLToPath(new URL('../', import.meta.url));
const out = path.join(root, 'research/deepseek/round1');
const sourcePath = path.join(root, 'data/families/deepseek/research/v3-round1.json');
const sourceRaw = await fs.readFile(sourcePath, 'utf8');
const d = JSON.parse(sourceRaw);
const sources = new Map(JSON.parse(await fs.readFile(path.join(root, 'data/families/deepseek/sources.json'), 'utf8')).sources.map(s => [s.id, s]));
const wb = Workbook.create();
const textValue = value => value == null ? 'unknown' : typeof value === 'string' ? value : JSON.stringify(value);
const parameterHeaders = ['section','item','numeric_value','text_value','tensor_template','logical_shape','logical_parameters_each','count','logical_parameters','expected_scale_shape','stored_shape','stored_dtype','payload_bytes','evidence','source_id','reference_url','scope'];
const parameterRows = [];
for (const f of d.configFields) {
  parameterRows.push(['config', f.field, typeof f.value === 'number' ? f.value : null, typeof f.value === 'number' ? null : textValue(f.value), null, null, null, null, null, null, null, 'unknown', null, f.evidence, f.source_id, sources.get(f.source_id).url, f.demoField ? `demo.${f.demoField}=${textValue(f.demoValue)}` : 'checkpoint 字段；无直接 demo 配置映射']);
}
for (const m of d.matrices) {
  parameterRows.push(['matrix', m.id, null, null, m.tensorTemplate, textValue(m.logicalShape), m.logicalParametersEach, m.count, m.logicalParameters, textValue(m.expectedScaleShape), 'unknown', 'unknown', null, m.evidence, m.source_ids.join(';'), d.proofs[m.proof_id].url, '逻辑形状来自配置+参数声明；索引仅核对名称，非文件头审计']);
}
for (const [key,value] of Object.entries(d.parameterTotals)) {
  if (typeof value !== 'number') continue;
  parameterRows.push(['summary', key, value, null, null, null, null, null, null, null, 'unknown', 'unknown', null, 'derived', 'hf-v3-config-json;hf-v3-modeling_deepseek-py', sources.get('hf-v3-modeling_deepseek-py').url, d.parameterTotals.scope]);
}
parameterRows.push(['mtp','num_nextn_predict_layers',d.mtp.declaredModules,null,null,null,null,null,null,null,'unknown','unknown',null,'official','hf-v3-config-json',sources.get('hf-v3-config-json').url,'MTP 不计入主干 61 层']);
parameterRows.push(['mtp','projection',null,null,'model.layers.61.eh_proj.weight',textValue(d.mtp.projectionLogicalShape),d.mtp.projectionLogicalParameters,1,d.mtp.projectionLogicalParameters,null,'unknown','unknown',null,'derived','v3-paper-v2;hf-v3-config-json',sources.get('v3-paper-v2').url+'#S2.SS2','论文公式21 + H；不是 checkpoint 文件头形状']);
parameterRows.push(['mtp','official_unique_approx',d.mtp.officialUniqueApprox,null,null,null,null,null,null,null,'unknown','unknown',null,'official','gh-deepseek-v3-readme_weights-md',sources.get('gh-deepseek-v3-readme_weights-md').url,'官方近似值；排除共享 embedding/head，不强行与 README 的 14B 配平']);
parameterRows.push(['mtp','exact_unique_parameters',null,'unknown',null,null,null,null,null,null,'unknown','unknown',null,'unknown','gh-deepseek-v3-readme_weights-md',sources.get('gh-deepseek-v3-readme_weights-md').url,'共享 norm 别名与实际张量 shape 未核验']);

const operatorHeaders = ['model_id','source_layer_0based','display_layer_1based','layer_kind','phase','attention_path','step_id','operation','input_shape','output_shape','input_shape_example','output_shape_example','dtype','math_relation','reference_url','reference_symbol','source_id','source_revision','evidence','backend','backend_evidence','duration','condition','note'];
const operatorRows = [];
const knownDims = Object.fromEntries(Object.entries(d.symbols).filter(([,v]) => v.value != null).map(([k,v]) => [k,v.value]));
// Dimension grammar contains only identifiers, integer arithmetic and parentheses, all from our research JSON.
// Numeric examples use an explicit parser rather than executing JavaScript expressions.
function dim(expression, env) {
  const tokens = String(expression).match(/[A-Za-z_][A-Za-z_0-9]*|\d+|\/\/|[()+*-]/g);
  if (!tokens || tokens.join('').replace(/\s/g,'') !== String(expression).replace(/\s/g,'')) throw new Error(`Invalid dimension: ${expression}`);
  let i = 0;
  const atom = () => {
    const token = tokens[i++];
    if (token === '(') { const x = sum(); if (tokens[i++] !== ')') throw new Error('Unclosed dimension'); return x; }
    if (/^\d+$/.test(token)) return Number(token);
    if (!(token in env)) throw new Error(`Unknown dimension ${token}`);
    return env[token];
  };
  const product = () => { let x = atom(); while (tokens[i] === '*' || tokens[i] === '//') { const op = tokens[i++]; const y = atom(); x = op === '*' ? x*y : Math.floor(x/y); } return x; };
  const sum = () => { let x = product(); while (tokens[i] === '+' || tokens[i] === '-') { const op = tokens[i++]; const y = product(); x = op === '+' ? x+y : x-y; } return x; };
  const result = sum();
  if (i !== tokens.length || !Number.isSafeInteger(result) || result < 0) throw new Error(`Invalid dimension result: ${expression}`);
  return result;
}
function shapeText(shapes, env = null) {
  return Object.entries(shapes).map(([name,shape]) => `${name}=[${shape.map(x => env ? dim(x,env) : x).join(',')}]`).join('; ');
}
for (const r of d.representatives) {
  for (const scenario of d.scenarios) {
    const env = {...knownDims, ...scenario, N_tok: scenario.B*scenario.L_q};
    for (const branch of ['naive','absorb']) {
      for (const s of d.steps.filter(s => ['both',r.kind].includes(s.layerKind) && ['both',branch].includes(s.branch))) {
        const p = d.proofs[s.proof_id];
        operatorRows.push([d.modelId,r.sourceLayer,r.displayLayer,r.kind,scenario.phase,branch,s.id,s.title,
          shapeText(s.inputs),shapeText(s.outputs),shapeText(s.inputs,env),shapeText(s.outputs,env),s.dtype,s.relation,
          p.url,p.symbol,p.source_id,sources.get(p.source_id).revision,s.evidence,'unknown',s.backendEvidence,null,
          `MP=1; B=${scenario.B}; L_q=${scenario.L_q}; L_kv=${scenario.L_kv}; start_pos=${scenario.start_pos}; N_e=${scenario.N_e} (动态代表例)`,s.note]);
      }
    }
  }
}
const tables = [
  {sheetName:'参数表',filename:'v3-parameters.csv',headers:parameterHeaders,rows:parameterRows},
  {sheetName:'代表层算子',filename:'v3-representative-operators.csv',headers:operatorHeaders,rows:operatorRows},
];
await fs.mkdir(out,{recursive:true});
for (const t of tables) {
  const sheet = wb.worksheets.add(t.sheetName);
  const values = [t.headers,...t.rows];
  const range = sheet.getRangeByIndexes(0,0,values.length,t.headers.length);
  range.values = values;
  range.format.font = {name:'Arial',size:10};
  sheet.showGridLines = false;
  range.format.columnWidth = 24;
  range.format.rowHeight = 20;
  sheet.getRangeByIndexes(0,0,1,t.headers.length).format = {fill:'#20344A',font:{name:'Arial',bold:true,color:'#FFFFFF'},rowHeight:28};
  sheet.freezePanes.freezeRows(1);
}
wb.recalculate();
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:5},maxChars:1000});
console.log(errors.ndjson);
for (const t of tables) {
  const sheet = wb.worksheets.getItem(t.sheetName);
  const values = sheet.getRangeByIndexes(0,0,t.rows.length+1,t.headers.length).values;
  if (JSON.stringify(values) !== JSON.stringify([t.headers,...t.rows])) throw new Error(`Artifact values changed: ${t.sheetName}`);
  const quote = value => {
    const s = value == null ? '' : String(value);
    return /[",\r\n]/.test(s) ? '"'+s.replaceAll('"','""')+'"' : s;
  };
  // Artifact Tool has no documented CSV export API in this runtime; serialize its verified range values.
  await fs.writeFile(path.join(out,t.filename),'\uFEFF'+values.map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n');
  const preview = await wb.render({sheetName:t.sheetName,range:'A1:H8',scale:1.5,format:'png'});
  await fs.writeFile(path.join(process.cwd(),t.filename+'.png'),new Uint8Array(await preview.arrayBuffer()));
  console.log(JSON.stringify({filename:t.filename,rows:t.rows.length,columns:t.headers.length}));
}
await fs.writeFile(path.join(out,'tables.json'),JSON.stringify({sourcePath:'data/families/deepseek/research/v3-round1.json',sourceSha256:createHash('sha256').update(sourceRaw).digest('hex'),tables},null,2)+'\n');
