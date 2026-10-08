import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import {exportTables} from './analysis_export_data.mjs';
const require=createRequire(path.join(process.cwd(),'artifact-loader.cjs'));
const {Workbook,SpreadsheetFile,FileBlob}=await import(require.resolve('@oai/artifact-tool'));
const root=fileURLToPath(new URL('../',import.meta.url)),data=path.join(root,'data/families/deepseek'),out=path.join(root,'dist/assets/deepseek');
const load=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const d=await load(path.join(data,'analysis.json')),scenarios=await load(path.join(data,'research/analysis-scenarios.json')),compare=await load(path.join(root,'data/analysis/cross-family.json'));
const comparisonOnly=process.argv.includes('--comparison-only'),previewOnly=process.argv.includes('--preview-comparison');
const definitions=exportTables(d,scenarios,compare),wb=comparisonOnly||previewOnly?await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(out,'DeepSeek-静态分析.xlsx'))):Workbook.create();
if(previewOnly){
  console.log((await wb.inspect({kind:'table',range:'跨家族比较!A43:G47',include:'values,formulas',tableMaxRows:5,tableMaxCols:7,maxChars:1500})).ndjson);
  const preview=await wb.render({sheetName:'跨家族比较',range:'A43:G47',scale:1.2,format:'png'});
  await fs.writeFile(path.join(process.cwd(),'analysis-comparison-before.png'),new Uint8Array(await preview.arrayBuffer()));
  process.exit(0);
}
if(comparisonOnly){
  const def=definitions.find(s=>s.name==='跨家族比较'),sheet=wb.worksheets.getItem(def.name);
  const manifest=await load(path.join(data,'research/analysis-export-manifest.json'));
  const prior=manifest.sheets.find(s=>s.name===def.name).rows;
  if(def.rows.length<prior)throw Error('Comparison append mode cannot remove existing rows');
  const existing=sheet.getRangeByIndexes(5,0,prior,def.headers.length).values;
  for(let i=0;i<prior;i++){
    if(JSON.stringify(existing[i])===JSON.stringify(def.rows[i]))continue;
    const model=compare.models[i];
    if(model.familyId!==d.familyId||!d.excludedModels?.some(m=>m.id===model.id))throw Error('Existing audited comparison rows changed; preserving the baseline');
    sheet.getRangeByIndexes(5+i,0,1,def.headers.length).values=[def.rows[i]];
  }
  for(let i=prior;i<def.rows.length;i++){
    const row=sheet.getRangeByIndexes(5+i,0,1,def.headers.length);
    row.copyFrom(sheet.getRangeByIndexes(4+prior,0,1,def.headers.length),'all');
    row.values=[def.rows[i]];row.format.autofitRows();
  }
  for(let i=0;i<def.rows.length;i++){
    const model=compare.models[i];
    if(model.familyId!==d.familyId||!d.excludedModels?.some(m=>m.id===model.id))continue;
    const row=sheet.getRangeByIndexes(5+i,0,1,def.headers.length);
    row.format.font={name:'Arial',size:10};row.format.wrapText=true;row.format.verticalAlignment='center';
    for(let c=0;c<def.headers.length;c++)if(typeof def.rows[i][c]==='number')sheet.getCell(5+i,c).format.numberFormat='#,##0';
    row.format.autofitRows();
  }
}
const renderSheets=process.env.ANALYSIS_RENDER_SHEETS?new Set(process.env.ANALYSIS_RENDER_SHEETS.split(',')):null;
await fs.mkdir(out,{recursive:true});
for(const def of definitions){
  assertGrid(def);
  if(comparisonOnly)continue;
  const sheet=wb.worksheets.add(def.name);sheet.showGridLines=false;
  sheet.getRange('A2').values=[[def.title]];sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true};
  sheet.getRange('A3').values=[[def.note]];sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#54647B'};
  const grid=sheet.getRangeByIndexes(4,0,def.rows.length+1,def.headers.length);grid.values=[def.headers,...def.rows];
  grid.format.font={name:'Arial',size:10};grid.format.verticalAlignment='center';grid.format.wrapText=true;grid.format.rowHeight=38;
  for(let c=0;c<def.headers.length;c++){
    const col=sheet.getRangeByIndexes(4,c,def.rows.length+1,1);col.format.columnWidth=def.widths[c];
    if(def.rows.some(row=>typeof row[c]==='number'))sheet.getRangeByIndexes(5,c,def.rows.length,1).format.numberFormat=def.name==='并行场景'&&c===11?'0.0%':def.name==='成本场景'&&[9,10,11].includes(c)?'0.0':'#,##0';
  }
  const header=sheet.getRangeByIndexes(4,0,1,def.headers.length);header.format={fill:'#20344A',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:38,wrapText:true};
  grid.format.autofitRows();sheet.freezePanes.freezeRows(5);sheet.freezePanes.freezeColumns(1);
}
function assertGrid(def){for(const row of def.rows){if(row.length!==def.headers.length)throw Error('Column mismatch '+def.name);for(const v of row){if(typeof v==='number'&&!Number.isFinite(v))throw Error('Non-finite '+def.name);if(typeof v==='string'&&v.length>32767)throw Error('Excel cell length '+def.name);}}}
wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:10},maxChars:1200});console.log(errors.ndjson);
const quality=[];
const quote=v=>{const s=v==null?'':String(v);return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
for(const def of definitions){
  const sheet=wb.worksheets.getItem(def.name),actual=sheet.getRangeByIndexes(5,0,def.rows.length,def.headers.length).values;
  if(JSON.stringify(actual)!==JSON.stringify(def.rows))throw Error('Range value mismatch '+def.name);
  if(def.file&&(!comparisonOnly||def.name==='跨家族比较'))await fs.writeFile(path.join(out,def.file),'\uFEFF'+[def.headers,...actual].map(row=>row.map(quote).join(',')).join('\r\n')+'\r\n');
  const ranges=comparisonOnly?(def.name==='跨家族比较'?[`A${def.rows.length+2}:G${def.rows.length+5}`,`J${def.rows.length+5}:M${def.rows.length+5}`]:[]):def.name==='成本场景'?['A1:B9','D5:K9','M5:Q9','S5:V9']:def.name==='并行场景'?['A1:E9','N5:R9']:def.name==='算子条件'?['A1:D9','E5:F9']:['A1:D9'];
  for(let i=0;i<ranges.length&&(!renderSheets||renderSheets.has(def.name));i++){
    const preview=await wb.render({sheetName:def.name,range:ranges[i],scale:1.2,format:'png'});
    await fs.writeFile(path.join(process.cwd(),`analysis-${def.name}-${i}.png`),new Uint8Array(await preview.arrayBuffer()));
  }
  quality.push({name:def.name,rows:def.rows.length,columns:def.headers.length,cells:def.rows.length*def.headers.length,csv:def.file??null});
}
const inspect=await wb.inspect({kind:'table',range:comparisonOnly?`跨家族比较!A${compare.models.length+5}:G${compare.models.length+5}`:'成本场景!M5:Q8',include:'values,formulas',tableMaxRows:4,tableMaxCols:7,maxChars:1500});console.log(inspect.ndjson);
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'DeepSeek-静态分析.xlsx'));
await fs.rm(path.join(out,'DeepSeek-静态分析.xlsx.inspect.ndjson'),{force:true});
await fs.writeFile(path.join(data,'research/analysis-export-manifest.json'),JSON.stringify({schemaVersion:1,snapshot:comparisonOnly?compare.snapshot:d.snapshot,path:'assets/deepseek/DeepSeek-静态分析.xlsx',
  analysisSha256:createHash('sha256').update(await fs.readFile(path.join(data,'analysis.json'))).digest('hex'),sheets:quality,
  dataCells:quality.reduce((n,s)=>n+s.cells,0),scope:'Fixed analytical scenarios and source rows; numbers typed; unknown measurements blank; no hardware validation'},null,2)+'\n');
console.log(JSON.stringify({sheets:quality,cells:quality.reduce((n,s)=>n+s.cells,0)}));
