// Independent enumerated shape/position references and synthetic algebra checks.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {costEstimate, parallelEstimate, mechanism, checkContract, sumFloor, sumCappedFloor, sumWindow, softmax} from '../dist/analysis-math.mjs';
const root=fileURLToPath(new URL('../',import.meta.url));
const load=async file=>JSON.parse(await fs.readFile(path.join(root,file),'utf8'));
const data=await load('data/families/deepseek/analysis.json');
const close=(a,b,label)=>assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-11*Math.max(1,Math.abs(a),Math.abs(b)),`${label}: ${a} != ${b}`);
let checks=0,costCases=0;
for(const r of [1,4,8,128])for(const n of [0,1,3,4,7,8,127,128,129,255]){
 const positions=Array.from({length:n},(_,i)=>i+1);
 close(sumFloor(n,r),positions.reduce((s,t)=>s+Math.floor(t/r),0),'floor enumeration');
 for(const cap of [0,1,3,128])close(sumCappedFloor(n,r,cap),positions.reduce((s,t)=>s+Math.min(cap,Math.floor(t/r)),0),'capped floor enumeration');
 close(sumWindow(n,r),positions.reduce((s,t)=>s+Math.min(r,t),0),'window enumeration');checks+=6;
}
const scenarios=[];
for(const model of data.costModels){
 const arch=await load(model.architecturePath),c=model.config,nodes=arch.nodes.filter(n=>n.group==='decoder');
 const H=c.hidden_size,NH=c.num_attention_heads;
 for(const B of [1,3])for(const Q of [1,5,128])for(const P of [0,7,127,4095])for(const pathMode of model.kind==='mla'||model.kind==='dsa'?['latent','expanded']:['latent']){
  const input={batch:B,query:Q,history:P,phase:Q===1?'decode':'prefill',path:pathMode,logits:'all',cacheBytes:2,activationBytes:2,weightBytes:2,layer:'all'};
  const actual=costEstimate(model,input);assert.equal(actual.valid,true,model.id);
  let projections=0,core=0,useful=0,index=0,cache=0,idxCache=0,states=0;
  for(const node of nodes){
   for(const mod of node.modules)for(const matrix of mod.matrices){
    if(matrix.role!=='weight'||matrix.logical_shape.length!==2||matrix.tensor_template.includes('.ape'))continue;
    const calls=mod.representative?mod.selectedCount:matrix.count;
    projections+=2*B*Q*matrix.logical_shape[0]*matrix.logical_shape[1]*calls;
   }
   let layerUsefulPairs=0;
   for(let j=1;j<=Q;j++){
    const position=P+j;
    if(model.kind==='v4'){
     const ratio=node.compressRatio;
     layerUsefulPairs+=Math.min(position,c.sliding_window)+(ratio?Math.min(Math.floor(position/ratio),ratio===4?c.index_topk:Infinity):0);
    }else layerUsefulPairs+=model.kind==='dsa'?Math.min(position,c.index_topk):position;
   }
   if(model.kind==='v4'){
    const ratio=node.compressRatio,C=ratio?Math.floor((P+Q)/ratio):0;
    const slots=c.sliding_window+(ratio?Math.min(C,ratio===4?c.index_topk:Infinity):0);
    core+=4*B*NH*c.head_dim*Q*slots;useful+=4*B*NH*c.head_dim*layerUsefulPairs;
    cache+=B*(c.sliding_window+C)*c.head_dim*2;
    if(ratio){
     const overlap=ratio===4?2:1;
     states+=2*B*overlap*ratio*overlap*c.head_dim*4;
     if(ratio===4){index+=2*B*Q*C*c.index_n_heads*c.index_head_dim;idxCache+=B*C*c.index_head_dim*2;states+=2*B*overlap*ratio*overlap*c.index_head_dim*4;}
    }
   }else if(model.kind==='gqa'){
    const D=c.head_dim??H/NH;
    core+=4*B*NH*D*Q*(P+Q);useful+=4*B*NH*D*layerUsefulPairs;
    cache+=B*(P+Q)*2*c.num_key_value_heads*D*2;
   }else{
    const latent=pathMode==='latent',R=c.kv_lora_rank,DN=c.qk_nope_head_dim,DR=c.qk_rope_head_dim,DV=c.v_head_dim;
    const width=latent?2*R+DR:DN+DR+DV;
    core+=2*B*NH*Q*(P+Q)*width;useful+=2*B*NH*layerUsefulPairs*width;
    cache+=B*(P+Q)*(latent?R+DR:NH*(DN+DR+DV))*2;
    if(model.kind==='dsa'){index+=2*B*Q*(P+Q)*c.index_n_heads*c.index_head_dim;idxCache+=B*(P+Q)*(c.index_head_dim+4);}
   }
  }
  for(const [field,expected]of Object.entries({projectionFlops:projections,coreFlops:core,usefulCoreFlops:useful,indexFlops:index,mainKvBytes:cache,indexCacheBytes:idxCache,compressionStateBytes:states}))close(actual[field],expected,model.id+' '+field);
  assert.ok(actual.usefulCoreFlops<=actual.coreFlops);
  close(actual.headFlops,2*B*Q*c.vocab_size*H,'all logits');
  assert.equal(actual.measuredLatency,null);assert.equal(actual.measuredPeakMemory,null);assert.equal(actual.measuredThroughput,null);
  const local=costEstimate(model,{...input,layer:nodes.at(-1).id});assert.equal(local.rows.reduce((n,r)=>n+r.layers,0),1);assert.equal(local.headFlops,0);
  costCases++;
 }
 for(const phase of ['prefill','decode','verify'])for(const pathMode of model.kind==='mla'||model.kind==='dsa'?['latent','expanded']:['latent']){
  const input={batch:1,query:phase==='prefill'?512:phase==='decode'?1:5,history:phase==='prefill'?0:4095,phase,path:pathMode,logits:phase==='verify'?'all':model.defaultLogits,cacheBytes:2,activationBytes:2,weightBytes:2,layer:'all'};
  const result=costEstimate(model,input);assert.ok(result.valid);scenarios.push({name:model.name,modelId:model.id,input,result});
 }
 for(const bad of [{batch:0},{query:0},{history:-1},{query:model.context,history:1},{cacheBytes:0.5},{phase:'decode',query:5},{phase:'verify',logits:'last'}]){
  assert.equal(costEstimate(model,{batch:1,query:1,history:0,phase:'decode',path:'latent',logits:'all',cacheBytes:2,activationBytes:2,weightBytes:2,layer:'all',...bad}).valid,false);checks++;
 }
}
const parallelScenarios=[];
for(const model of data.costModels)for(const [tp,dp,policy]of [[1,1,'replicated'],[1,8,'replicated'],[2,4,'uniform-tp']]){
 const input={tp,dp,policy,batch:2,query:1,history:4095,weightBytes:2,activationBytes:2,cacheBytes:2,redundantExperts:0,remoteFraction:'uniform',allReduces:1};
 const result=parallelEstimate(model,input);assert.ok(result.valid,JSON.stringify(result.errors));
 const s=model.parallelElements;const E=model.config.n_routed_experts??0,K=model.config.num_experts_per_tok??model.config.n_activated_experts??0;
 const ep=E?tp*dp:1,remote=E?(ep-1)/ep:0;
 close(result.logicalParametersPerRank,s.replicated+s.nonExpertLinears/(policy==='uniform-tp'?tp:1)+s.routedExperts/ep,'per-rank weights');
 const moeLayers=model.groups.filter(g=>g.ffn==='MoE').reduce((n,g)=>n+g.count,0);
 // Explicit enumerate logical routes; sum payload only when the destination is remote.
 const destinations=E?Array.from({length:ep},(_,i)=>i):[0];
 const meanRemote=destinations.filter(x=>x!==0).length/destinations.length;
 close(result.moeSendBytes,2*2*dp*K*model.config.hidden_size*2*meanRemote*moeLayers,'dispatch + combine');
 const message=2*model.config.hidden_size*2;
 const ringSendPerRank=tp===1?0:2*(tp-1)*message/tp*model.config.num_hidden_layers;
 close(result.tpRingSendBytesPerRank,ringSendPerRank,'ring per rank');close(result.tpRingSendBytes,ringSendPerRank*tp*dp,'ring all ranks');
 assert.equal(result.measuredCommunicationTime,null);parallelScenarios.push({name:model.name,modelId:model.id,input,result});checks+=5;
}
for(const rank of [2,3,4]){const m=mechanism('mla',{rank});assert.ok(m.maxError<1e-12);for(let i=0;i<4;i++)close(m.expandedScores[i],m.latentScores[i],'MLA score identity');checks+=5;}
for(let visible=1;visible<=8;visible++)for(let topK=1;topK<=8;topK++){const d=mechanism('dsa',{visible,topK});assert.ok(d.selected.every(i=>i<visible));assert.equal(d.selected.length,Math.min(visible,topK));close(d.probability.reduce((a,b)=>a+b,0),1,'DSA probability');checks+=3;}
for(const ratio of [4,8,128])for(let tokens=1;tokens<=24;tokens++){const c=mechanism('compression',{ratio,tokens});assert.equal(c.compressed.length,Math.floor(tokens/ratio));assert.equal(c.remainder.length,tokens%ratio);for(const r of c.compressed)close(r.weights.reduce((a,b)=>a+b,0),1,'compression gating');checks+=2;}
const mhc=mechanism('mhc',{iterations:40});for(const sum of [...mhc.rowSums,...mhc.columnSums])close(sum,1,'Sinkhorn');assert.equal(mhc.output.length,4);checks+=9;
for(let block=2;block<=6;block++)for(const threshold of [0.1,0.6,0.95]){
 const d=mechanism('dspark',{block,threshold});let expected=1;
 for(let i=0;i<block;i++){expected*=d.confidences[i];close(d.prefixSurvival[i],expected,'conditional prefix product');}
 assert.equal(d.committed.length,d.accepted+1);assert.deepEqual(d.committed.slice(0,d.accepted),d.drafts.slice(0,d.accepted));
 if(d.rejection){close(d.rejection.correctionDistribution.reduce((a,b)=>a+b,0),1,'correction mass');assert.equal(d.verification.at(-1).accepted,false);}
 checks+=4;
}
// Rejection sampler distribution identity for diverse normalized target/draft pairs.
for(let seed=1;seed<=100;seed++){
 const p=softmax(Array.from({length:5},(_,i)=>Math.sin(seed+i))),q=softmax(Array.from({length:5},(_,i)=>Math.cos(seed*0.7+i)));
 const accepted=q.map((v,i)=>Math.min(v,p[i])),rejected=1-accepted.reduce((a,b)=>a+b,0),residual=p.map((v,i)=>Math.max(0,v-q[i])),mass=residual.reduce((a,b)=>a+b,0);
 for(let i=0;i<5;i++)close(accepted[i]+rejected*residual[i]/mass,p[i],'rejection distribution identity');checks+=5;
}
const rms=data.cannContracts.find(c=>c.id==='rms-norm'),swiglu=data.cannContracts.find(c=>c.id==='swiglu'),mm=data.cannContracts.find(c=>c.id==='matmul');
assert.equal(checkContract(rms,{x:[2,128],gamma:[128],dtype:'BFLOAT16',layout:'ND'}).status,'incomplete_static_check');
for(const input of [{x:[2,128],gamma:[64],dtype:'BFLOAT16',layout:'ND'},{x:[2,0],gamma:[0],dtype:'BFLOAT16',layout:'ND'},{x:[2,128],gamma:[128],dtype:'INT8',layout:'ND'}])assert.equal(checkContract(rms,input).status,'contradiction');
assert.equal(checkContract(swiglu,{x:[2,127],dim:-1,dtype:'BFLOAT16',layout:'ND'}).status,'contradiction');
assert.equal(checkContract(swiglu,{x:[2,128],dim:2,dtype:'BFLOAT16',layout:'ND'}).status,'contradiction');
assert.equal(checkContract(mm,{x:[2,128],w:[64,256]}).status,'contradiction');
assert.equal(checkContract(mm,{x:[2,4,128],w:[2,128,256]}).results[0].status,'unknown');
checks+=8;
await fs.writeFile(path.join(root,'data/families/deepseek/research/analysis-scenarios.json'),JSON.stringify({snapshot:data.snapshot,costScenarios:scenarios,parallelScenarios,scope:'Explicit static assumptions; no runtime measurements'},null,2)+'\n');
const record={status:'passed',date:data.snapshot,costCases,independentChecks:checks,costExportScenarios:scenarios.length,parallelExportScenarios:parallelScenarios.length,
 scope:'Canonical matrix enumeration, explicit causal/slot position sums, analytical collectives and synthetic algebra/probabilities; no hardware experiment'};
await fs.writeFile(path.join(root,'data/families/deepseek/research/analysis-math-validation.json'),JSON.stringify(record,null,2)+'\n');
console.log(JSON.stringify(record));
