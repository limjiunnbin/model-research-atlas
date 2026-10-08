"""Validate the shipped DeepSeek contracts, source pointers, whole-layer maps and storage audit."""
import argparse
import ast
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/families/deepseek'
def read(p):return json.loads(p.read_text())
def parameters(c):
 H=c['hidden_size'];N=c['num_attention_heads'];L=c['num_hidden_layers'];V=c['vocab_size'];kind=c['model_type']
 if kind in ('qwen2','llama'):
  D=H//N;NK=c['num_key_value_heads'];assert N%NK==0
  bias=H+2*NK*D if kind=='qwen2' else 2*H+2*NK*D if c.get('attention_bias') else 0
  layer=2*H*H+2*NK*D*H+bias+3*H*c['intermediate_size']+2*H
  return L*layer+V*H*(1 if c['tie_word_embeddings'] else 2)+H,None
 if kind!='deepseek_v4':
  Q=c['q_lora_rank'];R=c['kv_lora_rank'];Dn=c['qk_nope_head_dim'];Dr=c['qk_rope_head_dim'];Dv=c['v_head_dim'];E=c['n_routed_experts'];J=c['moe_intermediate_size']
  attn=H*Q+Q+N*(Dn+Dr)*Q+H*(R+Dr)+R+N*(Dn+Dv)*R+H*N*Dv
  if kind=='deepseek_v32':attn+=c['index_n_heads']*c['index_head_dim']*Q+c['index_head_dim']*H+c['index_n_heads']*H+2*c['index_head_dim']
  dense=sum(i<c['first_k_dense_replace'] or i%c.get('moe_layer_freq',1)!=0 for i in range(L))
  moe=L-dense;router=E*H+(0 if kind=='deepseek_v2' else E)
  main=2*V*H+H+L*(attn+2*H)+dense*3*H*c['intermediate_size']+moe*(router+(E+c['n_shared_experts'])*3*H*J)
  mtp=c.get('num_nextn_predict_layers',0)*(attn+2*H+router+(E+c['n_shared_experts'])*3*H*J+2*H*H+3*H)
  return main,{'mtp':mtp} if mtp else None
 D=c['head_dim'];Q=c['q_lora_rank'];G=c['o_groups'];O=c['o_lora_rank'];E=c['n_routed_experts'];J=c['moe_intermediate_size'];S=c['hc_mult'];M=S*(S+2)
 assert N*D%G==0 and len(c['compress_ratios'])==L+(3 if c.get('dspark_block_size') else c['num_nextn_predict_layers'])
 attn=N+H*Q+Q+N*D*Q+D*H+D+N*D*O+H*G*O
 hc=2*(M*S*H+M+3);experts=(E+1)*3*H*J
 def comp(r,d):return 2*(2 if r==4 else 1)*d*H+r*(2 if r==4 else 1)*d+d
 def layer(i,r):
  value=attn+2*H+hc+experts+E*H+(0 if i<c['num_hash_layers'] else E)
  if r:value+=comp(r,D)
  if r==4:value+=c['index_n_heads']*c['index_head_dim']*Q+c['index_n_heads']*H+comp(4,c['index_head_dim'])
  return value
 main=2*V*H+H+S*S*H+S+1+sum(layer(i,r) for i,r in enumerate(c['compress_ratios'][:L]))
 if c.get('dspark_block_size'):
  rank=c['dspark_markov_rank'];draft=sum(layer(L+i,r) for i,r in enumerate(c['compress_ratios'][L:]))+H*H*len(c['dspark_target_layer_ids'])+2*H+S*S*H+S+1+2*V*rank+H+rank
  return main,{'mtp':0,'dspark':draft}
 mtp=sum(layer(L+i,r)+2*H*H+3*H+S*S*H+S+1 for i,r in enumerate(c['compress_ratios'][L:]))
 return main,{'mtp':mtp}

def dimension(text,env):
 def visit(n):
  if isinstance(n,ast.Constant) and type(n.value)is int:return n.value
  if isinstance(n,ast.Name):return env[n.id]
  if isinstance(n,ast.BinOp):
   a,b=visit(n.left),visit(n.right)
   if isinstance(n.op,ast.Add):return a+b
   if isinstance(n.op,ast.Mult):return a*b
   if isinstance(n.op,ast.FloorDiv):return a//b
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='min':return min(visit(a) for a in n.args)
  raise ValueError(text)
 result=visit(ast.parse(str(text),mode='eval').body);assert isinstance(result,int) and result>=0;return result

def validate(check_downloads=True):
 family=read(DATA/'family.json');compute=read(DATA/'compute.json');hardware=read(DATA/'hardware.json');evidence=read(DATA/'research/site-proofs.json');audit=read(DATA/'research/v3-header-audit.json');checks=collections.Counter()
 assert family['id']==compute['familyId']==hardware['familyId']=='deepseek'
 assert family['acceptanceScope']['deviceExperimentsRequired'] is False
 assert family['acceptanceScope']['deviceExperiments']=='excluded_from_current_scope_by_user'
 assert hardware['experimentScope']['required'] is False
 assert all(o['status']=='未来可选 · 不纳入本次验收' for o in hardware['optimizations'])
 assert compute['proofs']==evidence['proofs'] and compute['counts']['proofs']==len(evidence['proofs'])
 for proof in evidence['proofs'].values():
  source=evidence['sources'][proof['source_id']]
  assert proof['sourceSha256']==source['sha256'] and proof['revision']==source['revision'] and proof['path']==source['path']
  assert 0<proof['line']<=proof['end'] and len(proof['functionSha256'])==64
 checks['sourceFiles']=len(evidence['sources']);checks['functionProofs']=len(evidence['proofs'])
 ids={m['id'] for m in family['models']};assert len(ids)==21
 sections={s['id'] for s in read(DATA/'report.json')};assert all(set(m['sections'])<=sections for m in family['models'])
 for m in family['models']:
  c=read(ROOT/m['configPath']);a=read(ROOT/m['architecturePath']);expected,mtp=parameters(c)
  assert a['modelId']==m['id'] and a['mainLogicalParameters']==expected==m['facts']['parameters']['value'],m['id']
  language=[n for n in a['nodes'] if n['group']=='decoder'];assert [n['number'] for n in language]==list(range(1,c['num_hidden_layers']+1))
  assert len({n['id'] for n in a['nodes']})==len(a['nodes'])
  assert sum(n['parameters'] for n in a['nodes'] if n['group']not in ('mtp','dspark'))==expected
  if mtp is not None:
   assert a['mtpLogicalParameters']==mtp['mtp']
   if 'dspark'in mtp:assert a['dsparkLogicalParameters']==mtp['dspark'] and len([n for n in a['nodes'] if n['group']=='dspark'])==3
  if c['model_type']=='deepseek_v4':
   assert [n['compressRatio'] for n in language]==c['compress_ratios'][:c['num_hidden_layers']]
   assert sum(n['routing']=='hash' for n in language)==c['num_hash_layers']
   assert all(n['compressRatio']==0 for n in a['nodes'] if n['group'] in ('mtp','dspark'))
  header=read(ROOT/a['storageAuditPath']);assert header['revision']==m['revision'] and header['weightPayloadTransferredBytes']==0
  assert sum(n['payloadBytes'] for n in a['nodes'])==header['payloadBytes'] and header['payloadBytes']+header['metadataCapturedBytes']==header['fileBytes']
  hrows={(r['group'],r['sourceIndex'],r['tensorTemplate']):r for r in header['templates']};covered=set()
  for n in a['nodes']:
   hgroup='decoder' if n['group']=='decoder' else 'auxiliary' if c['model_type']=='deepseek_v4' and n['group'] in ('mtp','dspark') else 'mtp' if n['group']=='mtp' else 'global';source_index=n.get('sourceIndex',n['number']-1 if n['group']=='decoder' else None)
   assert sum(mod['parameters'] for mod in n['modules'])==n['parameters']
   for mod in n['modules']:
    assert mod['parameters']==sum(x['logical_parameters_each']*x['count'] for x in mod['matrices'])
    if mod['representative']:assert mod['count']==c['n_routed_experts'] and mod['selectedCount']==c['num_experts_per_tok']
    for x in mod['matrices']:
     assert x['logical_parameters_each']==(math.prod(x['logical_shape']) if x['role']=='weight' else 0)
     if x.get('proof'):assert x['proof'] in evidence['proofs']
     assert x['stored_shape'] is not None and x['stored_dtype'] is not None and x['payloadBytes'] is not None
     if x['role']=='weight':assert x['logical_shape']==x['stored_shape'] or x.get('storageLayout') and x['stored_dtype']=='I8' and x['logical_shape']==[x['stored_shape'][0],x['stored_shape'][1]*2]
     key=(hgroup,source_index,x['tensor_template']);r=hrows[key];covered.add(key)
     assert (x['stored_shape'],x['stored_dtype'],x['payloadBytes'],x['count'])==(r['storedShape'],r['storedDtype'],r['payloadBytesEach'],r['copies'])
     checks['matrixInstances']+=1
  assert covered==set(hrows),m['id']
  checks['auditedTensors']+=header['tensorCount'];checks['headerMetadataBytes']+=header['metadataCapturedBytes']
  assert all(v['value'] is not None or v['evidence']=='unknown' for v in m['facts'].values())
  checks['models']+=1;checks['languageLayers']+=len(language)
 assert audit['tensorCount']==91991 and len(audit['shards'])==163 and audit['mainParameterElements']==671026419200
 assert audit['weightPayloadTransferredBytes']==0 and audit['payloadBytes']+audit['metadataTransferredBytes']==audit['fileBytes']
 assert sum(r['tensorCount'] for r in audit['layers'].values())==audit['tensorCount']
 assert sum(r['payloadBytes'] for r in audit['layers'].values())==audit['payloadBytes']
 assert len(compute['models'])==21 and {m['id'] for m in compute['models']}==ids
 total=0
 for model in compute['models']:
  a=read(DATA/(model['id']+'-architecture.json'));nodes={n['id']:n for n in a['nodes']}
  assert {n['id'] for n in model['nodes']}==set(nodes)
  for n in model['nodes']:
   assert n['parameters']==nodes[n['id']]['parameters']
   t=compute['templates'][n['template']];assert t['modelId']==model['id']
   original={x['tensor_template']:(x['logical_shape'],x['count']) for mod in nodes[n['id']]['modules'] for x in mod['matrices']}
   for phase in ('prefill','decode'):
    observed={x['tensor_template']:([int(i) for i in x['logical_shape'].split('x')],x['multiplicity']) for s in t['steps'] if s['phase']==phase for x in s['tensors']}
    assert observed==original,(model['id'],n['id'],phase)
   for s in t['steps']:
    assert set(s['apis'])=={'torch','nvidia','amd','ascend'} and s['phase'] in ('prefill','decode')
    assert all(pid in compute['proofs'] for pid in s['reference']['proofs'])
    for api in s['apis'].values():assert all(pid in compute['proofs'] for pid in api['proofs'])
    env={'B':1,'L_q':128 if s['phase']=='prefill' else 1,'L_kv':128 if s['phase']=='prefill' else 256,'N_tok':128 if s['phase']=='prefill' else 1,'N_e':1}
    inputs=[tuple(dimension(v,env) for v in shape) for shape in s['inputDimensions'].values()];outputs=[tuple(dimension(v,env) for v in shape) for shape in s['outputDimensions'].values()]
    if s['op']=='linear':assert inputs[0][-1]==inputs[1][1] and outputs[0]==inputs[0][:-1]+(inputs[1][0],)
    elif s['op']=='identity':assert inputs==outputs
    elif s['op']=='elementwise':assert all(x==outputs[0] for x in inputs)
    elif s['op']=='einsum':
     from validate_deepseek import infer_einsum
     assert outputs[0]==infer_einsum(s['einsum'],inputs)
    elif s['op']=='hc_pre':assert inputs[0][:-2]+(inputs[0][-1],)==outputs[0] and outputs[1]==inputs[0][:-1] and outputs[2]==inputs[0][:-1]+(inputs[0][-2],)
    elif s['op']=='hc_post':assert inputs[1]==outputs[0] and inputs[0]==outputs[0][:-2]+(outputs[0][-1],)
    checks['expandedStepShapes']+=1
   total+=len(t['steps'])
 assert total==compute['counts']['steps']
 assert compute['counts']['languageLayers']==checks['languageLayers']
 source_ids={s['id'] for s in hardware['sources']};topics={m['id']:m for m in hardware['modules']}
 for item in hardware['modules']+hardware['platforms']+hardware['optimizations']:
  assert item['refs'] and all(r['id'] in source_ids for r in item['refs'])
 for m in family['models']:
  for link in m['implementationLinks']:assert m['id'] in topics[link['topic']]['models'] and link['path'].endswith('/'+m['id'])
  if 'distill'in m['id']:assert all(l['topic']!='moe' for l in m['implementationLinks'])
 if check_downloads:
  for download in family['downloads']:
   raw=(ROOT/'dist'/download['path']).read_bytes();assert len(raw)==download['bytes'] and hashlib.sha256(raw).hexdigest()==download['sha256'],download['path'];checks['downloads']+=1
  from validate_deepseek_exports import validate as validate_exports
  exports=validate_exports()
 else:exports=None
 for fig in family['figures']:
  import xml.etree.ElementTree as ET
  tree=ET.fromstring((ROOT/'dist'/fig['path']).read_bytes());assert tree.tag=='{http://www.w3.org/2000/svg}svg'
  assert tree.find('{http://www.w3.org/2000/svg}title') is not None and tree.find('{http://www.w3.org/2000/svg}desc') is not None
  checks['figures']+=1
 cann=read(DATA/'research/cann-operator-audit.json')
 assert cann['tag']=='v9.2.0-beta.2' and cann['package']['version']=='9.2.0-beta.2' and cann['package']['operatorExecution'] is False
 assert len(cann['operators'])==cann['counts']['operatorFamilies']==30
 assert len(cann['codeRanges'])==cann['counts']['codeRanges']
 for r in cann['codeRanges']:
  source=evidence['sources'][r['sourceId']];assert (r['repo'],r['revision'],r['sourceSha256'])==(source['repo'],source['revision'],source['sha256'])
  assert r['url']==source['url']+f'#L{r["line"]}-L{r["end"]}' and 0<r['line']<=r['end']
 for op in cann['operators']:
  assert op['references'] and all(r['sourceId'] in evidence['sources'] for r in op['references'])
 checks['cannOperatorFamilies']=30;checks['cannSourceRanges']=len(cann['codeRanges'])
 result={'status':'passed','date':'2026-10-01','scope':'21 checkpoint 全文件头/矩阵、独立 MTP/DSpark、CANN 固定来源/范围、shape、引用与导出','checks':dict(checks),'exports':exports,'downloadsChecked':check_downloads,'computeSteps':total,'computeTemplates':len(compute['templates']),'deviceExperiments':'excluded_from_current_scope_by_user','limits':['没有执行权重值加载或设备实验','Llama 原底座 gated，差异保留未知','header 不证明上游训练共享别名或权重值相同','CANN 相关源码/包内资料可查；实际库/分派、ABI、编译与设备数值仍未由静态研究证明']}
 write_json=__import__('collect_deepseek_round1').write_json
 write_json(DATA/'research/atlas-validation.json',result);print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-downloads',action='store_true');validate(not p.parse_args().skip_downloads)
