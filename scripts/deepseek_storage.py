"""Bind each logical matrix and auxiliary module to its own complete fixed header audit."""
import copy
import json
import math


def complete_storage(e,name,c,arch):
 from build_deepseek_atlas import DATA,node,module,matrix
 audit_path=f'data/families/deepseek/research/header-audits/{name}.json'
 audit=json.loads((DATA/'research/header-audits'/f'{name}.json').read_text());assert audit['revision']==next(s['revision'] for s in e.sources.values() if s.get('modelId')=='deepseek-ai/'+name and s.get('path')=='config.json')
 native=c['model_type']=='deepseek_v4';dsa=c['model_type']=='deepseek_v32';L=c['num_hidden_layers'];H=c['hidden_size']
 # MTP shares the main embedding/output matrix by the documented mathematical scope.
 # The selected vLLM reference constructs SharedHead.norm separately: this is a reference-count assumption, not a raw weight-value comparison.
 if not native and c.get('num_nextn_predict_layers'):
  prototype=next(n for n in reversed(arch['nodes']) if n['group']=='decoder')
  n=copy.deepcopy(prototype);n.update(id='mtp-1',group='mtp',number=0,type='MTP / DSA' if dsa else 'MTP / MLA',sourceIndex=L,implementationTopic='mtp')
  sid=next(sid for sid,s in e.sources.items() if s.get('repo')=='vllm-project/vllm' and s.get('path')=='vllm/model_executor/models/deepseek_mtp.py')
  extra=[matrix(e,sid,'DeepSeekMultiTokenPredictorLayer.__init__','model.layers.{i}.'+p,d) for p,d in [('enorm.weight',[H]),('hnorm.weight',[H]),('eh_proj.weight',[H,2*H])]]
  extra.append(matrix(e,sid,'SharedHead.__init__','model.layers.{i}.shared_head.norm.weight',[H]))
  n['modules'].append(module('mtp-input','MTP 输入融合与输出 norm',extra,'mtp'))
  aliases=[matrix(e,sid,'DeepSeekMultiTokenPredictor.__init__','model.layers.{i}.embed_tokens.weight',[c['vocab_size'],H],role='shared_alias'),matrix(e,sid,'SharedHead.__init__','model.layers.{i}.shared_head.head.weight',[c['vocab_size'],H],role='shared_alias')]
  n['modules'].append(module('shared-aliases','文档共享 embedding/head 的存储副本',aliases,'mtp'))
  n['parameters']=sum(m['parameters'] for m in n['modules']);n['activeLinearParameters']=sum(x['logical_parameters_each']*(m['selectedCount'] if m['representative'] else x['count']) for m in n['modules'] for x in m['matrices'] if x['role']=='weight' and len(x['logical_shape'])==2)
  n['parameterScope']='参考口径：共享 embedding/head 存储副本排除；选定 vLLM 的 SharedHead.norm 独立构造和加载。上游训练别名/权重数值相同与否未由 header 证明。'
  arch['nodes'].append(n);arch['mtpLogicalParameters']=n['parameters'];arch['mtpReferenceSource']=e.sources[sid]['url']
 def stored_name(name):
  if not dsa:return name
  if name in {'embed.weight','head.weight','norm.weight'}:return {'embed.weight':'model.embed_tokens.weight','head.weight':'lm_head.weight','norm.weight':'model.norm.weight'}[name]
  replacements=[('layers.{i}.attn.','model.layers.{i}.self_attn.'),('layers.{i}.ffn.','model.layers.{i}.mlp.'),('layers.{i}.attn_norm.','model.layers.{i}.input_layernorm.'),('layers.{i}.ffn_norm.','model.layers.{i}.post_attention_layernorm.'),('.shared_experts.w1.','.shared_experts.gate_proj.'),('.shared_experts.w3.','.shared_experts.up_proj.'),('.shared_experts.w2.','.shared_experts.down_proj.'),('.experts.{e}.w1.','.experts.{e}.gate_proj.'),('.experts.{e}.w3.','.experts.{e}.up_proj.'),('.experts.{e}.w2.','.experts.{e}.down_proj.'),('.mlp.w1.','.mlp.gate_proj.'),('.mlp.w3.','.mlp.up_proj.'),('.mlp.w2.','.mlp.down_proj.'),('.gate.bias','.gate.e_score_correction_bias')]
  for a,b in replacements:name=name.replace(a,b)
  if '.indexer.'not in name:
   for a,b in [('self_attn.wq_a.','self_attn.q_a_proj.'),('self_attn.q_norm.','self_attn.q_a_layernorm.'),('self_attn.wq_b.','self_attn.q_b_proj.'),('self_attn.wkv_a.','self_attn.kv_a_proj_with_mqa.'),('self_attn.kv_norm.','self_attn.kv_a_layernorm.'),('self_attn.wkv_b.','self_attn.kv_b_proj.'),('self_attn.wo.','self_attn.o_proj.')]:name=name.replace(a,b)
  return name
 rows={(r['group'],r['sourceIndex'],r['tensorTemplate']):r for r in audit['templates']};used=set()
 if native:
  sid=e.select(name.replace('-Base',''),'inference/model.py')
  for n in arch['nodes']:
   if n['group']not in ('mtp','dspark'):continue
   alias=[]
   for template in ('mtp.{i}.emb.tok_emb.weight','mtp.{i}.head.weight'):
    if ('auxiliary',n['sourceIndex'],template)in rows:
     alias.append(matrix(e,sid,'MTPBlock.__init__' if n['group']=='mtp' else 'DSparkBlock.__init__',template,[c['vocab_size'],H],role='shared_alias'))
   if alias:n['modules'].append(module('shared-aliases','共享 embedding/head 的实际存储副本',alias,'mtp' if n['group']=='mtp' else 'dspark'))
 for n in arch['nodes']:
  group='decoder' if n['group']=='decoder' else 'auxiliary' if native and n['group'] in ('mtp','dspark') else 'mtp' if n['group']=='mtp' else 'global';i=n.get('sourceIndex',n['number']-1 if n['group']=='decoder' else None)
  n['payloadBytes']=0
  for mod in n['modules']:
   metadata=[]
   for m in mod['matrices']:
    tensor=stored_name(m['tensor_template']);key=(group,i,tensor)
    r=rows[key];assert r['copies']==m['count'],(name,n['id'],tensor,r['copies'],m['count']);used.add(key)
    packed=r['storedDtype']=='I8' and c.get('expert_dtype')=='fp4' and '.experts.'in tensor and tensor.endswith('.weight')
    assert r['storedShape']==m['logical_shape'] or packed and r['storedShape']==[m['logical_shape'][0],m['logical_shape'][1]//2],(name,tensor,r['storedShape'],m['logical_shape'])
    m.update(tensor_template=tensor,stored_shape=r['storedShape'],stored_dtype=r['storedDtype'],payloadBytes=r['payloadBytesEach'],storageEvidence='完整固定 checkpoint header；实际 stored namespace',source=audit_path)
    if packed:m['storageLayout']='I8 容器，每个字节包含两个 FP4 逻辑元素；32 输入元素一组 scale；原始张量未加载'
    n['payloadBytes']+=r['payloadBytesEach']*r['copies']
    scale=tensor.replace('.weight','.scale') if native else tensor.replace('.weight','.weight_scale_inv')
    sk=(group,i,scale)
    if sk in rows and scale!=tensor:
     rs=rows[sk];assert rs['copies']==m['count'];used.add(sk)
     metadata.append({'tensor_template':scale,'logical_shape':rs['storedShape'],'stored_shape':rs['storedShape'],'stored_dtype':rs['storedDtype'],'logical_parameters_each':0,'count':m['count'],'role':'quantization_metadata','evidence':'official','source':audit_path,'payloadBytes':rs['payloadBytesEach'],'storageEvidence':'该 checkpoint 全分片 header；不计入逻辑参数'})
     n['payloadBytes']+=rs['payloadBytesEach']*rs['copies']
   mod['matrices'].extend(metadata)
 missing=set(rows)-used
 assert not missing,(name,'uncovered header templates',sorted(missing,key=str)[:12])
 assert sum(n['payloadBytes'] for n in arch['nodes'])==audit['payloadBytes'],name
 arch['storageAuditPath']=audit_path;arch['fullPublishedPayloadBytes']=audit['payloadBytes'];arch['headerTensorCount']=audit['tensorCount'];arch['storageEvidence']='全部分片 header、offsets、dtype/shape 和索引；未读取权重数值'
 arch['scope']='固定配置/参考构造与本 checkpoint 全分片 header 对应；主干、独立 MTP/DSpark、共享副本、scale 与非训练表分列；非设备实验或运行兼容结论。'
 arch['parameterScope']='主干浮点逻辑参数，排除 MTP/DSpark、scale、非训练 hash 表；I8/FP4 存储元素按明确打包还原逻辑矩阵。'
