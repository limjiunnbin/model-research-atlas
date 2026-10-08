"""Generate the DeepSeek website from fixed configs, source proofs, and the separate V3 header audit."""
import ast
import collections
import copy
import hashlib
import html
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/families/deepseek'; ASSETS=ROOT/'dist/assets/deepseek'
DATE='2026-10-01'
CACHES={'sources.json':Path.home()/'.cache/model-research-atlas/deepseek-round1',
        'extended-sources.json':Path.home()/'.cache/model-research-atlas/deepseek-extended',
        'backend-sources.json':Path.home()/'.cache/model-research-atlas/deepseek-backends',
        'followup-sources.json':Path.home()/'.cache/model-research-atlas/deepseek-followup',
        'cann-sources.json':Path.home()/'.cache/model-research-atlas/deepseek-cann'}

def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def slug(name):return name.replace('DeepSeek-','').lower()
def shape(x):return ' × '.join(map(str,x)) if isinstance(x,list) else str(x)

class Evidence:
 def __init__(self):
  self.sources={};self.cache={};self.proofs={}
  for file,cache in CACHES.items():
   if not (DATA/file).is_file():continue
   for s in read(DATA/file)['sources']:
    self.sources[s['id']]=s;self.cache[s['id']]=cache
 def text(self,sid):
  s=self.sources[sid];raw=(self.cache[sid]/s['cacheFile']).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==s['sha256'],sid
  return raw.decode()
 def select(self,model,path):
  candidates=[sid for sid,s in self.sources.items() if s.get('modelId')=='deepseek-ai/'+model and s.get('path')==path]
  return next((sid for sid in candidates if sid.startswith('ext-')),candidates[0])
 def proof(self,sid,symbol):
  key=sid+':'+symbol
  if key not in self.proofs:
   text=self.text(sid);nodes={}
   for n in ast.parse(text).body:
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):nodes[n.name]=n
    if isinstance(n,ast.ClassDef):
     nodes.update({n.name+'.'+f.name:f for f in n.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))})
   n=nodes[symbol];s=self.sources[sid];body='\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])
   self.proofs[key]={'repo':s.get('repo',s.get('modelId')),'revision':s['revision'],'path':s['path'],
     'source_id':sid,'symbol':symbol,'line':n.lineno,'end':n.end_lineno,'sourceSha256':s['sha256'],
     'functionSha256':hashlib.sha256(body.encode()).hexdigest(),'url':s['url']+f'#L{n.lineno}-L{n.end_lineno}',
     'verification':'固定源码位置与函数体哈希核验；未运行模型/设备'}
  return key


def matrix(e,sid,symbol,name,dims,count=1,role='weight',declared=None):
 return {'tensor_template':name,'logical_shape':dims,'stored_shape':None,'stored_dtype':None,
   'logical_parameters_each':math.prod(dims) if role=='weight' else 0,'count':count,'role':role,
   'evidence':'derived','proof':e.proof(sid,symbol),'source':e.sources[sid]['url'],
   'declaredFormat':declared,'payloadBytes':None}

def module(mid,title,matrices,topic,count=1,selected=1):
 return {'id':mid,'title':title,'count':count,'selectedCount':selected,'representative':count>1,
         'parameters':sum(m['logical_parameters_each']*m['count'] for m in matrices),
         'matrices':matrices,'implementationTopic':topic}

def node(nid,group,number,kind,ffn,modules,cache_key=None,topic=None):
 # Active linear weights exclude normalization vectors, fixed routing tables and metadata.
 active=sum(m['logical_parameters_each']*(mod['selectedCount'] if mod['representative'] else m['count'])
            for mod in modules for m in mod['matrices'] if m['role']=='weight' and len(m['logical_shape'])==2)
 if nid=='embedding':active=None
 return {'id':nid,'group':group,'number':number,'type':kind,'ffn':ffn,'modules':modules,
         'parameters':sum(m['parameters'] for m in modules),'activeLinearParameters':active,
         'payloadBytes':None,'cacheKey':cache_key,'implementationTopic':topic}


def mla_arch(e,name,c):
 H=c['hidden_size'];N=c['num_attention_heads'];Q=c['q_lora_rank'];R=c['kv_lora_rank'];Dn=c['qk_nope_head_dim'];Dr=c['qk_rope_head_dim'];Dv=c['v_head_dim']
 E=c['n_routed_experts'];K=c['num_experts_per_tok'];I=c['intermediate_size'];J=c['moe_intermediate_size'];S=c['n_shared_experts'];L=c['num_hidden_layers']
 dsa=c['model_type']=='deepseek_v32';v2=c['model_type']=='deepseek_v2'
 sid=e.select(name,'inference/model.py' if dsa else 'modeling_deepseek.py')
 attn_symbol='MLA.__init__' if dsa else ('DeepseekV2Attention.__init__' if v2 else 'DeepseekV3Attention.__init__')
 mlp_symbol='Expert.__init__' if dsa else ('DeepseekV2MLP.__init__' if v2 else 'DeepseekV3MLP.__init__')
 gate_symbol='Gate.__init__' if dsa else 'MoEGate.__init__'
 block_symbol='Block.__init__' if dsa else ('DeepseekV2DecoderLayer.__init__' if v2 else 'DeepseekV3DecoderLayer.__init__')
 prefix='layers.{i}.attn.' if dsa else 'model.layers.{i}.self_attn.'
 attn_fields=[('wq_a.weight' if dsa else 'q_a_proj.weight',[Q,H]),('q_norm.weight' if dsa else 'q_a_layernorm.weight',[Q]),
  ('wq_b.weight' if dsa else 'q_b_proj.weight',[N*(Dn+Dr),Q]),('wkv_a.weight' if dsa else 'kv_a_proj_with_mqa.weight',[R+Dr,H]),
  ('kv_norm.weight' if dsa else 'kv_a_layernorm.weight',[R]),('wkv_b.weight' if dsa else 'kv_b_proj.weight',[N*(Dn+Dv),R]),
  ('wo.weight' if dsa else 'o_proj.weight',[H,N*Dv])]
 attn=[matrix(e,sid,attn_symbol,prefix+k,d) for k,d in attn_fields]
 norm_prefix='layers.{i}.' if dsa else 'model.layers.{i}.'
 norms=[matrix(e,sid,block_symbol,norm_prefix+k,[H]) for k in (['attn_norm.weight','ffn_norm.weight'] if dsa else ['input_layernorm.weight','post_attention_layernorm.weight'])]
 fprefix='layers.{i}.ffn.' if dsa else 'model.layers.{i}.mlp.'
 names=['w1.weight','w3.weight','w2.weight'] if dsa else ['gate_proj.weight','up_proj.weight','down_proj.weight']
 dense=[matrix(e,sid,mlp_symbol,fprefix+k,d) for k,d in zip(names,[[I,H],[I,H],[H,I]])]
 routed=[matrix(e,sid,mlp_symbol,fprefix+'experts.{e}.'+k,d,E) for k,d in zip(names,[[J,H],[J,H],[H,J]])]
 shared=[matrix(e,sid,mlp_symbol,fprefix+'shared_experts.'+k,d) for k,d in zip(names,[[S*J,H],[S*J,H],[H,S*J]])]
 router=[matrix(e,sid,gate_symbol,fprefix+'gate.weight',[E,H])]
 if not v2:router.append(matrix(e,sid,gate_symbol,fprefix+('gate.bias' if dsa else 'gate.e_score_correction_bias'),[E]))
 index=[]
 if dsa:
  hi=c['index_n_heads'];di=c['index_head_dim']
  index=[matrix(e,sid,'Indexer.__init__',prefix+'indexer.'+k,d) for k,d in
   [('wq_b.weight',[hi*di,Q]),('wk.weight',[di,H]),('weights_proj.weight',[hi,H]),('k_norm.weight',[di]),('k_norm.bias',[di])]]
 nodes=[]
 for i in range(L):
  is_moe=i>=c['first_k_dense_replace'] and i%c.get('moe_layer_freq',1)==0
  modules=[module('attention','DSA 主 attention' if dsa else 'MLA 低秩投影与 attention',copy.deepcopy(attn),'dsa' if dsa else 'mla'),module('norm','主干 RMSNorm',copy.deepcopy(norms),'precision')]
  if index:modules.append(module('indexer','Lightning Indexer：打分与 top-k',copy.deepcopy(index),'dsa'))
  if is_moe:modules += [module('router','分组路由与校正',copy.deepcopy(router),'moe'),module('routed','路由专家模板',copy.deepcopy(routed),'moe',E,K),module('shared','共享专家',copy.deepcopy(shared),'moe')]
  else:modules.append(module('ffn','Dense SwiGLU',copy.deepcopy(dense),'ffn'))
  nodes.append(node(f'decoder-{i+1}','decoder',i+1,'DSA' if dsa else 'MLA','MoE' if is_moe else 'Dense',modules,'dsa' if dsa else 'mla','dsa' if dsa else 'mla'))
 global_symbol='Transformer.__init__' if dsa else ('DeepseekV2Model.__init__' if v2 else 'DeepseekV3Model.__init__')
 head_symbol='Transformer.__init__' if dsa else ('DeepseekV2ForCausalLM.__init__' if v2 else 'DeepseekV3ForCausalLM.__init__')
 for gid,symbol,tensor,dims in [('embedding',global_symbol,'embed.weight' if dsa else 'model.embed_tokens.weight',[c['vocab_size'],H]),('final-norm',global_symbol,'norm.weight' if dsa else 'model.norm.weight',[H]),('output-head',head_symbol,'head.weight' if dsa else 'lm_head.weight',[c['vocab_size'],H])]:
  nodes.append(node(gid,'global',0,gid,None,[module(gid,gid,[matrix(e,sid,symbol,tensor,dims)],'precision')]))
 cache={'mla':f'MP=1 吸收式 cache 每 token/层为 {R}+{Dr}={R+Dr} 个元素；展开式为 {N}×({Dn}+{Dr}+{Dv})。实际分页/精度须按运行框架核验。'}
 if dsa:cache['dsa']=f'主 attention 保留 {R}+{Dr} 元素的潜变量/位置 cache；参考 demo 主 KV 用 FP8 量化后回到默认 dtype 模拟，不能标为实际 FP8 存储。Indexer 另有 E4M3 key [B,L_kv,{c["index_head_dim"]}] 与 F32 scale [B,L_kv,{c["index_head_dim"]//128}]；top-k={c["index_topk"]}。'
 return nodes,cache,sid


def v4_arch(e,name,c):
 H=c['hidden_size'];N=c['num_attention_heads'];D=c['head_dim'];Q=c['q_lora_rank'];G=c['o_groups'];O=c['o_lora_rank'];E=c['n_routed_experts'];K=c['num_experts_per_tok'];J=c['moe_intermediate_size'];S=c['hc_mult'];M=S*(S+2);L=c['num_hidden_layers']
 code_model=name.replace('-Base','');sid=e.select(code_model,'inference/model.py')
 def mat(symbol,path,dims,count=1,role='weight',declared=None):return matrix(e,sid,symbol,path,dims,count,role,declared)
 attn=[mat('Attention.__init__','layers.{i}.attn.'+p,d) for p,d in
  [('attn_sink',[N]),('wq_a.weight',[Q,H]),('q_norm.weight',[Q]),('wq_b.weight',[N*D,Q]),('wkv.weight',[D,H]),('kv_norm.weight',[D]),('wo_a.weight',[G*O,N*D//G]),('wo_b.weight',[H,G*O])]]
 norms=[mat('Block.__init__','layers.{i}.'+p,[H]) for p in ['attn_norm.weight','ffn_norm.weight']]
 hc=[]
 for branch in ('attn','ffn'):
  for suffix,dims in [('fn',[M,S*H]),('base',[M]),('scale',[3])]:hc.append(mat('Block.__init__',f'layers.{{i}}.hc_{branch}_{suffix}',dims))
 routed=[mat('Expert.__init__','layers.{i}.ffn.experts.{e}.'+p,d,E,declared='参考 Linear：FP4 每容器 2 个逻辑元素，32 输入通道一组 UE8M0 scale；本版文件头核对 I8 容器与 F8_E8M0 scale' if c['expert_dtype']=='fp4' else '配置 expert_dtype=FP8；本版实际 shape/dtype/scale 已由文件头核对') for p,d in [('w1.weight',[J,H]),('w3.weight',[J,H]),('w2.weight',[H,J])]]
 shared=[mat('Expert.__init__','layers.{i}.ffn.shared_experts.'+p,d) for p,d in [('w1.weight',[J,H]),('w3.weight',[J,H]),('w2.weight',[H,J])]]
 def compressor(prefix,ratio,head):
  coff=2 if ratio==4 else 1
  return [mat('Compressor.__init__',prefix+p,d) for p,d in [('ape',[ratio,coff*head]),('wkv.weight',[coff*head,H]),('wgate.weight',[coff*head,H]),('norm.weight',[head])]]
 nodes=[]
 dspark=bool(c.get('dspark_block_size'));ratios=c['compress_ratios'];aux=len(ratios)-L
 if dspark:
  demo=json.loads(e.text(e.select(name,'inference/config.json')))
  assert aux==demo['n_mtp_layers']==3 and all(r==0 for r in ratios[L:])
 else:assert aux==c['num_nextn_predict_layers']
 for i,ratio in enumerate(ratios):
  hash_route=i<c['num_hash_layers'];kind={0:'SWA',4:'CSA',128:'HCA'}[ratio]
  mods=[module('attention',f'{kind}：head dim={D}，窗口 {c["sliding_window"]}',copy.deepcopy(attn),'compression'),module('norm','子层 RMSNorm',copy.deepcopy(norms),'precision'),module('residual',f'mHC：{S} 条残差流',copy.deepcopy(hc),'mhc')]
  if ratio:mods.append(module('compressor',f'压缩比 {ratio} 的 learned pooling',compressor('layers.{i}.attn.compressor.',ratio,D),'compression'))
  if ratio==4:
   hi=c['index_n_heads'];di=c['index_head_dim']
   idx=[mat('Indexer.__init__','layers.{i}.attn.indexer.'+p,d) for p,d in [('wq_b.weight',[hi*di,Q]),('weights_proj.weight',[hi,H])]]
   idx+=compressor('layers.{i}.attn.indexer.compressor.',ratio,di)
   mods.append(module('indexer',f'压缩 indexer：{hi} heads / top-k {c["index_topk"]}',idx,'compression'))
  router=[mat('Gate.__init__','layers.{i}.ffn.gate.weight',[E,H])]
  if hash_route:router.append(mat('Gate.__init__','layers.{i}.ffn.gate.tid2eid',[c['vocab_size'],K],role='index_table',declared='I32、requires_grad=False 的 token→expert 地址表；排除可学习逻辑权重计数'))
  else:router.append(mat('Gate.__init__','layers.{i}.ffn.gate.bias',[E]))
  mods += [module('router','Hash 路由（仍计算 gating scores）' if hash_route else 'sqrt(softplus) 分数路由',router,'moe'),module('routed','路由专家模板',copy.deepcopy(routed),'moe',E,K),module('shared','共享专家',copy.deepcopy(shared),'moe')]
  group='decoder' if i<L else 'dspark' if dspark else 'mtp';j=i-L
  n=node(f'decoder-{i+1}' if i<L else f'dspark-{j+1}' if dspark else 'mtp-1',group,i+1 if i<L else 0,kind if i<L or not dspark else 'DSpark / SWA','MoE',mods,kind.lower(),'compression' if i<L else 'dspark' if dspark else 'mtp');n.update(compressRatio=ratio,routing='hash' if hash_route else 'score',residualStreams=S)
  if i>=L:n['sourceIndex']=j
  nodes.append(n)
 for gid,symbol,tensor,dims in [('embedding','ParallelEmbedding.__init__','embed.weight',[c['vocab_size'],H]),('final-norm','Transformer.__init__','norm.weight',[H]),('output-head','ParallelHead.__init__','head.weight',[c['vocab_size'],H])]:
  nodes.append(node(gid,'global',0,gid,None,[module(gid,gid,[mat(symbol,tensor,dims)],'precision')]))
 head_hc=[mat('Transformer.__init__','hc_head_'+p,d) for p,d in [('fn',[S,S*H]),('base',[S]),('scale',[1])]]
 nodes.append(node('head-hc','global',0,'mHC-head',None,[module('head-hc','输出前 mHC 流折叠',head_hc,'mhc')],topic='mhc'))
 # Reference MTP explicitly shares embedding/head, but owns its projections and normalization.
 if dspark:
  draft=[n for n in nodes if n['group']=='dspark']
  for n in draft:
   for mod in n['modules']:
    mod['implementationTopic']='dspark'
    for m in mod['matrices']:m['tensor_template']=m['tensor_template'].replace('layers.{i}.','mtp.{i}.')
  first=draft[0];last=draft[-1];F=len(c['dspark_target_layer_ids']);R=c['dspark_markov_rank']
  first['modules'].append(module('dspark-context','目标隐藏特征融合',[mat('DSparkBlock.__init__','mtp.{i}.main_proj.weight',[H,F*H]),mat('DSparkBlock.__init__','mtp.{i}.main_norm.weight',[H])],'dspark'))
  tail=[mat('DSparkBlock.__init__','mtp.{i}.norm.weight',[H]),mat('DSparkMarkovHead.__init__','mtp.{i}.markov_head.markov_w1.weight',[c['vocab_size'],R]),mat('DSparkMarkovHead.__init__','mtp.{i}.markov_head.markov_w2.weight',[c['vocab_size'],R]),mat('DSparkConfidenceHead.__init__','mtp.{i}.confidence_head.proj.weight',[1,H+R])]
  tail += [mat('DSparkBlock.__init__','mtp.{i}.hc_head_'+p,d) for p,d in [('fn',[S,S*H]),('base',[S]),('scale',[1])]]
  last['modules'].append(module('dspark-heads','Markov 修正、confidence 与流折叠',tail,'dspark'))
  for n in draft:
   n['parameters']=sum(m['parameters'] for m in n['modules']);n['activeLinearParameters']=sum(x['logical_parameters_each']*(m['selectedCount'] if m['representative'] else x['count']) for m in n['modules'] for x in m['matrices'] if x['role']=='weight' and len(x['logical_shape'])==2 and not x['tensor_template'].endswith('markov_w1.weight'))
  cache={'swa':f'DSpark 的上下文 SWA cache 与目标主干 cache 分开；窗口 {c["sliding_window"]}、共享 KV {D} 维，draft block 内非因果；其余主干 CSA/HCA cache 仍独立。', 'csa':f'窗口 + floor(L_kv/4) 的主干压缩 KV/indexer；两个 FP32 compressor 状态另计。','hca':f'窗口 + floor(L_kv/128) 的主干压缩 KV；无 learned indexer。'}
  return nodes,cache,sid
 mtp=next(n for n in nodes if n['group']=='mtp');mtp_mods=mtp['modules']
 extra=[mat('MTPBlock.__init__','mtp.{i}.'+p,d) for p,d in [('e_proj.weight',[H,H]),('h_proj.weight',[H,H]),('enorm.weight',[H]),('hnorm.weight',[H]),('norm.weight',[H]),('hc_head_fn',[S,S*H]),('hc_head_base',[S]),('hc_head_scale',[1])]]
 mtp_mods.append(module('mtp-projections','MTP 输入、norm 与独立流折叠',extra,'mtp'))
 for mod in mtp_mods:
  for m in mod['matrices']:
   if m['tensor_template'].startswith('layers.{i}.'):m['tensor_template']=m['tensor_template'].replace('layers.{i}.','mtp.{i}.')
 mtp.update(node('mtp-1','mtp',0,'SWA / mHC','MoE',mtp_mods,'swa','mtp'))
 cache={
  'swa':f'滚动窗口 [B,{c["sliding_window"]},{D}]，不缓存 V3 式 512+64 潜变量。',
  'csa':f'窗口 {c["sliding_window"]} + floor(L_kv/4) 条 {D} 维 KV；另有 floor(L_kv/4)×{c["index_head_dim"]} 的 indexer cache。两个 compressor 都有 FP32 kv_state/score_state；ratio=4 overlap coff=2，形状 [B,2r,2d]。参考 cache 经 FP8/FP4 模拟后仍用默认 dtype 存储。',
  'hca':f'窗口 {c["sliding_window"]} + floor(L_kv/128) 条 {D} 维 KV，无 learned indexer；compressor 的两个 FP32 状态 [B,128,{D}] 另计。'
 }
 return nodes,cache,sid


def gqa_arch(e,name,c):
 H=c['hidden_size'];N=c['num_attention_heads'];NK=c['num_key_value_heads'];D=H//N;I=c['intermediate_size'];L=c['num_hidden_layers'];qwen=c['model_type']=='qwen2'
 sid='hf-transformers-modeling_qwen2-py' if qwen else 'hf-transformers-modeling_llama-py';cls='Qwen2' if qwen else 'Llama'
 fields=[('q_proj.weight',[N*D,H]),('k_proj.weight',[NK*D,H]),('v_proj.weight',[NK*D,H]),('o_proj.weight',[H,N*D])]
 if qwen:fields += [('q_proj.bias',[N*D]),('k_proj.bias',[NK*D]),('v_proj.bias',[NK*D])]
 elif c.get('attention_bias'):fields += [('q_proj.bias',[N*D]),('k_proj.bias',[NK*D]),('v_proj.bias',[NK*D]),('o_proj.bias',[H])]
 attn=[matrix(e,sid,cls+'Attention.__init__','model.layers.{i}.self_attn.'+p,d) for p,d in fields]
 ffn=[matrix(e,sid,cls+'MLP.__init__','model.layers.{i}.mlp.'+p,d) for p,d in [('gate_proj.weight',[I,H]),('up_proj.weight',[I,H]),('down_proj.weight',[H,I])]]
 norms=[matrix(e,sid,cls+'DecoderLayer.__init__','model.layers.{i}.'+p,[H]) for p in ['input_layernorm.weight','post_attention_layernorm.weight']]
 nodes=[node(f'decoder-{i+1}','decoder',i+1,'GQA','Dense',[module('attention',f'GQA：{N} Q heads / {NK} KV heads',copy.deepcopy(attn),'gqa'),module('ffn','Dense SwiGLU',copy.deepcopy(ffn),'ffn'),module('norm','RMSNorm',copy.deepcopy(norms),'precision')],'gqa','gqa') for i in range(L)]
 globals=[('embedding',cls+'Model.__init__','model.embed_tokens.weight',[c['vocab_size'],H]),('final-norm',cls+'Model.__init__','model.norm.weight',[H])]
 if not c['tie_word_embeddings']:globals.append(('output-head',cls+'ForCausalLM.__init__','lm_head.weight',[c['vocab_size'],H]))
 for gid,symbol,tensor,dims in globals:nodes.append(node(gid,'global',0,gid,None,[module(gid,gid,[matrix(e,sid,symbol,tensor,dims)],'precision')]))
 return nodes,{'gqa':f'KV cache [B,{NK},L_kv,{D}]，每 token/层为 2×{NK}×{D} 元素；Q heads 广播对应 KV heads，不复制新增权重。底座配置与蒸馏配置分别核对，运行框架布局/精度未知。'},sid


def apply_v3_audit(arch,audit):
 templates={(r['kind'],r['tensorTemplate']):r for r in audit['templates']}
 for n in arch['nodes']:
  if n['group']=='decoder':kind='dense' if n['ffn']=='Dense' else 'moe';n['payloadBytes']=audit['layers'][str(n['number']-1)]['payloadBytes']
  elif n['group']=='global':kind='global';n['payloadBytes']=0
  else:continue
  for mod in n['modules']:
   extra=[]
   for m in mod['matrices']:
    r=templates[(kind,m['tensor_template'])];assert r['storedShape']==m['logical_shape']
    m.update(stored_shape=r['storedShape'],stored_dtype=r['storedDtype'],payloadBytes=r['payloadBytesEach'],storageEvidence='safetensors header audit @ fixed revision')
    if n['group']=='global':n['payloadBytes']+=r['payloadBytesEach']
    scale_name=m['tensor_template'].replace('.weight','.weight_scale_inv');rs=templates.get((kind,scale_name))
    if rs:
     extra.append({'tensor_template':scale_name,'logical_shape':rs['storedShape'],'stored_shape':rs['storedShape'],'stored_dtype':rs['storedDtype'],'logical_parameters_each':0,'count':m['count'],'role':'quantization_metadata','evidence':'official','source':audit['shards'][0]['url'],'payloadBytes':rs['payloadBytesEach'],'storageEvidence':'全分片文件头审计'})
   mod['matrices'].extend(extra)
 arch['storageAuditPath']='data/families/deepseek/research/v3-header-audit.json'
 # Header-backed MTP is displayed separately; document sharing does not prove every loader alias.
 groups=collections.defaultdict(list)
 for row in audit['templates']:
  if row['kind']!='mtp':continue
  name=row['tensorTemplate'];alias=name.endswith('embed_tokens.weight') or name.endswith('shared_head.head.weight')
  role='shared_alias' if alias else 'quantization_metadata' if name.endswith('weight_scale_inv') else 'weight'
  mid='routed' if '.experts.'in name else 'shared' if '.shared_experts.'in name else 'router' if '.mlp.gate.'in name else 'attention' if '.self_attn.'in name else 'mtp-other'
  groups[mid].append({'tensor_template':name,'logical_shape':row['storedShape'],'stored_shape':row['storedShape'],'stored_dtype':row['storedDtype'],'logical_parameters_each':math.prod(row['storedShape']) if role=='weight' else 0,'count':row['copies'],'role':role,'evidence':'official','source':'assets/deepseek/v3-header-audit.json','proof':None,'payloadBytes':row['payloadBytesEach'],'storageEvidence':'全部 MTP header；共享 embedding/head 副本不重复计逻辑参数'})
 mods=[module(k,k+' / MTP header',v,'mtp',arch['config']['experts'] if k=='routed' else 1,arch['config']['topK'] if k=='routed' else 1) for k,v in groups.items()]
 n=node('mtp-1','mtp',0,'MTP','MoE',mods,'mla','mtp');n['sourceIndex']=61;n['payloadBytes']=audit['layers']['61']['payloadBytes'];n['parameterScope']='去掉文档明确共享 embedding/head 存储副本后的参数元素数；shared_head.norm 别名未知，不冒称精确 unique 参数'
 arch['nodes'].append(n);arch['mtpLogicalParameters']=None;arch['mtpStoredParameterElements']=audit['mtpStoredParameterElements'];arch['mtpEstimateExcludingSharedEmbeddingHead']=n['parameters']


def make_report(family,architectures,e,audit):
 esc=html.escape
 def table(headers,rows):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
 def section(sid,title,body):return {'id':sid,'title':title,'html':'<h2>'+esc(title)+'</h2>'+body}
 model_rows=[(m['name'],m['facts']['layers']['value'],m['facts']['attention']['value'],m['facts']['hidden']['value'],f'{m["facts"]["parameters"]["value"]:,}',m['baseModel'] or '不预设权重底座') for m in family['models']]
 relation=read(DATA/'research/config-relations.json')
 token=read(DATA/'research/r1-config-tokenizer-comparison.json') if (DATA/'research/r1-config-tokenizer-comparison.json').exists() else {'records':[]}
 token_rows=[]
 for row in token['records']:
  changes=row['configChanges'];tc=row['tokenizerConfigChanges']
  token_rows.append((row['model'],row['baseModel'],', '.join(sorted(changes)) if changes is not None else '底座 gated，未获取配置',', '.join(sorted(tc)) if tc is not None else '底座 gated，未获取 tokenizer 配置'))
 sections=[
  section('versions','1. 代表版本与底座关系','<p>当前覆盖 21 个选定 checkpoint，名称与 revision 来自官方目录。V2 是设计演进入口；V3 Base/后训练分列；R1/Zero 官方基于 V3-Base；Distill 保留各自 Qwen/Llama 底座。Max 推理档位没有新增为 checkpoint，另补 DSpark 挂接版与 Flash-0731/Pro-0813；V4.1 等仍属后续范围。</p>'+table(['模型','主干层','结构','hidden','主干逻辑参数','底座'],model_rows)),
  section('architecture','2. 结构、参数与缓存口径','<p>层从 1 开始展示，源码层号从 0 开始。MoE 用一个代表专家模板与完整专家数/top-k，不展开成数万个浏览器对象。逻辑参数由固定配置加参考源码的声明形状推导，排除量化 scale 与非训练 hash 地址表；MTP/DSpark 单列。激活线性参数代理不是 FLOPs 或性能。</p><p>当前 21 版实际存储全部来自各自全分片 header；逻辑形状、I8/FP4 容器、F8_E8M0/F32 scale 与非训练 I64 地址表分别核对。结构页按 MLA、DSA、CSA、HCA、SWA、GQA 各自的缓存说明显示，mHC 的 4 流轴不合并进 hidden。</p><p>本轮展开 21 版主干、实际发布的 MTP 与 DSpark。V3-Base、R1/Zero、V3.2/Exp 的 MTP 均有自己的文件头，并以固定 vLLM 独立 forward/loader 对应；共享 embedding/head 副本排除。SharedHead.norm 在所选参考后端独立构造/加载，上游训练别名和值是否相同仍不由 header 证明。</p>'),
  section('v3','3. V3 基线与文件头审计',f'<p>主干参数 {audit["mainParameterElements"]:,}，61 层 = 3 Dense + 58 MoE。文件头覆盖 {len(audit["shards"])} 个分片与 {audit["tensorCount"]:,} 个张量，实际全部发布载荷 {audit["payloadBytes"]:,} 字节；索引声明 total_size={audit["indexDeclaredTotalSize"]:,}，不作为实测载荷。此次 HTTP Range 只读取 {audit["metadataTransferredBytes"]:,} 字节元数据，没有读取权重载荷。</p>'+table(['存储类型','张量数量'],audit['dtypeTensorCounts'].items())+'<p>主 attention 的 Q/低秩 KV/NoPE/RoPE 与 naive/absorb 路径分别说明。MP=1 时吸收式 cache 为每 token/层 512+64 元素，展开式为 128×(192+128)。BF16 假设的字节公式不等于框架显存实测。</p><p>MTP 文件层号 61 与主干 0–60 分开；存储参数元素数 '+f'{audit["mtpStoredParameterElements"]:,}'+' 包含共享 embedding/head 的存储副本。官方 11.5B unique、14B/685B 发布口径和文件头统计分别保留；无法只由 header 证明所有加载别名或相同数值。V3 demo 转换显式跳过 MTP，不能宣称支持该路径。</p>'),
  section('increments','4. V3.2 / V4 增量','<p>V3.2 与 Exp 的所选配置和 inference/model.py 字节一致。DSA 新增 Indexer 的低秩 Q 投影、K 投影、LayerNorm、每头权重投影、Hadamard 旋转、FP8 index score 与 top-k。每层新增逻辑参数 13,959,424，61 层主干共 671,877,944,064。参考 prefill 使用展开 attention 加 index mask；decode 使用吸收式路径，不能把 V3 的两个自由实现分支直接当作 V3.2 分派。</p><p>V4-Flash 主干为 2 SWA + 21 CSA + 20 HCA；V4-Pro 为 30 CSA + 31 HCA。普通 V4 的 compress_ratios 含一个 MTP 条目，DSpark 版含三个 draft stage 条目；末尾 ratio=0 不计入主干。ratio=4 的 learned compressor 使用 overlap/coff=2，并有独立的压缩 indexer；ratio=128 使用全压缩前缀，无 learned indexer。窗口、压缩 KV、indexer cache 和两个 FP32 compressor 状态分别计数。</p><p>V4 的 query head dim=512，末尾 64 为位置分量；共享 KV dim=512，输出采用分组低秩 wo_a/wo_b。mHC 每子层有 [24,4H] 的控制投影、[24] base、[3] scale，维护 [B,L_q,4,H]；hc_pre 经 Sinkhorn 构造 pre/post/comb，hc_post 还原四流。前三层 hash 路由仍计算 gating scores，token→expert 表是非训练地址表，参考构造声明 I32，本轮实际 header 为 I64，不据此认定 Engram 集成。</p><p>Base 的 expert_dtype=FP8，后训练版 routed experts=FP4；FP4 每容器两元素、group32 UE8M0 scale 是参考 Linear 的条件格式，实际 header 已核对：routed FP4 存在 I8 容器，scale 为 F8_E8M0；Base/shared 与 routed 的 dtype/shape 分开，不以 I8 单独推断 FP4。V4 参考 MTP 共享 embedding/head，独立拥有投影、norm 与流折叠，普通 Transformer.forward 不自动执行 MTP。</p>'),
  section('r1','6. R1 与 Distill','<p>原始 R1、R1-Zero、V3-Base 的所选配置与原始 V3 字节相同，固定 HF modeling_deepseek.py 的 SHA-256 也相同。这里只复用主干形状与参考计算；后训练、权重内容、能力与服务条件不据此推定相同。</p>'+table(['checkpoint','配置字节相同','参考代码字节相同'],[(r['model'],r['configByteIdentical'],r['referenceCodeByteIdentical']) for r in relation])+'<p>R1-Zero 在 V3-Base 上直接进行大规模 RL，未以 SFT 作前置；R1 使用冷启动数据并采用两次 SFT/两阶段 RL。官方评测使用最大生成 32,768 token、temperature=0.6、top_p=0.95，并对采样任务生成 64 条响应估计 pass@1；这些是评测/采样条件，不是架构或服务容量。</p><p>Distill 六版都是 Dense GQA 底座，不使用 V3 MLA/MoE。Qwen2 固定参考类的 Q/K/V bias 与 Llama 的 attention_bias/mlp_bias 分别核对；词表、EOS/BOS、位置上限、tie_word_embeddings 来自实际蒸馏配置。名称 1.5B 等沿用底座档位，不强行把推导参数配平到名称。</p>'+table(['蒸馏模型','底座','配置变化字段','tokenizer_config 变化字段'],token_rows)+'<p>Qwen 四版完成固定底座配置与 tokenizer_config 比较；Llama 原底座 resolve 请求返回 gated/401，蒸馏配置与 tokenizer_config 已读取，原底座差异保留未知。没有借第三方镜像替代。现已逐条比较四个 Qwen 底座与蒸馏完整 tokenizer.json：151,643 个 vocab token 的 ID、151,387 个 merge 完全一致，added_tokens 不同；normalizer/pre_tokenizer/post_processor/decoder 一致。六个 Distill 的完整词表与特殊 ID/embedding 上限都核对，Llama 原底座的完整 tokenizer 同样 gated。</p>'),
  section('implementation','7. 参考实现与 Ascend 路径','<p>固定 vLLM、vLLM-Ascend 与 TileKernels 各自提交；这些快照未被组成或测试为兼容栈。模型 forward、框架模块、torch.ops 分派和设备实现分层记录，数学步骤与 kernel 不是一对一。实现、硬件、优化页按当前模型过滤专题。</p><p>原始 FP8、转换 MXFP8/MXFP4、BF16 fallback 分开说明。TileKernels 的 Ascend 后端要求 Ascend 950、CANN≥9.2、Python≥3.12、PyTorch≥2.13、TileLang≥0.1.15；不能推广为 A2/A3 或任意已安装环境。V3.2/V4 的 Indexer、压缩、mHC 与 MoE 通信分别记录条件和缺口。</p>'),
  section('evidence','8. 证据与验收','<p>每项已知事实使用 official、derived 或 interpretation，未知为 null/unknown。已读取范围、配置 revision、源码 commit、函数位置与文件/函数体 SHA-256 随下载保留。参数、shape、cache 与来源引用由 DeepSeek 专用验证器检查，普通 build 按 family 分派；Kimi 专用验证器不冒充覆盖 DeepSeek。</p><p>网页、报告与 CSV 使用相同研究数据。全层映射由代表模板展开，下载说明注明静态展开范围；它不等于权重加载、设备数值正确性或性能验证。检查窄屏、二维回退、版本比较、阶段/分支筛选与所有下载链接。</p>'),
  section('experiments','9. 本次验收范围与可选后续研究','<p>当前没有硬件，用户已将设备运行与性能验证移出本次范围。本次交付按模型结构、文件头、shape/参数公式、固定来源与 CANN 源码、导出一致性及网站功能验收；设备实验不作为未完成项或验收前置条件。</p><p>源码与 package 的只读核查仍是本次静态研究。实际运行分支、ABI、数值与性能标为未实测；duration、利用率、mte、scalar 与吞吐保持空值或未知。未来有硬件并另行发起时，可使用可选研究卡片与实验记录模板，不自动恢复实验。</p>')
 ]
 spec=read(DATA/'research/speculation.json')
 rows=[(m['name'],m['config']['declaredNextNLayers'],m['config']['stages'],m['config']['targetLayerIds'],m['config']['blockSize'],m['config']['markovRank'],f'{m["draftLogicalParameters"]:,}') for m in spec['models']]
 body='<p>DSpark 采用并行 draft backbone 加轻量顺序模块，并由 confidence 估计条件接受概率。V4-Flash/Pro-DSpark 官方声明是原 preview checkpoint 挂接 draft 模块；Flash-0731/Pro-0813 属于另外的正式发布，不由配置相同推定权重相同。全部固定目录、配置、native 代码和 header 独立保存。</p>'
 body+=table(['checkpoint','HF nextn 声明','实际 draft stages','目标层号(0-based)','block size','Markov rank','独立 draft 逻辑参数'],rows)
 body+='<p>存储仍使用 mtp.0/1/2 namespace，但模块是 DSparkBlock：首层 main_proj/main_norm 融合三处目标 hc_head 特征，三层都使用独立 context SWA cache，末层有 norm、流折叠、两个 [V,R] Markov 矩阵和 [1,H+R] confidence 投影；共享主干 embedding/head。MTP 的 eh_proj/enorm/hnorm 不套给 DSpark。HF num_nextn_predict_layers=1、compress_ratios 的 3 个尾零、inference n_mtp_layers=3 与真实 namespace 分别保留。</p>'
 body+='<p>native prefill 只初始化 context KV；draft 时一个 noisy block 在 backbone 内非因果并行处理。随后按前一已选 token 做 Markov embedding→logits bias→sample，形成块内顺序依赖。confidence proj 在 header 为 BF16，native 构造/计算 FP32；原生 forward_head 返回 logits，服务侧 sigmoid 后才是条件置信度。checkpoint block=5 与模型卡 vLLM 示例 num_speculative_tokens=7 是不同设置，原样保留。</p>'
 body+=table(['阶段','静态协议'],[(p['phase'],p['operation']) for p in spec['protocol']])
 body+='<p>论文的条件概率 c_k 给出前缀生存概率 S_k=∏_{j≤k}c_j，期望接受长度为 Σ_k S_k。作者的负载感知前缀 scheduler 结合 engine-specific throughput profile；公开 demo 的固定阈值、框架自适应配置和生产 scheduler 分开。前缀选择需保持 stopping-time/early-stopping 条件，任意看后续 token 再回退可能造成选择偏差。训练侧含 vanilla Markov/RNN 变体和 confidence 校准，本 V4 权重与代码采用 vanilla Markov，不推定包含 RNN。</p>'
 body+=table(['方法','草稿','target','参数/缓存'],[(r['method'],r['draft'],r['target'],r['parameters']+'；'+r['cache']) for r in spec['comparisons']])
 body+='<p>MTP 训练层数与服务循环生成次数不同，MTP-1 不等于只能配置 1 个 draft token。固定 vLLM 参考的 MTP 会 mask position=0 embedding，两次 norm/concat→eh_proj→独立 DecoderLayer；logits hidden 与回收 hidden 分别只执行一次 final norm。V3 native demo 跳过 MTP，不据此抹掉 checkpoint 已发布的 MTP。</p>'
 body+='<p>DSpark 作者在 V4 在线服务、匹配总吞吐条件下自述 Flash 单用户生成速度提高 60%–85%、Pro 57%–78%；这些数值属于作者环境，未作为本站或 Ascend 实测。静态小概率向量验证仅检查 rejection distribution 恒等式及条件概率乘积，不验证训练置信度、校准、设备或部署收益。Muon optimizer、SFT/RL/GRPO 和 on-policy distillation 是训练侧机制，独立于推测解码执行图。</p>'
 body+='<p>主要来源：<a href="https://arxiv.org/abs/2607.05147v1">DSpark 论文 v1</a>；<a href="https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2">DeepSpec 固定源码</a>；每个 checkpoint 的 config/inference/header 与框架 proof 链见 speculation.json / site-proofs.json。</p>'
 sections.insert(4,section('speculation','5. MTP、DSpark 与推测解码协议',body))
 trace=read(DATA/'research/backend-trace.json')
 implementation=next(s for s in sections if s['id']=='implementation')
 implementation['html']+=table(['开放后端路径','调用/注册链','约束'],[(r['id'],' → '.join(r['flow']),r['conditions']) for r in trace['paths']])+'<p>注册和 host/kernel 入口范围有独立 source/range 哈希，C++ 范围不冒充 Python AST 函数证明。CANN 相关算子已在匹配版本开源仓继续核查，具体来源见第 10 节；实际分派、ABI 与设备数值仍待验证。未看到 TileKernels 与该框架的调用接线，保留两套独立实现。</p>'
 cann=read(DATA/'research/cann-operator-audit.json')
 body='<p>此前将 CANN 相关缺口笼统写为闭源是不准确的，现按匹配版本开源仓和本地 package 继续核查。本地只读检查确认 CANN 9.2.0-beta.2、aarch64；固定 ops-transformer/ops-nn/ops-math 的 v9.2.0-beta.2 tag，并另固定 op-plugin bridge。</p>'
 body+=table(['来源','固定 revision'],[(repo,revision) for repo,revision in cann['repositories'].items()]+[('Ascend/op-plugin',cann['pluginRevision'])])
 rows=[]
 source_links=[]
 for op in cann['operators']:
  refs=' '.join(f'<a href="{esc(r["url"])}">{esc(r["role"])}</a>' for r in op['references'] if r['role'] in ('api','definition','tiling','kernel_entry'))
  rows.append((op['directory'],', '.join(op['apiSymbols'][:5]),op['frameworkContext'],op['relationship']))
  source_links.append(refs)
 body+='<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+esc(title)+'</th>' for title in ['算子源码目录','API 符号示例','框架上下文','接线证据范围','固定源码链接'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(value)+'</td>' for value in row)+'<td>'+links+'</td></tr>' for row,links in zip(rows,source_links))+'</tbody></table></div>'
 body+='<p>Python API 后缀不能用来猜测 CANN API 版本：所选 op-plugin 的 npu_fused_infer_attention_score_v2 按条件调用 aclnnFusedInferAttentionScoreV4/V5；MoE dispatch/combine 与 grouped_matmul 也存在版本/格式条件分支。源码候选集合不是执行时序。官方 MhcPre/Post 与框架 custom HcPre/Post 保留为独立实现。</p>'
 body+='<p>安装目录提供编译器/AscendC SDK 和可读取的 SwiGLU、RMSNorm、Matmul 高级 API 源码；所检查路径没有三个独立算子库的 libopapi 与算子头文件。SDK 高级 API 不等同完整 aclnn 算子库，缺少包内文件时使用对应 tag 的源文件继续追踪。未安装/更新虚拟机软件或执行算子，设备实验不在本次范围。</p>'
 body+=f'<p>本轮 CANN 数据包含 {cann["counts"]["operatorFamilies"]} 个源码家族、{cann["counts"]["sourceFiles"]} 个固定源文件和 {cann["counts"]["codeRanges"]} 个源码范围，原始字节同时核对 SHA-256 与 Git blob SHA-1。完整逐项链接与本地包证据见 <a href="assets/deepseek/DeepSeek-CANN-operators.json">CANN 算子核对 JSON</a>。源码存在、实际库/分派、ABI、设备数值与性能分别验收。</p>'
 sections.append(section('cann','10. CANN 开源算子与本地 SDK',body))
 if (DATA/'analysis.json').is_file():
  analysis=read(DATA/'analysis.json')
  body=f'<p>静态分析扩展覆盖 {analysis["counts"]["costModels"]} 个 checkpoint 的主干成本、{analysis["counts"]["cannFamilies"]} 个 CANN 契约家族、固定版本的并行/通信与五个机制教学演示。成本只统计声明矩阵收缩和逻辑容量，支持与数值边界保持。</p>'
  body+='<p>网站新增成本计算器、CANN 约束手册、并行/通信计算和机制步进页；主页另有跨家族比较，保留 DeepSeek、Kimi、GLM 的不同读取深度与统计口径。</p>'
  body+='<p>完整公式、条件、示例与来源见 <a href="assets/deepseek/DeepSeek-静态分析.md">静态分析说明</a>、<a href="assets/deepseek/DeepSeek-静态分析.xlsx">分析工作簿</a> 和 <a href="assets/deepseek/DeepSeek-静态分析.json">分析 JSON</a>。设备实验仍不在本次范围。</p>'
  sections.append(section('analysis','11. 静态成本、算子契约与系统分析',body))
 write(DATA/'report.json',sections)
 md=['# DeepSeek 模型研究图谱\n', '研究快照：'+DATE+'。P0–P5 静态研究与交付，设备验证不纳入本次验收。',
     table(['模型','主干层','结构','主干逻辑参数'],[(m['name'],m['facts']['layers']['value'],m['facts']['attention']['value'],f'{m["facts"]["parameters"]["value"]:,}') for m in family['models']])]
 # Markdown export is generated from the exact report sections, with readable text/table HTML retained.
 for sid,fig in zip(('versions','v3','increments','speculation'),family['figures']):
  s=next(s for s in sections if s['id']==sid)
  s['html']+=f'<figure><img src="{fig["path"]}" alt="{esc(fig["caption"])}" loading="lazy"/><figcaption>{esc(fig["source"])}</figcaption></figure>'
 write(DATA/'report.json',sections)
 for s in sections:md += ['## '+s['title'],s['html'].replace('<h2>'+esc(s['title'])+'</h2>','',1).replace('src="assets/deepseek/figures/','src="figures/').replace('href="assets/deepseek/','href="')]
 target=ASSETS/'DeepSeek-研究报告.md';target.write_text('\n\n'.join(md)+'\n')
 return sections


def build():
 e=Evidence();versions=read(DATA/'versions.json');versions['models']=versions['models']+read(DATA/'research/followup-checkpoints.json')['models'];audit=read(DATA/'research/v3-header-audit.json');models=[];architectures={};config_diffs=[]
 reference=read(DATA/'configs/DeepSeek-V3-config.json')
 for v in versions['models']:
  name=v['name'];mid=slug(name);c=read(DATA/'configs'/f'{name}-config.json');csid=e.select(name,'config.json')
  if c['model_type']=='deepseek_v4':nodes,cache,sid=v4_arch(e,name,c)
  elif c['model_type'] in ('qwen2','llama'):nodes,cache,sid=gqa_arch(e,name,c)
  else:nodes,cache,sid=mla_arch(e,name,c)
  main=sum(n['parameters'] for n in nodes if n['group'] not in ('mtp','dspark'));types=collections.Counter(n['type'] for n in nodes if n['group']=='decoder')
  quant=c.get('quantization_config');precision=('配置 '+str(quant['quant_method']).upper()+'；scale='+str(quant.get('scale_fmt','F32'))) if quant else '配置 torch_dtype='+c.get('torch_dtype','unknown')
  if c['model_type']=='deepseek_v4':precision+='；routed expert='+c['expert_dtype'].upper()+'；本版文件头已核对'
  arch={'schemaVersion':1,'modelId':mid,'label':name,'evidence':'derived','scope':'固定配置与参考源码的逻辑参数/矩阵；各版实际存储由自身文件头核对，MTP/DSpark 单列；非运行支持或性能结论。',
        'source':e.sources[csid]['url'],'referenceSource':e.sources[sid]['url'],'source_ids':[csid,sid],
        'nodes':nodes,'config':{'hidden':c['hidden_size'],'heads':c['num_attention_heads'],'experts':c.get('n_routed_experts'),'topK':c.get('num_experts_per_tok'),'shared':c.get('n_shared_experts'),'context':c['max_position_embeddings'],'precision':precision,'streams':c.get('hc_mult',1)},
        'cache':cache,'mainLogicalParameters':main,'mtpLogicalParameters':sum(n['parameters'] for n in nodes if n['group']=='mtp') if any(n['group']=='mtp' for n in nodes) else None if c.get('num_nextn_predict_layers') else 0,
        'parameterScope':'主干浮点逻辑权重、norm 与 routing correction；排除 quant scale、非训练 hash 地址表；MTP/DSpark 单列',
        'attentionTypeCounts':dict(types)}
  if c.get('dspark_block_size'):
   arch['dsparkLogicalParameters']=sum(n['parameters'] for n in nodes if n['group']=='dspark');arch['mtpLogicalParameters']=0
   arch['dspark']={'blockSize':c['dspark_block_size'],'markovRank':c['dspark_markov_rank'],'noiseTokenId':c['dspark_noise_token_id'],'targetLayerIds':c['dspark_target_layer_ids'],'stages':len([n for n in nodes if n['group']=='dspark']),'declaredNextNLayers':c['num_nextn_predict_layers'],'namespace':'mtp.*','limit':'HF num_nextn_predict_layers=1 与实际 3 个 DSpark stage 分开；MTP namespace 不等于独立 MTP。模型卡 num_speculative_tokens=7 与配置 block_size=5 分别记录，不静默配平。'}
  from deepseek_storage import complete_storage
  complete_storage(e,name,c,arch)
  path=f'data/families/deepseek/{mid}-architecture.json';write(ROOT/path,arch);architectures[mid]=arch
  def fact(value,kind='official',source=None,note=None):
   x={'value':value,'evidence':kind if value is not None else 'unknown','source':source or e.sources[csid]['url']}
   if note:x['note']=note
   return x
  attention=' / '.join(f'{k} {count} 层' for k,count in types.items())
  facts={k:fact(value) for k,value in {'layers':c['num_hidden_layers'],'hidden':c['hidden_size'],'heads':c['num_attention_heads'],'experts':c.get('n_routed_experts'),'topK':c.get('num_experts_per_tok'),'shared':c.get('n_shared_experts'),'expertInput':c['hidden_size'] if c.get('n_routed_experts') else None,'expertWidth':c.get('moe_intermediate_size'),'denseWidth':c.get('intermediate_size'),'context':c['max_position_embeddings'],'vocab':c['vocab_size'],'visionLayers':None,'visionHidden':None}.items()}
  facts['attention']=fact(attention,'derived');facts['quantization']=fact(precision);facts['parameters']=fact(main,'derived',note=arch['parameterScope']);facts['payload']=fact(sum(n['payloadBytes'] for n in arch['nodes'] if n['group'] not in ('mtp','dspark')),source=arch['storageAuditPath'],note='主干载荷；MTP/DSpark 独立，完整发布载荷见该 checkpoint 全分片 header')
  facts['qRank']=fact(c.get('q_lora_rank'));facts['kvRank']=fact(c.get('kv_lora_rank'));facts['streams']=fact(c.get('hc_mult',1));facts['headDim']=fact(c['qk_nope_head_dim']+c['qk_rope_head_dim'] if 'qk_nope_head_dim' in c else c.get('head_dim',c['hidden_size']//c['num_attention_heads']));facts['valueHeadDim']=fact(c.get('v_head_dim',c.get('head_dim',c['hidden_size']//c['num_attention_heads'])));facts['indexTopK']=fact(c.get('index_topk'))
  facts['mtpParameters']=fact(arch['mtpLogicalParameters'],'derived',source=arch['storageAuditPath'],note='参考构造口径；共享副本排除，上游训练别名不由 header 证明')
  facts['dsparkParameters']=fact(arch.get('dsparkLogicalParameters',0),'derived',source=arch['storageAuditPath'],note='DSpark 挂接 draft 参数独立于主干；不存在对应模块时为 0')
  facts['dsparkStages']=fact(arch.get('dspark',{}).get('stages',0),'derived',source=arch['storageAuditPath']);facts['draftBlock']=fact(c.get('dspark_block_size'));facts['markovRank']=fact(c.get('dspark_markov_rank'))
  summary=f'{c["num_hidden_layers"]} 层；{attention}；'+('DSpark 3-stage / Markov / confidence；主干与 draft 分开' if c.get('dspark_block_size') else 'mHC '+str(c['hc_mult'])+' 流，hash/score MoE 分层' if c['model_type']=='deepseek_v4' else 'Dense GQA 底座的 R1 蒸馏分支' if v['kind']=='distill' else 'MLA/MoE；MTP 与主干分开计数')
  branch='支线' if v['kind']=='distill' else '基础' if v['kind']=='base' else '后训练' if name.startswith('DeepSeek-R1') else '主线'
  models.append({'id':mid,'name':name,'branch':branch,'summary':summary,'tags':['长文档','通用对话']+(['推理'] if 'R1' in name else [])+(['DSpark','推测解码'] if c.get('dspark_block_size') else []),'source':v['url'],'revision':v['revision'],'releaseDate':None,
    'facts':facts,'sections':['versions','architecture','evidence','speculation','cann']+(['r1'] if 'R1' in name else ['increments'] if 'V3.2' in name or 'V4' in name else ['v3']),
    'configPath':f'data/families/deepseek/configs/{name}-config.json','architecturePath':path,'auditPath':arch['storageAuditPath'],
    'baseModel':v['baseModel'],'implementationLinks':[]})
  if name in ('DeepSeek-V3-Base','DeepSeek-R1','DeepSeek-R1-Zero'):
   diff={k:{'v3':reference.get(k),'candidate':c.get(k)} for k in set(reference)|set(c) if reference.get(k)!=c.get(k)}
   config_diffs.append({'model':name,'revision':v['revision'],'configDifferences':diff,'configByteIdentical':(DATA/'configs'/f'{name}-config.json').read_bytes()==(DATA/'configs/DeepSeek-V3-config.json').read_bytes(),
                        'referenceCodeByteIdentical':e.sources[sid]['sha256']==e.sources['hf-v3-modeling_deepseek-py']['sha256'],'limit':'配置与参考代码相同不等于权重、后训练、能力或服务条件相同'})
 family={'id':'deepseek','name':'DeepSeek','publisher':'DeepSeek','updated':DATE,'description':'V3 可复核基线、V3.2 DSA、V4 压缩注意力与 mHC、MTP/DSpark 推测解码，以及 R1/Distill 底座关系。',
   'scope':'21 个选定 checkpoint 的主干、MTP 与 DSpark，全部有独立完整文件头审计；配置/参考构造、权重存储、服务协议与设备实测分列。设备实验已按用户要求移出本次范围。',
   'models':models,'figures':[],'downloads':[],'reportPath':'data/families/deepseek/report.json','officialDirectory':'https://huggingface.co/deepseek-ai','documentCenterUrl':'#/family/deepseek/sources',
   'overview':{'title':'从 MLA、压缩注意力到 MTP/DSpark 推测解码','conclusion':'V3.2 增加 DSA indexer；V4 重建 KV 压缩、分组输出投影与 mHC，并区分 hash/score 路由。MTP 与 DSpark 的独立草稿、缓存、验证和调度分别研究；R1 后训练与蒸馏底座分开，设备兼容和性能保留具体条件。','highlights':[{'id':'v3','label':'V3 · 文件头与参考计算'},{'id':'v3.2','label':'V3.2 · DSA'},{'id':'v4-flash','label':'V4-Flash · 压缩与 mHC'},{'id':'v4-flash-dspark','label':'DSpark · 并行草稿与验证'},{'id':'r1','label':'R1 · 后训练与蒸馏'}]},
   'historyNote':'这是架构与训练关系路线，不证明所有相邻 checkpoint 直接继承权重。Max 档位是推理模式；V4.1 等后续条目仍在既定首版范围之外。',
   'auditNote':'当前 21 个 checkpoint 都审计全部分片的 header/data_offsets，各模型使用自己的实际存储；不读取权重数值，不由同形状推定同权重或共享别名。',
   'limitations':'逻辑结构不等于任意框架上的执行图；原始权重、转换精度、kernel 分派与设备实测分开记录。设备实验不在本次范围，不提供实测速度排名。',
   'technology':[{'id':k,'title':title,'section':k,'summary':summary} for k,title,summary in [('v3','V3 基线','MLA、Dense/MoE、主干与 MTP 的口径和文件头审计。'),('increments','V3.2 / V4 增量','DSA、CSA/HCA、mHC、hash 路由与 FP4/FP8。'),('speculation','MTP / DSpark','独立模块、Markov/confidence、draft/verify/update 与调度边界。'),('r1','R1 / Distill','相同底座、配置、完整 tokenizer/BPE 与后训练分开。'),('evidence','证据与缺口','源码存在、转换支持、运行正确性与性能分开。')]],
   'scenarios':[{'title':'理解长上下文计算','text':'对比 V3 MLA、V3.2 indexer 与 V4 的窗口/压缩路径。','models':['v3','v3.2','v4-flash','v4-pro'],'boundary':'配置上限和缓存公式不是实测质量或服务容量。'},{'title':'研究 R1 部署','text':'区分 V3-Base 后训练主干与 Qwen/Llama 蒸馏底座。','models':['r1','r1-zero','r1-distill-qwen-7b','r1-distill-llama-8b'],'boundary':'同名系列不能统一套用 MLA/MoE，也不能以参数规模推断吞吐。'}]}
 family['technology'].append({'id':'cann','title':'CANN 开源与 SDK','section':'cann','summary':'匹配版本的开源算子、op-plugin bridge 与本地安装资料；源码、库、ABI 与设备验证分开。'})
 family['acceptanceScope']={'mode':'static_research_and_website','deviceExperimentsRequired':False,'deviceExperiments':'excluded_from_current_scope_by_user','futureExperiments':'optional_separate_request','reason':'当前没有硬件；本次验收不要求设备运行、性能或数值实测'}
 write(DATA/'research/config-relations.json',config_diffs)
 from deepseek_speculation import build_speculation
 speculation=build_speculation(e,family,architectures)
 from deepseek_backend_trace import build_trace
 trace=build_trace(e)
 from deepseek_cann import build_cann
 cann=build_cann(e)
 from deepseek_figures import build_figures
 family['figures']=build_figures(family,architectures)
 make_report(family,architectures,e,audit)
 family['downloads'].append({'name':'DeepSeek-研究报告.md','path':'assets/deepseek/DeepSeek-研究报告.md','bytes':(ASSETS/'DeepSeek-研究报告.md').stat().st_size,'sha256':hashlib.sha256((ASSETS/'DeepSeek-研究报告.md').read_bytes()).hexdigest()})
 from deepseek_compute import build_compute
 compute=build_compute(e,family,architectures)
 from deepseek_hardware import build_hardware
 hardware=build_hardware(e,family,compute)
 family['hardwarePath']='data/families/deepseek/hardware.json'
 for path in hardware['downloads'].values():
  family['downloads'].append({'name':path.rsplit('/',1)[-1],'path':path,'bytes':0,'sha256':'0'*64})
 write(DATA/'hardware.json',hardware)
 family['computePath']='data/families/deepseek/compute.json'
 for m in family['models']:m['computePath']='#/family/deepseek/compute/'+m['id']
 write(DATA/'compute.json',compute)
 write(DATA/'family.json',family)
 write(DATA/'research/site-proofs.json',{'sources':e.sources,'proofs':e.proofs})
 if (DATA/'analysis-sources.json').is_file():
  from build_deepseek_analysis import build as build_analysis
  build_analysis()
  family=read(DATA/'family.json')
  make_report(family,architectures,e,audit)
 ASSETS.mkdir(parents=True,exist_ok=True)
 write(ASSETS/'v3-header-audit.json',audit)
 catalog=read(ROOT/'data/catalog.json');catalog['families']=[f for f in catalog['families'] if f['id']!='deepseek']+[{'id':'deepseek','name':'DeepSeek','publisher':'DeepSeek','description':family['description'],'path':'data/families/deepseek/family.json','updated':DATE,'modelCount':len(models),'figureCount':len(family['figures']),'topics':['MLA / DSA / CSA / HCA','R1 与 Distill','Ascend 源码研究'],'counts':[str(len(models))+' 个代表 checkpoint','全部 checkpoint 文件头审计','MTP / DSpark 与服务协议']}];write(ROOT/'data/catalog.json',catalog)
 print('DeepSeek architectures:',len(architectures),'source proofs:',len(e.proofs))
 return e,family,architectures

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--research-root',type=Path,help='Restored public input cache with round1/, extended/, backends/, followup/ and cann/');args=p.parse_args()
 if args.research_root:
  CACHES.update({name:args.research_root/group for name,group in {'sources.json':'round1','extended-sources.json':'extended','backend-sources.json':'backends','followup-sources.json':'followup','cann-sources.json':'cann'}.items()})
 build()
