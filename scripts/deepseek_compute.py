"""Build source-grounded mathematical step templates; deployment dispatch is a separate evidence layer."""
import collections
import copy
import hashlib
import json
import math


def build_compute(e,family,architectures):
 d={'schemaVersion':1,'familyId':'deepseek','snapshot':'2026-10-01','authored':'2026-10-01',
    'symbols':{'B':'padded 参考请求 batch','L_q':'本次新增/query token 数；常规 decode=1','L_kv':'历史+本次可见长度',
      'N_tok':'本轮 padded 参考路径 B×L_q；不冒称 ragged token 数','N_e':'代表专家实际收到的 token 数；无丢弃时 ΣN_e=N_tok×K',
      'C':'按实际层压缩比 floor(L_kv/r) 的已完成记录数；窗口和状态另计','K_idx':'min(index_topk,可见索引记录数)',
      'r':'该层真实压缩比 4 或 128；r=0 是纯窗口层','H':'单条 residual stream 宽度；mHC stream 轴单列',
      'MP':'本数学视图 MP=1；实际 TP/EP/DP 分片与通信由具体框架记录','{i}':'0-based 层号；{e} 专家号'},
    'notes':['全主干层通过模板展开；参数与存储字段引用同一 architecture。阶段重复引用同一份权重，不能把各行参数计数累加成模型总参数。',
      '逻辑步骤用于解释投影、状态与 attention 数据流，框架可以融合/重排；reference 是函数上下文，不是实测有序 kernel 时序。',
      '当前范围每个 checkpoint 都有独立全分片 header 审计；逻辑形状与 FP4/I8 打包、scale、共享副本分别记录，header 不证明权重值或加载共享。',
      '设备实验已移出本次范围；没有模型加载、GPU/NPU 数值或性能实验。duration、利用率与 profiler 指标保持未知。',
      'GQA 使用固定 Transformers eager 参考的数学关系；其配置声明的库版本与该源码快照不等同已测试兼容栈。'],
    'phaseOptions':['prefill','decode'],'pathOptions':[],'models':[],'templates':{},'proofs':{},'downloads':[]}
 def source_for(model):
  name=model['name'];kind=architectures[model['id']]['nodes'][0]['type']
  if name.startswith('DeepSeek-V4'):return e.select(name.replace('-Base',''),'inference/model.py'),'Attention.forward'
  if 'V3.2'in name:return e.select(name,'inference/model.py'),'MLA.forward'
  if 'Distill'in name:return ('hf-transformers-modeling_qwen2-py','Qwen2Attention.forward') if 'Qwen'in name else ('hf-transformers-modeling_llama-py','LlamaAttention.forward')
  return e.select(name,'modeling_deepseek.py'), 'DeepseekV2Attention.forward' if name=='DeepSeek-V2' else 'DeepseekV3Attention.forward'
 def add(t,phase,path,sid,symbol,sid_step,title,inputs,outputs,relation,tensors=None,op='descriptive',extra=None,note=''):
  pid=e.proof(sid,symbol) if symbol else None
  s={'id':phase+'-'+path+'-'+sid_step,'title':title,'phase':phase,'path':path,
     'input':'; '.join(k+'=['+','.join(map(str,v))+']' for k,v in inputs.items()),
     'output':'; '.join(k+'=['+','.join(map(str,v))+']' for k,v in outputs.items()),
     'inputDimensions':inputs,'outputDimensions':outputs,'op':op,'relation':relation,'intermediate':'按本步骤的分支与层配置；cache/状态不合并计数',
     'dtype':'激活/中间精度见固定参考函数；存储类型只由文件头证据填写',
     'tensors':tensors or [],'parameters':sum(x['logical_parameters_each']*x['multiplicity'] for x in (tensors or [])),
     'reference':{'chain':symbol+'（参考函数上下文）' if symbol else '固定文件头摘要：'+architectures[t['modelId']]['storageAuditPath'],'proofs':[pid] if pid else []},'note':note,
     'apis':{'torch':{'level':'参考函数 / 数学关系' if symbol else '存储元数据；没有 Torch 执行映射','evidence':'source' if symbol else 'unknown','chain':symbol or 'unknown','condition':'固定配置；本视图 MP=1；未加载 checkpoint','proofs':[pid] if pid else []},
             **{k:{'level':'未核验该单步设备映射','evidence':'unknown','chain':'unknown','condition':'生产模块/分派的条件证据另见实现专题；不猜测单步 kernel','proofs':[]} for k in ('nvidia','amd','ascend')}}}
  if extra:s.update(extra)
  t['steps'].append(s)
 def tensor(m):
  return {'tensor_template':m['tensor_template'],'logical_shape':'x'.join(map(str,m['logical_shape'])),
          'stored_shape':'x'.join(map(str,m['stored_shape'])) if m['stored_shape'] is not None else None,'stored_dtype':m['stored_dtype'],
          'logical_parameters_each':m['logical_parameters_each'],'multiplicity':m['count'],'role':m['role'],'proof':m.get('proof')}
 for model in family['models']:
  mid=model['id'];a=architectures[mid];sid,attn_forward=source_for(model)
  cfg=a['config'];H=cfg['hidden'];NH=cfg['heads'];native='V4'in model['name'];dsa='V3.2'in model['name'];gqa='Distill'in model['name']
  cm={'id':mid,'name':model['name'],'symbols':{'H':H,'N_head':NH,'E':cfg.get('experts'),'K':cfg.get('topK'),'streams':cfg.get('streams',1)},
      'weightEvidence':f'本 checkpoint 全分片 header：{a["headerTensorCount"]:,} 张量；逻辑主干、MTP/DSpark、共享副本和量化元数据分开。未读取权重值或运行模型。',
      'nodes':[]}
  for n in a['nodes']:
   signature=[n['group'],n['type'],n['ffn'],n.get('routing'),n.get('compressRatio'),[(m['id'],[(x['tensor_template'],x['logical_shape'],x['count'],x['stored_dtype']) for x in m['matrices']]) for m in n['modules']]]
   tid=mid+'-'+hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
   cn={'id':n['id'],'layer':n['number'],'sourceIndex':n.get('sourceIndex',n['number']-1 if n['group']=='decoder' else 0 if n['group']=='mtp' else None),'group':n['group'],'type':n['type'],'ffn':n['ffn'],'template':tid,'parameters':n['parameters'],'weightEvidence':cm['weightEvidence']}
   cm['nodes'].append(cn)
   if tid in d['templates']:continue
   t={'id':tid,'modelId':mid,'layerType':n['type'],'ffn':n['ffn'],'steps':[],'parameterScope':a['parameterScope']}
   for phase in ('prefill','decode'):
    path=n['type'].lower() if native or gqa else 'dsa-'+phase if dsa else 'mla-expanded-reference'
    if path not in d['pathOptions']:d['pathOptions'].append(path)
    # Parameter declarations are attached once per phase, including separate metadata rows.
    for mod in n['modules']:
     for j,m in enumerate(mod['matrices']):
      dims=m['logical_shape'];pid=m.get('proof');proof=e.proofs.get(pid);psid=proof['source_id'] if proof else sid;symbol=proof['symbol'] if proof else None if m.get('storageEvidence') else attn_forward
      tensors=[tensor(m)];mi=mod['id']+'-'+str(j);name=m['tensor_template']
      if m['role'] in ('quantization_metadata','index_table','shared_alias'):
       add(t,phase,path,psid,symbol,mi,'元数据 / 地址表：'+name,{'metadata':dims},{'metadata':dims},'存储/索引记录，不计入浮点逻辑权重参数',tensors,op='identity');continue
      if native and n['group']=='dspark' and name.endswith('markov_w1.weight'):
       width=read_config(model)['dspark_block_size'];add(t,phase,path,psid,symbol,mi,'Markov token embedding 查表',{'ids':['B',width]},{'embed':['B',width,dims[1]]},'按前一个已选 token 查 [V,R]；这是 lookup，不是将完整 V×R 权重作 dense GEMM',tensors,op='embedding_lookup');continue
      if len(dims)==2:
       outputs={'y':['B','L_q',dims[0]]};inputs={'x':['B','L_q',dims[1]],'W':dims};op='linear';extra=None
       relation='y=x @ W^T；权重按[输出,输入]；专家模板另用 N_e 动态 token 数'
       if mod['id']=='routed':inputs['x']=['N_e',dims[1]];outputs['y']=['N_e',dims[0]]
       if mod['id']=='router' or mod['id']=='shared' and (native or dsa):inputs['x']=['N_tok',dims[1]];outputs['y']=['N_tok',dims[0]]
       if native and n['group']=='mtp' and name.endswith('h_proj.weight'):
        streams=read_config(model)['hc_mult'];inputs['x']=['B','L_q',streams,dims[1]];outputs['y']=['B','L_q',streams,dims[0]]
       if native and name.endswith('wo_a.weight'):
        G=read_config(model)['o_groups'];O=read_config(model)['o_lora_rank']
        inputs={'x':['B','L_q',G,dims[1]],'W':[G,O,dims[1]]};outputs={'y':['B','L_q',G,O]};op='einsum';extra={'einsum':'bqgd,grd->bqgr'};relation='按 G 组低秩输出：einsum(bqgd,grd->bqgr)，W 由 [G*R_o,D_in] reshape'
       if native and mod['id']=='residual':relation='mHC flatten 4H 的控制投影；输出随后拆为 pre/post/comb，不是普通 hidden 输出'
       add(t,phase,path,psid,symbol,mi,'逻辑投影：'+name,inputs,outputs,relation,tensors,op=op,extra=extra,note=m.get('declaredFormat') or '')
      else:
       add(t,phase,path,psid,symbol,mi,'向量参数：'+name,{'parameter':dims},{'parameter':dims},'参数声明；具体 norm、bias、sink 或 mix scale 计算由后续数据流说明',tensors,op='identity')
    if n['group']=='mtp' and not native:
     msid=next(k for k,s in e.sources.items() if s.get('repo')=='vllm-project/vllm' and s.get('path')=='vllm/model_executor/models/deepseek_mtp.py')
     add(t,phase,'mtp-reference',msid,'DeepSeekMultiTokenPredictorLayer.forward','mtp-input','MTP 独立 forward 与隐藏态回收',{'hidden':['N_tok',H],'next_embedding':['N_tok',H]},{'logits_hidden':['N_tok',H],'recycled_hidden':['N_tok',H]},'position=0 embedding 置零；enorm/hnorm → concat 2H/eh_proj → MTP DecoderLayer → residual merge；SharedHead.norm 分别用于 logits 与回收态，不重复两次 norm',note=n.get('parameterScope','按固定后端独立 MTP 参考'))
     add(t,phase,'mtp-reference',msid,'DeepSeekMultiTokenPredictor.compute_logits','mtp-logits','MTP 共享输出 head',{'hidden':['N_tok',H]},{'logits':['N_tok',read_config(model)['vocab_size']]},'step_idx % num_mtp_layers 选层；shared_head.norm → logits_processor(shared head)；共享 embedding/head 存储副本不重复计逻辑参数')
    elif n['group']=='dspark':
     c=read_config(model);K=c['dspark_block_size'];R=c['dspark_markov_rank'];D=c['head_dim'];F=len(c['dspark_target_layer_ids'])
     if n['sourceIndex']==0:add(t,phase,'dspark-context',sid,'DSparkBlock.forward_embed','context','目标特征融合与 noisy draft 输入',{'target_features':['B','L_q',F*H],'last_token':['B']},{'main_x':['B','L_q',H],'draft_streams':['B',K,c['hc_mult'],H]},'target 选层 hc_head 折叠后 concat；main_proj/main_norm；block首位是已知 token，其余填 noise token；共享 embed 后扩为四流',note='本 checkpoint block_size=5；模型卡 vLLM num_speculative_tokens=7 是独立服务设置；不能把 n_nextn=1 当作只有一层 DSpark')
     add(t,phase,'dspark-context' if phase=='prefill' else 'dspark-draft',sid,'DSparkAttention.forward','context-kv','独立上下文 KV 与 block 非因果 attention',{'main_x':['B','L_q',H],'draft':['B',K,H]},{'context_cache':['B',c['sliding_window'],D],'draft_out':['B',K,H]},'prefill只投影/写context KV并返回x；decode将rolling context KV与全部draft KV拼接，block内允许非因果，经过Q/输出分组低秩与RoPE',note='context SWA cache不与target主干cache合并；prefill不执行完整draft Block，前列矩阵行是参数声明')
     if n['sourceIndex']==2:
      add(t,phase,'dspark-draft',sid,'DSparkBlock.forward_head','markov','轻量顺序 Markov logits 修正与 confidence',{'streams':['B',K,c['hc_mult'],H],'previous_token':['B']},{'token_ids':['B',K+1],'logits':['B',K,c['vocab_size']],'confidence_logits':['B',K]},'hc_head/norm/共享LM head形成并行基准logits；依次取前token的W1 embedding，经W2生成bias并加到对应位置，再sample；confidence线性投影[hidden,markov_embed]',note='native返回confidence logits；sigmoid/校准/验证调度在生成/服务侧，不能直接当作prefix survival概率')
    elif n['group']=='decoder' or n['group']=='mtp':
     X=['B','L_q',H]
     if native:
      c=read_config(model);S=c['hc_mult'];D=c['head_dim'];r=n['compressRatio'];G=c['o_groups'];Ro=c['o_lora_rank']
      add(t,phase,path,sid,'Block.hc_pre','hc-pre','mHC 子层前折叠',{'streams':['B','L_q',S,H]},{'x':X,'post':['B','L_q',S],'comb':['B','L_q',S,S]},'FP32 RMS statistic + F.linear 控制投影 → hc_split_sinkhorn → pre 加权折叠；attn/ffn 各自一套参数',op='hc_pre')
      add(t,phase,path,sid,'Attention.forward','qkv','分头 Q 与共享 KV',{'x':X},{'Q':['B','L_q',NH,D],'KV':['B','L_q',D]},'Q low-rank→逐头 RMS→RoPE；共享 KV norm/RoPE，NoPE 部分做 FP8 精度模拟',note='位置子维=64；head dim=512；不是 V3 的低秩 KV 512+64 缓存')
      if r:
       C='L_kv//'+str(r);coff=2 if r==4 else 1
       add(t,phase,path,sid,'Compressor.forward','compress','learned gated pooling 与增量状态',{'x':X},{'cache':['B',C,D],'kv_state':['B',coff*r,coff*D],'score_state':['B',coff*r,coff*D]},'两个 FP32 投影 + ape → 按压缩窗口 softmax pooling → norm/RoPE；decode 只在压缩边界写新记录',note='r=4 使用 overlap/coff=2；未完成窗口不生成压缩 token，状态与历史 cache 分开')
      if r==4:
       C='L_kv//4';ki='min('+str(c['index_topk'])+','+C+')'
       add(t,phase,path,sid,'Indexer.forward','index','压缩 indexer 打分与选择',{'Q':['B','L_q',c['index_n_heads'],c['index_head_dim']],'K':['B',C,c['index_head_dim']]},{'score':['B','L_q',C],'indices':['B','L_q',ki]},'dot → ReLU → 每头 weights 加权求和 → causal mask → top-k；FP4 QAT 模拟与实际缓存格式分开')
      slots='min(L_kv,'+str(c['sliding_window'])+')'+(' + min('+str(c['index_topk'])+',L_kv//4)' if r==4 else ' + L_kv//128' if r==128 else '')
      add(t,phase,path,sid,'Attention.forward','sparse-attn','窗口与压缩 KV 稀疏 attention',{'Q':['B','L_q',NH,D],'KV':['B','L_kv',D],'indices':['B','L_q',slots]},{'out':['B','L_q',NH,D]},'sparse_attn(Q,KV,attn_sink,topk_indices,scale)；prefill KV 包含本批+已完成压缩，decode 使用 rolling-window + prefix cache',note='ratio=128 无 learned indexer；无效位置为 -1；inverse RoPE 后进入 grouped O projection')
      add(t,phase,path,sid,'Gate.forward','routing','Hash / score 路由',{'x':['N_tok',H]},{'weights':['N_tok',cfg['topK']],'ids':['N_tok',cfg['topK']]},'FP32 gating → sqrt(softplus)；前三层 ids=tid2eid[input_ids]，其余 bias+top-k；最终权重取原分数归一化并缩放',note='hash 表不是 Engram 记忆模块；hash 层仍计算 gating scores')
      add(t,phase,path,sid,'Expert.forward','expert-activation','专家 clamp/SwiGLU/路由权重',{'gate':['N_e',c['moe_intermediate_size']],'up':['N_e',c['moe_intermediate_size']]},{'act':['N_e',c['moe_intermediate_size']]},'up clamp[-limit,limit]、gate clamp(max=limit)；SiLU(gate)*up；routed weight 在 down 投影前相乘',op='elementwise')
      add(t,phase,path,sid,'Block.hc_post','hc-post','mHC 子层后还原四流',{'x':X,'residual':['B','L_q',S,H],'post':['B','L_q',S],'comb':['B','L_q',S,S]},{'out':['B','L_q',S,H]},'post⊗x + Σ_i comb[i,j] residual[i]；attention/FFN 两处分别执行',op='hc_post')
      if n['group']=='mtp':add(t,phase,path,sid,'MTPBlock.forward','mtp-input','MTP 共享 embedding 与输入组合',{'hidden':['B','L_q',S,H],'embedding':X},{'combined':['B','L_q',S,H]},'enorm/e_proj(embedding).unsqueeze(2) + h_proj(hnorm(hidden)) → inherited Block → 独立 norm/hc_head，共享 output head',note='MTP 单独调用；普通 Transformer.forward 不执行该模块')
     elif gqa:
      c=read_config(model);NK=c['num_key_value_heads'];D=H//NH
      add(t,phase,path,sid,attn_forward,'gqa-scores','GQA Key/Value 广播与 attention',{'Q':['B',NH,'L_q',D],'K_cache':['B',NK,'L_kv',D],'V_cache':['B',NK,'L_kv',D]},{'scores':['B',NH,'L_q','L_kv'],'out':['B',NH,'L_q',D]},'RoPE → cache.update → repeat_kv(N_head/N_KV) → QK^T/√D → mask/FP32 softmax → PV',note='参考 eager 数学路径；实际 SDPA/FlashAttention、分页与设备分派另核验')
      add(t,phase,path,sid,'Qwen2MLP.forward' if 'Qwen'in model['name'] else 'LlamaMLP.forward','swiglu','Dense SwiGLU',{'gate':['N_tok',c['intermediate_size']],'up':['N_tok',c['intermediate_size']]},{'act':['N_tok',c['intermediate_size']]},'SiLU(gate)*up → down；两次 pre-norm residual 分开',op='elementwise')
     else:
      c=read_config(model);Q=c['q_lora_rank'];R=c['kv_lora_rank'];DN=c['qk_nope_head_dim'];DR=c['qk_rope_head_dim'];DV=c['v_head_dim']
      if dsa:
       HI=c['index_n_heads'];DI=c['index_head_dim'];ki='min('+str(c['index_topk'])+',L_kv)'
       add(t,phase,path,sid,'Indexer.forward','index','Lightning Indexer',{'Q':['B','L_q',HI,DI],'K_cache':['B','L_kv',DI]},{'score':['B','L_q','L_kv'],'indices':['B','L_q',ki]},'独立 RoPE/旋转 → FP8 quant → fp8_index → causal mask/top-k；64 heads 加权 ReLU dot；保留 F32 scale cache',note='参考代码含 dist.broadcast 对照，不能未经进程组初始化就宣称可单进程运行')
       if phase=='decode':
        add(t,phase,path,sid,attn_forward,'absorb','DSA 吸收式 decode attention',{'q_nope':['B','L_q',NH,DN],'Wk':[NH,DN,R],'C':['B','L_kv',R]},{'q_abs':['B','L_q',NH,R],'score':['B','L_q',NH,'L_kv']},'Q_nope·W_K → 与潜变量/位置 cache 打分 → index mask → softmax → 先聚合 C 再乘 W_V',note='reference 主 KV 量化后回到默认 dtype；解量化 Wkv_b 可按分支缓存')
       else:add(t,phase,path,sid,attn_forward,'expanded','DSA 展开式 prefill attention',{'Q':['B','L_q',NH,DN+DR],'K':['B','L_q',NH,DN+DR],'V':['B','L_q',NH,DV]},{'score':['B','L_q',NH,'L_q'],'out':['B','L_q',NH,DV]},'初始 causal prefill → QK 分数 + index mask → softmax → PV；KV/cache 与 indexer cache 分开')
      else:
       add(t,phase,path,sid,attn_forward,'mla','参考展开式 MLA attention',{'Q':['B',NH,'L_q',DN+DR],'K':['B',NH,'L_kv',DN+DR],'V':['B',NH,'L_kv',DV]},{'score':['B',NH,'L_q','L_kv'],'out':['B',NH,'L_q',DV]},'Q/K NoPE+RoPE 分量拼接；cache.update → QK^T → mask/FP32 softmax → PV',note='这是所选 HF eager 参考路径；吸收式数学等价与 demo 样例见首轮包，不把 demo 当作所有 checkpoint 的运行栈')
      ff='MLP.forward' if dsa and n['ffn']=='Dense' else 'MoE.forward' if dsa else ('DeepseekV2' if mid=='v2' else 'DeepseekV3')+('MLP.forward' if n['ffn']=='Dense' else 'MoE.forward')
      gate_note=''
      if n['ffn']=='MoE':
       if mid=='v2':gate_note=f'V2：FP32 softmax；{c["n_group"]} 组按组内最大分数选择 {c["topk_group"]} 组，再选 {c["num_experts_per_tok"]} 专家；norm_topk_prob=false，原分数乘 {c["routed_scaling_factor"]}。'
       else:gate_note=f'V3 系：FP32 sigmoid；bias 只影响选择，{c["n_group"]} 组以组内两个最大校正分数之和选 {c["topk_group"]} 组；原始分数 gather 后归一化，再乘 {c["routed_scaling_factor"]}。'
      add(t,phase,path,sid,ff,'ffn','Dense / MoE 与残差',{'x':['N_tok',H]},{'out':['N_tok',H]},'Dense 或 routing→专家 token gather/重排→SiLU(gate)*up→down→按原路由权重 combine + shared；block 两处 residual 分开',note=gate_note+'代表专家 token 数 N_e 动态；无丢弃/无padding时 ΣN_e=N_tok*K；EP 通信只按所选源码分支解释')
      if n['ffn']=='MoE':
       gate_symbol='Gate.forward' if dsa else 'MoEGate.forward';gate_proof=e.proof(sid,gate_symbol);s=t['steps'][-1]
       s['reference']['proofs'].append(gate_proof);s['reference']['chain']+='；'+gate_symbol+'（路由条件上下文）'
       s['apis']['torch']['proofs'].append(gate_proof)
    elif n['id']=='embedding':
     symbol='ParallelEmbedding.forward' if native or dsa else ('Qwen2Model.forward' if 'Qwen'in model['name'] else 'LlamaModel.forward') if gqa else 'DeepseekV2Model.forward' if mid=='v2' else 'DeepseekV3Model.forward'
     add(t,phase,path,sid,symbol,'embedding','Token embedding lookup',{'ids':['B','L_q']},{'hidden':['B','L_q',H]},'embedding[input_ids]；若 mHC 再增加 4-stream 轴，不新增 embedding 参数')
    elif n['id']=='output-head':
     last=native or dsa
     symbol=('ParallelHead.forward' if read_config(model).get('dspark_block_size') else 'ParallelHead.get_logits') if native else 'Transformer.forward' if dsa else ('Qwen2ForCausalLM.forward' if 'Qwen'in model['name'] else 'LlamaForCausalLM.forward') if gqa else 'DeepseekV2ForCausalLM.forward' if mid=='v2' else 'DeepseekV3ForCausalLM.forward'
     add(t,phase,path,sid,symbol,'logits','输出 logits',{'hidden':['B',H] if last else ['B','L_q',H]},{'logits':['B',read_config(model)['vocab_size']] if last else ['B','L_q',read_config(model)['vocab_size']]},'所选参考 demo 取最后 token logits' if last else '所选 HF eager 参考返回各 query token logits；服务框架的裁剪另核验')
   d['templates'][tid]=t
  d['models'].append(cm)
 d['proofs']=copy.deepcopy(e.proofs)
 d['pathOptions']=list(dict.fromkeys(s['path'] for t in d['templates'].values() for s in t['steps']))
 d['counts']={'languageLayers':sum(n['group']=='decoder' for m in d['models'] for n in m['nodes']),
              'visionLayers':0,'globalComponents':sum(n['group']!='decoder' for m in d['models'] for n in m['nodes']),
              'templates':len(d['templates']),'steps':sum(len(d['templates'][n['template']]['steps']) for m in d['models'] for n in m['nodes']),
              'proofs':len(d['proofs'])}
 return d

def read_config(model):
 from pathlib import Path
 return json.loads((Path(__file__).resolve().parents[1]/model['configPath']).read_text())
