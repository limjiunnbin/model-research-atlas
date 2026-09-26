"""Author the pinned compute atlas from audited shapes and inspected source proofs.

Run with --research-root pointing to the local, read-only research snapshot.
Normal website build consumes the generated JSON; it does not fetch sources.
"""
import argparse,collections,csv,gzip,hashlib,io,json,math,re,zipfile
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument('--research-root',type=Path,required=True);args=P.parse_args()
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/families/kimi';A=ROOT/'dist/assets/kimi/compute';A.mkdir(parents=True,exist_ok=True)
family=json.loads((D/'family.json').read_text());rawproof=json.loads((args.research_root/'proofs.json').read_text());proofs={}
def proof(file,symbol,calls=None):
 matches=[p for p in rawproof if file in p['file'] and p['name']==symbol]
 assert len(matches)==1,(file,symbol,len(matches))
 p=matches[0];calls=calls or []
 assert all(c in p['calls'] for c in calls),(file,symbol,calls)
 key=hashlib.sha256((p['file']+p['name']).encode()).hexdigest()[:12]
 proofs[key]={k:p[k] for k in ['repo','revision','path','url','line','end','sourceSha256','functionSha256']}
 proofs[key].update(symbol=symbol,verifiedCalls=calls or p['calls'],verification='AST函数定义+函数体调用核对',snapshot='2026-09-21')
 return key
def api(file,symbol,calls=None,condition='',level='框架函数 / 条件分派'):
 key=proof(file,symbol,calls); shown=calls or []; shown=[('torch.nn.functional.'+c[2:] if c.startswith('F.') else 'torch.'+c if c.startswith('nn.') else c) for c in shown]; return dict(chain=symbol+(' → '+' / '.join(shown) if shown else '（函数上下文，不是本步骤的有序调用链）'),level=level,condition=condition or '仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。',proofs=[key],evidence='源码函数及调用已核验')
unknown=lambda why:dict(chain='未核验',level='未知',condition=why,proofs=[],evidence='未知；不由数学表达式猜测设备API')
AP={}
AP['torch-linear']=api('pytorch--torch__nn__modules__linear.py','Linear.forward',['F.linear'],level='torch.nn.Linear → torch.nn.functional.linear')
AP['torch-embed']=api('pytorch--torch__nn__modules__sparse.py','Embedding.forward',['F.embedding'],level='torch.nn.Embedding → torch.nn.functional.embedding')
for k,file,symbol,calls in [
 ('gpu-linear','vllm--vllm__model_executor__layers__linear.py','ColumnParallelLinear.forward',['self.quant_method.apply']),
 ('gpu-row-linear','vllm--vllm__model_executor__layers__linear.py','RowParallelLinear.forward',['self.quant_method.apply','tensor_model_parallel_all_reduce']),
 ('gpu-fp8','__quantization__fp8.py','Fp8LinearMethod.apply',['self.fp8_linear.apply_weights']),
 ('gpu-int4','compressed_tensors_wNa16.py','CompressedTensorsWNA16.apply_weights',['self.kernel.apply_weights']),
 ('gpu-rms','vllm--vllm__model_executor__layers__layernorm.py','RMSNorm.forward_native',['ir.ops.rms_norm','ir.ops.fused_add_rms_norm.maybe_inplace']),
 ('gpu-embed','vocab_parallel_embedding.py','VocabParallelEmbedding.forward',['self.quant_method.embedding','tensor_model_parallel_all_reduce']),
 ('gpu-moe','vllm--vllm__model_executor__layers__fused_moe__fused_moe.py','fused_experts_impl',['dispatch_fused_moe_kernel','ops.moe_sum']),
 ('cuda-mxfp4','trtllm_mxfp4_moe.py','TrtLlmMxfp4ExpertsMonolithic.apply',['trtllm_fp4_block_scale_moe']),
 ('amd-mxfp4','aiter_mxfp4_w4a16_moe.py','aiter_triton_kernel_w4a16_moe_forward',['aiter_routing','moe_gemm_a16w4']),
 ('cuda-mla','__flashinfer_mla.py','FlashInferMLAImpl.forward_mqa',['trtllm_batch_decode_with_kv_cache_mla']),
 ('amd-mla','__rocm_aiter_mla.py','AiterMLAImpl.forward_mqa',['rocm_aiter_ops.mla_decode_fwd','mla_gluon']),
 ('cuda-kda','__nvidia__kda.py','KimiK3DeltaAttention._forward',['flashinfer_fused_kda_decode','_flashkda_prefill','fused_recurrent_kda']),
 ('amd-kda','__amd__kda.py','KimiK3DeltaAttention._forward',['ops.fused_kda_decode','chunk_kda_prefill','fused_recurrent_kda']),
 ('cuda-attnres','__nvidia__ops__attn_res.py','attn_res',['ops.kimi_k3_attn_res','_attn_res_kernel[num_tokens,]']),
 ('gpu-vision','__kimi_k25_vit.py','MoonViTEncoderLayer.attention_qkvpacked',['self.wqkv','self.attn','self.wo']),
 ('gpu-projector','__kimi_k25_vit.py','KimiK25MultiModalProjector.forward',['self.linear_1','self.linear_2']),
 ('asc-linear','vllm_ascend__ops__linear.py','AscendUnquantizedLinearMethod.apply',['torch.ops.vllm.unquantized_gemm']),
 ('asc-linear-body','vllm_ascend__ops__linear.py','unquantized_gemm',['torch.nn.functional.linear']),
 ('asc-rms','vllm_ascend__ops__layernorm.py','AscendRMSNorm.forward_oot',['torch_npu.npu_rms_norm','torch.ops._C_ascend.npu_add_rms_norm_bias']),
 ('asc-silu','vllm_ascend__ops__activation.py','AscendSiluAndMul.forward_oot',['torch_npu.npu_swiglu']),
 ('asc-rope','vllm_ascend__ops__rotary_embedding.py','AscendDeepseekScalingRotaryEmbedding.forward',['torch.ops.vllm.npu_rotary_embedding']),
 ('asc-mla-prefill','__attention__mla_v1.py','AscendMLAImpl._forward_prefill',['torch_npu.npu_fused_infer_attention_score']),
 ('asc-mla-decode','__attention__mla_v1.py','AscendMLAImpl._forward_decode',['torch_npu.npu_fused_infer_attention_score_v2']),
 ('asc-mla-cache','__attention__mla_v1.py','AscendMLAImpl.exec_kv_decode',['torch_npu.npu_kv_rmsnorm_rope_cache']),
 ('asc-kda-prefill','vllm_ascend__ops__kda.py','run_chunk_kda',['torch.ops._C_ascend.chunk_kda_fwd','l2norm_fwd']),
 ('asc-kda-decode','vllm_ascend__ops__kda.py','run_recurrent_kda',['torch.ops._C_ascend.recurrent_kda']),
 ('asc-conv','vllm_ascend__ops__kimi_kda.py','AscendKimiK3DeltaAttention._run_causal_conv1d',None),
 ('asc-router','grouped_topk_router.py','AscendGroupedTopKRouter._compute_routing',['torch.topk']),
 ('asc-dispatch','token_dispatcher.py','TokenDispatcherWithMC2.token_dispatch',['torch_npu.npu_moe_distribute_dispatch_v2']),
 ('asc-combine','token_dispatcher.py','TokenDispatcherWithMC2.token_combine',['torch_npu.npu_moe_distribute_combine_v2']),
 ('asc-gmm','__w4a8__w4a8.py','AscendW4A8DynamicFusedMoEMethod.apply_gmm1',['torch_npu.npu_grouped_matmul']),
 ('asc-gmm2','__w4a8__w4a8.py','AscendW4A8DynamicFusedMoEMethod.apply_gmm2',['torch_npu.npu_grouped_matmul']),
 ('amd-mxfp4-activation','aiter_mxfp4_w4a16_moe.py','AiterW4A16ExpertsMonolithic._supports_activation',None),
 ('asc-situ','__w4a8__w4a8.py','AscendW4A8DynamicFusedMoEMethod.apply_gmm1_act_quant',['torch.ops._C_ascend.dequant_situ_quant']),
 ('asc-attnres','__triton__kimi_k3__attention_residual.py','apply_attn_res',['_apply_attn_res_kernel[num_vectorcore,]']),
 ('fla-chunk','--fla__ops__kda__chunk.py','chunk_kda',['ChunkKDAFunction.apply']),
 ('fla-recurrent','--fla__ops__kda__fused_recurrent.py','fused_recurrent_kda',['fused_recurrent_kda_fwd']),
 ('alog-loader','__nvidia__kda.py','a_log_weight_loader',['loaded_weight.narrow','default_weight_loader']),
 ('amd-kda-support','__amd__ops__kda_decode.py','is_fused_kda_decode_supported',['on_gfx950','on_gfx942']),
 ('state-layout','__mamba__mamba_utils.py','MambaStateShapeCalculator.kda_state_shape',None),
]:AP[k]=api(file,symbol,calls)
AP['asc-linear']['chain']+=' → unquantized_gemm → torch.nn.functional.linear'
AP['asc-linear']['proofs']+=AP['asc-linear-body']['proofs']
AP['asc-linear']['condition']='仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。'
AP['gpu-rms']['level']='已核验native/IR分支；非已确认设备kernel'
AP['gpu-rms']['condition']='所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。'
AP['asc-gmm']['condition']='仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。'
AP['asc-gmm2']['condition']=AP['asc-gmm']['condition']+' 本步骤使用w2下投影；apply_gmm1只对应w1。'
AP['asc-situ']['condition']='W4A8 SiTU分支；不是所有BF16 SiTU步骤均落到此算子。'
AP['asc-dispatch']['condition']=AP['asc-combine']['condition']='仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。'
AP['amd-kda']['condition']='融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。'
AP['amd-kda']['proofs']+=AP['amd-kda-support']['proofs']
AP['cuda-kda']['condition']='FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。'
AP['cuda-mla']['condition']='FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。'
AP['amd-mla']['condition']='AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。'
for k in ['cuda-mxfp4','amd-mxfp4']:AP[k]['condition']='已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。'
MODELS=[];TEMPLATES={};SHAPECHECK=[];WARNINGS=[]
def shape(a):return '['+','.join(map(str,a))+']'
def dims(s):return [int(v) for v in s.split('x')] if s!='metadata' else []
def clone(x):return json.loads(json.dumps(x))
def step_span(parent,start,end,chain):
 """Evidence at statement granularity, scoped to an AST-resolved function."""
 p=proofs[parent];r=next(x for x in rawproof if x['repo']==p['repo'] and x['path']==p['path'] and x['name']==p['symbol'])
 lines=(args.research_root/r['file']).read_text().splitlines();body='\n'.join(lines[p['line']-1:p['end']]);a=body.find(start);assert a>=0,(p['symbol'],start)
 b=body.find(end,a+len(start)) if end else body.find('\n',a)
 if b<0:b=len(body)
 assert b>a
 first=p['line']+body[:a].count('\n');last=p['line']+body[:b].count('\n');id=parent+'-'+str(first)
 proofs[id]=dict(**p);proofs[id].update(symbol=p['symbol']+' / 步骤语句',line=first,end=last,url=p['url'].split('#')[0]+'#L'+str(first),verifiedCalls=[],verification='AST限定函数后，逐步骤语句区间人工核对；不从调用集合推断顺序',functionSha256=hashlib.sha256(body[a:b].encode()).hexdigest())
 proofs[id].update(parentProof=parent,startNeedle=start,endNeedle=end)
 return dict(chain=chain,level='参考实现本步骤；按源码语句顺序',condition='与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。',proofs=[id],evidence='步骤语句已核验')
def refine_torch(model,s):
 k3=model['id']=='k3';kind=s['kind'];title=s['title'];parent=s['reference']['proofs'][0]
 if kind=='combine':
  s['apis']['torch']=step_span(parent,'new_x[idxs] = outs','return final_out','索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)')
 elif kind=='mla':
  if k3:parent=proof('moonshotai--Kimi-K3--modeling_kimi_linear.py','eager_attention_forward')
  if title=='注意力得分与归一化':
   s['apis']['torch']=step_span(parent,'scores = torch.einsum' if k3 else 'attn_weights = (','out = torch.einsum' if k3 else 'attn_output = torch.matmul','torch.einsum("bhqd,bhkd->bhqk",Q,K) × scale → 加mask → torch.nn.functional.softmax(FP32).to(query.dtype) → torch.nn.functional.dropout（推理p=0）' if k3 else 'torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）')
  elif title=='对V加权汇聚':
   s['apis']['torch']=step_span(parent,'out = torch.einsum' if k3 else 'attn_output = torch.matmul','return out' if k3 else 'attn_output = self.o_proj','torch.einsum("bhqk,bhkd->bhqd",P,V) → Tensor.transpose(1,2).contiguous()；上层合并heads' if k3 else 'torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)')
 elif kind=='rms' and model['id']=='k3' and s['module']!='attention' or kind=='rms' and not k3 and s['module'] not in ['components']:
  # Keep group-specific norms separate: vision and projector use nn.RMSNorm/LayerNorm.
  p=proofs[parent]
  if p['symbol'] in ['KimiRMSNorm.forward','DeepseekV3RMSNorm.forward']:
   s['apis']['torch']['chain']=p['symbol']+'：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma'
   s['apis']['torch']['level']='参考RMSNorm源码步骤；不是单独设备API'
def reference(model,kind,group='decoder'):
 k3=model['id']=='k3';name=model['name']
 if group=='vision' or group in ['projector','vision_other']:
  file='moonshotai--'+name+'--modeling_kimi_'+('k3' if k3 else 'k25')+'.py'
  sym={'vision':'MoonViTEncoderLayer.attention_qkvpacked','projector':'PatchMergerMLPV2.forward' if k3 else 'PatchMergerMLP.forward','vision_other':'MoonVision3dPatchEmbed.forward'}[group]
  return api(file,sym,level='Moonshot参考模块forward；含多个步骤，不代表1:1设备调用')
 if k3:
  sym={'rms':'KimiRMSNorm.forward','mla':'KimiMLAAttention.forward','rope':'KimiMLAAttention.forward','kda':'KimiDeltaAttention.forward','conv':'KimiDeltaAttention.forward','router':'KimiMoEGate.forward','dispatch':'KimiSparseMoeBlock.moe_infer','combine':'KimiSparseMoeBlock.moe_infer','moe':'KimiSparseMoeBlock.moe_infer','latent':'KimiSparseMoeBlock.forward','attnres':'_apply_attn_res','activation':'SituAndMul.forward'}.get(kind,'KimiDecoderLayer.forward')
  return api('moonshotai--Kimi-K3--modeling_kimi_linear.py',sym)
 sym={'rms':'DeepseekV3RMSNorm.forward','mla':'DeepseekV3Attention.forward','rope':'apply_rotary_pos_emb','router':'MoEGate.forward','moe':'DeepseekV3MoE.moe_infer','dispatch':'DeepseekV3MoE.moe_infer','combine':'DeepseekV3MoE.moe_infer','activation':'DeepseekV3MLP.forward'}.get(kind,'DeepseekV3DecoderLayer.forward')
 return api('moonshotai--'+name+'--modeling_deepseek.py',sym)
def backends(model,kind,group,tensors,ref):
 tp=clone(AP['torch-linear'] if kind=='linear' else AP['torch-embed'] if kind=='embedding' else ref)
 if kind=='linear':
  dtype={t['stored_dtype'] for t in tensors};gpu=AP['gpu-fp8'] if 'F8_E4M3' in dtype else AP['gpu-int4'] if 'I32'in dtype else unknown('此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。');cuda=clone(gpu);amd=clone(gpu);asc=clone(AP['asc-linear']) if dtype<= {'BF16','F32'} else unknown('检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。')
 elif kind=='embedding':cuda=clone(AP['gpu-embed']);amd=clone(cuda);asc=unknown('参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。')
 elif kind=='rms':cuda=clone(AP['gpu-rms']);amd=clone(cuda);asc=clone(AP['asc-rms'])
 elif kind=='mla':cuda=clone(AP['cuda-mla']);amd=clone(AP['amd-mla']);asc=clone(AP['asc-mla-prefill']);asc['chain']+='；decode: '+AP['asc-mla-decode']['chain'];asc['proofs']+=AP['asc-mla-decode']['proofs']+AP['asc-mla-cache']['proofs']
 elif kind in ['kda','conv']:cuda=clone(AP['cuda-kda']);amd=clone(AP['amd-kda']);asc=clone(AP['asc-conv'] if kind=='conv' else AP['asc-kda-prefill']);
 elif kind in ['moe','dispatch','combine']:
  cuda=clone(AP['cuda-mxfp4'] if model['id']=='k3' else AP['gpu-moe']);amd=clone(AP['amd-mxfp4'] if model['id']=='k3' else AP['gpu-moe']);asc=clone(AP[{'moe':'asc-gmm','dispatch':'asc-dispatch','combine':'asc-combine'}[kind]])
 elif kind=='attnres':cuda=clone(AP['cuda-attnres']);amd=unknown('本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。');asc=clone(AP['asc-attnres'])
 elif kind=='router':cuda=unknown('路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。');amd=unknown('本轮未固定ROCm TopK内核选择。');asc=clone(AP['asc-router'])
 elif kind=='rope':cuda=unknown('参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。');amd=unknown('参考RoPE函数已定位；ROCm具体融合内核未逐一确认。');asc=clone(AP['asc-rope'])
 elif kind=='activation':cuda=unknown('参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。');amd=clone(cuda);asc=clone(AP['asc-situ'] if model['id']=='k3' else AP['asc-silu'])
 else:cuda=unknown('此步骤仅核验参考实现/shape；未确认CUDA底层映射。');amd=unknown('此步骤仅核验参考实现/shape；未确认ROCm底层映射。');asc=unknown('此步骤仅核验参考实现/shape；未确认Ascend底层映射。')
 if kind=='kda':asc['chain']+='；decode: '+AP['asc-kda-decode']['chain'];asc['proofs']+=AP['asc-kda-decode']['proofs'];tp['chain']+='；自定义库fla.ops.kda.chunk_kda / fused_recurrent_kda，无单个标准torch API直达';tp['proofs']+=AP['fla-chunk']['proofs']+AP['fla-recurrent']['proofs']
 if kind=='vision-attention':cuda=clone(AP['gpu-vision']);amd=clone(cuda);asc=unknown('Ascend视觉塔集成已知，但MMEncoderAttention到具体CANN内核未追踪；不借用语言MLA接口。')
 if model['id']=='k3' and kind in ['moe','dispatch','combine']:
  amd=unknown('所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。')
  amd['proofs']=AP['amd-mxfp4-activation']['proofs'];amd['evidence']='排除条件源码已核验；兼容路径未知'
 return dict(torch=tp,nvidia=cuda,amd=amd,ascend=asc)

def make_steps(model,node,cfg):
 k3=model['id']=='k3';text=cfg.get('text_config',cfg);H=text['hidden_size'];heads=text['num_attention_heads'];Dq=text['qk_nope_head_dim']+text['qk_rope_head_dim'];Dv=text['v_head_dim'];steps=[];used=set()
 tensors=[]
 for mod in node['modules']:
  for t in mod['matrices']:
   tensor=dict(**t,owner=mod['id'],multiplicity=mod['count'] if mod['representative'] else 1)
   tensor['tensor_template']=tensor['tensor_template'].replace('.experts.{i}.','.experts.{e}.')
   tensors.append(tensor)
 def take(suffix,contains=None):
  found=[t for t in tensors if t['tensor_template'] not in used and ('.'+suffix+'.' in '.'+t['tensor_template'] or t['tensor_template'].endswith('.'+suffix)) and (contains is None or contains in t['tensor_template'])]
  for t in found:used.add(t['tensor_template'])
  return found
 def emit(title,kind,inp,out,relation,ts=None,intermediate='无持久缓存',module=None,note='',refkind=None):
  ts=ts or [];ref=reference(model,refkind or kind,node['group']);id='s'+str(len(steps)+1)
  s=dict(id=id,title=title,module=module or (ts[0]['owner'] if ts else kind),kind=kind,input=inp,output=out,relation=relation,intermediate=intermediate,tensors=ts,parameters=sum(t['logical_parameters_each']*t['multiplicity'] for t in ts),reference=ref,
   phase='prefill: N=ΣT_i（padded参考为B×T）；decode: T=1,N=B；推测验证可T>1。视觉仅编码/预填充，不逐文本token重复。',
   dtype=('权重/辅助张量存储：'+'；'.join(sorted({t['stored_dtype'] for t in ts}))+'；实际激活、累加及cache dtype由运行分支决定，本轮未统一验证。I32/U8可能是量化打包容器。') if ts else '继承输入；归一化/评分通常显式FP32，详见源码；最终设备计算dtype未统一验证',
   note=note,apis=backends(model,kind,node['group'],ts,ref))
  if kind=='moe' and '下投影' in title:s['apis']['ascend']=clone(AP['asc-gmm2'])
  if k3 and kind=='linear' and any('.self_attn.o_proj.' in t['tensor_template'] for t in ts):
   s['apis']['nvidia']=clone(AP['gpu-row-linear'])
   initfile='__nvidia__kda.py' if node['type']=='KDA' else '__nvidia__mla.py';initsym='KimiK3DeltaAttention.__init__' if node['type']=='KDA' else 'MultiHeadLatentAttention.__init__'
   s['apis']['nvidia']['proofs']+=[proof(initfile,initsym,['RowParallelLinear'])]
   s['apis']['nvidia']['condition']='固定K3构造代码o_proj=RowParallelLinear；reduce_results、TP及可选融合gemm_rs_ar决定实际通信，非无条件all-reduce。'
  if k3 and kind=='activation' and (node['ffn']=='Dense' or '共享' in title):s['apis']['ascend']=unknown('Dense/共享专家SiTU不等于路由专家W4A8 apply_gmm1_act_quant；本步骤设备融合路径未核验。')
  refine_torch(model,s);steps.append(s);return s
 def weight(suffix,title=None,kind='linear',prefix=None,contains=None,note=''):
  ts=take(suffix,contains)
  if not ts:return
  w=next((t for t in ts if t['logical_shape']!='metadata' and (t['tensor_template'].endswith('.weight') or t['tensor_template'].endswith('.weight_packed'))),None)
  if not w:
   emit(title or suffix,'parameter',shape(['存储']),shape(['实现读取']), '参数/缓冲读取，无独立GEMM',ts,note=note);return
  ds=dims(w['logical_shape']);prefix=prefix or ('N_e' if w['owner']=='routed' else 'Nv' if node['group']=='vision' else 'N')
  if kind=='embedding':inp='[B,T] int token IDs';out=shape(['B','T',ds[1]]);rel='Y[b,t,:]=W[token_id[b,t],:]'
  elif len(ds)==2:
   inp=shape([prefix,ds[1]]);out=shape([prefix,ds[0]]);rel='Y=X @ Wᵀ'+(' + bias' if any(t['tensor_template'].endswith('.bias') for t in ts) else '')
   if w['owner']=='routed':kind='moe';rel+='；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时）'
  elif len(ds)==1:
   inp=out=shape([prefix,ds[0]]);kind='norm' if any(t['tensor_template'].endswith('.bias') for t in ts) else 'rms';rel='LayerNorm: (X-mean)/sqrt(var+eps)×γ+β' if kind=='norm' else 'RMSNorm: X×rsqrt(mean(X²)+eps)×γ'
  elif len(ds)==3 and 'conv1d' in w['tensor_template']:
   inp=out=shape(['B','T',ds[0]]);kind='conv';rel='逐通道因果conv1d(width=4)后SiLU；无跨通道混合'
  elif len(ds)==4:
   inp=shape(['Np',ds[1],ds[2],ds[3]]);out=shape(['Np',ds[0]]);kind='patch';rel='Conv2d(patch14×14,stride14)，展平视觉patch序列'
  else:inp=shape(['Nv',ds[-1]]);out=inp;kind='position';rel='按图像网格插值位置表，并加到patch特征；不是GEMM'
  s=emit(title or suffix,kind,inp,out,rel,ts,note=note)
  if len(ds)==2 and kind not in ['embedding','parameter']:
   s['linearCheck']={'inputLast':ds[1],'weight':[ds[0],ds[1]],'outputLast':ds[0]}
  return s
 def act(width,title='门控激活',prefix='N'):
  return emit(title,'activation',f'gate/up:[{prefix},{width}]',f'[{prefix},{width}]','SiTU = 4*tanh(gate/4)*sigmoid(gate) * (25*tanh(up/25))；beta=4, linear_beta=25（固定配置）' if k3 else 'SiLU(gate)×up',module='ffn',note='激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。')
 def residual(title):emit(title,'residual',f'residual,branch:[N,{H}]',f'[N,{H}]','Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块',module='residual')
 def attnres(which):
  ts=take(which+'_res_norm')+take(which+'_res_proj')
  emit(which+' 跨层残差读取','attnres',f'prefix:[N,{H}]; blocks:[N,R,{H}]',f'[N,{H}]','沿残差块维R归一化评分→softmax→对未归一化values加权和',ts,intermediate=f'values:[N,R+1,{H}]; scores:[N,R+1]; R由0-based层索引和block_size12决定',note='这里R是网络深度残差块数，不是历史token长度。首层无先前块时可跳过。')
 if node['group']=='decoder':
  if k3:attnres('self_attention')
  weight('input_layernorm','输入归一化')
  if node['type']=='MLA':
   for s in ['q_a_proj','q_a_layernorm','q_b_proj']:weight('self_attn.'+s,s)
   weight('self_attn.kv_a_proj_with_mqa','KV压缩投影')
   emit('拆分KV潜变量与旋转键','reshape','[N,576]','KV_c:[N,512]; Krope:[N,64]','split([kv_rank512,rope_dim64])',module='attention',refkind='mla')
   for s in ['kv_a_layernorm','kv_b_proj']:weight('self_attn.'+s,s)
   emit('拆分Q/K/V与布局变换','reshape',f'Q:[N,{heads*Dq}]; KV:[N,{heads*(128+Dv)}]; Krope:[N,64]',f'Q/K:[B,{heads},T,{Dq}]; V:[B,{heads},T,{Dv}]','view/transpose/split；Krope在head维广播，不新增权重',module='attention',refkind='mla')
   emit('RoPE / 位置处理','rope',f'Qrope:[B,{heads},T,64]; Krope:[B,1,T,64]',f'旋转后的同shape','K2参考调用apply_rotary_pos_emb；K3所读参考forward未直接调用RoPE，生产MLA wrapper中有rotary_emb',take('rotary_emb'),module='attention',note='历史inv_freq缓冲的存储长度不作为runtime旋转维度；runtime依赖config和重建/加载路径。')
   emit('注意力得分与归一化','mla',f'Q:[B,{heads},T,{Dq}]; K:[B,{heads},S,{Dq}]',f'P:[B,{heads},T,S]（数学逻辑）','P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P',intermediate=f'参考展开K/V:[B,{heads},S,{Dq}] / [B,{heads},S,{Dv}]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支',module='attention',note='prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。')
   emit('对V加权汇聚','mla',f'P:[B,{heads},T,S]; V:[B,{heads},S,{Dv}]',f'[B,T,{heads*Dv}]','O=P @ V，转置/合并heads',module='attention')
   if k3:
    weight('self_attn.g_proj','MLA输出门控投影');emit('MLA输出门控','elementwise',f'O,g:[N,{heads*Dv}]',f'[N,{heads*Dv}]','O←O×sigmoid(g)',module='attention',refkind='mla')
   weight('self_attn.o_proj','MLA输出投影')
  else:
   Pj=96*128
   for s in ['q_proj','k_proj','v_proj','q_conv1d','k_conv1d','v_conv1d','f_a_proj','f_b_proj','b_proj']:weight('self_attn.'+s,s)
   ts=take('A_log')+take('dt_bias')
   st=emit('KDA衰减与更新门','elementwise',f'raw_gate:[B,T,96,128]; beta:[B,T,96]',f'g:[B,T,96,128]; β:[B,T,96]','融合gate转换与beta sigmoid；Q/K逐head做L2归一化',ts,module='attention',refkind='kda',note='A_log存储[128]，config/参考参数预期[96]；vLLM加载器narrow按local heads取片。存储128保持审计值，不改写为96；未实测该检查点完整加载。')
   st['apis']['torch']['proofs']+=AP['alog-loader']['proofs']
   emit('KDA chunk / recurrence核心','kda','Q,K,V,g:[B,T,96,128]; beta:[B,T,96]','O:[B,T,96,128]','按key维衰减S̄=diag(exp(g))Sprev；δ=v−kᵀS̄；S=S̄+βkδᵀ；o=qᵀS（含配置scale）',intermediate='逻辑状态[B,96,128,128]；FLA transpose_state_layout=True和Ascend state_v_first=True采用V,K末轴；分片为heads/TP；vLLM合并卷积state为[B,3×12288/TP,3+num_spec]或转轴布局，窗口长度4不等于状态存储长度；FLA参考库的具体conv缓存长度未逐实现核验',module='attention',note='prefill分块与decode递归数学对应但算子不同；本配置K=V=128时shape相同仍必须注明轴含义。')
   weight('self_attn.g_proj','输出门控投影');weight('self_attn.o_norm','每head输出RMSNorm',prefix='B,T,96');emit('KDA门控输出','elementwise','normalized_O,g:[B,T,96,128]','[B,T,12288]','O←RMSNorm(O)×sigmoid(g)，再合并heads',module='attention',refkind='kda');weight('self_attn.o_proj','KDA输出投影')
  residual('注意力残差合并')
  if k3:attnres('mlp')
  weight('post_attention_layernorm','前馈前归一化')
  if node['ffn']=='Dense':
   weight('mlp.gate_proj','Dense门控投影');weight('mlp.up_proj','Dense上投影');act(text['intermediate_size']);weight('mlp.down_proj','Dense下投影')
  else:
   ts=take('gate',contains='.gate.');emit('MoE路由评分','router',f'[N,{H}]',f'logits:[N,{text.get("n_routed_experts",text.get("num_experts"))}]','FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放',ts,module='router')
   top=text.get('num_experts_per_tok',text.get('num_experts_per_token'));E=text.get('n_routed_experts',text.get('num_experts'));Eh=text.get('routed_expert_hidden_size',H);W=text['moe_intermediate_size']
   emit('Top-K专家选择','router',f'scores:[N,{E}]',f'ids,weights:[N,{top}]','grouped/noaux_tc路由规则；总专家E与每token选中K严格区分',module='router')
   if k3:weight('routed_expert_down_proj','潜空间下投影')
   emit('专家分发与token重排','dispatch',f'X:[N,{Eh}]; ids:[N,{top}]',f'每专家X_e:[N_e,{Eh}]','argsort/gather或EP dispatch；N_e由实际路由确定',module='routed',intermediate=f'Σ_e N_e=N×{top}（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定')
   weight('w1' if k3 else 'gate_proj','路由专家门控GEMM',contains='.experts.');weight('w3' if k3 else 'up_proj','路由专家上投影GEMM',contains='.experts.');act(W,prefix='N_e');weight('w2' if k3 else 'down_proj','路由专家下投影GEMM',contains='.experts.')
   emit('专家加权合并','combine',f'Y_e:[N_e,{Eh}]; weights:[N,{top}]',f'[N,{Eh}]','逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简',module='routed')
   if k3:weight('routed_expert_norm','潜空间输出归一化');weight('routed_expert_up_proj','潜空间上投影')
   shared=text.get('n_shared_experts',text.get('num_shared_experts'))
   for s in ['gate_proj','up_proj']:weight('shared_experts.'+s,'共享专家'+s)
   act(W*shared,title='共享专家门控激活');weight('shared_experts.down_proj','共享专家下投影');emit('共享/路由分支合并','elementwise',f'shared,routed:[N,{H}]',f'[N,{H}]','Y=Y_routed+Y_shared；共享专家使用原残差宽度输入',module='shared')
  residual('前馈残差合并')
 elif node['group']=='vision':
  vc=cfg['vision_config'];C=vc['vt_hidden_size'];Q=vc.get('qkv_hidden_size',C);nh=vc['vt_num_attention_heads'];d=Q//nh
  weight('norm0','视觉注意力前归一化');weight('wqkv','视觉融合QKV投影')
  emit('视觉QKV拆分 / RoPE','reshape',f'[Nv,{3*Q}]',f'Q,K,V:[Nv,{nh},{d}]','view(...,3,heads,d)→unbind→二维RoPE',module='attention')
  emit('视觉注意力核心','vision-attention',f'Q,K,V:[Nv,{nh},{d}]',f'[Nv,{Q}]','按cu_seqlens分段计算非因果视觉attention；不跨独立图像任意注意',module='attention',intermediate='逻辑P:[heads,Lv,Lv]按每视觉段；Flash实现不必物化；无文本自回归KV缓存')
  weight('wo','视觉注意力输出投影');emit('视觉注意力残差','residual',f'branch,residual:[Nv,{C}]',f'[Nv,{C}]','相加',module='residual');weight('norm1','视觉MLP前归一化');weight('mlp.fc0','视觉MLP上投影');emit('视觉GELU','gelu',f'[Nv,{vc["vt_intermediate_size"]}]',f'[Nv,{vc["vt_intermediate_size"]}]','GELU；具体tanh近似由vision config决定',module='ffn');weight('mlp.fc1','视觉MLP下投影');emit('视觉MLP残差','residual',f'branch,residual:[Nv,{C}]',f'[Nv,{C}]','相加',module='residual')
 elif node['group']=='embedding':weight('embed_tokens','Token嵌入查表',kind='embedding')
 elif node['group']=='output_head':weight('lm_head','词表输出头',note='可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。')
 elif node['group']=='final_norm_and_residual':
  if k3:
   ts=take('output_attn_res_norm')+take('output_attn_res_proj');emit('最终AttnRes','attnres',f'prefix/blocks:[N,R,{H}]',f'[N,{H}]','最终跨块加权混合',ts)
  weight('model.norm','最终RMSNorm')
 elif node['group']=='projector':
  vc=cfg['vision_config'];C=vc['mm_hidden_size'];G=math.prod(vc['merge_kernel_size']);width=G*C
  if not k3:weight('pre_norm','Patch合并前LayerNorm',prefix='Nm,G')
  emit('空间合并与时间池化','reshape',f'视觉片段:[frames,h,w,{C}]',f'grouped:[Nm,{G},{C}] → [Nm,{width}]','空间2×2打包；sd2_tpool对相应时间组求均值；Nm依赖processor网格，不固定等于原始Nv/4',module='projector')
  weight('proj.0','连接器第一线性',prefix='Nm');emit('连接器GELU','gelu',f'[Nm,{width}]',f'[Nm,{width}]','nn.GELU',module='projector');weight('proj.2','投影到语言宽度',prefix='Nm')
  if k3:weight('post_norm','连接器输出RMSNorm',prefix='Nm')
  emit('插入多模态embedding','scatter','text embedding:[B,T,7168]; visual:[Nm,7168]','[B,T,7168]','按media placeholder位置填入视觉特征；数量由processor与网格校验',module='projector')
 elif node['group']=='vision_other':
  weight('patch_embed.proj','Patch嵌入卷积');weight('patch_embed.pos_emb','插值位置嵌入');weight('encoder.final_layernorm','视觉最终归一化',prefix='Nv')
 for t in tensors:
  if t['tensor_template'] not in used:
   used.add(t['tensor_template']);emit('补充参数 / 缓冲：'+t['tensor_template'].split('.')[-2],'parameter',t['stored_shape'],t['logical_shape'],'保留审计事实；该张量的独立运算映射未确认',[t],note='不得将辅助参数强行解释成一次矩阵乘法。')
 assert len(used)==len(tensors),(model['id'],node['id'])
 assert sum(s['parameters'] for s in steps)==node['parameters'],(model['id'],node['id'],sum(s['parameters'] for s in steps),node['parameters'])
 return steps

for model in [m for m in family['models'] if m['branch']=='主线']:
 arch=json.loads((ROOT/model['architecturePath']).read_text());cfg=json.loads((ROOT/model['configPath']).read_text());text=cfg.get('text_config',cfg);nodes=[];signatures={}
 for node in arch['nodes']:
  sig=json.dumps([node['group'],node['type'],node['ffn'],node['modules']],sort_keys=True)
  if sig not in signatures:
   tid=model['id']+'-t'+str(len(signatures)+1);signatures[sig]=tid;TEMPLATES[tid]=dict(id=tid,modelId=model['id'],group=node['group'],layerType=node['type'],ffn=node['ffn'],steps=make_steps(model,node,cfg))
  nodes.append(dict(id=node['id'],layer=node['number'],group=node['group'],type=node['type'],ffn=node['ffn'],template=signatures[sig],parameters=node['parameters'],weightEvidence='同尺寸Instruct模板推导，非Base文件头审计' if model['id']=='k2-base' else '固定检查点文件头审计',source=model['source'],revision=model.get('revision')))
 MODELS.append(dict(id=model['id'],name=model['name'],nodes=nodes,config=model['configPath'],tensorTemplates=arch['templatePath'],revision=model.get('revision'),symbols=dict(H=text['hidden_size'],heads=text['num_attention_heads'],qRank=text['q_lora_rank'],kvRank=text['kv_lora_rank'],qNope=text['qk_nope_head_dim'],qRope=text['qk_rope_head_dim'],vDim=text['v_head_dim'],experts=text.get('n_routed_experts',text.get('num_experts')),topK=text.get('num_experts_per_tok',text.get('num_experts_per_token')))))
SYMBOLS={'B':'批大小；不同引擎可将序列打包','T':'本次新增token数；常规decode=1，prefill>=1','S':'注意力可见历史+本次token总长度','N':'本次有效token总数=ΣT_i；padded参考实现为B×T','N_e':'第e个专家收到的token数，动态；无丢弃/无padding时ΣN_e=N×TopK','H':'语言残差宽度7168','heads':'语言MLA/KDA头数；K2为64 MLA，K3为96','R':'AttnRes有效跨层残差块数；不是上下文长度','Nv':'视觉patch token总数，按图片/视频网格分段','Nm':'合并/池化后的视觉token数','Np':'输入patch数','G':'空间合并单元数=2×2=4','TP':'tensor-parallel分片数；局部head/投影维可能除以TP','Vocab':'词表163840','weight_shape':'逻辑权重按[输出,输入]；不是运行时NZ/转置布局','packed_shape':'safetensors文件存储形状；I32/U8是打包容器，非实际权重计算精度'}
SYMBOLS.update({'{i}':'模板中的0-based层索引；逐层CSV已展开','{e}':'专家索引0≤e<experts；份数列表示该模板覆盖的全部专家'})
NOTES=[
 '研究源码快照为2026-09-21，结果整理于2026-09-23；不同仓库固定提交是分别核对的研究快照，不声称它们组成已测试的兼容软件栈。',
 'PyTorch参考计算与vLLM服务实现分开。接口列按模块/分派/设备算子分层；并列API可能是互斥分支，不能当作依次执行或1:1映射。',
 'K2 Base没有独立完整文件头审计：尺寸模板来自同配置Instruct，保留推导证据；其余7版对齐既有检查点逐张量审计。',
 'FP8权重scale_inv是F32二维block表，块大小128×128；INT4权重I32按输入维打包8元素/容器，group32 scale为BF16；K3路由MXFP4每U8打包2元素，group32 scale为U8。具体每张量shape在步骤里保留。',
 'K3 A_log审计存储[128]，参考配置96 heads。vLLM a_log_weight_loader按param局部head数narrow取片；不把存储值修改成96。该差异并未通过整模型加载实测。',
 '旧检查点rotary_emb.inv_freq存储长度可能不同于config旋转维度所需频率长度；生成运行时RoPE的具体加载/重建路径尚未逐版验证，不据此推算运行cache。',
 'K3官方参考MLA forward在所读文件中未直接调用RoPE；生产wrapper存在rotary_emb调用。此处区分代码路径，不自动补造参考调用。',
 'Ascend W4A8接口列是转换检查点/对应量化方法的条件路径，不等同直接支持原始INT4/MXFP4。MC2通信、A5特化与A3部署条件分别适用。',
 '未执行GPU/NPU基准、权重加载或数值精度实验；未核验设备内核处明确留空。shape检查是静态相容性检查，不是运行测试。'
]
missing=[x for x in json.loads((args.research_root/'sources.json').read_text()) if 'error'in x]
data=dict(schemaVersion=1,familyId='kimi',snapshot='2026-09-21',authored='2026-09-23',symbols=SYMBOLS,notes=NOTES,models=MODELS,templates=TEMPLATES,proofs=proofs,missingPaths=missing)
(D/'compute.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
def source_text(ids):return '\n'.join(f"{i} {proofs[i]['repo']}@{proofs[i]['revision']} {proofs[i]['path']}::{proofs[i]['symbol']} L{proofs[i]['line']}-{proofs[i]['end']} {proofs[i]['url']}" for i in dict.fromkeys(ids))
flow_fields=['版本','model_id','组件','层号_1based','层类型','模板','步骤序号','模块','步骤','数学关系','输入shape','逻辑权重shape','打包shape与dtype','输出shape','中间与缓存','阶段','精度','逻辑参数计数_本步骤','权重证据','配置或检查点revision','PyTorch接口','来源标识与链接','边界说明']
api_fields=['版本','model_id','组件','层号_1based','层类型','模板','步骤序号','模块','步骤','输入shape','输出shape','PyTorch接口','PyTorch来源','NVIDIA_CUDA调用链','NVIDIA层级','NVIDIA条件','NVIDIA来源','AMD_ROCm调用链','AMD层级','AMD条件','AMD来源','Ascend调用链','Ascend层级','Ascend条件','Ascend来源','证据强度']
flow_rows=[];api_rows=[];expanded_models={}
for model in MODELS:
 fr=[];ar=[]
 for node in model['nodes']:
  for j,s in enumerate(TEMPLATES[node['template']]['steps'],1):
   def tn(t):return t['tensor_template'].replace('{i}',str(node['layer']-1))
   ids=s['reference']['proofs']+s['apis']['torch']['proofs']
   row=dict(zip(flow_fields,[model['name'],model['id'],node['id'],node['layer'] or '全局',node['type']+'/'+str(node['ffn']),node['template'],j,s['module'],s['title'],s['relation'],s['input'],'\n'.join(tn(t)+': '+t['logical_shape']+(' ×'+str(t['multiplicity'])+' experts' if t['multiplicity']>1 else '') for t in s['tensors']),'\n'.join(tn(t)+': '+t['stored_shape']+' '+t['stored_dtype'] for t in s['tensors']),s['output'],s['intermediate'],s['phase'],s['dtype'],s['parameters'],node['weightEvidence'],model['revision'] or 'Base未独立权重审计',s['apis']['torch']['chain'],source_text(ids),s['note']]))
   b=s['apis'];vals=[model['name'],model['id'],node['id'],node['layer'] or '全局',node['type'],node['template'],j,s['module'],s['title'],s['input'],s['output'],b['torch']['chain'],source_text(b['torch']['proofs'])]
   for backend in ['nvidia','amd','ascend']:
    a=b[backend];vals +=[a['chain'],a['level'],a['condition'],source_text(a['proofs'])]
   vals+=['源码已核验仅指列出的函数/调用；未知项明确标记，未做设备实测'];ar.append(dict(zip(api_fields,vals)));fr.append(row)
 flow_rows+=fr;api_rows+=ar;expanded_models[model['id']]=(fr,ar)
def csvbytes(rows,fields):
 b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=fields);w.writeheader();w.writerows(rows);return b.getvalue().encode('utf-8-sig')
(A/'Kimi-layer-compute-shapes.csv').write_bytes(csvbytes(flow_rows,flow_fields));(A/'Kimi-backend-api-map.csv').write_bytes(csvbytes(api_rows,api_fields))
with zipfile.ZipFile(A/'Kimi-per-model-tables.zip','w',zipfile.ZIP_DEFLATED) as z:
 for id,(fr,ar) in expanded_models.items():z.writestr(id+'-compute-shapes.csv',csvbytes(fr,flow_fields));z.writestr(id+'-backend-api.csv',csvbytes(ar,api_fields))
 z.writestr('字段与符号说明.md',('\n\n'.join(NOTES)+'\n\n'+'\n'.join(k+'：'+v for k,v in SYMBOLS.items())).encode())
data['counts']=dict(models=len(MODELS),languageLayers=sum(n['group']=='decoder' for m in MODELS for n in m['nodes']),visionLayers=sum(n['group']=='vision' for m in MODELS for n in m['nodes']),components=sum(len(m['nodes']) for m in MODELS),templates=len(TEMPLATES),steps=len(flow_rows),apiRows=len(api_rows),proofs=len(proofs))
(A/'Kimi-compute-atlas.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
report=['# Kimi K2—K3 逐层计算与后端接口对照','源码快照：2026-09-21；整理：2026-09-23']+NOTES+['## 符号与字段']+[f'- {k}：{v}' for k,v in SYMBOLS.items()]+['## 完整层映射']
for m in MODELS:
 report+=['### '+m['name'],f"语言层 {sum(n['group']=='decoder' for n in m['nodes'])}；视觉层 {sum(n['group']=='vision' for n in m['nodes'])}；全部组件 {len(m['nodes'])}。"]
 for n in m['nodes']:report.append(f"- {n['id']} / 1-based {n['layer'] or '全局'} / {n['type']} / {n['ffn']} → {n['template']}；参数 {n['parameters']}；{n['weightEvidence']}")
report+=['## 步骤模板（逐层CSV已完整展开）']
for tid,t in TEMPLATES.items():
 report+=['### '+tid]
 for s in t['steps']:
  report +=[f"#### {s['id']} {s['title']}",f"{s['input']} → {s['relation']} → {s['output']}",s['intermediate'],s['note']]
  for x in s['tensors']:report.append(f"- {x['tensor_template']}：logical {x['logical_shape']}；stored {x['stored_shape']} {x['stored_dtype']}；multiplicity {x['multiplicity']}")
  for b,a in s['apis'].items():report.append(f"- {b} / {a['level']}：{a['chain']}。条件：{a['condition']}。来源："+' '.join(a['proofs']))
report+=['## 精确函数来源']+[f"- {k} [{p['symbol']}]({p['url']}) — {p['repo']}@{p['revision']}，{p['path']} L{p['line']}-{p['end']}，函数体SHA256 {p['functionSha256']}" for k,p in proofs.items()]
(A/'Kimi-layer-api-report.md').write_text('\n\n'.join(report));(A/'字段与符号说明.md').write_text('\n\n'.join(NOTES)+'\n\n'+'\n'.join(k+'：'+v for k,v in SYMBOLS.items()))
data['downloads']=[dict(name=p.name,path='assets/kimi/compute/'+p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in A.iterdir() if p.is_file()]
data['counts']=dict(models=len(MODELS),languageLayers=sum(n['group']=='decoder' for m in MODELS for n in m['nodes']),visionLayers=sum(n['group']=='vision' for m in MODELS for n in m['nodes']),components=sum(len(m['nodes']) for m in MODELS),templates=len(TEMPLATES),steps=len(flow_rows),apiRows=len(api_rows),proofs=len(proofs))
(D/'compute.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
family['computePath']='data/families/kimi/compute.json'
for m in family['models']:
 if m['branch']=='主线':m['computePath']='#/family/kimi/compute/'+m['id']
family['downloads']=[x for x in family['downloads'] if not x['path'].startswith('assets/kimi/compute/')]+data['downloads']
(D/'family.json').write_text(json.dumps(family,ensure_ascii=False,indent=2))
print(json.dumps(data['counts']));print('Exports:',[(p.name,p.stat().st_size) for p in A.iterdir()])
