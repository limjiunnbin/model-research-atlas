"""Source-grounded MTP/DSpark contracts, config conflicts and small mathematical checks."""
import ast
import itertools
import json
import math


def build_speculation(e,family,architectures):
 from build_deepseek_atlas import DATA,write
 models=[];proofs=[]
 def proof(repo,path,symbol):
  sid=next(sid for sid,s in e.sources.items() if s.get('repo')==repo and s.get('path')==path)
  pid=e.proof(sid,symbol);proofs.append(pid);return pid
 ds=[m for m in family['models'] if architectures[m['id']].get('dspark')]
 for m in ds:
  a=architectures[m['id']];s=a['dspark'];cfg=e.select(m['name'],'inference/model.py')
  ids=[e.proof(cfg,x) for x in ['Transformer.forward','Transformer.forward_spec','DSparkBlock.forward_embed','DSparkAttention.forward','DSparkBlock.forward_head','DSparkMarkovHead.forward','DSparkConfidenceHead.forward']];proofs+=ids
  models.append({'modelId':m['id'],'name':m['name'],'revision':m['revision'],'mainLogicalParameters':a['mainLogicalParameters'],'draftLogicalParameters':a['dsparkLogicalParameters'],'config':s,'referenceProofs':ids,'conditions':['native prefill 只融合目标特征和写上下文 KV，不执行完整 draft Block','draft backbone 一次处理 noisy block，Markov head 依次条件于前一 token；3 个 stage 各自有 SWA context cache','confidence 原生返回 logits；服务路径 sigmoid 后使用阈值或可选负载/吞吐剖面调度','target verify/rejection/bonus token 与 KV commit/update 是服务协议，不能从 draft.forward 单函数推定'],'conflicts':[{'field':'num_nextn_predict_layers','hfConfig':1,'inferenceConfigStages':3,'headerStages':[0,1,2],'resolution':'mtp.* 是 DSpark 存储 namespace，不能当作 1 个传统 MTP 层'},{'field':'proposal_length','checkpointBlockSize':s['blockSize'],'modelCardVllmExample':7,'resolution':'native demo 按 checkpoint block=5；vLLM 服务 num_speculative_tokens=7 是另一项显式设置，须按该框架处理；不改写配置'}]})
 refs={
  'mtpForward':proof('vllm-project/vllm','vllm/model_executor/models/deepseek_mtp.py','DeepSeekMultiTokenPredictorLayer.forward'),
  'mtpLayerSelection':proof('vllm-project/vllm','vllm/model_executor/models/deepseek_mtp.py','DeepSeekMultiTokenPredictor.forward'),
  'mtpNormConstruction':proof('vllm-project/vllm','vllm/model_executor/models/deepseek_mtp.py','SharedHead.__init__'),
  'mtpLoadRewrite':proof('vllm-project/vllm','vllm/model_executor/models/deepseek_mtp.py','DeepSeekMTP._rewrite_spec_layer_name'),
  'ascendDraftForward':proof('vllm-project/vllm-ascend','vllm_ascend/models/deepseek_v4/dspark.py','DeepseekV4DSparkModel.forward'),
  'ascendContextKV':proof('vllm-project/vllm-ascend','vllm_ascend/models/deepseek_v4/dspark.py','DeepseekV4DSparkModel.precompute_and_store_context_kv'),
  'ascendConfidence':proof('vllm-project/vllm-ascend','vllm_ascend/models/deepseek_v4/dspark.py','DSparkDeepseekV4ForCausalLM.compute_confidence'),
  'ascendLoadMap':proof('vllm-project/vllm-ascend','vllm_ascend/models/deepseek_v4/dspark.py','DSparkDeepseekV4ForCausalLM._remap_dspark_name'),
  'ascendPropose':proof('vllm-project/vllm-ascend','vllm_ascend/worker/v2/spec_decode/dspark/speculator.py','AscendDSparkSpeculator.propose'),
 }
 # Inspect available exact symbols; retain all boundaries/hashes for downstream review without executing code.
 for path in ['vllm/v1/worker/gpu/spec_decode/adaptive_verification.py','vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py','vllm/v1/worker/gpu/spec_decode/dspark/speculator.py','deepspec/modeling/dspark/markov_head.py','deepspec/modeling/dspark/common.py','deepspec/eval/dspark/draft_ops.py']:
  sid=next(k for k,s in e.sources.items() if s.get('path')==path);text=e.text(sid)
  for n in ast.parse(text).body:
   symbols=[n.name] if isinstance(n,ast.FunctionDef) else [n.name+'.'+f.name for f in n.body if isinstance(f,ast.FunctionDef) and f.name in ('forward','propose','__init__')] if isinstance(n,ast.ClassDef) else []
   for symbol in symbols:proofs.append(e.proof(sid,symbol))
 # Distribution identity for ordinary rejection sampling on full-support synthetic vectors.
 # This is static algebra, not checkpoint generation, confidence calibration or device correctness.
 maximum=0.0;cases=0
 for p,q in [([.2,.3,.5],[.5,.2,.3]),([.1,.1,.8],[.2,.7,.1]),([.25]*4,[.1,.2,.3,.4])]:
  accepted=[min(a,b) for a,b in zip(p,q)];residual=[max(a-b,0.0) for a,b in zip(p,q)];reject=sum(residual)
  recovered=[a+reject*r/reject if reject else a for a,r in zip(accepted,residual)]
  maximum=max(maximum,max(abs(a-b) for a,b in zip(recovered,p)));cases+=1
 c=[.8,.6,.25,.1,.05];prefix=[];product=1.0
 for x in c:product*=x;prefix.append(product)
 expected=sum(prefix)
 # Prefix scheduler demonstration: queue fronts only, no peeking into later draft-token dependencies.
 result={'schemaVersion':1,'snapshot':'2026-10-01','scope':'MTP/DSpark 的参数、draft/verify/update 数据协议与固定源码；不运行模型、训练或设备实验','models':models,'protocol':[{'phase':'context-prefill','operation':'目标主干正常 prefill/验证并取指定层特征；融合后初始化独立 context KV','proofs':[refs['ascendContextKV']]},{'phase':'draft','operation':'noisy block → 多 stage 并行 backbone → 顺序 Markov logits 修正/sample → confidence','proofs':[refs['ascendDraftForward'],refs['ascendPropose']]},{'phase':'select-prefix','operation':'conditional confidence → prefix survival；阈值或 engine-specific throughput profile 分配验证长度，不能任意 peek 后续位置再回退','proofs':[refs['ascendConfidence']]},{'phase':'verify','operation':'target 对选中 prefix 做因果 forward；按所选 sampler 接受/拒绝，含首个拒绝位置或全部接受后的 bonus token','proofs':[],'limit':'采样方式、EOS/grammar、prefix 选择与 resampling 条件各自固定；没有单函数完整运行证明'},{'phase':'update','operation':'只提交已接受/修正 token 的状态；丢弃或覆盖未接受 draft cache，下一轮输入由 target hidden 驱动','proofs':[refs['ascendPropose']]}],
  'comparisons':[{'method':'AR','draft':'无附加 draft','target':'每步一个 target token','parameters':'只有主干','cache':'主干 cache'},{'method':'MTP','draft':'独立 next-token predictor 可循环重用；训练深度与服务 draft 次数不同','target':'验证候选 causal prefix','parameters':'共享 embedding/head，独立投影、block、norm','cache':'MTP 和主干各自按后端管理'},{'method':'DSpark','draft':'并行 backbone + 轻量顺序 Markov/RNN；本 V4 checkpoint 是 vanilla Markov','target':'可变验证 prefix，confidence/负载配置独立','parameters':'3 stage，首层 main_proj，末层 Markov/confidence，共享 embed/head','cache':'独立 SWA context + block 内非因果 draft；不沿用主干 CSA/HCA'}],
  'referenceProofs':list(dict.fromkeys(proofs)),'paperSource':'dspark-paper-v1','reportedPerformance':{'source':'DSpark paper §5.4','evidence':'作者生产服务自述，非本站或 Ascend 实测','flashGenerationSpeedAtMatchedThroughput':'60%–85%','proGenerationSpeedAtMatchedThroughput':'57%–78%','limit':'V4 在线用户流量与对应服务条件，不等于任意设备、batch 或 checkpoint 的收益'},
  'syntheticValidation':{'status':'passed','rejectionDistributionCases':cases,'maxProbabilityAbsError':maximum,'conditionalConfidence':c,'prefixSurvival':prefix,'expectedAcceptedDraftLength':expected,'limits':'只复算分布恒等式与条件概率乘积；不证明学习到的 confidence 正确、调度校准或 kernel 数值'}}
 write(DATA/'research/speculation.json',result);return result
