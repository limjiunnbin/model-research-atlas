"""Describe verified source contexts and conditional backend paths without claiming device execution."""
import ast


def build_hardware(e,family,compute):
 models=[m['id'] for m in family['models']];v3=[m for m in models if m in ('v2','v3','v3-base','r1','r1-zero')];v32=[m for m in models if m.startswith('v3.2')];v4=[m for m in models if m.startswith('v4')];distill=[m for m in models if 'distill'in m]
 def sid(repo,path):return next(k for k,s in e.sources.items() if s.get('repo')==repo and s.get('path')==path)
 A='vllm-project/vllm-ascend';T='deepseek-ai/TileKernels';bindings={}
 dspark=[m['id'] for m in family['models'] if m['facts'].get('dsparkStages',{}).get('value')]
 specifications={
  'mla':[(A,'vllm_ascend/ops/mla.py','AscendMultiHeadLatentAttention.forward'),(A,'vllm_ascend/attention/mla_v1.py','AscendMLAImpl.forward'),(A,'vllm_ascend/attention/mla_v1.py','AscendMLAImpl._forward_prefill'),(A,'vllm_ascend/attention/mla_v1.py','AscendMLAImpl._forward_decode')],
  'dsa':[(A,'vllm_ascend/ops/dsa.py','AscendDeepseekSparseAttention.forward'),(A,'vllm_ascend/attention/dsa_v1.py','AscendDSAImpl.forward'),(A,'vllm_ascend/attention/dsa_v1.py','AscendDSAImpl._forward_attention')],
  'compression':[(A,'vllm_ascend/models/deepseek_v4/model.py','DeepseekV4Attention.forward'),(A,'vllm_ascend/models/deepseek_v4/compressor.py','Compressor.forward'),(A,'vllm_ascend/models/deepseek_v4/indexer.py','AscendIndexerOps.select_topk')],
  'mhc':[(A,'vllm_ascend/models/deepseek_v4/model.py','DeepseekV4DecoderLayer.hc_pre'),(A,'vllm_ascend/models/deepseek_v4/model.py','DeepseekV4DecoderLayer.hc_post'),(T,'tile_kernels/mhc/pre_big_fuse_kernel.py','mhc_pre_big_fuse_fwd'),(T,'tile_kernels/mhc/pre_big_fuse_asc.py','mhc_pre_big_fuse_fwd_asc'),(T,'tile_kernels/mhc/sinkhorn_asc.py','mhc_sinkhorn_fwd_asc')],
  'moe':[(A,'vllm_ascend/ops/fused_moe/moe_mlp.py','apply_moe_mlp'),(A,'vllm_ascend/ops/fused_moe/token_dispatcher.py','TokenDispatcherWithMC2.token_dispatch'),(A,'vllm_ascend/ops/fused_moe/token_dispatcher.py','TokenDispatcherWithMC2.token_combine'),(A,'vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py','AscendW4A8MXFPDynamicFusedMoEMethod.apply_gmm1_act_quant'),(T,'tile_kernels/moe/topk_gate_kernel.py','topk_gate'),(T,'tile_kernels/moe/topk_gate_asc.py','get_topk_gate_kernel_asc')],
  'precision':[(A,'vllm_ascend/ops/layernorm.py','AscendRMSNorm.forward_oot'),(A,'vllm_ascend/ops/linear.py','AscendUnquantizedLinearMethod.apply'),(A,'vllm_ascend/quantization/methods/w8a8/fp8_block.py','AscendFp8BlockLinearMethod.process_weights_after_loading'),(A,'vllm_ascend/quantization/methods/w8a8/fp8_block.py','AscendFp8BlockLinearMethod.apply')],
  'mtp':[(A,'vllm_ascend/models/deepseek_v4/mtp.py','DeepSeekV4MTP.forward'),(A,'vllm_ascend/models/deepseek_v4/mtp.py','DeepSeekV4MTP.load_weights'),('vllm-project/vllm','vllm/model_executor/models/deepseek_mtp.py','DeepSeekMultiTokenPredictorLayer.forward'),('vllm-project/vllm','vllm/model_executor/models/deepseek_mtp.py','DeepSeekMTP.load_weights')],
  'dspark':[(A,'vllm_ascend/models/deepseek_v4/dspark.py','DeepseekV4DSparkModel.forward'),(A,'vllm_ascend/models/deepseek_v4/dspark.py','DeepseekV4DSparkModel.precompute_and_store_context_kv'),(A,'vllm_ascend/models/deepseek_v4/dspark.py','DSparkDeepseekV4ForCausalLM.compute_confidence'),(A,'vllm_ascend/models/deepseek_v4/dspark.py','DSparkDeepseekV4ForCausalLM._remap_dspark_name'),(A,'vllm_ascend/spec_decode/dspark_proposer.py','AscendDSparkProposer.set_inputs_first_pass'),(A,'vllm_ascend/worker/v2/spec_decode/dspark/speculator.py','AscendDSparkSpeculator.propose')],
  'gqa':[('huggingface/transformers','src/transformers/models/qwen2/modeling_qwen2.py','Qwen2Attention.forward'),('huggingface/transformers','src/transformers/models/llama/modeling_llama.py','LlamaAttention.forward')],
  'ffn':[(A,'vllm_ascend/ops/activation.py','AscendSiluAndMul.forward_oot'),(A,'vllm_ascend/ops/activation.py','AscendSiluAndMulWithClamp.forward_oot'),(A,'vllm_ascend/ops/linear.py','AscendUnquantizedLinearMethod.apply'),('huggingface/transformers','src/transformers/models/qwen2/modeling_qwen2.py','Qwen2MLP.forward'),('huggingface/transformers','src/transformers/models/llama/modeling_llama.py','LlamaMLP.forward')],
 }
 for topic,entries in specifications.items():
  rows=[]
  for repo,path,symbol in entries:
   source=sid(repo,path);pid=e.proof(source,symbol);p=e.proofs[pid];text=e.text(source);body='\n'.join(text.splitlines()[p['line']-1:p['end']]);tree=ast.parse(body if not body.startswith('    ') else '\n'.join(line[4:] for line in body.splitlines()))
   calls=sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call) and any(x in ast.unparse(n.func) for x in ('torch_npu.','torch.ops.','DeviceOperator.'))})
   rows.append({'proof':pid,'source':source,'symbol':symbol,'deviceCallContexts':calls,'level':'参考 eager 函数' if topic=='gqa' else 'TileLang Ascend kernel generator' if path.endswith('_asc.py') else '框架模块/分派函数',
                'condition':'固定版本的函数上下文；其中 API 可属于互斥条件分支，不表示顺序或整模型兼容性'})
  bindings[topic]=rows
 descriptions={
 'mla':('MLA 投影、cache 与 prefill/decode',v3,['模型 forward','Ascend MLA wrapper','mla_forward 分派','prefill/decode 后端','条件 attention/device API'],
        '低秩 Q/KV、位置 Key 与 cache 管理分开；prefill 与 decode 的数据布局/FA/图模式分支分别记录。',
        'wrapper.forward 调用 torch.ops.vllm.mla_forward；AscendMLAImpl.forward 再分派，_forward_decode 包含普通/NZ/FA quant cache 布局及 query padding/SpecDecoding 条件。',
        '支持条件包括 device、head 数/维度、cache 精度与 NZ、block table、TP/CP 和图模式；读取函数不证明这些条件组成可运行栈。',
        'CANN attention 的匹配版本开源实现与 API 已可查；实际所走分支、ABI 与数值尚未在设备验证。没有原始 FP8 权重加载、转换或设备测试。'),
 'dsa':('DSA 与 Lightning Indexer',v32,['低秩 Query','Indexer 与 scale cache','top-k / causal mask','稀疏 attention 分派'],
        'V3.2 在主 attention 外增加 Indexer，缓存、scale 与 top-k 长度独立。',
        'AscendDeepseekSparseAttention.forward 调用 dsa_forward；AscendDSAImpl 中 prolog、indexer、cache update 和 attention 各有单流/多流或设备条件。',
        'index_topk、key/head dim、量化 scale、padding、metadata 和硬件代际决定实现；模型卡的 DSA 算法不等于任意 npu kernel 支持。',
        'V4 也使用部分 DSA 基础设施，但其 cache 与压缩语义单独建模，不能照搬 V3.2 路径。'),
 'compression':('V4 窗口、CSA/HCA 与 compressor',v4,['mHC 折叠后的输入','Q/KV 投影','窗口与 learned 压缩','CSA indexer 或 HCA 前缀','稀疏 attention','分组低秩输出'],
        'ratio=4 具有 overlap 的 learned pooling 与独立 indexer；ratio=128 使用全部压缩前缀；ratio=0 是窗口/MTP 条件。',
        '框架 Compressor.forward 使用 _C_ascend.compressor 分派；Indexer.select_topk 使用 npu_quant_lightning_indexer_v2；source forward 与 cache plan/metadata 相互约束。',
        '压缩 record 数 floor(S/r)、窗口 offset、state layout、indexer 精度和更新边界必须对齐；source dispatcher 不是已经验证的 AscendC kernel。',
        '参考 inference 对 KV/indexer 做 QAT 精度模拟后保存默认 dtype；不能据此断言生产框架存储也为 BF16。'),
 'mhc':('mHC 控制投影、Sinkhorn 与 pre/post',v4,['四流残差','控制投影与 RMS statistic','pre/post/comb + Sinkhorn','子层计算','四流还原'],
        '两套 [24,4H] 控制投影；hc_pre 折叠四流，hc_post 使用 post/comb 还原。',
        '框架 hc_pre/hc_post 分派 npu_hc_pre_v2 / npu_hc_post；TileKernels wrapper 与 *_asc generator 为另一条独立实现来源，不认定框架使用了该库。',
        'TileKernels 所选快照明确要求 Ascend 950、CANN≥9.2、Python≥3.12、PyTorch≥2.13、TileLang≥0.1.15；不能推广到 A2/A3 或旧 torch_npu。',
        '源码生成器存在不等于编译、数值验证或融合收益；kernel 与参考之间的布局/迭代数需后续检查。'),
 'moe':('路由、专家 GEMM 与通信',[m for m in models if m not in distill],['gating scores','Hash/top-k','dispatch / token gather','专家 GEMM 与 clamp/SwiGLU','combine + shared'],
        'V3 的分组 sigmoid 路由与 V4 的 hash/sqrtsoftplus 路由分开；专家 count、选中数、每专家 N_e 与 EP placement 不混用。',
        'MC2 token_dispatch/combine 使用 npu_moe_distribute_dispatch_v2 / combine_v2；MXFP4 GMM1 分支调用 grouped_matmul、swiglu_group_quant，另有非 DeepSeek 的 SITU 分支，不能按调用集合串成执行序列。',
        '通信方式、EP/TP、group_list_type、group32 scale、SWIGLU clamp、shared overlap 决定实现；原始专家格式先经对应 loader/转换。',
        'Distill 是 Dense GQA，只适用 FFN/激活共同部分，不适用路由或专家通信；未测试任何通信或 grouped GEMM。'),
 'precision':('FP8/MXFP4、解量化、norm 与布局',models,['权重格式核对','加载/转换与 scale','动态量化或 BF16 fallback','GEMM / norm','输出精度'],
        'V3 原始 E4M3 + F32 block scale 来自文件头；MXFP4/UE8M0 等属于明确的转换或参考条件，不能只按名称等同。',
        'AscendFp8BlockLinearMethod 可选择 mxfp8_method，其他分支走 unquantized_gemm；process_weights_after_loading 必须与 apply 一起核对。RMSNorm.forward_oot 包含 npu_rms_norm 或 add-rms 条件。',
        '硬件原生格式、转换脚本、dtype、scale 布局、NZ 与 custom-op 包版本须固定；支持一条转换路径不等于直接加载原始 FP8/FP4。',
        '没有质量/误差或吞吐实验，不能填写量化收益。'),
 'mtp':('MTP 共享参数、加载与推测解码',[m['id'] for m in family['models'] if m['facts']['mtpParameters']['value']],['主干表示与 token embedding','MTP 投影/子层','共享 output head','draft / verify 分派'],
        '论文训练目标、发布额外层、普通主干 forward 与 speculative decoding 四项分开。',
        'V3 demo 跳过 layer 61；V4 官方参考构造共享 embed/head 的 MTPBlock，框架 MTP.forward/load_weights 是独立来源。',
        '需要对应 speculative config、MTP checkpoint 命名、layer offset、cache/metadata 与 scheduler 条件；不能以文件头存在推断启用。',
        '已核对五版缺失的独立 MTP 存储与固定 vLLM forward；SharedHead.norm 在该后端独立构造/加载，上游训练共享和值相同仍不由 header 证明。'),
 'dspark':('DSpark 半自回归 draft 与验证调度',dspark,['目标特征与 context KV','noisy block 并行 backbone','轻量顺序 Markov 修正','confidence / prefix survival','target verify / rejection','只提交已接受状态'],
        'DSpark 是推测解码机制；本 V4 发布有 3 个 stage 存在 mtp.* namespace，不是把 1 个 MTP 层改名。原始 preview 不默认包含 DSpark。',
        'Ascend draft model 用独立 SWA cache，main_proj 融合目标特征；load_weights 将 mtp stage 分派到首层/末层模块；confidence 使用 sigmoid，proposer 与 speculator 独立准备 block/slot mapping。',
        'block_size、num_speculative_tokens、feature层号、Markov rank、采样方式、EOS/grammar、KV提交、图/TP/EP和缓存布局须分别锁定；native block=5 与模型卡 vLLM example=7 不混为一项。',
        'DeepSpec 的 Qwen/Gemma 训练和评测代码不是 V4 生产系统；论文负载感知 scheduler 和框架阈值/自适应配置分别核验，作者收益不是本站或 Ascend 实测。'),
 'gqa':('Distill 的 Qwen/Llama GQA',distill,['Q/K/V 投影','RoPE 与 KV cache','repeat_kv / attention','Dense FFN','残差'],
        '六版 Distill 使用 Qwen2/Llama 原生 Dense GQA 结构，Q/K/V bias、KV heads 和词表分别核对。',
        '所选 Transformers v4.44.0 的 eager 参考作为矩阵与数学关系来源；不是声明本机已装该版本或所有服务使用 eager。',
        '运行框架、SDPA/FlashAttention、分页、tokenizer 与 chat_template 须按实际底座检查；不能套用 MLA cache。',
        '这里只定位参考模型与共同 norm/linear 条件，GQA 专属 Ascend attention 内核尚未追通。'),
 'ffn':('Dense FFN 与 SwiGLU',models,['gate/up 投影','SiLU(gate)×up','down 投影','残差'],
        'Dense GQA 模型不含专家路由；V3 前 Dense 层与共享专家也有这一数学结构。',
        '固定 Qwen/Llama MLP.forward 核对参考数学；框架未量化 Linear.apply 使用 unquantized_gemm 分派。',
        '三个投影、激活精度、bias、并行切分和量化布局各自固定；Linear 分派不证明某个 FFN 的完整设备时序。',
        '已定位 AscendSiluAndMul.forward_oot 到 torch_npu.npu_swiglu；clamp 版先按前后半维分别 clamp 再调用。Dense 分支不套用专家 MC2 通信，底层 CANN 数值仍未验证。'),
 }
 hardware={'schemaVersion':1,'familyId':'deepseek','updated':'2026-10-01','title':'DeepSeek 实现、硬件条件与可选后续研究','scope':'本次交付为固定来源的静态研究与网站；设备运行/性能验证已按用户要求移出本次范围。','experimentScope':{'required':False,'status':'excluded_from_current_scope_by_user','reason':'当前没有硬件；本次按静态研究与网站验收'},'optimizationNote':'以下是未来有硬件时可选的研究问题，不纳入本次验收，也不作为未完成项；源码观察与测量结果分开。','modules':[],'platforms':[],'optimizations':[],'sources':[],'protocol':[],'versionNotes':[],'downloads':{'report':'assets/deepseek/DeepSeek-implementation-hardware-ascend.md','sources':'assets/deepseek/DeepSeek-sources.csv'}}
 used=set()
 for topic,(title,ids,flow,meaning,code,hw,limit) in descriptions.items():
  rows=bindings[topic];refs=[{'id':r['source'],'line':e.proofs[r['proof']]['line']} for r in rows];used.update(r['source'] for r in rows)
  hardware['modules'].append({'id':topic,'title':title,'models':ids,'flow':flow,'meaning':meaning,'code':code,'hardware':hw,'limit':limit,'evidence':'固定源码静态核验；设备未实测','refs':refs,'bindings':rows})
  for m in family['models']:
   if m['id'] in ids:m['implementationLinks'].append({'topic':topic,'title':title,'path':'#/family/deepseek/implementation/'+topic+'/'+m['id']})
 from build_deepseek_atlas import DATA,read
 trace=read(DATA/'research/backend-trace.json')
 cann=read(DATA/'research/cann-operator-audit.json')
 cann_refs=[]
 for repo,revision in cann['repositories'].items():
  source=next(s['id'] for s in e.sources.values() if s.get('repo')=='cann/'+repo and s.get('path')=='README.md')
  cann_refs.append({'id':source});used.add(source)
 hardware['modules'].append({'id':'cann','title':'CANN 开源算子与本地 SDK','models':models,'flow':['固定框架调用','op-plugin 条件 API','匹配 CANN tag 的 API/host/tiling','kernel 与 SDK 高级 API','运行分支/ABI/数值验证属于未来可选研究'],
  'meaning':'CANN 相关算子公开源码可读，应按版本继续核查，不能将未完成的源代码追踪概括为闭源。本专题是所研究家族的来源清单，各版本启用的算子仍按模型和分派条件区分。',
  'code':f"本地 SDK {cann['package']['version']}；ops-transformer/ops-nn/ops-math 固定同版本 tag；{cann['counts']['operatorFamilies']} 个源码家族和独立 op-plugin bridge，含 API/host/tiling/kernel 范围哈希。Python v2 不自动等于 aclnn V2。",
  'hardware':'所检查安装为 aarch64 编译器/AscendC SDK，独立算子库文件未在该路径发现；源文件从对应 tag 核对。不能据 SDK 版本证明目标设备、框架或二进制兼容。',
  'limit':'官方 MhcPre/Post 与框架自定义 HcPre/Post 独立保留；TileKernels 仍是独立实现。没有安装/更新虚拟机软件、编译或运行算子；设备实验不在本次范围。',
  'evidence':'固定 tag 源文件和 Git blob/SHA-256、源码范围、本地包只读检查','refs':cann_refs,'bindings':[]})
 for model in family['models']:model['implementationLinks'].append({'topic':'cann','title':'CANN 开源算子与本地 SDK','path':'#/family/deepseek/implementation/cann/'+model['id']})
 for topic_module in hardware['modules']:
  matching=[op for op in cann['operators'] if topic_module['id'] in op['topics']]
  if matching:
   topic_module['code']+=' CANN 匹配 tag 来源：'+', '.join(op['directory'] for op in matching)+'。'
   for op in matching:
    r=next((r for r in op['references'] if r['role']=='definition'),op['references'][0])
    topic_module['refs'].append({'id':r['sourceId'],'line':r['line']});used.add(r['sourceId'])
 for item in trace['paths']:
  module=next(m for m in hardware['modules'] if m['id']==item['topic'])
  module['code']+=' 开放底层链：'+' → '.join(item['flow'])+'。'
  module['limit']+=' '+item['conditions']
  for r in item['ranges']:module['refs'].append({'id':r['sourceId'],'line':r['line']});used.add(r['sourceId'])
 requirements=sid(T,'README.md');used.add(requirements)
 hardware['platforms']=[{'id':'ascend-a2-a3','name':'Ascend A2/A3：设备条件独立确认','precision':'BF16/INT8 与其他格式由具体 CANN/torch_npu 和 loader 决定','support':'只记录源码分派条件，不以设备连接或 import 成功作为整模型支持证明。','stack':'设备环境与所选 vLLM-Ascend 提交未组成已测试兼容栈；固定驱动、CANN、torch_npu、框架与 custom-op 包后验证。','boundary':'不将 Ascend 950 的 FP4/FP8/TileKernels 条件推广到 A2/A3。设备实验不在本次范围。','refs':[{'id':bindings['precision'][0]['source']}]},
 {'id':'ascend-950','name':'Ascend 950：V4 / TileKernels 候选平台','precision':'按模块区分 FP4、FP8、BF16、FP32 与 UE8M0 scale；格式与转换单列','support':'TileKernels 明确包含 Ascend 后端，框架 V4 模块与 compressor/mHC dispatcher 存在；仍未验证整模型或两套实现的连接关系。','stack':'TileKernels：Python≥3.12、PyTorch≥2.13、TileLang≥0.1.15、CANN≥9.2；所选 vLLM-Ascend 是独立固定源码快照。','boundary':'硬件型号一致不足以证明软件栈兼容；source presence、编译、正确性与性能分别标记。','refs':[{'id':requirements}]}]
 candidates=[('P0','baseline','先建立可运行基线','precision','固定完整环境、loader 与权重转换，先小批正确性，再记录初始 prefill/decode。','启动、logits/模块误差、trace、p50/p95、每卡显存'),
 ('P1','mla-cache','MLA 投影与 cache 读写','mla','在固定 decode shape 对比分步与融合/缓存路径；检查重复解量化与中间读写。','kernel 时间、HBM 读写、cache bytes、logits/输出误差'),
 ('P1','dsa-index','Indexer 与 sparse attention','dsa','分别测 index score/top-k、attention 和 metadata，变化只作用于一项，保留选择结果质量。','indexer/attention 时间、top-k 一致性、质量/误差、workspace'),
 ('P1','v4-compress','压缩边界与 compressor 状态','compression','分别比较首次 prefill、decode 压缩边界与非边界；窗口/前缀/FP32 状态分别测量。','边界 p95、state/cache bytes、压缩 token 与数值误差'),
 ('P1','mhc-fusion','mHC pre/post 与 Sinkhorn 融合','mhc','以参考 mix/Sinkhorn 为正确性基线，再比较可用融合路径；不预设收益。','pre/post 时间、迭代误差、row/column 归一化、HBM 读写'),
 ('P1','moe-comm','专家 GEMM 与 dispatch/combine','moe','固定 EP/TP、路由和 N_e 分布，测通信、专家和 shared 部分，再评估重叠。','dispatch/GEMM/combine 分段时间、N_e 分布、吞吐、误差'),
 ('P2','quant-graph','量化、图模式与动态负载','precision','同模型固定质量参考，分别变化量化格式或图模式；warmup 与编译时间单列。','延迟分布、显存、质量、图重编译次数')]
 for priority,oid,title,topic,proposal,metric in candidates:
  refs=[{'id':bindings[topic][0]['source']}]
  hardware['optimizations'].append({'id':oid,'priority':priority,'title':title,'status':'未来可选 · 不纳入本次验收','scope':topic+' / 如另行开展，固定具体软硬件和负载','observation':descriptions[topic][4],'proposal':proposal,'metric':metric,'risk':'源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。','refs':refs})
 for source in sorted(used):
  s=e.sources[source];hardware['sources'].append({k:s[k] for k in ['id','title','url','revision','sha256','accessed','kind']})
 hardware['protocol']=['环境：设备型号、卡数/拓扑、驱动固件、CANN、torch_npu、框架、custom-op 包、容器 digest，禁止只用路径名冒充版本号。',
  '模型：checkpoint revision、原始与转换精度、scale/packing、转换脚本 commit、weight header 哈希/覆盖。',
  '负载：B、L_q、L_kv、并发、输入/输出长度、padding/ragged、TP/EP/DP、graph 与环境开关。',
  '正确性：先固定参考与种子，检查模块/logits 的绝对/相对误差；量化额外检查任务质量。',
  '性能：分别测 prefill/decode/verify/服务；记录 warmup、次数、时间单位、p50/p95、吞吐口径、峰值显存与原始 trace。',
  '一次只改变一个主要变量；保存成功、失败与未执行状态；不把上游 README 数字或静态公式当成本项目实测。']
 hardware['versionNotes']=[{'models':[m['id'] for m in family['models']],'text':'官方模型代码、框架源码、TileKernels 分别固定版本；函数调用集合是上下文证据，不是执行顺序或 1:1 kernel 对应。'},{'models':[m['id'] for m in family['models']],'text':'CANN 9.2.0-beta.2 的匹配开源仓和本地 SDK 已核查；源码可查、独立算子库是否安装、框架分派、ABI、设备正确性与性能分别记录。'},{'models':[m['id'] for m in family['models']],'text':'设备验证不纳入本次验收，所有性能字段保持未知。'}]
 # Link only module-level contexts applicable to each step; never guess individual device kernels.
 for template in compute['templates'].values():
  model=next(m for m in family['models'] if m['id']==template['modelId'])
  for s in template['steps']:
   title=(s['title']+' '+s['relation']).lower()
   topic='dspark' if 'dspark'in title or 'dspark'in s['path'] or 'DSpark'in template['layerType'] else 'mtp' if 'mtp'in s['path'] else 'mhc' if 'mhc'in title or 'hc-pre'in s['id'] or 'hc-post'in s['id'] else 'compression' if template['layerType'] in ('CSA','HCA','SWA','SWA / mHC') and ('attention'in title or 'index'in title or 'compress'in title) else 'dsa' if 'index'in title or template['layerType']=='DSA' and 'attention'in title else ('ffn' if template['ffn']=='Dense' else 'moe') if any(x in title for x in ('expert','路由','swiglu','ffn','mlp')) else 'gqa' if template['layerType']=='GQA' and 'attention'in title else 'mla' if 'attention'in title or 'mla'in title else 'precision'
   module=next(x for x in hardware['modules'] if x['id']==topic)
   if model['id'] not in module['models']:continue
   rows=bindings[topic];proof_ids=[r['proof'] for r in rows[:2]]
   s['apis']['ascend']={'level':'条件框架模块/分派上下文，非该单步 kernel','evidence':'source','chain':'；'.join(r['symbol'] for r in rows[:2]),'condition':module['hardware']+' '+module['limit'],'proofs':proof_ids}
 compute['proofs']=e.proofs.copy();compute['counts']['proofs']=len(e.proofs)
 return hardware
