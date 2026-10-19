# GLM 实现、硬件与 Ascend 优化研究

研究快照：2026-10-19

基于 cann-recipes-infer 固定快照 3b36eea 的源码与配方文档研究；没有执行任何 NPU 性能实验。代码存在、配方自述与本地实测严格分开记录。

## 混合注意力：34 层 KDA + 11 层 DSA

layer_types 决定本层走 KDA 还是 DSA（i%4==3 为 DSA） → KDA：q/k/v 投影 → 短卷积 + SiLU → 门控 delta 递推 → 门控 RMSNorm → o_proj → DSA：q_a/q_b 低秩 Q → kv_a 压缩潜在 → indexer 选 top-k → 吸收式稀疏注意力 → o_proj → KDA 持固定大小递推状态，DSA 持每 token 512 元素潜在缓存

两类层的缓存增长方式不同：KDA 的递推状态与序列长度无关，DSA 只缓存压缩潜在向量。把 34/45 的层换成 KDA，是把长上下文的缓存与带宽压力整体降下来，而不是把注意力做得更快一点。

KDA 实现在 kda.py：prefill 走 flash_kda（自述融合 l2norm、门控激活与 beta sigmoid），decode 走 fused_recurrent_kda；两者都消费未激活的原始 decay。DSA 在 modeling_glm53.py 用 npu_sparse_flash_attention 直接在潜在缓存上做稀疏注意力。

KDA 层应观察递推状态读写与短卷积；DSA 层应观察 indexer 的 top-k 选择与稀疏注意力内核。两类层的瓶颈不同，不能用同一个算子占比推断整层耗时。

层排布取自配置默认值，发布 config.json 可能覆盖。本站没有运行 profiling，不能判断哪一类层在真实负载下更贵。

来源：[cann-recipes-infer / models/glm_5_3/models/kda.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/kda.py)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py)；[cann-recipes-infer / models/glm_5_3/models/configuration_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/configuration_glm53.py#L155)

## mHC 超连接与两个 AscendC 融合算子

层间隐藏态是 4 条并行流 [T, 4, D] → npu_hc_pre：展平 → RMS 归一 → 与 hc_fn 矩阵乘 → 拆成 pre/post/comb → Sinkhorn 归一 → 按 pre 折叠成单流 → 子层（注意力或 FFN）在折叠后的单流上计算 → npu_hc_post：按 post 与 comb 把结果与残差重新展开回 4 条流

mHC 把残差通路从 1 条变成 4 条，每个子层前后各用一组学习到的混合矩阵决定怎么读、怎么写。代价是每层多两次全流量的矩阵混合，所以这两步必须融合成算子，否则会变成访存瓶颈。

modules.py 调用 torch.ops.custom.npu_hc_pre / npu_hc_post，每个解码层调用两对（注意力侧与 FFN 侧各一组独立参数）。算子随 opp/vendors/customize 包提供，set_env.sh 负责 source。算子文档约束 hc_mult=4、d=4096、hc_mix=24、sinkhorn 迭代 20，且可被 aclgraph 捕获。

hc_pre 文档说明：当 T/bs ≤ 128 且能被 16 整除时启用融合 kernel（自述“性能较高”），否则回退到 hc_pre_inv_rms + hc_pre_sinkhorn 两个拆分 kernel（自述“性能较低”）。这是形状相关的分支，batch 与序列切分会直接改变走哪条路径。

仓库没有给出 hc_pre / hc_post 的任何实测数字，融合前后的差异没有可引用的量化结果。Sinkhorn 的具体数值实现在算子内部，源码侧只能看到 I/O 契约。

来源：[cann-recipes-infer / models/glm_5_3/models/modules.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modules.py#L102)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L666)；[cann-recipes-infer / ops/ascendc/docs/custom-npu_hc_pre.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/ops/ascendc/docs/custom-npu_hc_pre.md)；[cann-recipes-infer / ops/ascendc/docs/custom-npu_hc_post.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/ops/ascendc/docs/custom-npu_hc_post.md)；[cann-recipes-infer / models/glm_5_3/set_env.sh](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/set_env.sh#L19)

## Hybrid HiF8-MXFP8-MXFP4 分模块量化

读入发布 FP8 权重（e4m3，128×128 块 scale）并反量化到 BF16 → classify() 按张量名判定归属：路由专家 / 共享专家 / 普通 Linear / 跳过 → 路由专家 → MXFP4 W4A8；共享专家 → MXFP8 W8A8；DSA 四个投影与 dense MLP → HiF8 W8A8 → KDA 全部权重、kv_b_proj、整个 indexer、mlp.gate、各 norm、embed/lm_head 原样保留

这不是"全模型降到某个位宽"，而是按模块的敏感度和计算占比分档：参数量最大的路由专家压到 4 bit，计算密集但数量少的注意力投影用 8 bit，对数值最敏感的递推状态、索引打分和路由 gate 完全不动。

convert_model.py 的 classify() 是唯一判定入口；quantize() 按 kind 分派到 quantize_mxfp4 / quantize_mxfp8 / quantize_hif8。HiF8 由 torch_npu.npu_dynamic_quant(dst_type=torch_npu.hifloat8) 编码，逐输出通道一个 float32 scale。MX 系列块大小固定 32，scale 为 E8M0 uint8；MXFP4 按 2 值/字节打包。运行时由 compressed_tensors 的 target 匹配还原成三个 config group。

HiF8 激活不做缩放，直接 npu_dtype_cast 后进 npu_quant_matmul；MX 系列激活走 npu_dynamic_mx_quant。路由专家的两段 GMM 之间用 npu_swiglu_mx_quant 融合 SwiGLU 与再量化。权重格式决定能否走 NZ 布局与哪一类 Cube 指令，换档位不是只改配置。

本站没有做精度评测，不能判断该分档对任务质量的影响。KDA 层完全不量化意味着 34/45 的注意力权重仍是 BF16，显存收益主要来自专家而非注意力。

来源：[cann-recipes-infer / models/glm_5_3/utils/convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/convert_model.py#L228)；[cann-recipes-infer / models/glm_5_3/utils/convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/convert_model.py#L445)；[cann-recipes-infer / models/glm_5_3/utils/convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/convert_model.py#L156)；[cann-recipes-infer / models/glm_5_3/utils/mx_quantize.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/mx_quantize.py#L438)；[cann-recipes-infer / module/quantization/compressed_tensors/compressed_tensors_w8a8_hif8.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/compressed_tensors/compressed_tensors_w8a8_hif8.py#L92)；[cann-recipes-infer / module/quantization/mxfp4.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/mxfp4.py#L200)；[cann-recipes-infer / module/quantization/compressed_tensors/compressed_tensors.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/compressed_tensors/compressed_tensors.py#L341)

## k-pool 压缩 Lightning Indexer

wk 投影 + k_norm 得到每 token 的索引 key → 每 index_kpool=4 个连续 token 为一组，用 softmax(gate + 位置嵌入) 在组内加权合并成一个池化 key → 池化 key 写入独立的压缩缓存，写入后不再更新 → decode 用 npu_lightning_indexer 在池化缓存上选 top-k，再还原成 token 级稀疏索引

Lightning Indexer 本身要对全部历史打分，长上下文下这一步会变成瓶颈。k-pool 把打分对象从 token 数降到 token 数 / 4，并且因为一个池在其 4 个 token 齐全后就不再变化，decode 每层只需读回已算好的池，代价与历史长度解耦。

indexer.py 的 _commit_pools 实现池化；decode 用 torch_npu.npu_lightning_indexer（layout_key="PA_BSND"，sparse_mode=3）。prefill 没有对应算子，仍走 torch matmul + topk，并按 QUERY_BLOCK=1024 分块以控制显存。池化缓存是独立的 cache manager，compress_ratio=index_kpool，并强制 block_size 能被 index_kpool 整除。

选择宽度 select_k = index_topk / kpool = 512，输出宽度在 always_select_tail 下为 2048+3。prefill 与 decode 走完全不同的实现，profiling 必须分开看，不能用 decode 的 indexer 占比推断 prefill。

仓库没有给出 k-pool 开/关的对比数据，压缩对召回质量的影响没有可引用的评测。prefill 路径是 torch 实现，是否成为长序列瓶颈需要实测。

来源：[cann-recipes-infer / models/glm_5_3/models/indexer.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/indexer.py#L179)；[cann-recipes-infer / models/glm_5_3/models/indexer.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/indexer.py#L214)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L506)；[cann-recipes-infer / models/glm_5_3/models/configuration_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/configuration_glm53.py#L60)

## 吸收式 MLA 与全程 NoPE

加载后把 kv_b_proj 拆成 w_k [N, 256, 512] 与 w_v [N, 512, 256]，并释放原权重 → q_nope 经 einsum 吸收进潜在空间得到 q_latent [T, 64, 512] → npu_sparse_flash_attention 直接在 512 维潜在缓存上做稀疏注意力 → 输出侧再用 w_v 展开回 v 头并进 o_proj

吸收式 MLA 让注意力全程在压缩潜在空间进行，每 token 只缓存 512 个元素，不需要把 K/V 展开。qk_rope_head_dim=0 表示 DSA 层完全不做 RoPE，缓存里也没有位置分量。

prepare_absorbed_weights 在 process_weights_after_loading 阶段执行，之后 kv_b_proj.weight 被置空。因为要参与这次拆分，kv_b_proj 被显式排除在量化之外（quant_config=None），转换脚本的 ignore 列表同步排除。SFA 内核仍要求 rope 操作数，由常零张量 SharedZeroRope 合成，不进缓存。

每 token 每 DSA 层的缓存是 512 元素（bf16 约 1KB）。11 层 DSA 的缓存总量远小于同规模全注意力模型，这是长上下文配置上限能提高的结构原因之一。

本站未实测缓存占用与带宽。常零 rope 张量是内核接口的适配手段，不代表模型里存在被置零的位置编码参数。

来源：[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L521)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L569)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L439)；[cann-recipes-infer / models/glm_5_3/models/configuration_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/configuration_glm53.py#L236)

## 共享专家多流与 EP 通信重叠

decode 且 moe_ep_size > 1 时进入 dispatch / combine 路径 → 路由专家的 GMM 下发后，record_event 标记 → 共享专家切到侧流执行，与路由专家的 dispatch / GMM / combine 重叠 → combine 之前 wait_event 汇合

共享专家每个 token 都要算，但它与路由专家之间没有数据依赖。把它放到侧流，可以用它的计算填住 EP 全局通信的空窗，而不是串行等待。

enable_multi_streams 默认 True，在 custom_params 中配置。侧流与 event 在模型初始化时创建一次，重叠逻辑在 _moe_infer_dispatch_combine；共享专家自身的 down_proj 还会等待 shared_expert_event。prefill 与 moe_ep_size==1 时改为内联执行。

收益取决于路由专家 GMM 与 EP 通信的耗时是否足以覆盖共享专家。batch 越小通信占比越高，重叠的相对收益通常越明显，但这需要 trace 确认。

仓库没有给出开/关多流的对比数据。README 只描述了重叠对象，没有量化结论。

来源：[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L340)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L827)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)；[cann-recipes-infer / models/glm_5_3/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/README.md#L95)

## npugraph_ex 图模式与 decode 路径约束

config 选择 exe_mode: npugraph_ex 或 eager → 图模式只捕获 decode step → decode 分支不做 host 侧读取，以便 fullgraph 追踪 → 常零 rope 缓冲按 is_prefill 分键，避免穿插的 prefill 重新分配已捕获的 decode 缓冲

图模式的收益来自消除逐算子下发开销，但要求 decode 路径完全静态。仓库里多处代码是为满足这个约束而写的，不是普通的实现选择。

build_step_metadata 明确注释 decode 分支要在 fullgraph=True 下可追踪；SharedZeroRope 按 is_prefill 分键缓冲；indexer 的 build_pool_gather 只在 prefill 使用，因为它在 host 上读 .max()，图模式不允许。

图模式只覆盖 decode，prefill 仍是 eager。比较两种 exe_mode 时必须分别看 prefill 与 decode，不能只看端到端吞吐。

仓库没有给出 npugraph_ex 与 eager 的对比数字。

来源：[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L733)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L372)；[cann-recipes-infer / models/glm_5_3/models/indexer.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/indexer.py#L38)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)

## Profiling 通路与两条路径的实测分布

YAML 只有 enable_profiler 一个开关，采集参数硬编码在 executor/utils/profiler_context.py → torch_npu.profiler，Level1 + PipeUtilization，active=3 / repeat=1 / skip_first=3 → 按阶段分目录落盘到 <output_path>/prof/{prefill,decode,mm_encode} → 离线由 msprof 解析；仓库不提交任何 trace 产物

必须区分“能打开 profiler”与“已提交实测数据”。本配方（多卡 HiF8）只有一张 README decode 表；另一条单卡 CPU-MoE offload 路径（SGLang + KTransformers，INT8 W8A8 加 MXFP4）才有 profiler 导出的逐组件分解与带置信区间的 A/B。两条路径的量化、卡数与框架都不同，数据不可互相挪用。

create_profiler 固定 profiler_level=Level1、aic_metrics=PipeUtilization、with_stack/record_shapes/profile_memory 全关；ProfilerManager 按 mm_encode / prefill / decode 分阶段切换，prefill 与 mm_encode 用 active=1、skip_first=0，decode 沿用 3/1/3。

单卡路径的 trace 给出 GLM-5.3 的结构开销量级：单步 NPU 关键路径 33.3 ms 中 KDA 10.4、MoE 9.6、DSA 5.4、mHC 3.3 ms。这是该路径的数字，只能当作结构开销的量级参考，不能当作本配方在 950DT 上的指标。

decode 阶段无法冷启动单独采集（必须先过 prefill，且进入 decode 后不再切回）；离线 msprof 常只产出截断的 trace_view.json 而 CSV 全缺，仓库把根因归到输出目录的文件系统。profiler 自身会抬高单步耗时（该路径 profiling 下 20.89–28.72 ms，生产 20.57 ms）。ops/ascendc/docs/ 下 20 份算子文档全部没有实测数字；benchmark/ 只做精度评测且无 GLM 条目。

来源：[cann-recipes-infer / executor/utils/profiler_context.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/executor/utils/profiler_context.py#L50)；[cann-recipes-infer / docs/integration/sglang/glm53-flash-single-npu-moe-offload/glm53_flash_single_card_design.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/docs/integration/sglang/glm53-flash-single-npu-moe-offload/glm53_flash_single_card_design.md#L158)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)；[cann-recipes-infer / models/glm_5_3/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/README.md#L119)

## 并行策略：注意力与 dense 走 DP，并行度来自 MoE EP

world_size=8，moe_tp_size=1 → 框架据此派生 moe_ep_size = world_size / moe_tp_size = 8 → 路由专家按 Expert 维切分，288 个专家分到 8 张卡 → cp / attn_tp / dense_tp / moe_tp / o_proj_tp / shared_tp 必须为 1，否则直接报错

这是一种把并行度全部压在专家维的选择：注意力与 dense 部分每张卡各算各的（DP），只有 MoE 做 EP 切分。好处是注意力侧没有 TP 通信，代价是每张卡都要完整持有注意力与 dense 权重。

派生在 inference_config.py；约束在 Glm53ForCausalLM.check_model_settings 集中检查，KDA 与 DSA 的 __init__ 另有本地保护。dense/shared FFN 与各投影直接硬编码 tp_size=1。950 平台注册 moe_ep_group 时 group_type=3，非 950 平台的 dispatch 会附加 comm_alg="fullmesh_v2"。

每卡需容纳全部 KDA/DSA 与 dense 权重加上 288/8 = 36 个专家。改卡数会同时改变专家分布与 HCCL 缓冲大小，不是线性缩放。

num_experts 必须能被 moe_ep_size 整除，这限制了可选卡数。本站没有测试其他 world_size。

来源：[cann-recipes-infer / executor/core/config/inference_config.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/executor/core/config/inference_config.py#L400)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L858)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L887)

## 硬件与支持边界

### 昇腾 950DT 系列（GLM-5.3-Flash 目标平台）

Hybrid HiF8-MXFP8-MXFP4：HiF8 需要 SoC 原生 hifloat8 支持，MX 系列需要 32 元素块与 E8M0 scale 的内核。

models/glm_5_3 配方直接面向该平台，config 中 platform_version: "950"；world_size 默认 8，MoE EP = 8。

CANN 安装到固定路径；另需 opp/vendors/customize（mHC 算子）与 opp/vendors/custom_transformer（causal_conv1d）两个算子包，由 set_env.sh 载入。

配方存在不等于本站完成部署。转换脚本的 HiF8 编码依赖 torch_npu 原生 hifloat8，文档指出 910B/A2 会拒绝该 dst_type。

来源：[cann-recipes-infer / models/glm_5_3/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/README.md#L19)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)；[cann-recipes-infer / models/glm_5_3/set_env.sh](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/set_env.sh)；[cann-recipes-infer / models/glm_5_3/utils/convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/convert_model.py#L163)

### Atlas A3 系列（GLM-5 / GLM-5.2 目标平台）

GLM-5：FP8 → W8A8（int8）或 MXFP8；GLM-5.2：BF16 → W8A8。与 5.3 的 HiF8/MXFP4 分档不是同一套方案。

models/glm_5 与 models/glm_5_2 配方面向 A3，提供 Docker 镜像与 weight_convert.sh。

GLM-5 镜像 cann9.1_pt2.8.0_glm_aarch_image_v0.2；GLM-5.2 为 v0.1；两者均要求 Ascend HDK 25.2.0。GLM-5.2 的真实注意力维度需 CANN 9.1 的 mla_prolog_v3 内核。

A3 与 950DT 是不同代际，量化档位、算子包与图模式支持都不同，三个版本的吞吐数字不可跨平台比较。

来源：[cann-recipes-infer / models/glm_5/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5/README.md)；[cann-recipes-infer / models/glm_5_2/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_2/README.md)；[cann-recipes-infer / models/glm_5_2/models/configuration_glm.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_2/models/configuration_glm.py)

## Ascend 优先验证与优化

### P0 先固定可复现的基线

未实测 · 仅源码与配方核验 · GLM-5.3-Flash / 950DT / Hybrid HiF8-MXFP8-MXFP4

观察：README 给出 8 卡 8K 序列下 BS 1/16/64/128 的 decode 每步耗时与 TPS，但没有写明 CANN、torch_npu、固件驱动版本、重复次数与预热方式，也没有给出精度结果。

方案：锁定 CANN、torch_npu、驱动固件、两个自定义算子包版本、模型与转换脚本 revision，先跑小批准确性与短服务烟测，再按 README 的 BS 档位复现吞吐。

指标：启动成功率；固定输入的输出一致性；decode 每步耗时与 TPS 的 p50/p95；每卡峰值显存；完整环境清单。

风险：算子包与 CANN 版本不匹配会影响 ABI 与图捕获。未记录版本的吞吐数字无法归因，也无法作为后续优化的对照。

来源：[cann-recipes-infer / models/glm_5_3/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/README.md#L119)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)；[cann-recipes-infer / models/glm_5_3/set_env.sh](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/set_env.sh)

### P0 验证分模块量化的精度代价

未实测 · 仓库无精度数据 · 路由专家 MXFP4 W4A8 / 共享专家 MXFP8 / DSA 投影与 dense MLP HiF8

观察：转换脚本按张量名把权重分成三个精度档，最大的一档（路由专家）只有 4 bit。仓库没有任何针对本配方（HiF8）的精度或困惑度结果。根 README 的困惑度 3.5888 属另一条单卡 offload 路径（INT8 W8A8 加 MXFP4，wikitext-2 32 窗，相对基线 +1.13%），该路径另有 GSM8K 97.35% 与 GPQA-Diamond 84.85%；这些都不能归到本配方。

方案：以 BF16 为基准，逐档打开量化（先 HiF8，再 MXFP8，再 MXFP4），比较 logits 误差与固定任务集质量，定位哪一档带来主要损失。

指标：相对 BF16 的 logits/隐藏态误差分布；固定评测集准确率；困惑度；每档的显存与吞吐变化。

风险：把不同来源的精度数字混为一谈会得出错误结论。W4A8 与 W4A4 的落盘权重相同，仅靠 config 切换，比较时必须确认实际激活位宽。

来源：[cann-recipes-infer / models/glm_5_3/utils/convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/convert_model.py#L445)；[cann-recipes-infer / models/glm_5_3/utils/convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/convert_model.py#L306)；[cann-recipes-infer / module/quantization/mxfp4.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/mxfp4.py)

### P1 确认 mHC 融合算子的形状分支是否命中

未实测 · 算子文档仅有定性描述 · npu_hc_pre / npu_hc_post，每层两对，共 90 次调用

观察：算子文档说明 T/bs ≤ 128 且能被 16 整除时走融合 kernel，否则回退到两个拆分 kernel，并分别标注“性能较高 / 较低”，但没有给出任何数字。mHC 每层要做两次全流量混合，调用次数与层数同阶。可参照的量级来自另一条单卡 offload 路径的 trace：单步 NPU 关键路径 33.3 ms 中 mHC 占 3.3 ms（约 10%）——那是 INT8 加 MXFP4 的单卡路径，不是本配方在 950DT 上的数字。

方案：在目标 BS 与序列切分下抓 trace，确认实际走的是融合还是回退分支；对比不同 batch 切分下 hc_pre/hc_post 的耗时占比。

指标：hc_pre / hc_post 在单步 decode 中的耗时占比；融合与回退分支的耗时差；随 BS 变化的曲线。

风险：形状分支由运行时 shape 决定，换 BS 或并发可能静默落到慢路径。仓库没有实测数据，不能预设融合一定更快。

来源：[cann-recipes-infer / ops/ascendc/docs/custom-npu_hc_pre.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/ops/ascendc/docs/custom-npu_hc_pre.md)；[cann-recipes-infer / ops/ascendc/docs/custom-npu_hc_post.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/ops/ascendc/docs/custom-npu_hc_post.md)；[cann-recipes-infer / models/glm_5_3/models/modules.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modules.py#L102)

### P1 量化共享专家多流的实际重叠收益

未实测 · 仓库无 A/B 数据 · decode 且 moe_ep_size > 1 的 dispatch / combine 路径

观察：enable_multi_streams 默认开启，README 描述它与 dispatch / GMM / combine 重叠，但本配方没有给出开关对比。prefill 与单卡 EP 下该路径不生效。另一条单卡 offload 路径上，侧流重叠的实测增益约 12%（A5 上 −5.97 ms/token），并由 trace 数出每步 44.1 个 >50 µs 空隙合计 21,038 µs 的结构上限——同样不可挪用到本配方。

方案：固定其他条件，只切换 enable_multi_streams，在多个 BS 档位比较 decode 每步耗时；同时抓 trace 确认侧流是否真正与通信重叠。

指标：decode 每步耗时差（开/关）；通信与计算的重叠比例；共享专家在时间线上的位置。

风险：小 batch 下通信占比高，收益可能明显；大 batch 下路由专家 GMM 变长，重叠窗口关系会变化。单点结论不能外推。

来源：[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L340)；[cann-recipes-infer / models/glm_5_3/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/README.md#L95)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)

### P1 检查 indexer 在 prefill 的 torch 路径是否成为长序列瓶颈

未实测 · prefill 无对应融合算子 · 11 层 DSA 的 indexer，prefill 阶段

观察：decode 使用 npu_lightning_indexer 在压缩池上选 top-k，prefill 没有等价算子，仍走 torch matmul 与 topk，并按 QUERY_BLOCK=1024 分块以控制显存。配方的吞吐表只给了 decode 指标。

方案：在 8K 与更长输入下分别抓 prefill trace，测量 indexer 相关算子占比，评估是否需要为 prefill 补融合实现。

指标：prefill 阶段 indexer 的耗时占比与显存峰值；随输入长度的增长曲线；与 KDA 层短卷积的占比对照。

风险：只看 decode 吞吐会掩盖 prefill 的问题，长输入场景下 TTFT 可能由这一段主导。

来源：[cann-recipes-infer / models/glm_5_3/models/indexer.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/indexer.py#L210)；[cann-recipes-infer / models/glm_5_3/models/indexer.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/indexer.py#L73)；[cann-recipes-infer / models/glm_5_3/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/README.md#L119)

### P2 分别测量 npugraph_ex 对 decode 与 prefill 的影响

未实测 · 仓库无对比数据 · exe_mode: npugraph_ex 与 eager

观察：图模式只捕获 decode step，prefill 仍是 eager；多处代码为满足 fullgraph 约束而特意写成无 host 读取。仓库没有给出两种模式的对比数字。

方案：固定量化与并行配置，只切换 exe_mode，分别记录 prefill 与 decode 指标，并确认图捕获是否成功、有无回退。

指标：decode 每步耗时（两种模式）；首步与稳态的差异；图捕获成功率与回退次数；TTFT。

风险：穿插的 prefill 可能影响已捕获的 decode 缓冲，代码里已有针对性处理，测试时要覆盖 prefill 与 decode 交替的真实调度。

来源：[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L733)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L372)；[cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)

### P2 核实路由专家 GMM 的 swiglu clamp 是否真正生效

已核验的代码一致性疑点 · 路由专家 MXFP4 GMM 的 SwiGLU 段

观察：模型向量化配置传入 {"swiglu_limit": ..., "enable_custom_ops": True}，但仓库内除该行外没有任何位置读取 enable_custom_ops；MXFP4 的 GMM 因此走默认的 npu_swiglu_mx_quant 分支，clamp_limit 未参与。共享专家的 MXFP8 路径则显式传了 clamp_limit。

方案：确认这是否是预期行为：若 clamp 对路由专家同样必要，需要补上；若不必要，应移除这个不被消费的参数以免误导。

指标：开启/关闭 clamp 时路由专家输出的数值分布与溢出计数；对最终精度的影响。

风险：这是源码阅读得出的疑点，未经运行确认。也可能由算子包内部默认处理，需要结合算子实现核对。

来源：[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L150)；[cann-recipes-infer / module/quantization/mxfp4.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/mxfp4.py#L200)；[cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py#L90)

## 统一实验协议

固定完整软硬件清单：CANN、torch_npu、固件驱动、两个自定义算子包版本、镜像 digest、模型 revision 与转换脚本 revision。不同量化档位不能混作同一条件。

分别测试 prefill 与 decode。配方吞吐表只覆盖 decode、8K 输入、BS 1/16/64/128；超出该范围的测试单列为探索性结果，不与原表并列。

每项优化单独 A/B：enable_multi_streams、exe_mode、量化档位各自独立切换，先抓逐模块 trace 再做单项对比。记录重复次数与误差范围。

低精度必须同时报告质量：相对 BF16 的 logits 误差与固定任务集结果。只报吞吐不报精度的低位宽结论不予采用。

跨卡数或跨平台比较时同时报告总卡数、拓扑与实际采集口径。950DT 与 Atlas A3 的数字不互相换算，也不合并排名。

## 版本边界

GLM-5.3-Flash 的层排布已与智谱发布的 config.json 核对一致：layer_types 为 34 个 linear_attention 加 11 个 deepseek_sparse_attention，linear_attn_config.full_attn_layers 为 [3,7,11,15,19,23,27,31,35,39,43]，mlp_layer_types 为前 3 层 dense 加 42 层 sparse，first_k_dense_replace=3 与实现默认值相同，没有覆盖。据此推导的解码主干加 embed/lm_head/末层归一为 313,326,811,966。配方 README 自述“约 306B”与该口径不符；仓库未给出推导过程，本站不替任一方消解该差异。

发布检查点是多模态的，含 24 层视觉塔（vision_config.depth=24，hidden_size=1024，out_hidden_size=4096），以及 1 个 MTP 层（num_nextn_predict_layers=1，位于第 45 层，同为 DSA + MoE）。本配方两者都不构建：配置类丢弃 vision_config、加载时跳过 model.visual.*，运行时直接拒绝 next_n>0。因此本站对这两部分只记录配置级信息，不给逐矩阵形状。

发布权重的量化布局有官方逐张量记录：quantization_config 为 FP8 e4m3、动态激活，modules_to_not_convert 列出 1509 条不量化张量。其覆盖面与本配方 convert_model.py 的 classify() 判定一致——KDA 全部投影、DSA 的 kv_b_proj、整个 indexer、mlp.gate、各 norm、mHC 参数、embed/lm_head 均在排除之列，而路由专家、共享专家、dense MLP 与 DSA 的四个投影不在其中。该列表同时出现 self_attn.qkv_proj 与 fused_qkvbfg_a_proj 两个融合名，说明发布检查点可能以融合张量存储 KDA 的 q/k/v 与 a 投影；本站按实现侧的拆分名记录形状。

GLM-5 与 GLM-5.2 仅通过配方的配置类默认值与 README 记录，没有逐矩阵结构，也未与厂商 config.json 核对。两者的配置默认值在本快照中完全一致，差异体现在 GLM-5.2 新增的 IndexShare 调度参数（index_topk_freq 等）与其 README 说明。

实现与部署材料来自 Huawei 的 CANN 推理配方仓库，不是厂商文档；这部分事实记为“计算推导”。GLM-5.3-Flash 的模型规格另有厂商 config.json 支撑，记为“官方披露”。本站未运行任何 NPU 实验，仓库内也没有 profiling 数据。

## 来源索引

- g1 [cann-recipes-infer / models/glm_5_3/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/README.md)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g2 [cann-recipes-infer / models/glm_5_3/config/glm_5_3_flash_ep8.yaml](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/config/glm_5_3_flash_ep8.yaml)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g3 [cann-recipes-infer / models/glm_5_3/models/configuration_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/configuration_glm53.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g4 [cann-recipes-infer / models/glm_5_3/models/modeling_glm53.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g5 [cann-recipes-infer / models/glm_5_3/models/kda.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/kda.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g6 [cann-recipes-infer / models/glm_5_3/models/indexer.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/indexer.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g7 [cann-recipes-infer / models/glm_5_3/models/modules.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modules.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g8 [cann-recipes-infer / models/glm_5_3/utils/convert_model.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/convert_model.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g9 [cann-recipes-infer / models/glm_5_3/utils/mx_quantize.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/utils/mx_quantize.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g10 [cann-recipes-infer / module/quantization/compressed_tensors/compressed_tensors_w8a8_hif8.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/compressed_tensors/compressed_tensors_w8a8_hif8.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g11 [cann-recipes-infer / models/glm_5_3/set_env.sh](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/set_env.sh)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g12 [cann-recipes-infer / module/quantization/mxfp4.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/mxfp4.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g13 [cann-recipes-infer / module/quantization/mxfp8.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/mxfp8.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g14 [cann-recipes-infer / module/quantization/compressed_tensors/compressed_tensors.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/module/quantization/compressed_tensors/compressed_tensors.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g15 [cann-recipes-infer / executor/core/config/inference_config.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/executor/core/config/inference_config.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g16 [cann-recipes-infer / ops/ascendc/docs/custom-npu_hc_pre.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/ops/ascendc/docs/custom-npu_hc_pre.md)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g17 [cann-recipes-infer / ops/ascendc/docs/custom-npu_hc_post.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/ops/ascendc/docs/custom-npu_hc_post.md)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g18 [cann-recipes-infer / models/glm_5/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5/README.md)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g19 [cann-recipes-infer / models/glm_5_2/README.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_2/README.md)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g20 [cann-recipes-infer / models/glm_5/models/configuration_glm.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5/models/configuration_glm.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g21 [cann-recipes-infer / models/glm_5_2/models/configuration_glm.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_2/models/configuration_glm.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g22 [cann-recipes-infer / docs/integration/sglang/glm53-flash-single-npu-moe-offload/glm53_flash_single_card_design.md](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/docs/integration/sglang/glm53-flash-single-npu-moe-offload/glm53_flash_single_card_design.md)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- g23 [cann-recipes-infer / executor/utils/profiler_context.py](https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/executor/utils/profiler_context.py)；3b36eeac4cde5e608f00cdd0560a715ff8925e15；访问 2026-10-19

- hf53 [zai-org/GLM-5.3-Flash / config.json（厂商发布配置）](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)；main；访问 2026-10-19