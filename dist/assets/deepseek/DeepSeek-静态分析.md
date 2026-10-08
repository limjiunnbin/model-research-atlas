# DeepSeek 静态成本、算子契约与系统分析

资料更新：2026-10-08。21 个 checkpoint 的计算与并行模型、V4.1 部署缓存模型分别说明；设备实验不属于当前范围。

## 1 成本模型

覆盖 21 个 checkpoint 的全部 1,122 个主干层。参数来自 canonical 架构和各自完整文件头；计算器按声明矩阵收缩计数，主干、共享副本和 MTP/DSpark 独立。

| 项目 | 公式 |
| --- | --- |
| 投影收缩 | 2×B×Q×Σ(in×out×调用份数)；routed 专家按 top-k，APE 位置表与 embedding lookup 不算矩阵乘法 |
| Dense/GQA attention | 2×B×N_head×Q×L_kv×(D_key+D_value)；有效因果位置为 Q×history + Q×(Q+1)/2 |
| DSA | Indexer = 2×B×Q×L_kv×N_index_head×D_index；参考 dense score/mask 路径与 top-k 有效位置分列 |
| V4 | C=floor(L_kv/r)；每 query 有效位置=min(window,t)+min(index_topk,floor(t/r))（HCA 不设 index_topk） |
| MLA cache | latent=B×L_kv×(R_kv+D_rope)；expanded=B×L_kv×N_head×(D_nope+D_rope+D_value) |
| GQA cache | B×L_kv×2×N_KV×D_head 个元素 |
| V4 cache/state | 主 KV 按 window+C；CSA index cache 独立；每 compressor 的两个 FP32 状态为 2×B×(coff×r)×(coff×D)，coff=2 当 r=4 |
| 输出 | head=2×B×logits_token_count×V×H；verify 必须选择全部 query logits |

- 成本使用完整模型、MP=1 的逻辑形状；buffer 按所选 batch 与上下文容量假设，实际 max_batch/max_seq、分页、分片与预留另计。
- 仅统计声明的矩阵收缩；norm、softmax、RoPE、routing/top-k、Sinkhorn、非线性、量化转换与通信算术另列，不称完整 FLOPs。
- V4 槽位乘法是上界，masked 槽位是否跳过依实际 kernel；有效数学位置另列。
- 参考 logits/score 张量的逻辑大小不证明设备分配；cache 精度是显式假设，FP8/FP4 模拟不当作实际缓存量化。
- 投影输入+输出字节是逻辑消费代理，重复消费重复计，不作为物理 HBM 流量。
- 主干与 MTP/DSpark 分开；未测延迟、吞吐、峰值显存和通信耗时始终为 null。

成本表在 [静态分析工作簿](DeepSeek-静态分析.xlsx) 与 [示例 CSV](DeepSeek-cost-scenarios.csv) 中。网页可修改 batch、query/history、logits 和精度假设；每个结果保留输入，不写入未测时间或吞吐。

## 2 DeepSeek-V4.1-Flash 部署缓存

按真实 FP4/FP8 数据与块 scale 的打包容量统计。主 KV 为 NVFP4-like 变体，E2M1 数据加每 16 元素 E4M3 scale，省略第二级全局 scale；窗口为 MXFP8，E4M3 数据加每 32 元素 E8M0 scale；index 为 MXFP4，E2M1 数据加每 32 元素 E8M0 scale。

| 项目 | 每请求容量 / 字节 |
| --- | --- |
| 全局 KV 与 index | (3 × floor(N/2) + N) × 356；偶数 N 为 890N |
| Main KV 记录 | 288 |
| Index K 记录 | 68 |
| 窗口 KV 记录 | 528 |
| 40 层 × 128 token 窗口 | 2703360 |
| 3-stage DSpark context（可选） | 202752 |
| FP32 压缩状态 | 24576 |

N 为已缓存序列长度，batch B 按请求数相乘。B=1、N=1,048,576 时，全局缓存为 890 MiB；加语言窗口、DSpark context 和压缩状态约为 892.795 MiB。权重、分页对齐、临时工作区、候选池/分数张量和通信缓冲另计。

参考代码的量化/反量化模拟与实际打包容量分开说明。V4.1 的 FLOPs 和并行通信未沿用旧 21 版公式；完整结构与边界见 [V4.1 Flash 研究](DeepSeek-V4.1-Flash-研究.md)，网页入口为 `#/family/deepseek/cost/v4.1-flash`。

## 3 CANN 算子契约

覆盖 30 个已固定源码家族、4207 条 API 参数表/补充条件/API guard 记录。参数表逐项保存所在文档和源码范围，不把同文档中不同平台/模式的条件合成无条件支持。

网页有按 API 文档、参数和类别筛选的手册，以及已编码必要条件的检查器。检查器发现矛盾时指出条件；未发现矛盾只表示该子集没有冲突，其他条件保持未核对。

下表用于定位算子家族。全部条件、章节上下文与固定行号见 [完整条件 CSV](DeepSeek-CANN-contracts.csv)、[静态分析 JSON](DeepSeek-静态分析.json) 和工作簿“算子条件”页。

| 算子家族 | API 文档数 | 条件数 | 框架上下文 | 固定来源 |
| --- | --- | --- | --- | --- |
| fused-infer-attention | 5 | 1134 | npu_fused_infer_attention_score / v2 / workspace / out | [aclnnFusedInferAttentionScore](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScore.md)；[aclnnFusedInferAttentionScoreV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV2.md)；[aclnnFusedInferAttentionScoreV3](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md)；[aclnnFusedInferAttentionScoreV4](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md)；[aclnnFusedInferAttentionScoreV5](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md) |
| sparse-flash-attention | 2 | 64 | sparse attention dispatcher | [aclnnSparseFlashAttention](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_attention/docs/aclnnSparseFlashAttention.md)；[aclnnSparseFlashAttentionV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_attention/docs/aclnnSparseFlashAttentionV2.md) |
| sparse-flash-mla | 1 | 83 | V4 sparse MLA | [aclnnSparseFlashMla](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_mla/docs/aclnnSparseFlashMla.md) |
| mixed-quant-sparse-mla | 1 | 123 | V4 mixed-quant sparse MLA | [aclnnMixedQuantSparseFlashMla](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/mixed_quant_sparse_flash_mla/docs/aclnnMixedQuantSparseFlashMla.md) |
| sparse-mla-metadata | 1 | 117 | metadata preparation | [aclnnSparseFlashMlaMetadata](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_mla_metadata/docs/aclnnSparseFlashMlaMetadata.md) |
| lightning-indexer | 1 | 26 | indexer | [aclnnLightningIndexer](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/lightning_indexer/docs/aclnnLightningIndexer.md) |
| lightning-indexer-v2 | 1 | 38 | V4 indexer | [aclnnLightningIndexerV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/lightning_indexer_v2/docs/aclnnLightningIndexerV2.md) |
| quant-lightning-indexer | 1 | 29 | quant indexer | [aclnnQuantLightningIndexer](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/quant_lightning_indexer/docs/aclnnQuantLightningIndexer.md) |
| quant-lightning-indexer-v2 | 1 | 40 | _C_ascend.npu_quant_lightning_indexer_v2 | [aclnnQuantLightningIndexerV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md) |
| compressor | 1 | 48 | _C_ascend.compressor / aclnnCompressor | [aclnnCompressor](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/compressor/docs/aclnnCompressor.md) |
| mhc-pre | 2 | 56 | official MhcPre; framework custom HcPreV2 kept separate | [aclnnMhcPre](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mhc/mhc_pre/docs/aclnnMhcPre.md)；[aclnnMhcPreV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mhc/mhc_pre/docs/aclnnMhcPreV2.md) |
| mhc-post | 1 | 14 | official MhcPost; framework custom HcPost kept separate | [aclnnMhcPost](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mhc/mhc_post/docs/aclnnMhcPost.md) |
| grouped-matmul | 6 | 744 | npu_grouped_matmul | [aclnnGroupedMatmul](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/docs/aclnnGroupedMatmul.md)；[aclnnGroupedMatmulV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/docs/aclnnGroupedMatmulV2.md)；[aclnnGroupedMatmulV3](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/docs/aclnnGroupedMatmulV3.md)；[aclnnGroupedMatmulV4](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md)；[aclnnGroupedMatmulV5](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)；[aclnnGroupedMatmulWeightNz](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md) |
| grouped-matmul-swiglu | 2 | 58 | grouped_matmul_swiglu_quant | [aclnnGroupedMatmulSwigluQuant](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)；[aclnnGroupedMatmulSwigluQuantWeightNZ](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md) |
| grouped-matmul-swiglu-v2 | 2 | 144 | V2 API alternative | [aclnnGroupedMatmulSwigluQuantV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md)；[aclnnGroupedMatmulSwigluQuantWeightNzV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantWeightNzV2.md) |
| moe-dispatch-v2 | 3 | 361 | npu_moe_distribute_dispatch_v2 | [aclnnMoeDistributeDispatchV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV2.md)；[aclnnMoeDistributeDispatchV3](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)；[aclnnMoeDistributeDispatchV4](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md) |
| moe-dispatch-v3 | 0 | 1 | plugin V3/V4 alternatives | [aclnn_moe_distribute_dispatch_v5.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v3/op_api/aclnn_moe_distribute_dispatch_v5.cpp#L29-L39) |
| moe-combine-v2 | 3 | 377 | npu_moe_distribute_combine_v2 | [aclnnMoeDistributeCombineV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV2.md)；[aclnnMoeDistributeCombineV3](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV3.md)；[aclnnMoeDistributeCombineV4](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md) |
| moe-combine-v3 | 0 | 1 | plugin V3/V4 alternatives | [aclnn_moe_distribute_combine_v5.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v3/op_api/aclnn_moe_distribute_combine_v5.cpp#L27-L40) |
| rotary-position | 2 | 42 | official rotary operator; custom inplace partial rotary is distinct | [aclnnRotaryPositionEmbedding](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbedding.md)；[aclnnRotaryPositionEmbeddingV2](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbeddingV2.md) |
| rms-norm | 1 | 24 | npu_rms_norm | [aclnnRmsNorm](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/norm/rms_norm/docs/aclnnRmsNorm.md) |
| add-rms-norm | 1 | 29 | npu_add_rms_norm; custom bias path is distinct | [aclnnAddRmsNorm](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/norm/add_rms_norm/docs/aclnnAddRmsNorm.md) |
| swiglu | 1 | 13 | npu_swiglu generated op_api | [aclnnSwiGlu](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swi_glu/docs/aclnnSwiGlu.md) |
| swiglu-group-quant | 2 | 71 | npu_swiglu_group_quant | [aclnnSwigluGroupQuant](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swiglu_group_quant/docs/aclnnSwigluGroupQuant.md)；[torchapi_swiglu_group_quant](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swiglu_group_quant/docs/torchapi_swiglu_group_quant.md) |
| matmul | 5 | 174 | Mm/Matmul API candidates; not an unconditional GEMM dispatch | [aclnnAddmm&aclnnInplaceAddmm](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/docs/aclnnAddmm&aclnnInplaceAddmm.md)；[aclnnAddmmWeightNz](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/docs/aclnnAddmmWeightNz.md)；[aclnnMatmul](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/docs/aclnnMatmul.md)；[aclnnMatmulWeightNz](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/docs/aclnnMatmulWeightNz.md)；[aclnnMm](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/docs/aclnnMm.md) |
| quant-matmul | 1 | 158 | quantized GEMM candidate | [aclnnQuantMatmulV5](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md) |
| dynamic-mx-quant | 3 | 86 | MX quantization | [aclnnDynamicMxQuant](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/quant/dynamic_mx_quant/docs/aclnnDynamicMxQuant.md)；[aclnnDynamicMxQuantV2](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/quant/dynamic_mx_quant/docs/aclnnDynamicMxQuantV2.md)；[aclnnDynamicMxQuantV3](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/quant/dynamic_mx_quant/docs/aclnnDynamicMxQuantV3.md) |
| add | 4 | 108 | mathematical building block; not a per-step kernel assignment | [aclnnAdd&aclnnInplaceAdd](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/add/docs/aclnnAdd&aclnnInplaceAdd.md)；[aclnnAddV3&aclnnInplaceAddV3](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/add/docs/aclnnAddV3&aclnnInplaceAddV3.md)；[aclnnAdds](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/add/docs/aclnnAdds.md)；[aclnnInplaceAdds](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/add/docs/aclnnInplaceAdds.md) |
| mul | 1 | 27 | mathematical building block; not a per-step kernel assignment | [aclnnMul&aclnnInplaceMul](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/mul/docs/aclnnMul&aclnnInplaceMul.md) |
| cast | 1 | 17 | experimental Cast; availability not production selection | [aclnnCast](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/experimental/math/cast/docs/aclnnCast.md) |

## 4 并行、通信与 P/D

| 项目 | 公式与条件 |
| --- | --- |
| EP | EP = TP × DP（所选 vLLM EP 配置） |
| 权重容量 | bytes/rank = w × [P_replicated + P_nonexpert / TP + P_expert × (1 + R/E) / EP]；复制假设将第二项除数改为 1 |
| 远端路由 | 均匀放置/路由假设：remote_fraction = (EP−1)/EP；也可手填 [0,1] |
| MoE 逻辑发送 | 每 MoE 层 dispatch = B×Q×DP×K×H×activation_bytes×remote_fraction；combine 同宽；所有层再乘 MoE 层数 |
| Ring AllReduce | 每 rank 每次发送 = 2×(TP−1)/TP × message_bytes；全组发送量再乘 TP |
| P/D | 按一个 DP replica 的有效主 KV 记录计；不含预留窗口空槽、FP32 压缩状态或其他协议数据 |

### TP、DP 与 EP 的进程组

所选 vLLM 开启 EP 后按 TP×DP 构造专家范围；attention 仍在各 DP replica 内按 TP 分片。PP 是另外的层分片维度，本计算器不建模 PP。

来源：[docs/serving/expert_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/docs/serving/expert_parallel_deployment.md#L37-L51)；[vllm/distributed/parallel_state.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/parallel_state.py#L1973-L1987)；[vllm/config/parallel.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/config/parallel.py#L171-L185)

### 每 rank 权重与冗余专家

容量公式采用两种明确假设：非专家权重复制，或非专家二维线性权重/embedding/head 理想均分到 TP；向量和位置表复制。具体模型存在复制的低秩投影和独立分派，不将理想均分写成实际 loader 布局。专家及冗余副本按 EP 均分；量化 scale、对齐、workspace、辅助模型另计。

来源：[docs/serving/expert_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/docs/serving/expert_parallel_deployment.md#L177-L191)；[vllm/config/parallel.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/config/parallel.py#L74-L88)；[vllm/distributed/eplb/eplb_state.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/eplb/eplb_state.py#L232-L246)

### MoE dispatch 与 combine

逻辑 token→expert 分配数为 N_global×top-k。发送量按远端路由比例和 hidden 向量字节计算；combine 使用相同宽度假设。每条逻辑 payload 只计发送一次，不再把接收重复相加。全局平均不代表最忙 rank，真实通信还受量化、padding、容量、token 去重及 SP 影响。

来源：[docs/serving/expert_parallel_deployment.md](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/docs/serving/expert_parallel_deployment.md#L18-L32)；[vllm/distributed/device_communicators/all2all.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/device_communicators/all2all.py#L50-L64)；[vllm/distributed/device_communicators/all2all.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/device_communicators/all2all.py#L111-L131)；[vllm/distributed/device_communicators/all2all.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/device_communicators/all2all.py#L146-L158)

### TP collective 与算法边界

展示 ring AllReduce 的每 rank 发送字节公式。每层次数由用户指定，默认只是分析假设；不把 ring 公式推广成实际 HCCL/NCCL 算法或链路时间。AllGather/ReduceScatter 与 all-to-all 是独立算法，backend 名称不证明物理链路选择。

来源：[vllm/distributed/parallel_state.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/parallel_state.py#L1545-L1559)；[vllm_ascend/distributed/device_communicators/pyhccl.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/distributed/device_communicators/pyhccl.py#L138-L152)

### EPLB 的映射、量化与 runner 条件

EPLB 基于专家负载调整逻辑专家与物理副本映射，冗余副本消耗额外容量。所选 Ascend 文档明确区分 MRv1 与 MRv2 配置，且列出各量化格式的 Enabled/Rejected 条件；A2 不支持冗余专家的文档声明单独保存，不由其他平台代填。作者验证声明不等于本项目验证。

来源：[vllm/distributed/eplb/eplb_state.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/eplb/eplb_state.py#L858-L872)；[docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md#L5-L27)；[docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md#L40-L60)

### Prefill/decode 分离与 cache 迁移

P/D 分离把预填充和生成调度到不同实例，connector 协调 cache 保存/加载及完成信号。逻辑迁移量按有效主 KV 记录计，index cache 可独立列出；压缩状态、格式转换、分片重排、传输副本、网络协议和隐藏特征另计。需要两端布局与位置语义相容，不能只比较总字节。

来源：[docs/features/disagg_prefill.md](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/docs/features/disagg_prefill.md#L1-L13)；[vllm/distributed/kv_transfer/kv_connector/v1/base.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L311-L325)；[vllm/distributed/kv_transfer/kv_connector/v1/base.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L364-L378)；[docs/source/developer_guide/Design_Documents/disaggregated_prefill.md](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md#L17-L31)

## 5 跨家族统一比较

沿用现有三个家族的 43 个条目，包含历史/未核对条目；网页可按家族筛选、添加版本，并保留每个事实的证据类别和来源。

- 家族统计口径和读取深度不同；参数、发布载荷和辅助模块范围保留各自说明。
- null 表示未知或未收录，不代表 0 或无该机制。
- 配置相同、参数相近和模型名称相近不证明权重、能力或速度相同。
- 不提供缺少统一实测条件的速度、价格或能力排名。

## 6 机制演示

### MLA：展开与吸收

教学合成矩阵，以参考展开投影为起点做代数吸收；不包含真实权重、低精度误差、位置外推或 kernel 融合。

来源：[DeepseekV3Attention.forward](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/afb92e1fa402c2be2a9eb085312bb02e0384d6c7/modeling_deepseek.py#L750-L856)

### DSA：Indexer 与选择

Indexer 分数与 attention 分数分开；教学选择不证明模型质量。

来源：[Indexer.forward](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/194c67e12b1b0d6df0ef373ddcf215bc84027409/inference/model.py#L457-L487)；[MLA.forward](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/194c67e12b1b0d6df0ef373ddcf215bc84027409/inference/model.py#L545-L608)

### V4：分块、overlap 与状态

固定教学 gating，仅展示分块/overlap/尾部状态；不模拟真实投影、RoPE、量化或全部 cache 管理。

来源：[Compressor.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L316-L377)；[Attention.forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L484-L543)

### mHC：四流与 Sinkhorn

pre/post 使用指定教学数值，子层为 tanh 示例；comb 用 Sinkhorn 展示行列和，不代表真实控制投影。

来源：[Block.hc_pre](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L673-L681)；[Block.hc_post](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L683-L686)

### DSpark：并行草稿与顺序验证

教学词表、logits、采样与 confidence；不体现真实训练校准或生产 scheduler。checkpoint block、native stage 和服务 token 数仍分别保存。

来源：[DSparkBlock.forward_embed](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py#L851-L858)；[DSparkBlock.forward_head](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py#L860-L874)

演示包含前后步进和可调输入。MLA 比较两条路径的分数和输出；DSA 分开 indexer 分数与 attention 分数；压缩保留未满窗口状态；mHC 展示 comb 的行列和；DSpark 提交接受前缀及校正/bonus，丢弃未验证草稿。

## 7 复现与验证

运行 collect_deepseek_analysis.py 获取固定公开输入，--verify-only 仅核对缓存；运行 build_deepseek_analysis.py 从既有 canonical 数据生成分析 JSON、手册和跨家族表。V4.1 输入由 build_deepseek_v41.py 固定并恢复。validate_deepseek_analysis.py 与 check_deepseek_analysis.mjs 分别验证来源/字段和独立数学基准，validate_deepseek_v41.py 核对本版结构和打包公式，check_deepseek_site.mjs 验证网页交互与下载。

普通校验仅输出终端结果，浏览器截图使用临时目录；只有明确传入 --write-record 才保存校验记录。重建分析工作簿前用 check_deepseek_analysis.mjs --export-scenarios 生成当前导出输入。完整命令和当前附件覆盖范围见 [复现说明](README.md)。设备实验不属于本次验收。
