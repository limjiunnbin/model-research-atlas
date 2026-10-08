# DeepSeek 实现、硬件与 Ascend 优化研究

研究快照：2026-10-01

本次交付为固定来源的静态研究与网站；设备运行/性能验证已按用户要求移出本次范围。

## MLA 投影、cache 与 prefill/decode

模型 forward → Ascend MLA wrapper → mla_forward 分派 → prefill/decode 后端 → 条件 attention/device API

低秩 Q/KV、位置 Key 与 cache 管理分开；prefill 与 decode 的数据布局/FA/图模式分支分别记录。

wrapper.forward 调用 torch.ops.vllm.mla_forward；AscendMLAImpl.forward 再分派，_forward_decode 包含普通/NZ/FA quant cache 布局及 query padding/SpecDecoding 条件。 CANN 匹配 tag 来源：attention/fused_infer_attention_score, posembedding/rotary_position_embedding。

支持条件包括 device、head 数/维度、cache 精度与 NZ、block table、TP/CP 和图模式；读取函数不证明这些条件组成可运行栈。

CANN attention 的匹配版本开源实现与 API 已可查；实际所走分支、ABI 与数值尚未在设备验证。没有原始 FP8 权重加载、转换或设备测试。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/mla.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/mla.py#L269)；[vllm-project/vllm-ascend / vllm_ascend/attention/mla_v1.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/attention/mla_v1.py#L2133)；[vllm-project/vllm-ascend / vllm_ascend/attention/mla_v1.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/attention/mla_v1.py#L1365)；[vllm-project/vllm-ascend / vllm_ascend/attention/mla_v1.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/attention/mla_v1.py#L1640)；[cann/ops-transformer / attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp#L20)；[cann/ops-transformer / posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp#L16)

## DSA 与 Lightning Indexer

低秩 Query → Indexer 与 scale cache → top-k / causal mask → 稀疏 attention 分派

V3.2 在主 attention 外增加 Indexer，缓存、scale 与 top-k 长度独立。

AscendDeepseekSparseAttention.forward 调用 dsa_forward；AscendDSAImpl 中 prolog、indexer、cache update 和 attention 各有单流/多流或设备条件。 CANN 匹配 tag 来源：attention/sparse_flash_attention, attention/lightning_indexer, attention/quant_lightning_indexer, posembedding/rotary_position_embedding。

index_topk、key/head dim、量化 scale、padding、metadata 和硬件代际决定实现；模型卡的 DSA 算法不等于任意 npu kernel 支持。

V4 也使用部分 DSA 基础设施，但其 cache 与压缩语义单独建模，不能照搬 V3.2 路径。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/dsa.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/dsa.py#L151)；[vllm-project/vllm-ascend / vllm_ascend/attention/dsa_v1.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/attention/dsa_v1.py#L1769)；[vllm-project/vllm-ascend / vllm_ascend/attention/dsa_v1.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/attention/dsa_v1.py#L2145)；[cann/ops-transformer / attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp#L17)；[cann/ops-transformer / attention/lightning_indexer/op_host/lightning_indexer_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/lightning_indexer/op_host/lightning_indexer_def.cpp#L17)；[cann/ops-transformer / attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp#L16)；[cann/ops-transformer / posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp#L16)

## V4 窗口、CSA/HCA 与 compressor

mHC 折叠后的输入 → Q/KV 投影 → 窗口与 learned 压缩 → CSA indexer 或 HCA 前缀 → 稀疏 attention → 分组低秩输出

ratio=4 具有 overlap 的 learned pooling 与独立 indexer；ratio=128 使用全部压缩前缀；ratio=0 是窗口/MTP 条件。

框架 Compressor.forward 使用 _C_ascend.compressor 分派；Indexer.select_topk 使用 npu_quant_lightning_indexer_v2；source forward 与 cache plan/metadata 相互约束。 CANN 匹配 tag 来源：attention/sparse_flash_mla, attention/mixed_quant_sparse_flash_mla, attention/sparse_flash_mla_metadata, attention/lightning_indexer_v2, attention/quant_lightning_indexer_v2, attention/compressor。 开放底层链：Python Compressor.forward → torch.ops._C_ascend.compressor → PrivateUse1 注册 → vllm_ascend::compressor → EXEC_NPU_CMD(aclnnCompressor) → Compressor host/tiling 与 arch35 kernel 入口。

压缩 record 数 floor(S/r)、窗口 offset、state layout、indexer 精度和更新边界必须对齐；source dispatcher 不是已经验证的 AscendC kernel。

参考 inference 对 KV/indexer 做 QAT 精度模拟后保存默认 dtype；不能据此断言生产框架存储也为 BF16。 ratio/coff、FP32 state、norm/rope shapes、量化 scale、slot mapping 与 token update 在 wrapper/host 层分别检查；arch35 源存在不推定设备编译成功。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/model.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/model.py#L662)；[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/compressor.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/compressor.py#L193)；[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/indexer.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/indexer.py#L199)；[cann/ops-transformer / attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp#L17)；[cann/ops-transformer / attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp#L17)；[cann/ops-transformer / attention/sparse_flash_mla_metadata/docs/aclnnSparseFlashMlaMetadata.md](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_mla_metadata/docs/aclnnSparseFlashMlaMetadata.md#L1)；[cann/ops-transformer / attention/lightning_indexer_v2/op_host/lightning_indexer_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/lightning_indexer_v2/op_host/lightning_indexer_v2_def.cpp#L17)；[cann/ops-transformer / attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_def.cpp#L17)；[cann/ops-transformer / attention/compressor/op_host/compressor_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/compressor/op_host/compressor_def.cpp#L12)；[vllm-project/vllm-ascend / csrc/torch_binding.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/torch_binding.cpp#L3226)；[vllm-project/vllm-ascend / csrc/torch_binding.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/torch_binding.cpp#L910)；[vllm-project/vllm-ascend / csrc/attention/compressor/op_host/compressor_def.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/attention/compressor/op_host/compressor_def.cpp#L12)；[vllm-project/vllm-ascend / csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp#L24)；[vllm-project/vllm-ascend / csrc/attention/compressor/op_kernel/arch35/compressor_kernel.h](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/attention/compressor/op_kernel/arch35/compressor_kernel.h#L24)

## mHC 控制投影、Sinkhorn 与 pre/post

四流残差 → 控制投影与 RMS statistic → pre/post/comb + Sinkhorn → 子层计算 → 四流还原

两套 [24,4H] 控制投影；hc_pre 折叠四流，hc_post 使用 post/comb 还原。

框架 hc_pre/hc_post 分派 npu_hc_pre_v2 / npu_hc_post；TileKernels wrapper 与 *_asc generator 为另一条独立实现来源，不认定框架使用了该库。 CANN 匹配 tag 来源：mhc/mhc_pre, mhc/mhc_post。 开放底层链：DeepseekV4DecoderLayer.hc_pre → torch.ops._C_ascend.npu_hc_pre_v2 → PrivateUse1 注册/shape 检查 → hc_pre host/tiling → 公开 AscendC hc_pre 入口。

TileKernels 所选快照明确要求 Ascend 950、CANN≥9.2、Python≥3.12、PyTorch≥2.13、TileLang≥0.1.15；不能推广到 A2/A3 或旧 torch_npu。

源码生成器存在不等于编译、数值验证或融合收益；kernel 与参考之间的布局/迭代数需后续检查。 wrapper 显式限 hc_mult=4、mix=24；fn/base/scale、dtype、pre_mix 与 Sinkhorn 迭代等分开。TileKernels *_asc 是另一独立实现，没有看到该框架调用 TileKernels 的接线。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/model.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/model.py#L745)；[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/model.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/model.py#L751)；[deepseek-ai/TileKernels / tile_kernels/mhc/pre_big_fuse_kernel.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/mhc/pre_big_fuse_kernel.py#L10)；[deepseek-ai/TileKernels / tile_kernels/mhc/pre_big_fuse_asc.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/mhc/pre_big_fuse_asc.py#L25)；[deepseek-ai/TileKernels / tile_kernels/mhc/sinkhorn_asc.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/mhc/sinkhorn_asc.py#L28)；[cann/ops-transformer / mhc/mhc_pre/op_host/mhc_pre_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mhc/mhc_pre/op_host/mhc_pre_def.cpp#L16)；[cann/ops-transformer / mhc/mhc_post/op_host/mhc_post_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mhc/mhc_post/op_host/mhc_post_def.cpp#L16)；[vllm-project/vllm-ascend / csrc/torch_binding.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/torch_binding.cpp#L3344)；[vllm-project/vllm-ascend / csrc/torch_binding.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/torch_binding.cpp#L1415)；[vllm-project/vllm-ascend / csrc/moe/hc_pre/op_host/hc_pre_def.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/moe/hc_pre/op_host/hc_pre_def.cpp#L18)；[vllm-project/vllm-ascend / csrc/moe/hc_pre/op_kernel/hc_pre.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/moe/hc_pre/op_kernel/hc_pre.cpp#L31)

## 路由、专家 GEMM 与通信

gating scores → Hash/top-k → dispatch / token gather → 专家 GEMM 与 clamp/SwiGLU → combine + shared

V3 的分组 sigmoid 路由与 V4 的 hash/sqrtsoftplus 路由分开；专家 count、选中数、每专家 N_e 与 EP placement 不混用。

MC2 token_dispatch/combine 使用 npu_moe_distribute_dispatch_v2 / combine_v2；MXFP4 GMM1 分支调用 grouped_matmul、swiglu_group_quant，另有非 DeepSeek 的 SITU 分支，不能按调用集合串成执行序列。 CANN 匹配 tag 来源：gmm/grouped_matmul, gmm/grouped_matmul_swiglu_quant, gmm/grouped_matmul_swiglu_quant_v2, mc2/moe_distribute_dispatch_v2, mc2/moe_distribute_dispatch_v3, mc2/moe_distribute_combine_v2, mc2/moe_distribute_combine_v3, activation/swi_glu, activation/swiglu_group_quant。 开放底层链：MoE/量化路径按条件选择融合算子 → _C_ascend grouped_matmul_swiglu_quant 注册 → host dtype/scale/group-list 条件 → aclnn wrapper/workspace/launch。

通信方式、EP/TP、group_list_type、group32 scale、SWIGLU clamp、shared overlap 决定实现；原始专家格式先经对应 loader/转换。

Distill 是 Dense GQA，只适用 FFN/激活共同部分，不适用路由或专家通信；未测试任何通信或 grouped GEMM。 这条 custom-op 与 torch_npu 分步 grouped_matmul + swiglu_group_quant 是不同候选分支；不串成一个执行链。FP4 容器、NZ/ND、group size 和量化 loader 要独立对应。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/fused_moe/moe_mlp.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/fused_moe/moe_mlp.py#L27)；[vllm-project/vllm-ascend / vllm_ascend/ops/fused_moe/token_dispatcher.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/fused_moe/token_dispatcher.py#L227)；[vllm-project/vllm-ascend / vllm_ascend/ops/fused_moe/token_dispatcher.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/fused_moe/token_dispatcher.py#L327)；[vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py#L352)；[deepseek-ai/TileKernels / tile_kernels/moe/topk_gate_kernel.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/moe/topk_gate_kernel.py#L7)；[deepseek-ai/TileKernels / tile_kernels/moe/topk_gate_asc.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/moe/topk_gate_asc.py#L9)；[cann/ops-transformer / gmm/grouped_matmul/op_host/grouped_matmul_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/op_host/grouped_matmul_def.cpp#L16)；[cann/ops-transformer / gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp#L16)；[cann/ops-transformer / gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp#L17)；[cann/ops-transformer / mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp#L17)；[cann/ops-transformer / mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp#L17)；[cann/ops-transformer / mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp#L17)；[cann/ops-transformer / mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp#L17)；[cann/ops-nn / activation/swi_glu/op_host/swi_glu_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swi_glu/op_host/swi_glu_def.cpp#L17)；[cann/ops-nn / activation/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp#L21)；[vllm-project/vllm-ascend / csrc/torch_binding.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/torch_binding.cpp#L2990)；[vllm-project/vllm-ascend / csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp#L16)；[vllm-project/vllm-ascend / csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp#L409)

## FP8/MXFP4、解量化、norm 与布局

权重格式核对 → 加载/转换与 scale → 动态量化或 BF16 fallback → GEMM / norm → 输出精度

V3 原始 E4M3 + F32 block scale 来自文件头；MXFP4/UE8M0 等属于明确的转换或参考条件，不能只按名称等同。

AscendFp8BlockLinearMethod 可选择 mxfp8_method，其他分支走 unquantized_gemm；process_weights_after_loading 必须与 apply 一起核对。RMSNorm.forward_oot 包含 npu_rms_norm 或 add-rms 条件。 CANN 匹配 tag 来源：attention/mixed_quant_sparse_flash_mla, gmm/grouped_matmul, norm/rms_norm, norm/add_rms_norm, activation/swiglu_group_quant, matmul/mat_mul_v3, matmul/quant_batch_matmul_v4, quant/dynamic_mx_quant, math/add, math/mul, experimental/math/cast。

硬件原生格式、转换脚本、dtype、scale 布局、NZ 与 custom-op 包版本须固定；支持一条转换路径不等于直接加载原始 FP8/FP4。

没有质量/误差或吞吐实验，不能填写量化收益。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/layernorm.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/layernorm.py#L83)；[vllm-project/vllm-ascend / vllm_ascend/ops/linear.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/linear.py#L133)；[vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w8a8/fp8_block.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/methods/w8a8/fp8_block.py#L198)；[vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w8a8/fp8_block.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/methods/w8a8/fp8_block.py#L240)；[cann/ops-transformer / attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp#L17)；[cann/ops-transformer / gmm/grouped_matmul/op_host/grouped_matmul_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/op_host/grouped_matmul_def.cpp#L16)；[cann/ops-nn / norm/rms_norm/op_host/rms_norm_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/norm/rms_norm/op_host/rms_norm_def.cpp#L17)；[cann/ops-nn / norm/add_rms_norm/op_host/add_rms_norm_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/norm/add_rms_norm/op_host/add_rms_norm_def.cpp#L17)；[cann/ops-nn / activation/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp#L21)；[cann/ops-nn / matmul/mat_mul_v3/op_host/mat_mul_v3_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/op_host/mat_mul_v3_def.cpp#L16)；[cann/ops-nn / matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_def.cpp#L17)；[cann/ops-nn / quant/dynamic_mx_quant/op_host/dynamic_mx_quant_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/quant/dynamic_mx_quant/op_host/dynamic_mx_quant_def.cpp#L25)；[cann/ops-math / math/add/op_host/add_def.cpp](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/add/op_host/add_def.cpp#L16)；[cann/ops-math / math/mul/op_host/mul_def.cpp](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/mul/op_host/mul_def.cpp#L17)；[cann/ops-math / experimental/math/cast/op_host/cast_def.cpp](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/experimental/math/cast/op_host/cast_def.cpp#L16)

## MTP 共享参数、加载与推测解码

主干表示与 token embedding → MTP 投影/子层 → 共享 output head → draft / verify 分派

论文训练目标、发布额外层、普通主干 forward 与 speculative decoding 四项分开。

V3 demo 跳过 layer 61；V4 官方参考构造共享 embed/head 的 MTPBlock，框架 MTP.forward/load_weights 是独立来源。

需要对应 speculative config、MTP checkpoint 命名、layer offset、cache/metadata 与 scheduler 条件；不能以文件头存在推断启用。

已核对五版缺失的独立 MTP 存储与固定 vLLM forward；SharedHead.norm 在该后端独立构造/加载，上游训练共享和值相同仍不由 header 证明。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/mtp.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/mtp.py#L274)；[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/mtp.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/mtp.py#L293)；[vllm-project/vllm / vllm/model_executor/models/deepseek_mtp.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/model_executor/models/deepseek_mtp.py#L113)；[vllm-project/vllm / vllm/model_executor/models/deepseek_mtp.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/model_executor/models/deepseek_mtp.py#L304)

## DSpark 半自回归 draft 与验证调度

目标特征与 context KV → noisy block 并行 backbone → 轻量顺序 Markov 修正 → confidence / prefix survival → target verify / rejection → 只提交已接受状态

DSpark 是推测解码机制；本 V4 发布有 3 个 stage 存在 mtp.* namespace，不是把 1 个 MTP 层改名。原始 preview 不默认包含 DSpark。

Ascend draft model 用独立 SWA cache，main_proj 融合目标特征；load_weights 将 mtp stage 分派到首层/末层模块；confidence 使用 sigmoid，proposer 与 speculator 独立准备 block/slot mapping。 CANN 匹配 tag 来源：attention/fused_infer_attention_score。

block_size、num_speculative_tokens、feature层号、Markov rank、采样方式、EOS/grammar、KV提交、图/TP/EP和缓存布局须分别锁定；native block=5 与模型卡 vLLM example=7 不混为一项。

DeepSpec 的 Qwen/Gemma 训练和评测代码不是 V4 生产系统；论文负载感知 scheduler 和框架阈值/自适应配置分别核验，作者收益不是本站或 Ascend 实测。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/dspark.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/dspark.py#L250)；[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/dspark.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/dspark.py#L224)；[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/dspark.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/dspark.py#L399)；[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/dspark.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/dspark.py#L527)；[vllm-project/vllm-ascend / vllm_ascend/spec_decode/dspark_proposer.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/spec_decode/dspark_proposer.py#L226)；[vllm-project/vllm-ascend / vllm_ascend/worker/v2/spec_decode/dspark/speculator.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py#L280)；[cann/ops-transformer / attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp#L20)

## Distill 的 Qwen/Llama GQA

Q/K/V 投影 → RoPE 与 KV cache → repeat_kv / attention → Dense FFN → 残差

六版 Distill 使用 Qwen2/Llama 原生 Dense GQA 结构，Q/K/V bias、KV heads 和词表分别核对。

所选 Transformers v4.44.0 的 eager 参考作为矩阵与数学关系来源；不是声明本机已装该版本或所有服务使用 eager。 CANN 匹配 tag 来源：attention/fused_infer_attention_score, posembedding/rotary_position_embedding。

运行框架、SDPA/FlashAttention、分页、tokenizer 与 chat_template 须按实际底座检查；不能套用 MLA cache。

这里只定位参考模型与共同 norm/linear 条件，GQA 专属 Ascend attention 内核尚未追通。

来源：[Transformers v4.44.0 / src/transformers/models/qwen2/modeling_qwen2.py](https://github.com/huggingface/transformers/blob/984bc11b0882ff1e5b34ba717ea357e069ceced9/src/transformers/models/qwen2/modeling_qwen2.py#L282)；[Transformers v4.44.0 / src/transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/984bc11b0882ff1e5b34ba717ea357e069ceced9/src/transformers/models/llama/modeling_llama.py#L364)；[cann/ops-transformer / attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp#L20)；[cann/ops-transformer / posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp#L16)

## Dense FFN 与 SwiGLU

gate/up 投影 → SiLU(gate)×up → down 投影 → 残差

Dense GQA 模型不含专家路由；V3 前 Dense 层与共享专家也有这一数学结构。

固定 Qwen/Llama MLP.forward 核对参考数学；框架未量化 Linear.apply 使用 unquantized_gemm 分派。 CANN 匹配 tag 来源：activation/swi_glu, matmul/mat_mul_v3。

三个投影、激活精度、bias、并行切分和量化布局各自固定；Linear 分派不证明某个 FFN 的完整设备时序。

已定位 AscendSiluAndMul.forward_oot 到 torch_npu.npu_swiglu；clamp 版先按前后半维分别 clamp 再调用。Dense 分支不套用专家 MC2 通信，底层 CANN 数值仍未验证。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/activation.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/activation.py#L36)；[vllm-project/vllm-ascend / vllm_ascend/ops/activation.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/activation.py#L42)；[vllm-project/vllm-ascend / vllm_ascend/ops/linear.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/linear.py#L133)；[Transformers v4.44.0 / src/transformers/models/qwen2/modeling_qwen2.py](https://github.com/huggingface/transformers/blob/984bc11b0882ff1e5b34ba717ea357e069ceced9/src/transformers/models/qwen2/modeling_qwen2.py#L222)；[Transformers v4.44.0 / src/transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/984bc11b0882ff1e5b34ba717ea357e069ceced9/src/transformers/models/llama/modeling_llama.py#L291)；[cann/ops-nn / activation/swi_glu/op_host/swi_glu_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swi_glu/op_host/swi_glu_def.cpp#L17)；[cann/ops-nn / matmul/mat_mul_v3/op_host/mat_mul_v3_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/op_host/mat_mul_v3_def.cpp#L16)

## CANN 开源算子与本地 SDK

固定框架调用 → op-plugin 条件 API → 匹配 CANN tag 的 API/host/tiling → kernel 与 SDK 高级 API → 运行分支/ABI/数值验证属于未来可选研究

CANN 相关算子公开源码可读，应按版本继续核查，不能将未完成的源代码追踪概括为闭源。本专题是所研究家族的来源清单，各版本启用的算子仍按模型和分派条件区分。

本地 SDK 9.2.0-beta.2；ops-transformer/ops-nn/ops-math 固定同版本 tag；30 个源码家族和独立 op-plugin bridge，含 API/host/tiling/kernel 范围哈希。Python v2 不自动等于 aclnn V2。

所检查安装为 aarch64 编译器/AscendC SDK，独立算子库文件未在该路径发现；源文件从对应 tag 核对。不能据 SDK 版本证明目标设备、框架或二进制兼容。

官方 MhcPre/Post 与框架自定义 HcPre/Post 独立保留；TileKernels 仍是独立实现。没有安装/更新虚拟机软件、编译或运行算子；设备实验不在本次范围。

来源：[cann/ops-transformer / README.md](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/README.md)；[cann/ops-nn / README.md](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/README.md)；[cann/ops-math / README.md](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/README.md)

## 硬件与支持边界

### Ascend A2/A3：设备条件独立确认

BF16/INT8 与其他格式由具体 CANN/torch_npu 和 loader 决定

只记录源码分派条件，不以设备连接或 import 成功作为整模型支持证明。

设备环境与所选 vLLM-Ascend 提交未组成已测试兼容栈；固定驱动、CANN、torch_npu、框架与 custom-op 包后验证。

不将 Ascend 950 的 FP4/FP8/TileKernels 条件推广到 A2/A3。设备实验不在本次范围。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/layernorm.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/layernorm.py)

### Ascend 950：V4 / TileKernels 候选平台

按模块区分 FP4、FP8、BF16、FP32 与 UE8M0 scale；格式与转换单列

TileKernels 明确包含 Ascend 后端，框架 V4 模块与 compressor/mHC dispatcher 存在；仍未验证整模型或两套实现的连接关系。

TileKernels：Python≥3.12、PyTorch≥2.13、TileLang≥0.1.15、CANN≥9.2；所选 vLLM-Ascend 是独立固定源码快照。

硬件型号一致不足以证明软件栈兼容；source presence、编译、正确性与性能分别标记。

来源：[deepseek-ai/TileKernels / README.md](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/README.md)

## Ascend 优先验证与优化

### P0 先建立可运行基线

未来可选 · 不纳入本次验收 · precision / 如另行开展，固定具体软硬件和负载

观察：AscendFp8BlockLinearMethod 可选择 mxfp8_method，其他分支走 unquantized_gemm；process_weights_after_loading 必须与 apply 一起核对。RMSNorm.forward_oot 包含 npu_rms_norm 或 add-rms 条件。

方案：固定完整环境、loader 与权重转换，先小批正确性，再记录初始 prefill/decode。

指标：启动、logits/模块误差、trace、p50/p95、每卡显存

风险：源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/layernorm.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/layernorm.py)

### P1 MLA 投影与 cache 读写

未来可选 · 不纳入本次验收 · mla / 如另行开展，固定具体软硬件和负载

观察：wrapper.forward 调用 torch.ops.vllm.mla_forward；AscendMLAImpl.forward 再分派，_forward_decode 包含普通/NZ/FA quant cache 布局及 query padding/SpecDecoding 条件。

方案：在固定 decode shape 对比分步与融合/缓存路径；检查重复解量化与中间读写。

指标：kernel 时间、HBM 读写、cache bytes、logits/输出误差

风险：源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/mla.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/mla.py)

### P1 Indexer 与 sparse attention

未来可选 · 不纳入本次验收 · dsa / 如另行开展，固定具体软硬件和负载

观察：AscendDeepseekSparseAttention.forward 调用 dsa_forward；AscendDSAImpl 中 prolog、indexer、cache update 和 attention 各有单流/多流或设备条件。

方案：分别测 index score/top-k、attention 和 metadata，变化只作用于一项，保留选择结果质量。

指标：indexer/attention 时间、top-k 一致性、质量/误差、workspace

风险：源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/dsa.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/dsa.py)

### P1 压缩边界与 compressor 状态

未来可选 · 不纳入本次验收 · compression / 如另行开展，固定具体软硬件和负载

观察：框架 Compressor.forward 使用 _C_ascend.compressor 分派；Indexer.select_topk 使用 npu_quant_lightning_indexer_v2；source forward 与 cache plan/metadata 相互约束。

方案：分别比较首次 prefill、decode 压缩边界与非边界；窗口/前缀/FP32 状态分别测量。

指标：边界 p95、state/cache bytes、压缩 token 与数值误差

风险：源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/model.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/model.py)

### P1 mHC pre/post 与 Sinkhorn 融合

未来可选 · 不纳入本次验收 · mhc / 如另行开展，固定具体软硬件和负载

观察：框架 hc_pre/hc_post 分派 npu_hc_pre_v2 / npu_hc_post；TileKernels wrapper 与 *_asc generator 为另一条独立实现来源，不认定框架使用了该库。

方案：以参考 mix/Sinkhorn 为正确性基线，再比较可用融合路径；不预设收益。

指标：pre/post 时间、迭代误差、row/column 归一化、HBM 读写

风险：源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/model.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/model.py)

### P1 专家 GEMM 与 dispatch/combine

未来可选 · 不纳入本次验收 · moe / 如另行开展，固定具体软硬件和负载

观察：MC2 token_dispatch/combine 使用 npu_moe_distribute_dispatch_v2 / combine_v2；MXFP4 GMM1 分支调用 grouped_matmul、swiglu_group_quant，另有非 DeepSeek 的 SITU 分支，不能按调用集合串成执行序列。

方案：固定 EP/TP、路由和 N_e 分布，测通信、专家和 shared 部分，再评估重叠。

指标：dispatch/GEMM/combine 分段时间、N_e 分布、吞吐、误差

风险：源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/fused_moe/moe_mlp.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/fused_moe/moe_mlp.py)

### P2 量化、图模式与动态负载

未来可选 · 不纳入本次验收 · precision / 如另行开展，固定具体软硬件和负载

观察：AscendFp8BlockLinearMethod 可选择 mxfp8_method，其他分支走 unquantized_gemm；process_weights_after_loading 必须与 apply 一起核对。RMSNorm.forward_oot 包含 npu_rms_norm 或 add-rms 条件。

方案：同模型固定质量参考，分别变化量化格式或图模式；warmup 与编译时间单列。

指标：延迟分布、显存、质量、图重编译次数

风险：源码分支/量化/布局改变可能影响数值或 graph/ABI；一次改变一个变量，超出测试条件不推广。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/layernorm.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/layernorm.py)

## 统一实验协议

环境：设备型号、卡数/拓扑、驱动固件、CANN、torch_npu、框架、custom-op 包、容器 digest，禁止只用路径名冒充版本号。

模型：checkpoint revision、原始与转换精度、scale/packing、转换脚本 commit、weight header 哈希/覆盖。

负载：B、L_q、L_kv、并发、输入/输出长度、padding/ragged、TP/EP/DP、graph 与环境开关。

正确性：先固定参考与种子，检查模块/logits 的绝对/相对误差；量化额外检查任务质量。

性能：分别测 prefill/decode/verify/服务；记录 warmup、次数、时间单位、p50/p95、吞吐口径、峰值显存与原始 trace。

一次只改变一个主要变量；保存成功、失败与未执行状态；不把上游 README 数字或静态公式当成本项目实测。

## 版本边界

官方模型代码、框架源码、TileKernels 分别固定版本；函数调用集合是上下文证据，不是执行顺序或 1:1 kernel 对应。

CANN 9.2.0-beta.2 的匹配开源仓和本地 SDK 已核查；源码可查、独立算子库是否安装、框架分派、ABI、设备正确性与性能分别记录。

设备验证不纳入本次验收，所有性能字段保持未知。

## 来源索引

- aux-vllm-project-vllm-ascend-activation-py [vllm-project/vllm-ascend / vllm_ascend/ops/activation.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/activation.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- aux-vllm-project-vllm-ascend-vllm_ascend-models-deepseek_v4-dspark-py [vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/dspark.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/dspark.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- aux-vllm-project-vllm-ascend-vllm_ascend-spec_decode-dspark_proposer-py [vllm-project/vllm-ascend / vllm_ascend/spec_decode/dspark_proposer.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/spec_decode/dspark_proposer.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- aux-vllm-project-vllm-ascend-vllm_ascend-worker-v2-spec_decode-dspark-speculator-py [vllm-project/vllm-ascend / vllm_ascend/worker/v2/spec_decode/dspark/speculator.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-attention-compressor-op_host-arch35-compressor_tiling-cpp [vllm-project/vllm-ascend / csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-attention-compressor-op_host-compressor_def-cpp [vllm-project/vllm-ascend / csrc/attention/compressor/op_host/compressor_def.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/attention/compressor/op_host/compressor_def.cpp)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-attention-compressor-op_kernel-arch35-compressor_kernel-h [vllm-project/vllm-ascend / csrc/attention/compressor/op_kernel/arch35/compressor_kernel.h](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/attention/compressor/op_kernel/arch35/compressor_kernel.h)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-gmm-grouped_matmul_swiglu_quant-op_host-grouped_matmul_swiglu_quant_def-cpp [vllm-project/vllm-ascend / csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-gmm-grouped_matmul_swiglu_quant-op_host-op_api-aclnn_grouped_matmul_swiglu_quant-cpp [vllm-project/vllm-ascend / csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-moe-hc_pre-op_host-hc_pre_def-cpp [vllm-project/vllm-ascend / csrc/moe/hc_pre/op_host/hc_pre_def.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/moe/hc_pre/op_host/hc_pre_def.cpp)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-moe-hc_pre-op_kernel-hc_pre-cpp [vllm-project/vllm-ascend / csrc/moe/hc_pre/op_kernel/hc_pre.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/moe/hc_pre/op_kernel/hc_pre.cpp)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- backend-deep-csrc-torch_binding-cpp [vllm-project/vllm-ascend / csrc/torch_binding.cpp](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/csrc/torch_binding.cpp)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- cann-cann-ops-math-README-md [cann/ops-math / README.md](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/README.md)；0a2ce5b57caec6068d9e5658b740c2d41482aa15；访问 2026-10-01

- cann-cann-ops-math-experimental-math-cast-op_host-cast_def-cpp [cann/ops-math / experimental/math/cast/op_host/cast_def.cpp](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/experimental/math/cast/op_host/cast_def.cpp)；0a2ce5b57caec6068d9e5658b740c2d41482aa15；访问 2026-10-01

- cann-cann-ops-math-math-add-op_host-add_def-cpp [cann/ops-math / math/add/op_host/add_def.cpp](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/add/op_host/add_def.cpp)；0a2ce5b57caec6068d9e5658b740c2d41482aa15；访问 2026-10-01

- cann-cann-ops-math-math-mul-op_host-mul_def-cpp [cann/ops-math / math/mul/op_host/mul_def.cpp](https://atomgit.com/cann/ops-math/blob/0a2ce5b57caec6068d9e5658b740c2d41482aa15/math/mul/op_host/mul_def.cpp)；0a2ce5b57caec6068d9e5658b740c2d41482aa15；访问 2026-10-01

- cann-cann-ops-nn-README-md [cann/ops-nn / README.md](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/README.md)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-nn-activation-swi_glu-op_host-swi_glu_def-cpp [cann/ops-nn / activation/swi_glu/op_host/swi_glu_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swi_glu/op_host/swi_glu_def.cpp)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-nn-activation-swiglu_group_quant-op_host-swiglu_group_quant_def-cpp [cann/ops-nn / activation/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/activation/swiglu_group_quant/op_host/swiglu_group_quant_def.cpp)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-nn-matmul-mat_mul_v3-op_host-mat_mul_v3_def-cpp [cann/ops-nn / matmul/mat_mul_v3/op_host/mat_mul_v3_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/mat_mul_v3/op_host/mat_mul_v3_def.cpp)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-nn-matmul-quant_batch_matmul_v4-op_host-quant_batch_matmul_v4_def-cpp [cann/ops-nn / matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_def.cpp)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-nn-norm-add_rms_norm-op_host-add_rms_norm_def-cpp [cann/ops-nn / norm/add_rms_norm/op_host/add_rms_norm_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/norm/add_rms_norm/op_host/add_rms_norm_def.cpp)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-nn-norm-rms_norm-op_host-rms_norm_def-cpp [cann/ops-nn / norm/rms_norm/op_host/rms_norm_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/norm/rms_norm/op_host/rms_norm_def.cpp)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-nn-quant-dynamic_mx_quant-op_host-dynamic_mx_quant_def-cpp [cann/ops-nn / quant/dynamic_mx_quant/op_host/dynamic_mx_quant_def.cpp](https://atomgit.com/cann/ops-nn/blob/30ef7dd563c8a4b74c3161835c8e47d1d96f87b6/quant/dynamic_mx_quant/op_host/dynamic_mx_quant_def.cpp)；30ef7dd563c8a4b74c3161835c8e47d1d96f87b6；访问 2026-10-01

- cann-cann-ops-transformer-README-md [cann/ops-transformer / README.md](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/README.md)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-compressor-op_host-compressor_def-cpp [cann/ops-transformer / attention/compressor/op_host/compressor_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/compressor/op_host/compressor_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-fused_infer_attention_score-op_host-fused_infer_attention_score_def-cpp [cann/ops-transformer / attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-lightning_indexer-op_host-lightning_indexer_def-cpp [cann/ops-transformer / attention/lightning_indexer/op_host/lightning_indexer_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/lightning_indexer/op_host/lightning_indexer_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-lightning_indexer_v2-op_host-lightning_indexer_v2_def-cpp [cann/ops-transformer / attention/lightning_indexer_v2/op_host/lightning_indexer_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/lightning_indexer_v2/op_host/lightning_indexer_v2_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-mixed_quant_sparse_flash_mla-op_host-mixed_quant_sparse_flash_mla_def-cpp [cann/ops-transformer / attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-quant_lightning_indexer-op_host-quant_lightning_indexer_def-cpp [cann/ops-transformer / attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-quant_lightning_indexer_v2-op_host-quant_lightning_indexer_v2_def-cpp [cann/ops-transformer / attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-sparse_flash_attention-op_host-sparse_flash_attention_def-cpp [cann/ops-transformer / attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_attention/op_host/sparse_flash_attention_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-sparse_flash_mla-op_host-sparse_flash_mla_def-cpp [cann/ops-transformer / attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_mla/op_host/sparse_flash_mla_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-attention-sparse_flash_mla_metadata-docs-aclnnSparseFlashMlaMetadata-md [cann/ops-transformer / attention/sparse_flash_mla_metadata/docs/aclnnSparseFlashMlaMetadata.md](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/attention/sparse_flash_mla_metadata/docs/aclnnSparseFlashMlaMetadata.md)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-gmm-grouped_matmul-op_host-grouped_matmul_def-cpp [cann/ops-transformer / gmm/grouped_matmul/op_host/grouped_matmul_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul/op_host/grouped_matmul_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-gmm-grouped_matmul_swiglu_quant-op_host-grouped_matmul_swiglu_quant_def-cpp [cann/ops-transformer / gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-gmm-grouped_matmul_swiglu_quant_v2-op_host-grouped_matmul_swiglu_quant_v2_def-cpp [cann/ops-transformer / gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-mc2-moe_distribute_combine_v2-op_host-moe_distribute_combine_v2_def-cpp [cann/ops-transformer / mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-mc2-moe_distribute_combine_v3-op_host-moe_distribute_combine_v3_def-cpp [cann/ops-transformer / mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_combine_v3/op_host/moe_distribute_combine_v3_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-mc2-moe_distribute_dispatch_v2-op_host-moe_distribute_dispatch_v2_def-cpp [cann/ops-transformer / mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-mc2-moe_distribute_dispatch_v3-op_host-moe_distribute_dispatch_v3_def-cpp [cann/ops-transformer / mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mc2/moe_distribute_dispatch_v3/op_host/moe_distribute_dispatch_v3_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-mhc-mhc_post-op_host-mhc_post_def-cpp [cann/ops-transformer / mhc/mhc_post/op_host/mhc_post_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mhc/mhc_post/op_host/mhc_post_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-mhc-mhc_pre-op_host-mhc_pre_def-cpp [cann/ops-transformer / mhc/mhc_pre/op_host/mhc_pre_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/mhc/mhc_pre/op_host/mhc_pre_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- cann-cann-ops-transformer-posembedding-rotary_position_embedding-op_host-rotary_position_embedding_def-cpp [cann/ops-transformer / posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp](https://atomgit.com/cann/ops-transformer/blob/5f33f1e23d41fe0a047d01b7c7ae278777cac1f5/posembedding/rotary_position_embedding/op_host/rotary_position_embedding_def.cpp)；5f33f1e23d41fe0a047d01b7c7ae278777cac1f5；访问 2026-10-01

- deepseek-ai-TileKernels-README-md [deepseek-ai/TileKernels / README.md](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/README.md)；66258df6175d2f630ffecb04c5ab66bff8a2ae6a；访问 2026-10-01

- deepseek-ai-TileKernels-tile_kernels-mhc-pre_big_fuse_asc-py [deepseek-ai/TileKernels / tile_kernels/mhc/pre_big_fuse_asc.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/mhc/pre_big_fuse_asc.py)；66258df6175d2f630ffecb04c5ab66bff8a2ae6a；访问 2026-10-01

- deepseek-ai-TileKernels-tile_kernels-mhc-pre_big_fuse_kernel-py [deepseek-ai/TileKernels / tile_kernels/mhc/pre_big_fuse_kernel.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/mhc/pre_big_fuse_kernel.py)；66258df6175d2f630ffecb04c5ab66bff8a2ae6a；访问 2026-10-01

- deepseek-ai-TileKernels-tile_kernels-mhc-sinkhorn_asc-py [deepseek-ai/TileKernels / tile_kernels/mhc/sinkhorn_asc.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/mhc/sinkhorn_asc.py)；66258df6175d2f630ffecb04c5ab66bff8a2ae6a；访问 2026-10-01

- deepseek-ai-TileKernels-tile_kernels-moe-topk_gate_asc-py [deepseek-ai/TileKernels / tile_kernels/moe/topk_gate_asc.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/moe/topk_gate_asc.py)；66258df6175d2f630ffecb04c5ab66bff8a2ae6a；访问 2026-10-01

- deepseek-ai-TileKernels-tile_kernels-moe-topk_gate_kernel-py [deepseek-ai/TileKernels / tile_kernels/moe/topk_gate_kernel.py](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/tile_kernels/moe/topk_gate_kernel.py)；66258df6175d2f630ffecb04c5ab66bff8a2ae6a；访问 2026-10-01

- hf-transformers-modeling_llama-py [Transformers v4.44.0 / src/transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/984bc11b0882ff1e5b34ba717ea357e069ceced9/src/transformers/models/llama/modeling_llama.py)；984bc11b0882ff1e5b34ba717ea357e069ceced9；访问 2026-10-01

- hf-transformers-modeling_qwen2-py [Transformers v4.44.0 / src/transformers/models/qwen2/modeling_qwen2.py](https://github.com/huggingface/transformers/blob/984bc11b0882ff1e5b34ba717ea357e069ceced9/src/transformers/models/qwen2/modeling_qwen2.py)；984bc11b0882ff1e5b34ba717ea357e069ceced9；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-attention-dsa_v1-py [vllm-project/vllm-ascend / vllm_ascend/attention/dsa_v1.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/attention/dsa_v1.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-attention-mla_v1-py [vllm-project/vllm-ascend / vllm_ascend/attention/mla_v1.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/attention/mla_v1.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-models-deepseek_v4-compressor-py [vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/compressor.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/compressor.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-models-deepseek_v4-indexer-py [vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/indexer.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/indexer.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-models-deepseek_v4-model-py [vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/model.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/model.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-models-deepseek_v4-mtp-py [vllm-project/vllm-ascend / vllm_ascend/models/deepseek_v4/mtp.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_v4/mtp.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-ops-dsa-py [vllm-project/vllm-ascend / vllm_ascend/ops/dsa.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/dsa.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-ops-fused_moe-moe_mlp-py [vllm-project/vllm-ascend / vllm_ascend/ops/fused_moe/moe_mlp.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/fused_moe/moe_mlp.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-ops-fused_moe-token_dispatcher-py [vllm-project/vllm-ascend / vllm_ascend/ops/fused_moe/token_dispatcher.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/fused_moe/token_dispatcher.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-ops-layernorm-py [vllm-project/vllm-ascend / vllm_ascend/ops/layernorm.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/layernorm.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-ops-linear-py [vllm-project/vllm-ascend / vllm_ascend/ops/linear.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/linear.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-ops-mla-py [vllm-project/vllm-ascend / vllm_ascend/ops/mla.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/ops/mla.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-quantization-methods-w4a8-w4a8_mxfp4-py [vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-ascend-vllm_ascend-quantization-methods-w8a8-fp8_block-py [vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w8a8/fp8_block.py](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/methods/w8a8/fp8_block.py)；a8fcedb03d93e60efceddbfc912406f7fa491d57；访问 2026-10-01

- vllm-project-vllm-vllm-model_executor-models-deepseek_mtp-py [vllm-project/vllm / vllm/model_executor/models/deepseek_mtp.py](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/vllm/model_executor/models/deepseek_mtp.py)；bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f；访问 2026-10-01