# Kimi 实现、硬件与 Ascend 优化研究

研究快照：2026-09-19

公开源码、配置和厂商资料研究；没有执行 GPU/NPU 性能实验。代码存在、部署配方、厂商实验与本地实测分开。

## MLA 与 Gated MLA

低秩 Q / KV 投影 → RoPE 与注意力 → 压缩缓存 / 内核布局 → 输出投影与可选门控

MLA 压缩的是注意力中的表示；权重大小、KV 缓存大小与运行时带宽是三个不同指标。K3 的 gated MLA 在注意力输出端增加门控。

官方 K3 KimiMLAAttention 读取 mla_use_output_gate；vLLM 分开维护 NVIDIA 和 AMD MLA，Ascend 提供 MLA 后端和 Kimi 适配。参考实现的展开张量不代表所有服务引擎都用同一缓存布局。

长上下文 decode 应观察缓存读带宽和并行切分；prefill 应观察 attention GEMM 与投影占比。不能用单个矩阵峰值预测两阶段吞吐。

K2 系列在此按 MLA 路线归类，不把 K3 输出门控自动套用给 K2。实际 KV dtype、分页与 DCP 要从引擎配置记录。

来源：[moonshotai/Kimi-K3 / modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L335)；[vllm-project/vllm / vllm/models/kimi_k3/nvidia/mla.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/mla.py)；[vllm-project/vllm / vllm/models/kimi_k3/amd/mla.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/mla.py)；[vllm-project/vllm-ascend / vllm_ascend/attention/mla_v1.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/attention/mla_v1.py)

## KDA：分块预填充与递归解码

Q / K / V、门控投影 → 短卷积 → chunk prefill / recurrent decode → 门控归一化与输出投影

KDA 用递归状态承载历史，MLA 层仍保留 KV；混合模型的缓存不会整体变成常数。Kimi Linear 的低秩门控与 K3 的 full-rank gate 路径需分开。

官方代码在 use_cache 且 q_len=1 时选 fused_recurrent，其余走 chunk；NVIDIA 路径检测 FlashInfer 可用性，AMD 路径按 shape/dtype 检测融合 conv+recurrence+norm；Ascend 调 run_chunk_kda / run_recurrent_kda。

分块预填充有矩阵工作与状态跨块传递；解码小 batch 可能更受启动、状态读写约束。Cube/Vector、CUDA 或 HIP 内核都需要按形状分桶。

存在专用算子不代表所有 batch、推测解码、变长和长前缀条件都进入最快分支。应同时记录输出误差与最终状态误差。

来源：[moonshotai/Kimi-K3 / modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L477)；[vllm-project/vllm / vllm/models/kimi_k3/nvidia/kda.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/kda.py#L56)；[vllm-project/vllm / vllm/models/kimi_k3/amd/kda.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/kda.py#L94)；[vllm-project/vllm-ascend / vllm_ascend/ops/kimi_kda.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/kimi_kda.py#L411)；[vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/design.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/design.md)

## MoE：路由、分发、Grouped GEMM 与合并

路由 Top-K → 按专家重排 / 跨卡分发 → Grouped GEMM 与激活 → 合并并加共享专家

专家总数影响权重驻留，Top-K 影响选中工作量；实际每个专家收到多少 token 决定 GEMM 的有效 M 维。低并发下大量小矩阵与通信可能比算力更关键。

vLLM 的 MXFP4 oracle 分列 W4A16、W4A8、W4A4 后端；K3 NVIDIA latent runner 只在特定 TP、共享专家和输出约简条件下进入融合路径。后端名称必须与实际 dispatcher 选择一起保存。

NVIDIA、AMD、Ascend 都需要专家并行通信与计算配合；比较 NCCL/RCCL/HCCL 路径时先固定拓扑、EP/TP 和路由分布。

相同“4 bit”不是相同计算格式；INT4 与 MXFP4、NVFP4 不能按位宽互换。权重重排的加载时间也不能算作每步解码成本。

来源：[vllm-project/vllm / vllm/model_executor/layers/fused_moe/oracle/mxfp4.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py#L113)；[vllm-project/vllm / vllm/models/kimi_k3/nvidia/latent_moe_runner.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/latent_moe_runner.py#L179)；[vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w4a8/w4a8.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/quantization/methods/w4a8/w4a8.py)；[AMD：K2.5 / K2.6 / K2.7-Code 的 MXFP4 服务](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-code-in-mxfp4-on-amd-gpus.html)

## K3 Stable LatentMoE

残差宽度 → 专家潜空间 → 选中专家计算 → 归一化与上投影 → 共享分支 / 通信合并

路由专家在较小输入空间工作，但额外下投影、归一化、上投影和通信仍需计入。3D 矩阵表保留真实专家输入宽度，不能把残差宽度代入每个专家。

官方 KimiSparseMoeBlock 条件创建 routed_expert_down_proj 与输出变换；Ascend 把输入/输出变换交给 FusedMoE。NVIDIA runner 有按 token 数选择的 tail，AMD runner 有独立约简与输出路径。

小 batch 可研究归一化/投影/collective 融合，大 batch 关注 GEMM 利用率。跨厂商可借鉴数据流，不能直接复制特定 warp 或 Cube 布局。

源码可见不同融合结构，但尚不能据此断言 Ascend 延迟更差；需要量化 tail 占端到端关键路径的比例。

来源：[moonshotai/Kimi-K3 / modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L762)；[vllm-project/vllm-ascend / vllm_ascend/models/kimi_k3.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py#L219)；[vllm-project/vllm / vllm/models/kimi_k3/nvidia/latent_moe_runner.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/latent_moe_runner.py#L238)；[vllm-project/vllm / vllm/models/kimi_k3/amd/latent_moe_runner.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/latent_moe_runner.py#L60)

## K3 AttnRes：跨层表示混合

保存块残差 → 归一化与学习评分 → 块维度 softmax → 残差加权混合

Attention Residuals 沿网络深度混合较早层的表示，不是对历史 token 做额外全注意力；其工作区随块数、token 数和隐藏维度变化。

Ascend 在 NPU、非空且内核可用时调用 apply_attn_res，其他情况回到显式 cat / FP32 归一化 / softmax / matmul。已有 Triton 专用实现，不能直接标成“缺失融合”。

主要关注中间张量读写、FP32 数值步骤与启动次数；可按 prefill token 数、残差块数分桶测量。NVIDIA 有独立 AttnRes 实现供数据流对照。

应先确认实际走专用路径；参考 fallback 多个算子不等于生产默认存在同样开销。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/kimi_k3.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py#L93)；[vllm-project/vllm-ascend / vllm_ascend/ops/triton/kimi_k3/attention_residual.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/triton/kimi_k3/attention_residual.py)；[vllm-project/vllm / vllm/models/kimi_k3/nvidia/ops/attn_res.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/ops/attn_res.py)

## 视觉编码器与多模态预填充

图像 / 视频分块 → 视觉 Transformer → Patch Merger / Projector → 拼接视觉 token 进入语言主干

图像分辨率、视频帧数和裁剪策略改变视觉 token 与首 token 延迟。文本 decode 加速不一定改善多图场景的完整响应时间。

官方 K3 模型定义 MoonViT3dPretrainedModel 与 PatchMerger；Ascend 显式构造 vision_tower，并按是否忽略量化配置选择视觉精度。

单独测视觉编码、投影、主干 prefill，并观察多模态调度和显存峰值。量化专家并不表示视觉层也被同样量化。

K2.6 / K2.7-Code 的归类来自已有配置审计；本次没有逐行审阅这两版独立视觉实现。不同图片输入必须统一 processor revision。

来源：[moonshotai/Kimi-K3 / modeling_kimi_k3.py](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L649)；[moonshotai/Kimi-K2.5 / modeling_kimi_k25.py](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py)；[vllm-project/vllm-ascend / vllm_ascend/models/kimi_k3.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py#L867)

## BF16、FP8、INT4 与 MXFP4

检查点格式 → 加载转换与重排 → 激活量化 / 反量化 → 算术累积与输出 dtype

W4A16 表示四位权重与十六位激活，W4A8 / W4A4 另有不同激活精度。文件中的 U8/I32 是打包容器，不是原始数值格式。

Ascend 的 W4A8 MXFP4 文件有 NZ 布局与 scale 重排；这些加载期路径不能证明所有 A2/A3 都原生支持任意 MXFP4 算术。K3 A3 教程明确使用 Eco-Tech W4A8 转换检查点。

Hopper 与 Blackwell 分代选核；AMD MI300X 的 W4A16 路线与 MI355X 的 MXFP4 路线分列；Ascend 按具体芯片、CANN 和算子支持矩阵确认。

转换检查点应另记校准集、精度验证、转换工具版本与 revision；不覆盖原始模型审计精度字段。当前未进行转换或质量复测。

来源：[vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py#L106)；[vllm-project/vllm / vllm/model_executor/layers/fused_moe/oracle/mxfp4.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py#L113)；[vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K3.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K3.md#L25)；[AMD：MI300X K2.5 算子优化](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)；[AMD：K2.5 / K2.6 / K2.7-Code 的 MXFP4 服务](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-code-in-mxfp4-on-amd-gpus.html)；[AMD CDNA4 / MI350 系列架构](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)

## Prefill / Decode、缓存与多卡服务

请求调度与分块预填充 → KV / KDA 状态管理 → TP / EP / DP / PP 通信 → 逐 token 解码与可选推测

Prefill 影响首 token 时间，decode 影响 token 间延迟；吞吐与交互性应分别报告。KDA 状态缓存、MLA KV、AttnRes 工作区和权重需要分别核算。

Ascend K3 教程给出 TP16/DP4/EP64 的四节点参考部署；SGLang 配置按 H100/H200/B200/B300/MI35x/A3 分硬件配方，也针对 DCP 与缓存组合设置限制。

通信重叠只有在独立工作可并行且缓冲生命周期安全时才有效。先固定负载和显存占用，再比较调度与推测解码。

部署示例 max-model-len=133120 不能作为已验证一百万上下文的证据。权重能装入并不意味着峰值并发、缓存和工作区都能装入。

来源：[vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K3.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K3.md)；[sgl-project/sglang / docs/src/snippets/configs/moonshotai/kimi-k3.jsx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/src/snippets/configs/moonshotai/kimi-k3.jsx#L18)；[vllm-project/vllm-ascend / vllm_ascend/ops/kimi_kda.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/kimi_kda.py#L461)

## 硬件与支持边界

### NVIDIA H100 / H200 · Hopper

按算子选择 BF16 / FP8、低位权重解包路径；不可照搬 Blackwell 选核。

SGLang K3 配方列有 H100 与 H200；vLLM 有 NVIDIA KDA / MLA / LatentMoE 实现。

SGLang 567d5925fe89；vLLM 4cc15f2121b3。配方存在 ≠ 本站完成部署。

H100 与 H200 的容量、带宽和节点数不同；代码路径仍须检查 GPU capability 与具体 shape。

来源：[NVIDIA Hopper 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)；[sgl-project/sglang / docs/src/snippets/configs/moonshotai/kimi-k3.jsx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/src/snippets/configs/moonshotai/kimi-k3.jsx#L18)；[vllm-project/vllm / vllm/models/kimi_k3/nvidia/kda.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/kda.py)

### NVIDIA B200 / B300 · Blackwell

MXFP4 后端存在于 vLLM oracle；权重 MXFP4 与 NVFP4 转换方案分开记录。

SGLang K3 分列 B200、B300、GB200、GB300 配方；网络与并行方式不能互换。

同上固定提交；硬件指南 CUDA 13.4 仅用于代际说明。

数据中心 Blackwell 与消费级 SM12.x 不是同一约束；禁止按品牌统一推导支持。

来源：[NVIDIA Blackwell 调优指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)；[vllm-project/vllm / vllm/model_executor/layers/fused_moe/oracle/mxfp4.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py#L113)；[sgl-project/sglang / docs/src/snippets/configs/moonshotai/kimi-k3.jsx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/src/snippets/configs/moonshotai/kimi-k3.jsx#L18)

### AMD MI300X · CDNA3

AMD K2.5 文章描述 W4A16 + BF16 混合路径，不能当作 MI355X 原生 MXFP4 路线。

厂商公开 SGLang + AITER / FlyDSL 的 K2.5 优化案例。

文章 2026-03-24；示例镜像标签含 v0.5.8-rocm720-mi30x-kimi-k2.5-opt-20260224。

案例只覆盖文中模型与配置；没有由此核验 K3 在 MI300X 上完整服务。

来源：[AMD：MI300X K2.5 算子优化](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)

### AMD MI350X / MI355X · CDNA4

硬件新增 OCP MXFP8 / MXFP6 / MXFP4；仍需合适内核、布局和激活精度。

SGLang K3 有 MI35x 配方；vLLM 独立 AMD KDA；AMD 报告 ATOM + AITER 的 K2.5/2.6/2.7-Code MXFP4。

vLLM 4cc15f2121b3；AITER c15adc7a6d1a；AMD ATOM 案例 2026-07-21。

ATOM 案例不能自动证明相同优化已合入任意 vLLM/SGLang 发行版。

来源：[AMD CDNA4 / MI350 系列架构](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)；[AMD：K2.5 / K2.6 / K2.7-Code 的 MXFP4 服务](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-code-in-mxfp4-on-amd-gpus.html)；[vllm-project/vllm / vllm/models/kimi_k3/amd/kda.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/kda.py)；[sgl-project/sglang / docs/src/snippets/configs/moonshotai/kimi-k3.jsx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/src/snippets/configs/moonshotai/kimi-k3.jsx#L18)

### Ascend Atlas 800 A2

已核验 K2.5 / K2.6 转换后的 W4A8 部署文档。

K2.5 文档列 2 节点、每节点 8×64GB；KDA 设计文档也覆盖 A2 数学路径。

K2.5 教程 v0.17.0rc1；K2.6 教程 v0.20.0rc1；源码快照 e139b7d573d3。

单个算子有 A2 路径不等于 K3 全模型 A2 服务已验证；此次 K3 教程明确仅 A3。

来源：[vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K2.5.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K2.5.md#L9)；[vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K2.6.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K2.6.md)；[vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/design.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/design.md)

### Ascend Atlas 800 A3

K3 参考部署用 Eco-Tech/Kimi-K3-w4a8；不要写成直接运行原始 MXFP4 的验证结论。

K3 文本、多模态、TP/DP/EP、Prefix Cache 与 FULL_DECODE_ONLY ACL Graph 有教程；四节点、16 逻辑 NPU/节点。

教程 main + vLLM 0.27.1；同快照 Dockerfile.a3 默认 CANN9.1.0 / vLLM0.28.0；匹配上游 commit 文件见来源。

文档与构建默认版本漂移必须显式解决；nightly 镜像需固定 digest。本次未运行硬件测试。

来源：[vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K3.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K3.md#L9)；[vllm-project/vllm-ascend / Dockerfile.a3](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/Dockerfile.a3#L21)；[vllm-project/vllm-ascend / .github/vllm-main-verified.commit](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/.github/vllm-main-verified.commit)

### Ascend A5 · 算子级证据

ChunkKdaFwd 设计描述 A5 BF16、chunk64、K=V=128 的对齐快路径。

设计文档列 arch35 regbase 双发射、特定场景单 L0，其余多 chunk 四阶段提交。

vLLM-Ascend e139b7d573d3 的设计与 API 文档。

仅确认源码设计范围；未核验完整产品规格、量产配置或 K3 全模型部署，不把 A5 特性套用到 A3。

来源：[vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/design.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/design.md#L1)；[vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/api.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/api.md)

## Ascend 优先验证与优化

### P0 先固定能复现的基线

已核验的文档一致性问题 · K3 / A3 / 转换 W4A8

观察：教程为 vLLM0.27.1，Docker 默认0.28.0；verified.commit 指向 84030bbe3d74d99bad477a3d2e37a973ccd8865c。三者不能混作一个已测试环境。

方案：锁定引擎、插件、CANN、torch_npu、驱动、镜像 digest、模型与 processor revision；运行小批准确性与短服务烟测，再扩大。

指标：启动成功率；固定输入输出误差；TTFT/TPOT p50、p95；每卡峰值内存；保存环境清单。

风险：版本不匹配可能影响接口、算子 ABI 和图捕获。源码快照检查不替代运行确认。

来源：[vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K3.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K3.md#L9)；[vllm-project/vllm-ascend / Dockerfile.a3](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/Dockerfile.a3#L21)；[vllm-project/vllm-ascend / .github/vllm-main-verified.commit](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/.github/vllm-main-verified.commit)

### P1 测量 KDA 状态收集与回写

代码可见额外数据搬运；瓶颈待测 · K3 KDA prefill / A3；保持原状态精度

观察：prefill 取 recurrent_state[state_indices].contiguous()，算子后 final_state 转 dtype 并按索引写回；布局转换有实际代码依据。

方案：用 trace 确认 gather/copy/scatter 和算子时长；评估索引直接输入或原位输出接口，优先减少非连续状态搬运。

指标：状态搬运字节与时间；kernel 数；prefill tokens/s；最终状态最大/均方误差；长续写偏差。

风险：缓存索引复用、前缀重用与 speculative rollback 可能导致别名或写错状态；不能直接删 contiguous。

来源：[vllm-project/vllm-ascend / vllm_ascend/ops/kimi_kda.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/kimi_kda.py#L461)；[vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/design.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/design.md)

### P1 按 token 桶检查 KDA 融合与阶段调度

需要 profiling 的假设 · K3 / Kimi Linear；分别测试 prefill、decode；BF16 基线

观察：NVIDIA/AMD 有特定 fused decode；Ascend 已有 chunk/recurrent 专用实现，A5 快路径条件明确。当前未证明某一平台更慢。

方案：统计输入投影、卷积、门控、递归、归一化、输出投影的占比；对小 batch 评估相邻向量步骤融合，对大 prefill 检查 Cube/Vector 流水。

指标：算子启动数；有效带宽；Cube/Vector 利用率；TPOT p95；变长尾块效率；状态误差。

风险：融合增加寄存器/片上存储压力，可能降低大 batch 效率；A5 优化不能直接搬到 A3。

来源：[vllm-project/vllm / vllm/models/kimi_k3/amd/kda.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/kda.py#L146)；[vllm-project/vllm-ascend / vllm_ascend/ops/kimi_kda.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/kimi_kda.py#L287)；[vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/design.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/design.md)

### P1 联合测 MoE 分发、专家 GEMM 与合并

需要 profiling 的假设 · K2 / K3；明确 W4A8、EP/TP 与节点拓扑

观察：已有量化 MoE 路径；总专家数量本身不能说明热点，必须测实际每专家 token 分布及 all-to-all 等待。

方案：按专家 token 数统计 GEMM；比较 dispatch/combine 与计算重叠；保持精度条件一致后评估分桶与路由均衡。

指标：路由偏斜；Grouped GEMM M 分布；通信有效带宽/等待比例；吞吐与 p95 TPOT；转换质量。

风险：padding 和通信缓冲可能吃掉节省的内存；重叠须保证正确的 stream 同步与权重驻留。

来源：[vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w4a8/w4a8.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/quantization/methods/w4a8/w4a8.py)；[vllm-project/vllm / vllm/model_executor/layers/fused_moe/oracle/mxfp4.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)；[AMD：K2.5 / K2.6 / K2.7-Code 的 MXFP4 服务](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-code-in-mxfp4-on-amd-gpus.html)

### P2 评估 LatentMoE 尾部融合

需要 profiling 的假设 · 仅 K3；固定 TP / EP / shared expert 设置

观察：NVIDIA runner 对小/大 token 数采用不同 tail；Ascend 通过 FusedMoE 传递 latent 变换；不能仅按 Python 层判断底层融合缺失。

方案：追踪上投影、RMSNorm、共享专家合并及约简；若占关键路径，再评估融合或通信重叠。

指标：tail 占步长比例；collective 次数与延迟；有效 GEMM M；端到端 TPOT。

风险：约简与归一化顺序不可交换；模型并行与序列并行条件可能使融合不合法。

来源：[vllm-project/vllm / vllm/models/kimi_k3/nvidia/latent_moe_runner.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/latent_moe_runner.py#L179)；[vllm-project/vllm / vllm/models/kimi_k3/amd/latent_moe_runner.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/latent_moe_runner.py#L60)；[vllm-project/vllm-ascend / vllm_ascend/models/kimi_k3.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py#L219)

### P2 确认 AttnRes 快路径与工作区

已核验条件回退；触发频率待测 · 仅 K3；FP32 评分/归一化数值基线

观察：NPU 非空且导入成功时用专用 apply_attn_res；回退有显式 cat 与 FP32 临时张量。不能把回退当作默认缺陷。

方案：记录分支命中率，测块数和 token 数增长时工作区；仅在 trace 显示收益空间后改融合/复用缓冲。

指标：快路径命中率；HBM字节；工作区峰值；AttnRes 延迟；结果误差。

风险：softmax/归一化降精度可能损害稳定性；工作区复用必须遵守图捕获与并发生命周期。

来源：[vllm-project/vllm-ascend / vllm_ascend/models/kimi_k3.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py#L93)；[vllm-project/vllm-ascend / vllm_ascend/ops/triton/kimi_k3/attention_residual.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/triton/kimi_k3/attention_residual.py)

### P2 分离长上下文与视觉首 token 代价

需要 profiling 的假设 · K3 / K2.5+ 多模态；文本与视觉负载分组

观察：K3 示例长度133120；模型宣称1M不等于该部署已经测过1M。视觉编码与主干 prefill 是不同阶段。

方案：从8K/32K/128K逐级扩展；记录MLA KV、KDA状态、AttnRes工作区；图片按分辨率/帧数分组；再研究 chunked prefill、prefix cache、PD分离。

指标：TTFT p95；排队时间；decode抖动；缓存命中率；每阶段内存峰值；OOM边界。

风险：超长上下文和跨节点缓存传输可能放大质量与延迟问题；避免用纯文本吞吐替代视觉结果。

来源：[vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K3.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K3.md)；[sgl-project/sglang / docs/src/snippets/configs/moonshotai/kimi-k3.jsx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/src/snippets/configs/moonshotai/kimi-k3.jsx#L51)；[vllm-project/vllm-ascend / vllm_ascend/models/kimi_k3.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py#L867)

## 统一实验协议

固定完整软硬件清单、模型与 processor revision、量化和缓存 dtype；不要把不同权重格式混为同一条件。

分别测试 prefill / decode：输入长度8K、32K、128K（内存允许时）；输出256/1024；并发1/4/16/64。超过部署文档范围时单列探索性测试。

使用相同提示集、输出长度策略和采样参数；冷启动、预热、prefix cache 命中/未命中分开。记录重复次数与误差范围。

先查逐模块 trace，再作单项 A/B；报告 TTFT/TPOT p50与p95、总吞吐、每卡内存、通信占比、质量回归。未实测字段留空。

低精度须比较 logits/状态误差与任务质量；跨卡数比较同时报告总卡数、拓扑、功率或成本的实际采集口径。没有数据就不排名。

## 版本边界

K2 Base / Instruct / 0905：本次以官方部署说明与现有配置/权重审计定位 MLA + MoE。官方 K2 GitHub 固定树没有 inference/model.py；不伪造该文件引用。

K2 Thinking：INT4 权重审计与部署量化路径分开。Ascend 教程对 quantization_config targets 有特殊要求，须按锁定版本验证。

K2.5 / K2.6 / K2.7-Code：原始检查点、Ascend 转换 W4A8、AMD ATOM MXFP4 是不同实验条件。视觉和 MoE 精度分别记录。

Kimi Linear：官方文档给出 KDA/MLA 3:1 及 fla-core 要求；不能借用 K3 全秩门控、LatentMoE 或 AttnRes。

K3：官方语言参考代码与三平台实现交叉阅读。69 KDA +24 gated MLA、LatentMoE、AttnRes、视觉主干在3D与代码专题间互相链接。

## 来源索引

- s1 [vllm-project/vllm-ascend / vllm_ascend/models/kimi_k3.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/models/kimi_k3.py)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s2 [vllm-project/vllm-ascend / vllm_ascend/ops/kimi_kda.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/kimi_kda.py)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s3 [vllm-project/vllm-ascend / vllm_ascend/ops/kimi_mla.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/kimi_mla.py)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s4 [vllm-project/vllm-ascend / vllm_ascend/attention/mla_v1.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/attention/mla_v1.py)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s5 [vllm-project/vllm-ascend / vllm_ascend/ops/triton/kimi_k3/attention_residual.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/ops/triton/kimi_k3/attention_residual.py)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s6 [vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w4a8/w4a8.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/quantization/methods/w4a8/w4a8.py)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s7 [vllm-project/vllm-ascend / vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s8 [vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/design.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/design.md)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s9 [vllm-project/vllm-ascend / csrc/attention/chunk_kda_fwd/docs/api.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/csrc/attention/chunk_kda_fwd/docs/api.md)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s10 [vllm-project/vllm-ascend / Dockerfile.a3](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/Dockerfile.a3)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s11 [vllm-project/vllm-ascend / .github/vllm-main-verified.commit](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/.github/vllm-main-verified.commit)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s12 [vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K3.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K3.md)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s13 [vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K2.5.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K2.5.md)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s14 [vllm-project/vllm / vllm/models/kimi_k3/nvidia/kda.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/kda.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s15 [vllm-project/vllm / vllm/models/kimi_k3/amd/kda.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/kda.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s16 [vllm-project/vllm / vllm/models/kimi_k3/nvidia/latent_moe_runner.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/latent_moe_runner.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s17 [vllm-project/vllm / vllm/models/kimi_k3/amd/latent_moe_runner.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/latent_moe_runner.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s18 [vllm-project/vllm / vllm/models/kimi_k3/nvidia/mla.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/mla.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s19 [vllm-project/vllm / vllm/models/kimi_k3/amd/mla.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/mla.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s20 [vllm-project/vllm / vllm/model_executor/layers/fused_moe/oracle/mxfp4.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s21 [vllm-project/vllm / vllm/models/kimi_k3/amd/ops/kda_prefill.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/amd/ops/kda_prefill.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s22 [vllm-project/vllm / vllm/models/kimi_k3/nvidia/ops/attn_res.py](https://github.com/vllm-project/vllm/blob/4cc15f2121b3421478f3186aac38de901b4a44ef/vllm/models/kimi_k3/nvidia/ops/attn_res.py)；4cc15f2121b3421478f3186aac38de901b4a44ef；访问 2026-09-19

- s23 [sgl-project/sglang / docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx)；567d5925fe896a7b70043271f5621d6fe6e650d4；访问 2026-09-19

- s24 [sgl-project/sglang / docs/src/snippets/configs/moonshotai/kimi-k3.jsx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/src/snippets/configs/moonshotai/kimi-k3.jsx)；567d5925fe896a7b70043271f5621d6fe6e650d4；访问 2026-09-19

- s25 [sgl-project/sglang / docs/cookbook/autoregressive/Moonshotai/Kimi-Linear.mdx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/cookbook/autoregressive/Moonshotai/Kimi-Linear.mdx)；567d5925fe896a7b70043271f5621d6fe6e650d4；访问 2026-09-19

- s26 [sgl-project/sglang / docs/src/snippets/autoregressive/kimi-k27-code-deployment.jsx](https://github.com/sgl-project/sglang/blob/567d5925fe896a7b70043271f5621d6fe6e650d4/docs/src/snippets/autoregressive/kimi-k27-code-deployment.jsx)；567d5925fe896a7b70043271f5621d6fe6e650d4；访问 2026-09-19

- s27 [MoonshotAI/Kimi-K3 / README.md](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/README.md)；3cb39dfd32e51c3328e2e4b4af21341247d06c43；访问 2026-09-19

- s28 [MoonshotAI/Kimi-Linear / README.md](https://github.com/MoonshotAI/Kimi-Linear/blob/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3/README.md)；8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3；访问 2026-09-19

- s29 [ROCm/aiter / aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py](https://github.com/ROCm/aiter/blob/c15adc7a6d1ad4422202424983aed97cab93db77/aiter/ops/triton/kimi_delta_attn/chunk_delta_attn.py)；c15adc7a6d1ad4422202424983aed97cab93db77；访问 2026-09-19

- s30 [ROCm/aiter / aiter/ops/triton/gated_delta_net/fused_kda_decode.py](https://github.com/ROCm/aiter/blob/c15adc7a6d1ad4422202424983aed97cab93db77/aiter/ops/triton/gated_delta_net/fused_kda_decode.py)；c15adc7a6d1ad4422202424983aed97cab93db77；访问 2026-09-19

- s31 [MoonshotAI/Kimi-K2 / README.md](https://github.com/MoonshotAI/Kimi-K2/blob/1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4/README.md)；1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4；访问 2026-09-19

- s32 [MoonshotAI/Kimi-K2 / docs/deploy_guidance.md](https://github.com/MoonshotAI/Kimi-K2/blob/1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4/docs/deploy_guidance.md)；1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4；访问 2026-09-19

- s33 [vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K2.6.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K2.6.md)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s34 [vllm-project/vllm-ascend / docs/source/tutorials/models/Kimi-K2-Thinking.md](https://github.com/vllm-project/vllm-ascend/blob/e139b7d573d3769fd1407d5027d7d4831f5469f0/docs/source/tutorials/models/Kimi-K2-Thinking.md)；e139b7d573d3769fd1407d5027d7d4831f5469f0；访问 2026-09-19

- s35 [moonshotai/Kimi-K3 / modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)；f831ab66814297da540d832a5235f8e904f29d06；访问 2026-09-19

- s36 [moonshotai/Kimi-K3 / modeling_kimi_k3.py](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py)；f831ab66814297da540d832a5235f8e904f29d06；访问 2026-09-19

- s37 [moonshotai/Kimi-K2.5 / modeling_kimi_k25.py](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py)；4d01dfe0332d63057c186e0b262165819efb6611；访问 2026-09-19

- hw-amd4 [AMD CDNA4 / MI350 系列架构](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)；滚动文档：页面标为 ROCm 10.0.0；不是本研究验证的软件栈；访问 2026-09-19

- hw-hopper [NVIDIA Hopper 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)；CUDA 13.4 文档快照；访问 2026-09-19

- hw-blackwell [NVIDIA Blackwell 调优指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)；CUDA 13.4 文档快照；访问 2026-09-19

- amd-mi300-k25 [AMD：MI300X K2.5 算子优化](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)；2026-03-24；厂商实验，未在本地复现；访问 2026-09-19

- amd-mxfp4 [AMD：K2.5 / K2.6 / K2.7-Code 的 MXFP4 服务](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-code-in-mxfp4-on-amd-gpus.html)；2026-07-21；ATOM + AITER，厂商实验；访问 2026-09-19