# K3：SGLang、vLLM、TorchTitan 与定制 Runtime 方案

研究基线：2026-10-07；TorchTitan 官方模型列表于 2026-10-08 复核。本文把 K3 模型证据与三个上游框架的公开设计结合，给出训练、推理和调度的工程方案。上游 API 仍在演进；上线前必须锁定源码 commit、依赖和设备固件。文中架构建议是方案推导，不代表已经实现或跑过 K3 训练/推理。

## 结论

建议做一套 **K3 专用控制面 + 上游执行引擎适配层**，不要从零重写训练和推理引擎：推理以 SGLang 或 vLLM 作为可替换的 GPU/NPU worker backend；训练以 TorchTitan 的分布式训练原语、配置和 checkpoint 路径为基础，先建立 K3 模型实现与数值收敛测试；两侧共享模型拓扑、权重/精度元数据、设备能力目录、placement 约束和观测 schema。定制化重点放在模型算子、KDA 状态生命周期、Latent MoE EP 通信布局、硬件感知 placement，以及可插拔的调度策略。

“一个框架”应指统一的控制面和配置契约，不是把训练与在线推理塞进同一个进程。二者有不同的目标函数、内存对象和故障恢复语义。

## K3 对系统的硬约束

本站 K3 配置/模型审计显示文本主干 93 层，其中 69 层 KDA、24 层 MLA；隐藏维度 7168，896 个路由专家、每 token top-16，另有视觉塔。总逻辑参数约 2.78T；这是模型文件/配置审计结果，不意味着单卡可驻留，也不等价于活跃参数或运行吞吐。分布式布局、精度和激活规模要按每种阶段单独预算。

- **两类注意力状态并存**：KDA 是带递归状态的线性注意力，MLA 层有 latent KV cache。调度器不能只用“每 token KV 字节数”估算所有层的缓存；需要分别管理按请求分配的 KDA 状态和 MLA 分页缓存，并定义抢占、迁移、复用及恢复语义。
- **稀疏专家通信占主导之一**：896 个专家、top-16 需要统计 token 热度、专家负载偏斜和 all-to-all 开销。Latent MoE 的投影/激活语义、SiTU、路由权重不能被当作普通 DeepSeek MoE 替换。
- **层类型不均匀**：69/24 的 KDA/MLA 混合层、视觉塔、MoE 层在计算、通信和状态内存上不同。均匀 PP 切层、统一 chunk 或单一显存水位可能导致长尾或 OOM。
- **模型精度分层**：权重格式、activation/累加精度、KV/KDA 状态精度必须是独立字段；量化检查点兼容性要靠 kernel/校准/数值测试确认，不能从权重字节数反推可用精度。

这些约束来自本仓库 K3 的逐层结构和文件头审计、算子映射与独立审计。硬件吞吐、收敛、峰值显存和通信效率尚未在本文中实测。

## 三个框架各自适合什么

| 框架 | 核心强项 | K3 定制落点 | 主要缺口 / 风险 |
|---|---|---|---|
| **SGLang** | 面向在线 serving；RadixAttention/prefix cache、连续批处理、chunked prefill、PD/EPD 解耦、运行时调度与模型服务功能 | KDA/MLA 混合缓存管理；Latent MoE 专用 dispatch/combine；K3 多模态 processor；prefill/decode/vision 角色拆分；K3-aware admission 与状态传输 | cache 命中并不代表 KDA recurrent state 可安全共享；PD 必须传递/重建正确的 KDA 状态和 MLA cache；硬件后端、内核和功能成熟度按 commit/设备矩阵逐项核实 |
| **vLLM** | V1 token-budget scheduler、Paged KV、前缀缓存、structured serving；明确的 scheduler 与 KV connector 扩展边界；广泛的 backend 集成 | 插入 K3 scheduler policy；hybrid KV manager 外接 KDA-state allocator；针对 MLA、KDA、Latent MoE 的模型 runner/kernel；按设备选择适配器 | 通用 paged-KV 不能自动解决 recurrent state 的状态语义；scheduler 单步 token 预算需结合 KDA/MLA/MoE 异构成本校准；vLLM 与 vLLM-Ascend 的版本耦合需要锁定 |
| **TorchTitan** | PyTorch 原生训练；FSDP/TP/PP/CP/EP 等并行组合；配置注册、ModelSpec/扩展接口、分布式 checkpoint、数值收敛方法 | 实现 K3 模型与 sharding 描述；KDA/MLA attention 与 Latent MoE；MoE EP；分层精度、activation checkpointing、PP stage placement；DCP reshard | TorchTitan 是训练平台，不是线上推理 server；官方 README 已列出 K3；先盘点现有实现与目标版本差距，再补齐数据/loss、并行切分、优化器/checkpoint 和收敛验证；支持算子组合依赖 PyTorch/设备后端 |

SGLang 最新 K3 cookbook/分支公开描述 KDA、MLA、Latent MoE、DCP/DSpark、HiCache 与多模态路径；vLLM 当前 API 也列出 K3 的 hybrid、multimodal、PP、quant 和 inner-state 能力，vLLM-Ascend 有 A3 K3 指南。它们证明相关实现正在出现，不证明所有版本/硬件/精度组合均已达到生产可用。2026-10-08 复核时，TorchTitan 官方 README 已把 Kimi K3 列为核心模型家族，并介绍 TitanRL 的训练/生成协同。该事实更新了此前仅依据通用扩展接口的判断；本文尚未审计其 K3 全路径或验证目标设备的训练与收敛，不能从列名推定全部硬件和精度组合可用。

## 目标架构

```text
                K3 Job API / ModelSpec / Hardware & Capability Registry
                                     │
                   Planner + Admission + Placement Controller
                   ├── 训练计划：DP/FSDP × TP × PP × CP × EP
                   └── 推理计划：prefill/decode/vision pools + cache budgets
                                     │
       ┌─────────────────────────────┴─────────────────────────────┐
       │                                                           │
 Training adapter                                             Serving adapter
 TorchTitan Trainer / DCP                                     SGLang or vLLM
 K3 model + sharding + loss                                   K3 runner + scheduler
       │                                                           │
       └──────────── Shared K3 ops / kernels / topology ───────────┘
       KDA state · MLA KV · Latent MoE · SiTU · attention residual · vision
```

各层职责：

1. **K3 ModelSpec**：唯一的模型真相来源，登记层型和顺序、专家路由契约、张量形状、模态入口、参数映射、状态类型、checkpoint key map 和算子精度约束。不同运行时可生成自己的 adapter 配置，但不能各自重新猜模型拓扑。
2. **Kernel/collective 层**：按 device backend 提供 KDA prefill/decode/update、MLA prefill/decode、Latent MoE dispatch/compute/combine、SiTU/attention residual、量化 GEMM 与多模态算子。每个实现声明支持的 shape、dtype、设备、layout、fallback 和数值容差。
3. **训练适配器**：TorchTitan ModelSpec 将层映射到 DTensor mesh；dense/attention 与 MoE 使用不同轴布局；长序列按 CP，模型深度按 PP，稀疏专家按 EP，参数/梯度状态按 FSDP sharding。 checkpoint 采用可重分片格式，并验证不同并行网格间的恢复。
4. **推理适配器**：保留 SGLang/vLLM 的 request lifecycle、tokenizer/API 与 engine loop，接入 K3 状态管理器、模型专用预算模型和策略接口。控制面只向 worker 下发 placement 和配额，不进入每 token 热路径。
5. **可观测性**：统一采集每步 token、队列等待、TTFT/TPOT、KDA 状态字节、MLA KV block、MoE 各专家 token/字节、collective 时长、重算/抢占、显存水位和错误码。训练另采 step time、MFU、tokens/s、梯度范数、loss、checkpoint 时间与恢复点。

## 定制 Runtime 调度

调度分两个尺度：集群控制器负责分钟级资源编排和副本伸缩；engine scheduler 负责毫秒级请求/微批调度。不要让集群控制器参与每 token 决策。

### 在线推理

请求 admission 先估计工作量向量，而不是只看 prompt token 数：

```text
cost(request) = (prefill_tokens, expected_decode_tokens, image_patches,
                 MLA_KV_bytes, KDA_state_bytes, expected_expert_tokens,
                 SLO_class, prefix_reuse_estimate)
```

每轮由 scheduler 在总 token 预算、预填充预算、显存 headroom、KDA 状态容量、视觉编码预算和 MoE 通信预算之内做决策。先筛选前缀命中与可行性，再按 SLO deficit/等待时间给优先级，最后用短 chunk prefill 填充空隙。按负载动态改变 prefill/decode 配额，避免长 prompt 持续压制 decode TPOT，也避免 decode 占满后长 prompt 饥饿。优先级需 aging，防止低优先级永久饿死。

- **缓存键和隔离**：prefix cache 按模型 revision、tokenizer/模板、adapter、精度和租户安全域分区。复用只覆盖经过模型语义证明可共享的前缀；KDA 的递归状态必须与 token 前缀、层/版本和精度绑定，不能误当作可任意切片的 KV block。
- **抢占策略**：按重算成本比较保留、换出、迁移与重新 prefill。MLA 可按 page/block 管理；KDA 可按请求状态整体迁移或在安全 checkpoint 边界重算。具体方案需测状态字节及恢复时延后决定。
- **PD/EPD**：prefill 输出交接包包含模型版本、token/位置边界、MLA cache 元数据与 KDA state、精度/layout、校验和；接收端校验兼容后才接纳。视觉 encoder 可分离成 EPD worker，但图像特征缓存和 PD 状态传输分开预算。
- **MoE 负载**：记录路由 token 数和 bytes 的 P50/P95/max；基于近期 expert queue/链路负载选择 EP placement。对热门专家考虑复制/重映射需要模型权重、路由语义一致性及设备容量验证，初版可先用负载感知分配和 admission 限流。
- **策略接口**：`estimate(request, cache_state) -> cost`、`admit(cost, budgets) -> decision`、`schedule(queue, budgets) -> batch`、`preempt(request, reason) -> action`。默认策略可复用上游 scheduler；K3 策略作为可配置 policy/plugin，而不是长期维护一份 engine 核心 fork。

优化目标采用约束式多目标：满足 TTFT/TPOT SLO 和错误率上限后，提升有效输出 token/s 与 GPU/NPU 利用率；公平性和长 prompt 等待上限作为约束。单纯最大化吞吐会恶化尾延迟，单纯按 token 数均衡会忽略视觉、KDA、MoE 差异。

### 训练作业

训练计划以拓扑和显存为输入，候选并行网格为 DP/FSDP × TP × PP × CP × EP；先筛除设备拓扑不支持的组合，再依据逐层计算/通信/激活估算选择 PP 边界和 EP 映射。KDA 与 MLA 层成本不同，视觉塔应独立估算；按每 stage 的峰值激活、参数/优化器分片、通信量和重计算成本做均衡，不用“层数均分”代替 profile。

执行时让每个 rank 保持固定训练计划，采用可复现 sampler 和共同 checkpoint step；只在 optimizer/update/checkpoint 边界做弹性伸缩或重分片。故障恢复读取最近一致 checkpoint，写出 mesh 元数据、模型 revision、优化器状态、RNG、数据游标、loss scaler 和训练配置哈希。训练“调度”主要是集群作业排队、拓扑 placement、节点故障恢复和 checkpoint 协调；微批与 pipeline schedule 仍由训练引擎执行。

## 分阶段落地与验收

1. **基线锁定与模型契约**：固定 K3 配置/权重 revision、SGLang/vLLM/TorchTitan/PyTorch commit、驱动与 kernel 版本；把本站 K3 逐层表转换成有 schema 的 ModelSpec；定义文本、图片、工具调用和输出格式测试集。
2. **推理正确性**：选一个硬件目标和一个上游 backend，先单机/小规模跑纯文本、图像、长上下文、并发、取消/超时、前缀命中/失效、混合 KDA/MLA、专家路由。对参考实现比较 logits/输出及 KDA state，覆盖量化格式。先做 correctness 与故障恢复，暂不调吞吐。
3. **TorchTitan 单机到多维并行**：debug model → 小型 K3 结构仿真 → 真权重小规模 forward/backward → 1D FSDP 数值基线 → 逐项加入 TP、PP、CP、EP、activation checkpoint 与目标精度；每增加一个并行维度跑 loss convergence 和 checkpoint reshard。TorchTitan 上游建议用可信单维基线与多维测试验证收敛；fake backend 只做配置/控制流检查，不可替代真实数据数值验证。
4. **Runtime policy A/B**：固定模型、权重、kernel、设备数与请求 trace；分别比较上游默认调度和 K3 成本感知策略。报告吞吐、TTFT/TPOT P50/P95/P99、SLO 达标率、队列等待、显存峰值、prefix 命中、KDA/MLA 占用、MoE skew、通信时间和错误/重试。
5. **灰度与版本治理**：按模型 revision × runtime commit × backend × dtype × topology 建兼容矩阵；逐单元标记已跑正确性、已测性能、已生产验证，未知状态保持未知。内核/框架升级都重新跑差分和回归。

### 最小验收门槛

- 数值：固定输入下与可信参考输出在设定 tolerance 内；训练 loss 曲线、梯度和 optimizer step 无异常，长跑能恢复并继续收敛。
- 状态：连续请求不会串扰 KDA 状态；取消、抢占、迁移、prefix invalidation 与 PD 重连后没有跨请求污染。
- 资源：无 OOM/无限重试；KDA、MLA、视觉和 MoE 各自的容量统计与实际 allocator/通信观测相符。
- 调度：目标负载 trace 下满足预设 SLO 与公平性；任何吞吐提升都同时报告尾延迟与资源变化。
- 可复现：结果关联配置、代码、容器、权重和数据哈希；区分仿真、源码能力、功能验证、性能测量、生产验证。

## 推荐决策

第一阶段选择 **vLLM/SGLang 其一作为推理执行基线**，按目标设备支持矩阵及 K3 正确性测试决定，不预设赢家；另一套保留为可比 backend。K3 目前公开 cookbook 与 device-specific 指南表明两边都值得做候选，不意味着无需集成工作。**TorchTitan 作为训练基础**，先盘点上游 K3 实现、补齐模型契约并做短程 loss convergence；若设备通信、低精度或长序列算子不满足，再把需要的基础设施贡献回 PyTorch/TorchTitan 或由后端插件补齐。

先做一个共享 K3 schema、一个可测算子/状态 API 和统一 benchmark harness。只有 profiling 证明上游扩展点成为真实瓶颈，再把对应部件抽成独立 K3 runtime 库；不建议起步就 fork 三个大型框架并承担长期同步成本。

## 上游与站内资料

- [SGLang Kimi-K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- [SGLang K3 roadmap](https://github.com/sgl-project/sglang/issues/32607)
- [vLLM K3 模型 API](https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/)
- [vLLM scheduler 配置与策略](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/)
- [vLLM Ascend K3 部署指南](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html)
- [TorchTitan 项目与并行训练](https://github.com/pytorch/torchtitan)
- [TorchTitan 扩展接口](https://github.com/pytorch/torchtitan/blob/main/docs/extension.md)
- [TorchTitan 数值收敛验证方法](https://github.com/pytorch/torchtitan/blob/main/docs/converging.md)
- [本站 K3 深度研究报告](Kimi-K2至K3深度研究报告.md)
- [本站 K3 逐层算子/API 映射](compute/Kimi-layer-api-report.md)
- [本站超节点资源与阶段模型](../../supernode/index.html)

