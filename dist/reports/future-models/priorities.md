# 未来模型研发的五项重点 技术机制与开发任务

技术研究基准：2026-10-08。重点与阅读结构整理：2026-10-09；TorchTitan 开发接口另锁定下文注明的源码版本。

这份报告面向要做 K3 算子、训练适配和推理调度的开发者，也用 DeepSeek 4.1 作为稀疏与共享缓存的专项对象。重点收敛到五项：**递归状态、稀疏共享缓存、MoE、低精度、训练推理 Runtime**。它们直接决定现有模型能否正确执行、能承载多少请求，以及计算和通信成本能否转化为服务收益。

每项先解释数学或数据流，再列出按依赖推进的开发工作包、交付物和验收条件。这里的接口是开发规格；列出工作包不表示相关实现或硬件实验已经完成。形状与字节算例区分逻辑量、物理分配和实测结果。

<a id="focus-toc"></a>
## 五项重点与阅读入口

| 重点 | 适用对象与关键问题 | 第一个交付物 |
|---|---|---|
| [1 递归状态算子](#focus-state) | K3 的 KDA 如何分块计算、续写并在拒绝后恢复；PGDN/PKDA 是否值得重新训练 | 单层数值参照与完整状态契约 |
| [2 稀疏与共享缓存](#focus-attention) | DS4.1 的 CSA2 是否真正少读、少算；CED/replay 如何恢复可用状态 | 逐层 owner/consumer 表与物理稀疏索引链 |
| [3 MoE 计算与通信](#focus-moe) | K3 的 top-16 分配如何避免小矩阵碎片和最慢专家拖尾 | 路由到归并的完整可往返链路 |
| [4 OCP/MX 低精度](#focus-precision) | 权重、激活和缓存的 scale/布局如何正确转换并调用目标设备 | 格式解码、转换 golden 向量与能力表 |
| [5 K3 训练推理 Runtime](#focus-runtime) | 如何把算子接入 SGLang/vLLM 与 TorchTitan，并保证预算、提交和恢复正确 | 一个后端的执行计划与状态提交适配器 |

第一次阅读，先看这张表，再逐项阅读“技术机制”和“开发事项”。负责某个算子的读者可直接进入对应项；负责整体框架的读者先读第 5 项的接口与预算，再回到前四项补齐执行模块。[42 章背景报告](report.md)保留原始论文、其他架构路线和推导，并按主题折叠目录，供需要时继续深入。

## 开发范围与推进顺序

**P0** 是数值、布局、状态与恢复正确性；**P1** 是保持已声明模型及量化语义的执行优化；**P2** 是改变结构、精度配方或长期状态表示的模型实验，需独立训练或适应及质量验证。PGDN/PKDA、无 Softmax、不同 Top-k 和把 CED 移植到 K3 都不能当作现成检查点的透明加速开关。

五项是重点范围，实际顺序按依赖推进：先冻结第 5 项的后端与契约，完成第 1/3/4 项的正确性基线；再以完整链路 profiling 决定性能热点。第 2 项作为 DS4.1 独立适配，最后回到第 5 项做混合负载与训练恢复验收。新增注意力函数和缩放研究留在受控实验中，不能挤占现有模型正确执行的工作。

以下开发验收需要目标设备和实际模型/数据。可接受误差、任务质量、首次响应、逐 token 延迟和等待上限须在比较前固定；没有这些条件，就只能报告机制、兼容性或算例，不能宣称性能收益。

<a id="s04"></a>
<a id="s06"></a>
<a id="s25"></a>
<a id="s26"></a>
<a id="s27"></a>
<a id="s28"></a>
<a id="s29"></a>
<a id="s30"></a>
<a id="s31"></a>
<a id="s32"></a>
<a id="focus-state"></a>
## 1 先把 KDA 状态算子做好，再决定是否研究 PGDN/PKDA

**这项工作的核心是：K3 的 69 层 KDA 必须在分块输入、并发、迁移和投机拒绝后保持同一条历史。** 错一次状态，即使当前 token 没有异常，后续输出也会继续偏离。先做现有 K3 的 P0 正确性与 P1 性能；PGDN/PKDA 另开 P2 训练实验，不能作为加载原权重时的开关。

### 技术机制

省略投影、短卷积、归一化和输出门，按 `S∈R^(d_k×d_v)` 的单头约定，KDA 核心可写为：

```text
Sbar_t = D_t S_(t−1)                 # D_t 为 key 通道遗忘对角阵
e_t    = v_t − Sbar_tᵀ k_t
S_t    = Sbar_t + β_t k_t e_tᵀ
o_t    = S_tᵀ q_t
```

它先遗忘，再按当前记忆的预测误差写入。Decode 每步读改写一个矩阵；Prefill/训练把块内相互影响转成矩阵运算，再传递块边界状态。块内 token 使用不同的因果状态，不能把块最终状态用于所有位置。以上是核心解释，实际门参数化、短卷积和 Q/K 预处理必须逐项按权重对应实现对齐。[Kimi Linear](https://arxiv.org/abs/2510.26692)、[本站 K3 配置](../../data/families/kimi/configs/Kimi-K3-config.json)。

建议公开的算子契约采用 `Q/K:[B,T,H,d_k]`、`V:[B,T,H,d_v]`、逐通道遗忘参数 `[B,T,H,d_k]`、主写入门 `[B,T,H]`，输入/返回 `S:[B,H,d_k,d_v]` 和 `O:[B,T,H,d_v]`；packed 输入另有样本边界。这里是建议的逻辑布局，后端若采用转置或扁平布局，转换和 stride 必须登记。短卷积状态属于完整层状态，不能从算子输出中遗漏。按 K3 的 `H=96,d_k=d_v=128`，FP32 主状态每层 6 MiB，69 层约 **414 MiB/请求**，尚未计算分片、卷积和快照。

PKDA 在主状态旁增加 `A:[B,H,d_k]`，记录 key 的逐坐标统计：

```text
A_t = αP_t A_(t−1) + βP_t (k_t ⊙ k_t)
r_t = log(A_t+ε) − c;  u_t = r_t/(1+|r_t|)
m_t = exp(−log(x)u_t);  k_write = m_t ⊙ k_t
S_t = D_t S_(t−1) + β_t k_write e_tᵀ
```

其中 `αP/βP` 是统计支路的门，`c` 是可学习中心，`ε>0` 防止对零取对数，`x>1` 决定缩放范围 `(1/x,x)`。主状态的误差仍为 `e_t=v_t−(D_t S_(t−1))ᵀk_t`。误差 `e_t` 仍用原 `k_t` 读取，写入才用 `k_write`；把两处都替换会改变模型。PGDN 使用标量遗忘，PKDA 保留逐通道遗忘；新统计门与主写入门分开。按上例，A 每层 48 KiB，69 层约 3.23 MiB。额外容量小，但 log/exp、块扫描和反向并不免费。上述 A 是按 K3 形状的假设增量，不代表已有 K3 检查点包含它。[固定 PGDN 参考](https://github.com/fla-org/flash-linear-attention/blob/37a6b1c6290e5240f6f0d80419d08a7aac27e548/fla/ops/precond_gated_delta_rule/naive.py)、[固定 PKDA 参考](https://github.com/fla-org/flash-linear-attention/blob/37a6b1c6290e5240f6f0d80419d08a7aac27e548/fla/ops/precond_kda/naive.py)。

### 开发事项

按以下顺序做四个工作包；每个先交付可检查结果，再进入下一步。

1. **P0：数值与布局契约。** 交付现有 K3 单层逐 token 参照、前处理顺序、状态布局和输入样本边界定义。验收比较输出与最终 S，覆盖非零初态、1/63/64/65 token、打包样本、空 suffix 与中途拆分；预填充后续写必须与连续输入一致。误差阈值由固定 dtype/参照基线预先确定，不能在看见失败后放宽。
2. **P0→P1：Decode 与状态事务。** 交付逐 token 更新及会话槽管理，原子提交 `S+卷积状态+token位置`。前缀分叉使用隔离副本；投机验证只提交已接受边界。拒绝后恢复已保存边界，并从该边界重算必要 token，不能对最终矩阵做“减去被拒绝 token”。验收全接受、零接受、部分接受、取消、槽位复用和迁移后 logits；同时测更新、快照和复制字节，不能只测计算 kernel。
3. **P0→P1：Chunk 与训练反向。** 交付块前向、状态边界、反向及 activation 重算方案，比较不同块长的输出、初态梯度和参数梯度。先盘点 TorchTitan 已有 K3 路径：固定 README 声明 KDA 使用 Attention Gym 并支持其 native CP，CUDA capability 要求≥9.0；这是已有后端能力，不能与 FLA 的 PKDA 限制混同。[固定 TorchTitan K3 README](https://github.com/pytorch/torchtitan/blob/fb45f5e878463f64a1a70ab80e773b7916de52f0/torchtitan/models/kimi_k3/README.md)。目标设备不匹配时才补后端；任何 CP 移植都要传正确边界状态或等价变换，多 rank 对照单 rank 的损失与梯度后才扩大规模。优化重点是块矩阵计算与向量门控的布局、临时张量和融合，而非无条件压低 S 精度。
4. **P2：PKDA 配对实验。** 交付含 A 的独立模型分支、`O/S_end/A_end` 契约与完整反向。key 梯度同时经过原 key 读取和预条件写入；A 的时间递推、中心参数及初始/最终 A 的梯度不能漏掉。S、A、卷积状态必须联合快照和恢复。固定 FLA 快照的 PGDN/PKDA chunk 均拒绝 `cp_context`，PKDA 要求 key dim≤256、初始 S 为 FP32；先明确移植缺口，不宣称自动继承基础 KDA 的 CP。[Chunk 限制](https://github.com/fla-org/flash-linear-attention/blob/37a6b1c6290e5240f6f0d80419d08a7aac27e548/fla/ops/precond_kda/chunk.py)、[ATK 反向](https://github.com/fla-org/flash-linear-attention/blob/37a6b1c6290e5240f6f0d80419d08a7aac27e548/fla/ops/atk/chunk_atk_bwd.py)。

### 验收与停止条件

S 与 A 先保持 FP32 基线，投影/块乘精度另测。PKDA 先用匹配数据、参数与训练预算的小模型，对照 KDA 的长检索、反复覆盖写入、训练时间和服务总成本；不能把论文约 340M/1B 的结果外推为 K3 收益。[原论文实验](https://arxiv.org/html/2604.21100v1#S4)。状态一致性或梯度不过，停止性能调优；质量提升不能覆盖新增时间与恢复成本，停止主干替换。无 Softmax 的 Sigmoid/ReLU 等稠密替代若仍计算全部 `QKᵀ`，复杂度仍为 O(T²)；核特征、Delta 或多尺度状态是不同路线。先在同训练与状态预算下比较长检索和梯度，再决定是否扩展本项，不能仅以“省掉 Softmax”申请主干替换。[分类与开发代价](report.md#s29)。

进一步阅读：背景报告 [04 章](report.md#s04)、[06 章](report.md#s06)、[25 章](report.md#s25)及[新兴注意力分类](report.md#s29)。

<a id="s05"></a>
<a id="s36"></a>
<a id="s37"></a>
<a id="s38"></a>
<a id="s39"></a>
<a id="s40"></a>
<a id="s41"></a>
<a id="focus-attention"></a>
## 2 DeepSeek 4.1：把“筛选、消费、恢复缓存”做成真实执行路径

**这项工作的核心是：共享缓存能少存，候选池能少算，但只有布局和执行路径一起改变才有实际收益。** 对现成 DS4.1，先实现官方模型及恢复规则；对 K3，缓存生命周期经验可用于 P0/P1，CSA2/CED 架构移植属于 P2，不能直接复用旧权重。

### 技术机制

固定 DS4.1 Flash 是 40 层语言主干，前后各 20 层。CSA2 含 4 个 Full、4 个 Reindex、30 个 Reuse，另有 2 个纯 SWA 层。Full 发布 main KV、index K 和选择结果；Reindex 复用前两者、重新选择；Reuse 连选择结果也复用。**三者仍计算本层 Q、SWA 和 attention 输出**。encoder 的全局 owner 为 2/8/14 层，压缩 ratio=2；decoder owner 为 20 层，ratio=1，24/28/32/36 层重新索引。[固定配置与数据流](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py)。

逻辑 main KV 为 `[B,M,512]`，index K 为 `[B,M,128]`；当前 query 的 index Q 为 `[B,Q,32,128]`。索引分数按 32 个头的 ReLU 点积加权归约，实际缩放与权重按参照。decoder 第一次 Full 扫描全部因果范围，按每 8 个位置的块最大值选最多 2048 块，得到最多 16384 候选；后续 Reindex 只应打分这些候选，再选 top-512。候选集与最终 attention 集不是同一对象。主 attention 的 64 个 Q head、512 维向量消费选中的 global KV 与本层 128-token SWA，仍使用 Softmax。[固定配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/config.json)。

固定 reference 的 `Indexer.forward` 对完整 index K 做 einsum 后才 mask 候选池外分数；逻辑正确，却未省池外乘法。生产优化应改为 **候选位置→Gather index K→局部 score→Top-K→Gather main KV→attention**，或让稀疏 kernel 直接跳过池外 tile。百万 token 时五个 decoder indexer 的打分位置数由 `5N` 变成 `N+4×16384`，约为前者 21.25%；这只数位置，不是全模型提速率，第一次扫描仍随 N 增长。[Reference 索引路径](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py)。

CED 让 decoder 的全局 KV 来自最后一个语言 encoder 表示，本层局部 KV 仍来自 decoder 输入。官方优化只回放 prompt 尾部 128 个 encoder 表示，以恢复 decoder SWA；按层-token 近似为 `20N+20min(N,128)`，不同于完整参考 `40N`。多层 SWA 的依赖会超出一个窗口，因此 bounded replay 是规定近似。现成 DS4.1 应按官方模式实现和评测；K3 若引入这种结构与恢复方式，才需要新训练或适应证据。不能从“KV都叫cache”推导透明兼容。[官方 CED/replay 设计](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/DeepSeek_V41_Tech_Report.pdf)。

### 开发事项

1. **P0：共享状态依赖表。** 交付逐层 owner/consumer 表与 cache 描述，包含样本、权重版本、压缩位置、因果长度和 query/分支身份。每个 consumer 验收是否读到正确 owner 和本 query 的选择结果；复用不能用一份跨请求全局可变 Top-K。
2. **P1：物理稀疏索引链。** 交付块最大归约、块 Top-K、候选 Gather、候选内打分和最终 Top-K，处理短尾块、因果边界、重复位置与平分规则。与同规则 reference 比较选择结果、attention 输出和 logits；实际打分位置数与 kernel 读取范围必须表明池外已跳过。记录 Gather 字节、临时 score 峰值与总索引时间，若不规则搬运抵消省算，保留密集回退。
3. **P0→P1：缓存打包与消费。** 交付 main KV 的 E2M1/16 通道 E4M3 scale、index K 的 E2M1/32 通道 E8M0 scale、SWA 的 FP8/32 通道 E8M0 三种独立布局。main 一条为 288 B，index 为 68 B；四 owner 的全局逻辑容量为 `(3floor(N/2)+N)×356 B`，偶数 N 即 890 B/token。验收真实分配、scale 解码、量化位置与融合解包/Gather；reference 浮点 buffer 模拟量化不能作为实际容量证据。[量化内核](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/kernel.py)。
4. **P0：CED/replay 与拒绝恢复。** 交付 encoder prefill、decoder replay、完整评分、decode/verify 的明确阶段接口。命中 global KV 但缺 SWA 时，encoder replay 不得覆盖已命中 global；新 suffix 才发布新条目。事务包含 global 长度、compressor 状态、SWA 环形内容/指针和索引版本，拒绝后恢复并使失效分支选择结果不可见；仅截断全局长度无法恢复被环形写覆盖的 SWA。
5. **P0：训练与并行边界。** 若目标包含训练，交付共享张量的梯度归属、micro-batch 生命周期及跨 PP 的缓存/梯度 payload，先让小配置的单 stage 与跨 stage 梯度对齐。共享表示不能因复用而 detach；owner 跨 stage 的通信量与重计算要纳入切分。CP 对压缩绝对位置、因果边界和候选编号的处理需单独验证，不把框架支持 CP 当作 CSA2 已适配。

### 验收与停止条件

分三组验收：完整前向参照检查基础算子；可信后端执行同一 replay 算法检查实现；replay 对完整前向测长检索、数字与工具链质量。覆盖短/长 prompt、不同命中边界、全接受/部分接受/拒绝与迁移。索引结果或 owner 版本错误，停止速度比较；质量未满足规定 replay 模式，不能用缓存节省掩盖差异。通过后分别报告首次全扫描、候选打分、解包、replay、TTFT 和 TPOT，才判断收益。以上为待开发规格，未声称已完成 GPU/NPU 实现。

进一步阅读：背景报告 [36 章](report.md#s36)、[37 章](report.md#s37)、[38 章](report.md#s38)、[39 章](report.md#s39)、[40 章](report.md#s40)、[41 章](report.md#s41)。

<a id="s07"></a>
<a id="s08"></a>
<a id="s09"></a>
<a id="s10"></a>
<a id="focus-moe"></a>
## 3 MoE：优先解决小专家矩阵与跨设备尾部等待

**为什么排进前五：** 对 K3，MoE 不是一个可选研究模块，而是大部分语言层的真实执行路径。896 个路由专家、每 token 选 16 个，意味着大容量、碎片化计算和高扇出通信同时存在。优先投入应是把现有路由结果高效执行；“增加专家数”“降低 Top-k”属于模型研究，不能作为服务调度器的透明优化。

### 3.1 技术核心：把一个 token 的 16 条路径正确送出再合回来

以下维度来自 K3 固定 revision `f831ab66814297da540d832a5235f8e904f29d06`。设这一层处理 T 个有效 token，路由在主干输入 `X[T,7168]` 上计算，返回 `ids[T,16]` 和 `weights[T,16]`；共享下投影把输入变为 `Z[T,3584]`，再进入路由专家。专家门控/上投影的逻辑权重均为 `[3072,3584]`，下投影为 `[3584,3072]`。加权归并得到 `[T,3584]` 后，先潜空间 RMSNorm，再共享上投影回 `[T,7168]`，与共享专家分支组合。[固定配置](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)、[固定路由与专家实现](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L703)。

完整数据流必须拆开看：

```text
route → count[e] → exclusive_prefix_sum[e] → permute
      → 按目标 rank 打包/dispatch → 按专家 grouped GEMM
      → SiTU → grouped GEMM → combine → inverse_permute
      → Top-k 加权归并 → RMSNorm → 上投影
```

`count[e]` 决定每个专家真实 M 维；前缀和给出专家连续区间，排列同时保存 `(原 token、Top-k 槽、专家、目标 rank)` 的映射。专家 e 接收 `Z_e[M_e,3584]`，两条投影产生 `[M_e,3072]`，经模型规定的 SiTU 后下投影，最后恢复每个 token 的 16 条输出。SiTU 和聚合后归一化不能替换成普通 SwiGLU。LatentMoE 的收益边界也在这里：把收发宽度 7168 降为 3584，减少这一项载荷，但新增共享投影和归一化，不代表整层计算或通信延迟减半。

均匀路由仅给出 `mean(M_e)=16T/896`：T=32 时约 0.571，T=1024 时约 18.29。因此要同时准备紧凑小 M 路径和吞吐路径。若为矩阵 tile 补齐到 `ceil(M_e/b)×b`，必须区分有效计数与 padded 计数，padding 输出不能进入 combine。空专家不启动计算；热点不能靠丢 token 或改选低负载专家解决。

算子契约至少交付：有效/补齐计数、send/recv rank offsets、排列与逆排列、路由权重施加位置、量化数据与 scale 描述、completion event。设备级 token 去重可以减少同一 token 发给同一 rank 的重复载荷，但要由接收端重新展开到各专家；须测额外索引成本。忽略去重时，dispatch 加 combine 的有效载荷近似为 `2×T×16×3584×元素字节数`，另计 scale、映射、对齐和协议。训练反向还要逆向收发 dZ、计算各专家 dW、恢复 dX 和所选路由权重梯度；Top-k 的离散选择沿模型原训练规则处理。专家副本的梯度归并与权重版本同步是训练额外成本。

前缀和输出应有 E+1 个边界，最后一项等于真实分配总数；EP 再建立目标 rank 的边界，不能混用这两组 offsets。通信完成只代表数据到达，计算完成还须独立事件；buffer 复用必须等消费结束。跟踪最慢 rank 的“接收—计算—回传”关键路径，才能区分热点来自专家路由、消息启动、矩阵碎片还是链路拥塞，避免对所有热点一律增加副本。

### 3.2 按顺序开发的五个工作包

| 顺序 | 开发事项与交付物 | 验收依据 |
|---|---|---|
| 1，P0 | 固定路由、SiTU、共享分支和潜空间归一化语义；交付单设备参考层及形状/排列契约 | 对齐参考 logits 与各阶段张量；空专家、同分 Top-k、masked token、奇数 M、全 token 热点均正确；同分规则以固定实现为准 |
| 2，P0 | 实现 count/prefix-sum/permute/inverse 与 EP dispatch/combine；交付可往返的 token/scale 数据包 | 全局 `ΣM_e=16T`；每条分配不丢不重、scale 随原块迁移；从单 rank 扩到多 rank，与单设备数学结果在声明容差内一致 |
| 3，P1 | 按真实 M_e 分布选择紧凑或 grouped GEMM 路径，融合安全的排列/量化和 SiTU/量化；交付形状分派表与阶段 profiler | 扫测 T、热点、空专家和 tile 边界；报告 padding 字节、转换、通信和两次 GEMM 的合计延迟，不能只报 GEMM 峰值 |
| 4，P1 | 在保持专家身份与权重一致的前提下做分块重叠、放置及热点副本；交付 EP 计划与等待预算 | 同一请求 trace 下记录最大/平均负载、每 rank 排队、通信时间及 TPOT P99；证明收益超过复制/迁移成本且无输出变化、无请求饥饿 |
| 5，训练 P0 | 补齐 permutation/collective 反向、专家 dgrad/wgrad 与副本同步；交付 TorchTitan EP 适配和可恢复 checkpoint | 与非 EP 基线比较输入、权重和路由权重梯度；累积多步、空专家、恢复重分片及短训 loss 一致；再验证规模迁移 |

**投入决策：** 先实现工作包 1–2，再由完整层 profiler 决定包 3–4 的收益排序。验收同时看有效 token/s、P99、实际常驻权重与网络字节。改变潜空间宽度、Top-k、专家数量或训练均衡规则列为 P2，独立重训后比较质量与总成本；它们不进入现有 K3 检查点的默认优化路径。本文未实测这些工作包的硬件收益。

进一步阅读：背景报告 [07 章](report.md#s07)、[08 章](report.md#s08)、[09 章](report.md#s09)、[10 章](report.md#s10)。

<a id="s14"></a>
<a id="s15"></a>
<a id="s16"></a>
<a id="s33"></a>
<a id="s34"></a>
<a id="s35"></a>
<a id="focus-precision"></a>
## 4 OCP/MX 低精度：先把数值、布局和后端选择做成可验证契约

**为什么排进前五：** K3 路由专家已使用 MXFP4 权重，DeepSeek 4.1 又把低精度用于不同缓存对象。最紧迫的问题是“能否按原格式正确、高效执行”，不是再增加一个 FP4 名称。位数降低后的 scale、转换和小矩阵 padding，可能抵消理论节省；长期递归状态的误差还会进入下一步更新。

### 4.1 技术核心：同为 FP4，不能按同一字节解释

| 使用对象 | 元素与缩放契约 | 工程含义 |
|---|---|---|
| OCP MXFP4 | E2M1，每 32 元素共享 E8M0 scale | scale 是 2 的幂；必须登记分块轴与尾块 |
| NVFP4 | E2M1，一维配方每 16 元素 E4M3 scale，另有全局 FP32 scale；权重可用 16×16 二维缩放 | 不能套用 MX 解码器；二维权重的元数据摊销另算 |
| DS 4.1 main KV | E2M1，每 16 通道 E4M3 scale，不带完整 NVFP4 的 global scale | 特定缓存存储格式，attention 前解量化；不承诺可直接喂原生 FP4 GEMM |
| DS 4.1 index K / SWA | 前者 MXFP4，后者 MXFP8，均每 32 通道 E8M0 scale | 同一模型也须保留独立缓存描述与转换路径 |

来源：[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)、[Transformer Engine NVFP4 配方](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)、[DS 4.1 固定量化调用](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/model.py)。

采用逻辑权重 `W[N,K]`、输入 `X[M,K]`，前向为 `Y=XWᵀ`。MX 行内 K 轴分块时，scale 的逻辑形状分别为 `[M,ceil(K/32)]` 与 `[N,ceil(K/32)]`；点积须逐块使用两侧 scale，再归并块结果。缩放随块变化，不能在完整 K 归约后随意乘一个全局系数。训练中 `dX=dY W`、`dW=dYᵀ X` 改变了归约轴；转置 packed 数据和 scale 指针，不会自动得到新轴的合法量化表示，须按配方重新分组/量化或维护经过验证的双布局。

互换格式规定数值含义，后端执行布局还包含 tile、scale 排列、stride、对齐与累加类型。NVIDIA cuDNN 的 scale tile 与 CANN ND/NZ 组织不同；仅比较 dtype 名称不足以选择 kernel。[cuDNN scale 布局](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/)、[CANN 9.1 MxMatmul](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/mxmatmul_scenario.md)。例如 `[1,4096]` 的 MXFP4 逻辑载荷为 2048 字节数据+128 字节 scale；若 scale 单独补到 128 行，则总计 18432 字节，反而超过该输入的 BF16 8192 字节。这是填充风险算例，不是已测设备结果。

张量精度要分开：权重可离线校准并打包；激活在线统计与转换；KV 多为写入后读取；递归 S/A 状态持续读写，局部误差近似满足 `E_t≈J_t E_(t−1)+ε_t`。后一项必须测长度、分支和回滚后的漂移，不能用一次 GEMM 的均方误差验收。路由、门、归一化、状态与梯度累积保留 BF16/FP32 参照。

profiling 要覆盖 Decode 的一两个专家 token、Prefill 的大批及训练反向，而不是选单个整齐大矩阵。分开记录离线权重转换和在线激活转换：前者可在模型加载时摊销，后者每步都付出成本。两种路径即使使用相同矩阵单元，也可能因量化统计、scale 填充或布局维护而有相反结果，默认路径须按真实负载决定。

### 4.2 按顺序开发的五个工作包

| 顺序 | 开发事项与交付物 | 验收依据 |
|---|---|---|
| 1，P0 | 登记元素/scale 编码、块和轴、global scale、舍入/饱和、layout 版本、累加与尾块；交付检查点解码器和 golden 向量 | 未知编码明确拒绝；测试 ±0、次正规数、中点、最大值邻域、NaN/Inf 和尾块；跨后端解码位模式对齐 |
| 2，P0 | 将量化、打包、transpose/reblock、后端 scale 布局转换分开；交付每种目标 GEMM 的合法输入生成路径 | 行列双表示分别与高精度参照比对；转置不能复用错误分块；有效字节包含 scale、对齐、工作区及双布局 |
| 3，P0→P1 | 按设备/软件版本、shape、编码、布局及 SiTU 支持分派原生算子；交付能力表与已验证的解码后 BF16 fallback | profiler 证明实际执行路径；原生与 fallback 数值达标；报告“转换+GEMM+epilogue”总时间，并记录 fallback 比例与峰值内存 |
| 4，P1 | 在原配方边界内融合激活统计/量化与专家排列，元素和 scale 一起通信；交付 MoE 的小 M/大 M 成本曲线 | 专家热点 trace 上测吞吐与 P99；all-to-all 搬运数值正确，all-reduce 不直接相加不同 scale 的编码；同步 amax 和元数据成本全部计入 |
| 5，P2 | 把增强缩放、状态降精度、低精度反向分别设为研究实验；交付质量曲线、长序列状态误差与达到目标 loss 的训练成本 | 同数据/种子/质量预算比较；覆盖罕见实体、路由边界、长无关段、分支回滚与恢复；短训稳定后再做长训和规模验证 |

转换 goldens 应纳入正式的 [OFP8 Revision 1.1 FINAL](https://github.com/opencomputeproject/FP8/blob/29141cc56f8b04841a13390a5cbe6ac59e1cb24c/OCP%208-bit%20Floating%20Point%20Specification%20%28OFP8%29%20Revision%201.1%20FINAL.pdf)：先舍入再判断溢出，E4M3 最大有限值 448 与中点 464、E5M2 的 57344 与中点 61440 必须测试，饱和和非饱和分别建预期结果。该正式文件与 MX 的版本独立；本报告尚未核实官方 MX v2.0，FP4/FP6 已在 MX v1.0。增强缩放及附加元数据论文只能进入工作包 5，不能提前当作“2.0 新标准”或硬件支持承诺。

**投入决策：** 先完成包 1–3，使原检查点可解释、可调用、可回退，再优化实际热点。P1 保持已声明的模型/量化语义；新的 PTQ/QAT、缩放配方与长期状态降精度要另设质量门槛，不能凭 dtype 相同划为透明优化。只有端到端成本下降且质量达标，才将实验升为默认路径；本文未复现低精度训练收敛或 GPU/NPU 性能。

进一步阅读：背景报告 [14 章](report.md#s14)、[15 章](report.md#s15)、[16 章](report.md#s16)、[33 章](report.md#s33)、[34 章](report.md#s34)、[35 章](report.md#s35)。

<a id="s01"></a>
<a id="s02"></a>
<a id="s03"></a>
<a id="s11"></a>
<a id="s12"></a>
<a id="s13"></a>
<a id="s17"></a>
<a id="s18"></a>
<a id="s19"></a>
<a id="s20"></a>
<a id="s21"></a>
<a id="s22"></a>
<a id="s23"></a>
<a id="s24"></a>
<a id="s42"></a>
<a id="focus-runtime"></a>
## 5 K3 训练与推理 Runtime 把前四项接成可恢复的执行系统

### 5.1 技术重点是一次执行计划与一次状态提交

K3 的 69 层 KDA、24 层 MLA、LatentMoE 和视觉编码器有不同的内存与阶段成本。推理引擎每轮需要回答三个具体问题：哪些请求能进入本轮；为它们分配哪些状态与临时缓冲；计算结束后哪些状态可以提交。训练引擎则需要确定参数和激活如何切分，以及失败后从哪个一致的更新点恢复。

建议先选 SGLang 或 vLLM **其中一个**作为目标设备的执行基线。这里要开发的是成本与状态适配器，复用引擎已有的队列、模型执行和请求生命周期。TorchTitan 承担训练；两侧共享权重版本、层型、算子精度和分片描述，不共享线上队列或训练中的可变状态。[SGLang K3 指南](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)、[vLLM K3 模块](https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/)。

先冻结下面三份开发契约。它们是建议的接口字段，不是声称上游已有同名 API。

| 契约 | 必须传递的信息 | 拒绝条件 |
|---|---|---|
| 模型与后端清单 | 权重 revision；93 层的类型与顺序；专家放置；每种算子的 shape、dtype、scale、layout 和累加规则 | 未知布局、未验证的 dtype、设备不支持且没有明确回退 |
| 每轮执行计划 | 请求 ID、阶段、token/position、序列边界；MLA block table；KDA 状态句柄；Verify 分支和暂存空间 | 位置与状态边界不一致；资源预留不足；前缀属于不同模型或精度 |
| 提交记录 | 实际接受边界；需要发布的状态句柄；待释放分支；迁移版本与完成事件 | 计算/传输未完成；只迁移了部分层；取消请求仍试图提交 |

Worker 完成计算后，先确认事件与有效输出，再更新提交边界。取消、超时或草稿拒绝都从已提交边界恢复。控制器负责副本和设备放置；每轮请求选择留在引擎内，避免一次 token 生成依赖远端控制器往返。

### 5.2 容量预算必须包含状态槽和临时分支

令每个请求的本地单槽递归状态为 `S_local`，需要常驻或缓存的槽数为 `n_slot`，验证暂存为 `V_local`，分页 KV 和草稿 KV 为 `K_local`。单请求容量至少要记为：

    M_request = n_slot × S_local + V_local + K_local + convolution_state

另加本轮视觉特征、MoE 路由/通信缓冲、图捕获工作区和资源余量。`S_local` 取本地实际分片布局；不能把第 1 项的未分片 414 MiB 直接当成每张卡的分配量。也不能假定每个请求只有一个 KDA 槽。SGLang 当前 K3 指南区分 Radix 缓存策略的状态槽和投机验证中间状态；布局、策略、精度或分片变化后必须重新读取分配器容量。[SGLang 状态池与比例说明](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)。

每轮的预算至少分别检查输入处理 token、生成请求、递归状态容量、KV 容量和视觉/通信临时空间。预估输出长度只影响计划，不能替代硬容量上限。为长输入加入等待时间提升优先级，并限制单轮输入处理占用，避免短请求持续挤走长请求或长输入拖慢所有生成请求。

### 5.3 开发工作包按依赖顺序推进

**R1 锁定一个可重复执行的后端基线。** 固定权重、引擎、PyTorch、内核、驱动和固件版本，盘点 KDA、MLA、LatentMoE、视觉与投机路径。交付兼容清单，每项标注已执行、回退或缺失；用相同请求记录数值、容量和时间。上游文档列出模型支持不能替代这份清单。另一套推理引擎等首个适配器验收后再接，避免同时调试两套生命周期。

**R2 接入执行计划与状态提交。** 在模型 runner 和状态分配器之间接入上述契约。交付可重放的生命周期测试：连续生成、每个草稿接受位置、全部拒绝、计算中取消、内存不足、迁移失败和重连。接收端校验完整状态后才能切换所有权；旧 worker 在交接确认前保留已提交状态。验收比较接续输出与状态，并检查句柄泄漏和跨请求污染。第 1 项负责状态计算与恢复正确性，本包负责引擎何时调用它们。

**R3 接入实际资源成本，再改调度策略。** 先给上游默认策略补齐状态槽、KV 页、MoE token 分布与传输字节的观测，再用目标设备测量更新成本估计。交付一条可重放的请求 trace 和一份逐轮预算账本；使用同一权重与内核比较默认策略和新策略。资源估计若长期低估，就先修预算；只提高平均输出 token/s 而使尾延迟或等待上限恶化的策略不通过。

**R4 接入 TorchTitan 已有 K3 模型并验证训练。** 固定 `fb45f5e878463f64a1a70ab80e773b7916de52f0` 的官方扩展文档采用 `BaseModel`、模型包中的 `MODEL_FLAVORS` / `build_model_config(flavor)` 和独立训练 recipe；不要把历史方案中的通用 ModelSpec 叫法当作此版本可调用的接口。优先盘点现有 `kimi_k3` 的模型、sharding 与 `state_dict_adapter`，用 `debugmodel` 核验前向、梯度和权重映射，再建立可信 1D FSDP 基线。每增加一种目标并行组合，保持每个更新步的总 token 与数据条件可比，比较梯度、loss 和恢复后的继续训练。[固定扩展接口](https://github.com/pytorch/torchtitan/blob/fb45f5e878463f64a1a70ab80e773b7916de52f0/docs/extension.md)、[K3 现有实现](https://github.com/pytorch/torchtitan/tree/fb45f5e878463f64a1a70ab80e773b7916de52f0/torchtitan/models/kimi_k3)、[官方收敛验证方法](https://github.com/pytorch/torchtitan/blob/fb45f5e878463f64a1a70ab80e773b7916de52f0/docs/converging.md)。

K3 的流水线边界还要携带 Attention Residuals 的块残差栈，只有 hidden state 不够。该固定版本的 `pipeline_parallel/layout.py`、`stage.py` 与 `cache.py` 管理块的传输、缓存和梯度归并；启用缓存时要求循环虚拟 stage 放置，不能随意换成 V 形调度。KDA 的 Context Parallel 路径依赖具体内核：该版 TorchTitan 使用 Attention Gym 的原生 CP，不能将 FLA 预条件算子的 CP 限制泛化为所有 KDA 后端。交付 PP 的块传输/梯度对照和所选 CP 的数值测试；更换并行网格恢复时保存模型、优化器、RNG、数据游标、精度配方与 mesh 元数据。[固定 K3 前提与并行支持](https://github.com/pytorch/torchtitan/blob/fb45f5e878463f64a1a70ab80e773b7916de52f0/torchtitan/models/kimi_k3/README.md)、[AttnRes 流水线缓存契约](https://github.com/pytorch/torchtitan/blob/fb45f5e878463f64a1a70ab80e773b7916de52f0/torchtitan/models/kimi_k3/pipeline_parallel/PP_ATTN_RES_CACHE.md)。

**R5 集成验收与失败回退。** 固定模型、设备数、质量目标和请求 trace，混合短文本、长输入、图像、投机分支及专家热点。交付完整的准确性与服务账本：首次响应、逐 token 延迟、P99、等待上限、有效任务吞吐、各类状态峰值、通信耗时和错误率。回退只能发生在明确的提交边界，不能带着已污染状态切换内核或重复提交 token。训练的验收另列目标质量、更新步时间、checkpoint 时间和恢复后的 loss，避免用线上指标替代训练质量。

### 5.4 本项的完成与停止条件

完成意味着：一个目标后端能依据实际状态和资源预算执行 K3，请求可以取消、回滚和恢复；一条明确的训练配置能从可信基线扩展到所需并行并恢复继续训练。源码检查、CPU 控制流模拟和短程小模型测试分别记账，不将它们标成完整 K3 性能或训练收敛验证。

若目标设备缺少关键反向、状态布局不兼容、迁移不能原子完成，或新调度无法同时满足质量与尾延迟约束，就保留已经验收的组合，缩小支持矩阵。不要通过省略状态、静默改精度或忽略失败请求制造吞吐收益。

进一步阅读：[原 K3 框架专题](../../assets/kimi/K3-SGLang-vLLM-TorchTitan与定制runtime方案.md)；背景报告 [22 章](report.md#s22)、[23 章](report.md#s23)、[24 章](report.md#s24)、[42 章](report.md#s42)。

[返回五项重点目录](#focus-toc) · [查阅完整 42 章背景报告](report.md)
