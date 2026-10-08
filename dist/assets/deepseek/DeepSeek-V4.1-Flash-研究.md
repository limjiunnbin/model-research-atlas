# DeepSeek-V4.1-Flash：CED、CSA2、Engram 与原生视觉

本研究核对官方技术报告、Hugging Face 配置和可读推理实现，固定模型 revision 为 `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`。本版不是 V4-Flash-0731 的同架构后训练更新：它重构了语言主干的缓存来源、跨层复用和残差数据流，并加入条件内存和视觉输入。本版未读取 safetensors 文件头或权重载荷，存储形状、真实载荷、设备兼容性和实测性能均保持未知。

## 1 配置与参数口径

| 项目 | V4.1 Flash | 依据与范围 |
| --- | --- | --- |
| 语言主干 | 40 层：20 层因果 encoder + 20 层 decoder | 技术报告 §2.1、§4.2.1；两段都保持因果计算 |
| 隐藏宽 / 词表 / 上下文 | 5120 / 129280 / 1048576 token | `config.json` 的 `text_config` |
| Attention | 64 Q heads，head dim=512，RoPE 64 维，Q rank=1280 | KV 为共享的 512 维表示，不是 V3 的 512+64 MLA cache |
| 输出投影 | 8 组，组内 rank=1024 | `wo_a` 为分组投影，不能当作普通完整 dense 矩阵收缩 |
| MoE | 每层 384 个路由专家，选 6；1 个共享专家；专家宽 2304 | 全部 40 层为 MoE；文本/视觉使用分别校正的路由 bias |
| 参数披露 | 约 552B 主干，另有约 196B Engram | 两者分列；不把主干、条件内存、视觉、DSpark、量化元数据或存储副本混为一个总数 |
| 激活参数 | prefill 约 8B；decode 约 16B | 论文优化的 CED/replay 部署口径；不是所有 reference forward 的直接计数或统一 FLOPs |
| mHC | 4 流，20 次 Sinkhorn 迭代，Single-Pass 数据依赖 | 每个 attention / FFN 子层的 mixing 控制投影仍为 `[24,20480]` |
| Engram | 0-based 第 1、14 层；2/3/4-gram，每阶 8 hash heads | 24 个地址，各取 256 维，拼接为 6144 维；通过 `[25600,6144]` 投影产生四流 key 和共享 value |
| 视觉 | ViT 32 层，hidden=1024，heads=16，patch=14 | 2D RoPE、SwiGLU、RMSNorm；两层 projector 为 9216→5120→5120 |
| DSpark | 3 个 stage，128 个 draft 专家选 3；block=5；Markov rank=256 | 与主干 384 选 6 分开；HF nextn=3 与 native n_mtp_layers=3 一致 |

精确的参考构造逻辑矩阵计数在本版结构 JSON 中；这些计数是 MP=1 参数声明的推导，不能替代实际 checkpoint 存储审计。特别是 Engram 的大表属于按地址访问的条件内存，不能按每 token 执行整个表的 GEMM 计算。DSpark 引用主干 embedding/head 对象，不在逻辑参数中重复计数；这不证明发布文件如何处理存储副本或训练别名。

来源：[固定配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/config.json)、[native 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/config.json)、[技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/DeepSeek_V41_Tech_Report.pdf)。

## 2 CED 与 SWA Bounded Replay

语言 encoder 处理输入序列，decoder 的全局 KV 直接投影自最后一层 encoder hidden states。因此，处理长 prompt 时不必对所有输入 token 执行完整的后半段 decoder。decoder 各层的局部 SWA KV 仍来源于各自 hidden states，不能一并替换为 encoder cache。

论文的 decoder bounded replay 只将 prompt 最后 `n_win=128` 个 token 的 encoder 输出送入后 20 层，恢复供解码使用的 SWA 窗口。忽略投影等附加成本的主干规模为 `N×20 + min(N,128)×20`，并非任意长度都恰好减半；当 `N≫128` 才接近半个主干的 prefill 计算。短输入、cache hit/miss、视觉输入和输出 logits 范围也会改变实际开销。

encoder bounded replay 处理另一种情况：命中长期 global KV、但活跃会话的 SWA KV 已丢失。它回放命中前缀的最后 128 token，与新 suffix 一起处理；回放部分复用既有 global KV，不能覆盖它，新增 suffix 才生成新的全局与局部缓存。

两种 bounded replay 都截断了跨层累积的 SWA 依赖，属于**近似恢复**，不是完整前向的逐值等价变换。论文报告响应质量影响很小，并在后训练中模拟 decoder replay；本站没有复现该质量结论。

来源：技术报告 §2.2、§3.2.1–3.2.2，尤其第 9、19–20 页。生产部署中的 EPD 是 **视觉 Encoder / Prefill / Decode** 分离，与 CED 中的语言 encoder/decoder 是不同的划分。

## 3 CSA2 的四份缓存与八个 indexer

CSA2 把“持有全局 KV”与“重算稀疏索引”分开。所有模式都重算本层 Q 和本层 SWA KV，不能把 Reuse 理解为跳过整层 attention。

| 模式 | 本层全局 KV / index K | 本层 index Q / Top-K | 当前层仍执行 |
| --- | --- | --- | --- |
| Full | 生成并持有 | 重算 | Q、SWA、选定 global KV 的 attention、输出投影 |
| Reindex | 复用最近 Full | 重算 | Q、SWA、重新选择后的 attention、输出投影 |
| Reuse | 复用最近 Full | 复用最近 Full/Reindex 的 Top-K | Q、SWA、attention、输出投影 |

固定配置的主干分布如下，表中层号均为源码的 **0-based** 索引，网站语言层编号为这些索引加一。

| 子系统 | 层型与模式 | 全局 KV owner | Indexer owner |
| --- | --- | --- | --- |
| encoder 0–19 | 前 2 层 SWA；其余 18 层 CSA2，ratio=2 | 2、8、14：Full；每组余下 5 层 Reuse | 2、8、14 |
| decoder 20–39 | 20 层 CSA2，ratio=1 | 20：Full；其余层复用这份 KV | 20：Full；24、28、32、36：Reindex |
| 总计 | 2 SWA + 4 Full + 4 Reindex + 30 Reuse | 4 份全局 main KV 和对应 index K | 8 个 indexer；30 层没有新的 index Q/score |

与 V4 CSA 相比，CSA2 不再用 `2m` overlap 或压缩位置 APE；ratio=2 用两个 token 的 pooling，ratio=1 是直接投影。index K 由主 KV 的 RoPE 前 latent 投影，替代独立的 hidden-state 压缩路径。只有 Full 层拥有 compressor 和 index K cache。

decoder 第一个 Full 层仍遍历全部因果可见的 main KV。它按每块 8 个位置取最大 index score，最多选择 2048 个块，形成最多 16384 个候选位置；后续 Reindex 在各自 query 对应的候选池内重新选 top-512。16384 是候选上限，512 是实际 attention 选择长度，两者不是一个参数。首次全范围扫描仍随上下文增长。

来源：技术报告 §2.3、图 4–5；固定 `Attention.__init__`、`Indexer.__init__/forward`、`select_candidate_blocks`、`SharedAttentionRuntime`。这些函数的行号和 SHA-256 附在结构 JSON 的 `referenceProofs` 中。

## 4 890 bytes/token 的复算与存储边界

论文部署的 global main KV 使用 **NVFP4 风格的变体**：E2M1 FP4 数据，每 16 个通道一个 E4M3 scale，并省略 NVFP4 的第二级 global scale。因此不能把它写成未经修改的完整 NVFP4。主 KV 在 RoPE 后量化。**SWA 使用 MXFP8**：固定参考代码的 `fp8_block_size=32`、`scale_fmt="ue8m0"`、`scale_dtype=float8_e8m0fnu` 传入 `act_quant`，其数据为 E4M3 FP8，每 32 个通道一个 E8M0 scale。index K 使用 MXFP4：E2M1 数据、每 32 通道一个 E8M0 scale；不要与 main KV 的 E4M3 scale 混用。

| 单条记录 | 数据 | Scale | 合计 |
| --- | --- | --- | --- |
| main KV，512 维 | 512/2=256 B | 512/16=32 B | 288 B |
| index K，128 维 | 128/2=64 B | 128/32=4 B | 68 B |
| 每个 Full owner 的记录 |  |  | 356 B |

对长度 `N`，三份 encoder 缓存各有 `floor(N/2)` 条，一份 decoder 缓存有 `N` 条。因此打包的全局缓存为：

```text
global_KV_bytes(N) = (3 × floor(N/2) + N) × (288 + 68)
N 为偶数时：global_KV_bytes(N) = 890 × N
N = 1048576 时：933232640 B = 890 MiB
```

这个数不含 40 层 SWA 窗口、DSpark context 窗口、压缩中间状态、query/score、Top-K、candidate masks、分页/对齐、batch、TP/CP 布局或 metadata。前三个 ratio=2 compressor 还分别持有两个 `[B,2,512]` FP32 状态，合计 `24576×B` 字节；它们与按长度增长的全局缓存分开。

官方给出的 HBM 约为 V4 Flash 的 1/4、SSD persistent cache 约为 1/8 是两个部署范围。后者还依赖长期 SSD 不存 SWA、转为短 TTL 的分布式主机 DRAM 窗口池及缺失时的 bounded replay，不是单靠 FP4 再省一半。长期 global KV、活跃 SWA、DSpark 与临时张量必须分别管理。

**部署容量采用真实 FP4 / FP8 打包口径，包含 scale。** 40 个主干 SWA 窗口的容量为 `40×128×(512+512/32)=2703360 B`，三个 DSpark context 窗口另为 `3×128×528=202752 B`。每请求缓存 1048576 token 时，全局缓存为 890 MiB；加主干窗口和 24576 B 压缩状态为 935960576 B，启用 DSpark context 后为 936163328 B。这是列示缓存与状态的打包容量，不包含权重、临时 workspace、分页/对齐或通信。

可读模型当前调用量化/反量化的 inplace 模拟路径；量化函数也提供真实 FP4/FP8 数据和 scale 的输出路径。参考 buffer 的默认 dtype 是实现差异，**不用于本研究的部署容量计算**。网站提供独立的 V4.1 部署缓存页面，可以调整请求数、缓存长度与 DSpark context 开关。

来源：技术报告 §2.4.4、§3.2；固定 `Attention._compress_kv` 的 16/E4M3 分组、`Attention._window_kv` 的 32/E8M0 MXFP8 分组、`Indexer.forward` 的 32/E8M0 MXFP4 分组，以及 [kernel.py 的 act_quant / fp4_act_quant](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/kernel.py)。

## 5 Single-Pass mHC 与 Engram

Single-Pass mHC 改变数据依赖：当前子层使用上一子层产生的 input-mixing 系数，而当前预测出的系数给下一子层使用。参考 `Block.forward` 中，attention 消费传入的 `pre_mix`，FFN 消费 attention 的 `attn_pre`，并返回 `ffn_pre` 给下一层。初始 mixing 为 one-hot，最后再折叠四流。

论文 Mega-mHC 融合 residual update、input mixing、系数预测、输入 norm 与 FP8 conversion。在 `n=4` 时，论文的激活访存从原多次实现的 `(4n+4)d=20d` 降至理想单次映射的 `(2n+2)d=10d`。这是特定融合实现的流量口径，不是“整模型快两倍”，也不是可读 reference 的实测结果。

Engram 用 token-based deterministic addressing 把记忆容量和主干计算分开。每个模块有 24 个 n-gram/hash-head 地址，查表后拼接 6144 维，再投影成四流 key 与一个共享 value；归一化点积、signed square root 和 sigmoid 决定各 residual stream 的写入门控。视觉位置既不参与文本 n-gram，也不接收 Engram contribution。V4.1 移除了旧 Engram 的短因果卷积。

两张大表分别为 `[384006168,256]` 与 `[384016682,256]`，合计 196613849600 个表参数；加入本地 KV 投影和门控向量后，参考 Engram 模块为 196928504320 个逻辑参数。官方的“196B”是近似披露，不应当用来强行改写配置推导值。

生产路径可把 deterministic lookup 预取到 host memory，通过后台 RDMA 与计算重叠。固定 reference 的 `ParallelEngramEmbedding.forward` 仍是本地分片 lookup、按块解量化和 all-reduce，没有实现这套后台预取服务。因此其参考显存、通信和延迟不能代表生产的 Engram 放置策略。

来源：技术报告 §2.4.1–2.4.2、§3.1.3；固定 `Block.forward`、`Engram.__init__/forward`、`ParallelEngramEmbedding.forward` 与 `EngramLayout`。

## 6 原生视觉与 DSpark

ViT 使用线性 patch embedding、32 层双向 attention、2D RoPE、RMSNorm 和 SwiGLU。14×14 patch 投影到 1024 维；3×3 pixel-unshuffle 把邻域拼成 9216 维后，经 9216→5120→5120 的 GELU projector 进入语言主干。实际 token 数按图像 patch 网格和 padding 计算，不能把语言上下文 `L_KV` 当作图像 attention 长度 `L_img`。图像 span delimiters 使用独立 learned embeddings，语言 MoE 对文本与视觉 token 使用不同的路由校正 bias。

DSpark 三个 draft stage 的专用 MoE 为 128 选 3。它读取主干 0-based 37/38/39 层 **attention 输入的 residual 流均值**，拼接 15360 维后投影到 5120；不是读取这三层的输出，也不是拼接三份全局 KV。prefill 只写入草稿 context SWA；decode 的三个块并行生成五个位置的 base logits，再用 rank=256 的 Markov head 按已采样前 token 顺序修正，confidence head 预测每位置条件接受概率。

论文 confidence scheduler 利用接受概率的前缀乘积和实测 engine throughput curves 选择各请求 verification 长度。草稿候选仍需 target verification、accept/reject、截断和缓存提交；不能用 confidence 本身代替 target 接受判据。固定 `generate.py` 是普通自回归采样，`forward_spec` 是草稿接口，并没有完整的 rejection sampling 或服务 scheduler。

来源：技术报告 §2.1.1、§2.4.3；固定 `vision.py`、`Transformer.merge_image_embeddings/forward`、`DSparkBlock`、`DSparkAttention`、`DSparkMarkovHead` 和 `DSparkConfidenceHead`。

## 7 训练、能力披露与 API

论文主干从头进行多模态预训练，语料 45T token，文本与多模态 token 比为 7:1；稀疏 attention 从 64K 序列训练，34T token 时扩展到 1M。视觉编码器有独立的对比预训练与自回归适配阶段，然后参与联合预训练。主干预训练不带传统 MTP；DSpark 在独立阶段冻结主干训练，后训练随策略更新但其目标梯度不传回主干。

优化器包括按 head 分组的 Muon、RMSNorm/非矩阵参数的 AdamW、embedding/head 的 momentum + Sinkhorn balancing；Engram 学习率倍率为 5。后训练沿用 SFT→RL→OPD，主要扩展 agent 任务、环境和 rollout 数据，不把数据工程升级写成新的 RL 算法。

官方模型卡的代表结果：GPQA Diamond 90.9、Terminal-Bench 2.1 90.6、DeepSWE v1.1 74.2、HLE with tools 63.9。它们来自官方评测；thinking effort=100、temperature=1.0、top_p=0.95、不同 harness 和上下文条件应保留。DeepSWE 的 mini-SWE 与 Terminal-Bench 的 DSH Minimal 不是同一 harness；跨 scaffold 表为 DeepSWE 每题 N=8、Terminal-Bench 每题 N=3，不能把所有分数当成统一一次测试的结果。本站未复现这些分数或计算速度排名。

官方 API 在 2026-09-10 发布本版，以 `deepseek-flash` 调用；兼容别名 `deepseek-v4-flash`、`deepseek-v4-flash-vision-exp` 暂时路由到 V4.1 Flash。须以当前更新日志为准：日志已说明 9 月 14 日之后继续提供 V4-Pro 服务，不沿用发布新闻中较早的“全面转路由”计划。

对来源差异保持原样：固定模型卡的 NL2Repo-Bench 为 64.0，API 更新日志列 65.4；没有足够说明来把两者归为同一评测条件。模型卡说发布时没有 Jinja template，但本固定 revision 的目录已经有 `chat_template.jinja`；不能继续把发布文案当作当前文件目录。prompt 语义以 `encoding` reference / maintained `deepseek-recipe` 为准，框架自动生成的 HF 使用示例不等于本仓已核验的整栈支持。

来源：[固定模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/README.md)、[官方 API 更新日志](https://api-docs.deepseek.com/zh-cn/updates/)、[encoding reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/encoding/encoding.py)。

## 8 与 V4 Flash 的结构比较及 Ascend 静态分析

| 维度 | V4 Flash（原始所选配置） | V4.1 Flash |
| --- | --- | --- |
| 语言层 / hidden | 43 / 4096 | 40 / 5120，CED 20+20 |
| 主干参数披露 | 约 284B | 约 552B，另有约 196B Engram |
| MoE | 256 选 6，shared=1，专家宽2048 | 384 选 6，shared=1，专家宽2304 |
| Attention | 2 SWA + 21 CSA ratio4 + 20 HCA ratio128 | 2 SWA + 38 CSA2；四个 Full、四个 Reindex、三十个 Reuse |
| 稀疏索引 | CSA 自有 indexer / 独立压缩路径 | Full 共享 index K；Reindex 更新 Top-K；decoder 候选池 |
| 残差 | mHC 当前子层系数依赖 | Single-Pass mHC 前移 input mixing 依赖 |
| 主 KV / 索引格式 | 各层压缩缓存；旧报告另列量化模拟与部署 | 四份共享全局 cache；main 为省略 global scale 的 NVFP4 变体，index MXFP4/32，SWA MXFP8/32 |
| 多模态与记忆 | 所选文本 checkpoint 不包含本版视觉/Engram | 32 层 ViT + projector；两处 Engram |
| 草稿 | 普通 MTP 与独立 DSpark checkpoint 分开 | 本版含 3-stage DSpark，128 选 3，block5 |

Ascend 分析仍停留在可核对的结构和数据契约层。现有 CANN 开源/SDK 的 Gather、MatMul、量化、RoPE、RMSNorm、MoE 路由与 attention 等材料，可作为基础算子的候选映射；它们没有证明 V4.1 的 CSA2 跨层共享、hierarchical gather/score、Single-Pass Mega-mHC、Engram 预取、视觉 EPD 或 DSpark scheduler 已在某个 NPU 栈形成可运行组合。特定精度、scale 格式、布局、owner/consumer cache 生命周期与并行约束须逐项匹配。

实现研究页面为本版提供五个独立专题，旧 V4/MLA 结论不自动套用。旧 21-model 逐层计算包和 FLOPs/通信估算器保持其已核验范围；本版给出 phase-aware 规模解释、按真实打包格式计算的交互部署缓存容量与独立逻辑矩阵 CSV。参考模拟只作为实现差异备注，部署容量以实际 FP4/FP8 数据与 scale 为依据。没有硬件，设备实测仍不属于本次任务。

## 9 复现

```sh
python3 scripts/build_deepseek_v41.py --fetch
python3 scripts/validate_deepseek_v41.py --sources
```

生成器只获取固定的小型配置、参考代码与论文，存入用户级缓存；不会下载权重。仓库仅保留本版当前配置、带源码位置的逻辑结构和研究正文；不生成按运行累积的历史台账或验证日志。普通 `scripts/build.py` 校验本版结构后将当前 canonical JSON 复制到网站。跨家族比较 XLSX/CSV 由既有分析导出脚本重建，报告包用 `build_deepseek_v41.py --package` 原位更新。
