# Kimi API、硬件、量化与缓存独立审计

本审计范围内已确认的 AH-01 至 AH-09 均已修复，修正内容已进入旧逐层 CSV/JSON/ZIP、新模型 CSV/JSON/XLSX 和硬件报告。最终复验为 **0 个未解的本范围已知实质性错误**。这个结论是源码与导出内容的一致性结论，不是整模型加载、设备支持或性能验收。

审计对象是 `local-artifact/2026-09-19/model-research-atlas/outputs/model-research-atlas`。最终冻结提交为 **`083af0a41b1134b7424fe514c30120597df10a8f`**，2026-09-26 01:54:51 UTC 复验确认 HEAD 匹配、工作区干净，全部检查零差异。71 个审计输入文件的准确哈希与仓库状态保存在 `frozen-artifacts-audit.json`。

相较首个候选 `40346969…`，Kimi 源 compute/hardware、旧导出、新 CSV/XLSX 内容均未变化。八份新 JSON 只增加 `isRetrieval=false`、`roleSource=null`、`encodingAxis=L_text` 元数据；已逐字段比较，没有算子表/来源/配置变化。`dist/model-documents.js` 仅修订 encoder 分支的 S/L_KV 提示文字，未改变 Kimi 接口与数学关系。差异证据保存在 `final-baseline-diff.json`、`final-json-semantic-diff.json`。

| 关键输入 | SHA-256 |
|---|---|
| data/families/kimi/compute.json | `86fdf9709a1fd9d83f484ba16de62d1f4613fd132093bc78aeec678c2d81ec77` |
| data/families/kimi/hardware.json | `6dbaa0814ec4e451fca224125ca005a3c91a28710d72e7cb2e74c0e919801e94` |
| dist/downloads/model-documents/kimi-kimi-k3.csv | `109f116e42f90c030bc760aa3f9ea400bf7cf757402ef003ca98082d4bb77746` |
| dist/downloads/model-documents/kimi-kimi-k3.json | `e24a849bffcdf44a53dc0fb27acf59404fac66f116dd6c593eb8f8017fb20ee0` |
| dist/downloads/model-documents/kimi-kimi-k3.xlsx | `66e543a5e9adfa7d346af565864bfdfda7be956bde95972ea5364a94fd68a3c9` |

## 审计方法与全量覆盖

审计只写入独立审计目录，没有修改或发布被审资料。没有运行或导入制作方的 `review-compute-check`、validator 或构建代码。独立脚本分别做源码 AST/区间/哈希检查、平台字段逐条比对、CSV/JSON 展开检查、OOXML 原始单元格检查和 ZIP 内容检查。语义判断另行阅读固定版本的原始源码，不以验证报告里的“通过”代替核验。

| 覆盖项 | 结果 |
|---|---|
| compute 数据 | 8 个主模型、53 模板、550 模板步骤、2,200 平台条目、660 组件、16,978 展开步骤全部遍历 |
| 不同平台声明 | 184 种完整声明变体、64 个不同 chain 字符串全部编目；重复行按模板全量展开检查 |
| 源码证明 | 160 条函数/语句证据全部通过独立函数定位、区间、声称调用、源码与函数内容哈希检查 |
| 原始文件真实性 | API 45 个源码文件；结合硬件证据共 79 个固定 URL，79/79 远端内容 SHA-256 与本地证据匹配 |
| 滚动 primary 来源 | NVIDIA Hopper/Blackwell、AMD CDNA4、MI300X K2.5 和 Kimi MXFP4 厂商页逐页读取；仅用于其直接支持的有限主张 |
| 旧逐层导出 | API 16,978 行的全部 chain 对齐；101,868 个 level/condition 单元对齐；shape CSV 的 101,868 个标题/输入/输出/数学/缓存/精度单元对齐 |
| 旧分模型 ZIP | 16 个 CSV 共 33,956 行与对应总表完全一致 |
| 新统一导出 | 8 模型 CSV/JSON 共 16,978 行；四个平台各自的 chain、level、condition 全部完整保留；没有条件遗漏 |
| 新工作簿 | 8 个 Kimi XLSX 的“算子表”共 509,580 个单元格（含表头）与 CSV 逐格一致，无误作 Excel 公式的文本或错误单元格 |
| 硬件衍生物 | 来源 CSV 42 行、294 单元与数据对齐；中文报告中 167 条模块/平台/建议/协议/版本声明与数据对齐 |
| 页面与缓存 | compute/hardware/structure 展示逻辑与 11 份 architecture JSON 的 API、精度、缓存边界已检查；data/dist 主数据一致；完整 atlas 下载只合理省略自引用 downloads 字段 |

全部 23 类步骤均有覆盖：linear 179、rms 80、reshape 42、residual 42、mla 30、moe 27、activation 26、router 18、rope 14、elementwise 14、norm 12、dispatch 9、combine 9、embedding 8、gelu 8、attnres 7、conv 6、vision-attention 4、scatter 4、patch 4、position 4、kda 2、identity 1。以上是完整模板统计，并不是任意平台的“已支持算子”数量。

仍明确未知的模板平台条目有 NVIDIA 320/550、AMD 340/550、Ascend 197/550。其余条目还包括 NoPE“不适用”、参考函数和条件分派，不能全部计为真实设备内核支持。

## 已确认问题的最终状态

compute JSON 为单行文件；下列位置以字段路径定位。固定源码的完整文件路径、修订和链接在 `proof-file-integrity.json` 与 `remote-source-check.json` 中保留。

| ID | 原问题与独立证据 | 最终处理与复验 |
|---|---|---|
| AH-01 | K3 AMD MoE 指向不接受 SiTU 的 AITER W4A16 Triton 路径。vLLM `97dc6b19…/aiter_mxfp4_w4a16_moe.py:316–317` 只接受 SWIGLUOAI/SILU；:320–327 另有限制。K3 config 的 hidden_act 是 situ。 | k3-t2 的 s24/s25/s26/s28/s29、k3-t3 的 s23/s24/s25/s27/s28 已改未知。没有把另一条 AITER BF16 SiTU 分支混成原 W4A16 分支。源、旧 API CSV、新 CSV/JSON/XLSX 均修正。 |
| AH-02 | K3 被列成执行 RoPE。config:173 为 mla_use_nope=true；Ascend `kimi_k3.py:432,443` 断言 NoPE 并传 use_rope=False，:323–324 不旋转提前返回。NVIDIA `nvidia/mla.py:26–27,143` 区分 NoPE 目标与备用 RoPE。 | k3-t3.s11 明确身份路径/不旋转，各设备旋转接口不适用；24 个 MLA 展开行全部修正，硬件报告也注明例外。64 维共享键和 512+64 缓存未误删。 |
| AH-03 | SiTU 漏掉 up 的 linear_beta=25 饱和分支。官方 `modeling_kimi_linear.py:75–82,88–91` 与 config:20–21 支持双参数。 | 五个 K3 activation 模板已使用 `4*tanh(gate/4)*sigmoid(gate)*(25*tanh(up/25))`，旧新导出一致。 |
| AH-04 | 仅按 BF16 dtype 把不同投影统映到 ColumnParallelLinear。NVIDIA KDA/MLA 的 o_proj 实为 RowParallel（kda.py:718 / mla.py:295）；DeepSeek down 为 RowParallel、qa/kva 为 Replicated。 | 已确认的 K3 o_proj 写正确并行类；未追通路径退回未知；PyTorch标准Linear与Moonshot模块上下文明确分层。未将未知再外推成不支持。 |
| AH-05 | Ascend routed down GEMM 错写 gmm1。固定 w4a8.py:506–527 是 w1，:532–548 的 apply_gmm2 才取 w2。 | 9 个 routed down 模板已全部写 gmm2；转换 W4A8 条件仍保留。 |
| AH-06 | 视觉 norm/残差/MLP、池化、scatter 引用了不执行相应运算的函数；KDA o_norm 错用普通 KimiRMSNorm。官方 KDA :539–541 构造 FusedRMSNormGated，:656 调 o_norm(o,g)。 | 视觉步骤分别引用 encoder layer forward / MLP2.forward / tpool_patch_merger / embedding merge；KDA 两模板 s15 明确实际 fused 调用，设备映射未知，并说明相邻门控行是同一调用的逻辑展开。所有旧新导出同步。 |
| AH-07 | K2.5/2.6/2.7 projector 将 pre_norm 放在时间池化前，均值与规范化一般不可交换。真实 MoonViT.forward 先 tpool，然后 PatchMergerMLP.forward pre_norm。 | 池化 → LayerNorm → flatten 的顺序正确；四模型的独立 flatten 步骤完整加入，故总展开由16,974变为16,978。 |
| AH-08 | 新导出把 KDA 状态矩阵 S 替换成上下文长度 L_KV，69 行产生 `L_KV=L_KV̄+βkδᵀ`。 | 源与导出统一使用 H_state / H_state_bar / H_state_prev。69 KDA 行逐条正确；L_KV 仅表达上下文长度。 |
| AH-09 | 新表原来只携入 NVIDIA/AMD chain，丢掉设备条件，运行条件中仅有 Ascend 限定。 | 四个平台各自单元格完整带 chain + level + condition。8模型67,912个平台行组合逐项比较通过。AMD KDA 架构/head/dtype/speculation 限制、MLA decode/DCP/cache 限制、MoE dispatcher 限制保留。 |

新增视觉最终归一化也独立核实：K2.5/K2.6 的 `MoonViT3dEncoder.forward:602`、K2.7 的 :636、K3 的 :617 都在 blocks 循环结束后调用 `self.final_layernorm(hidden_states)`。K3 config 为 norm_type=rmsnorm，构造 :589–591 对应 `nn.RMSNorm`；其他三版是 `nn.LayerNorm`。当前四个模板的归一化类型与来源均正确。

## KDA、MLA 与量化核验

K3 gate_lower_bound=-5.0。官方 chunk/recurrent 调用都携带该下界与 A_log、dt_bias。固定 FLA `fused_recurrent.py:164–173` 的目标分支为 `g=-5*sigmoid(exp(A_log)*(raw_g+dt_bias))`；无下界才使用 `-exp(A_log)*softplus(raw_g+dt_bias)`。站内没有把后者误写成 K3 配置事实，但门变换说明仍较概括；本报告补充精确公式，不将这种细化缺口当成已有错误。

FLA recurrent 的 key 维衰减、以衰减后状态求 delta、乘 beta 的 rank-one update、q 读出顺序与 H_state 公式一致。transpose_state_layout/state_v_first 的物理 VK 布局和逻辑 KV 表达需区分，K=V=128 不能用数字相同抹去轴含义。vLLM `MambaStateShapeCalculator.kda_state_shape:303–326` 给出的卷积缓存长度是 `conv_kernel_size-1+num_spec`，已与卷积窗口4分开；每卡 local heads 随 TP 变化。

MLA 512+64=576 元素是优化实现的语义压缩缓存，BF16 假设下每 token 每层1152字节；官方 reference 的 `KimiDynamicCache.update:119–177` 保存展开 K/V，不能将1152字节当作所有实现的真实分配。K3 的24个MLA层仍有随上下文增长的缓存；69个KDA层的固定状态仅指每序列的递归矩阵，不包括工作区、卷积、前缀和推测状态。

K2 FP8、K2 Thinking/2.5/2.6/2.7 INT4 group32、K3 MXFP4 group32 的检查点声明与固定配置相符。I32/U8 是打包容器，不能当计算 dtype；MXFP4 U8 scale 与 U8 权重容器也不是同一种数值语义。kv_cache_scheme=null 没有被外推成真实服务 cache dtype。输入存储精度、设备计算/累加、输出精度与物理 format 未混作已实测事实。

## 硬件结论与剩余限制

全部7个硬件行、8个实现专题、7项优化建议、5条协议和5条版本说明已逐项核验。A2 的K2.5/K2.6转换W4A8案例未外推为完整K3；A3是特定转换检查点的DP4/TP16/EP64教程；A5仅有特定chunk算子设计，不是完整服务支持。AMD MI300X W4A16案例与MI355X原生MX格式能力/ATOM案例分开。NVIDIA Hopper与Blackwell配方也未当作相互可换的软件栈。没有发现把厂商案例或理论峰值写成本地实测，亦未作无依据跨厂商性能排名。

原始依据包括 [Hopper官方指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)、[Blackwell官方指南](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)、[AMD CDNA4](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)、[MI300X K2.5案例](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)、[AMD Kimi MXFP4案例](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-code-in-mxfp4-on-amd-gpus.html)。滚动网页不是固定代码快照；报告保留该区别。

- 未执行GPU/NPU加载、前向推理、数值精度、吞吐/延迟、功耗、容量或多卡运行实验。所有新表duration/计算利用率/mte/scalar均为未实测。
- API与硬件研究使用不同日期的固定提交，未证明它们组合成兼容软件栈。教程版本、Docker默认版本和verified.commit的不一致仍明确保留。
- 未穷尽所有设备内核的capability、形状、dtype、layout与调度分支；未证实项仍写未知。尤其 source wrapper 的存在不能证明当前模型实际经过它。
- K3 A_log检查点存储128而配置96 heads的差别已记录；loader narrow行为不能代替真实整模型加载验证。
- 本子审覆盖8个Kimi主模型API导出及11份架构的缓存边界，不声称独立复做全部156模型tensor参数审计。全156模型及K3模板对照由主审负责，OpenBMB旧站一致性由另一子审负责。
- 本轮XLSX复验针对保存后的“算子表”内容与错误类型，没有重新执行Excel公式引擎或作工作簿视觉排版验收；工作簿参数与模板对照表的全量内容由主审独立检查。

证据文件：`inventory-summary.json`、`proof-file-integrity.json`、`remote-source-check.json`、`hardware-source-integrity.json`、`document-export-audit.json`、`frozen-artifacts-audit.json`。`semantic-audit.md` 与带时间戳的 export-baseline 文件保留历史中间问题，不替代本最终报告。
