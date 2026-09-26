# 旧网站与历史研究资产独立只读审计

审计日期：2026-09-26。状态：已完成本子审范围的最终只读复验。本报告核对的本地提交为 `6613c6f584ab41eec0eed8c4c3981032595b66e5`，工作区 clean；未发布。该提交相对先前冻结 `40346969effacb530ea9d25a37b01b54dc85c1ac` 的变更均属新模板/新文档页范围，旧站问题修正文件保持相同。中间提交 `083af0a41b1134b7424fe514c30120597df10a8f` 至此提交仅改新模板总ZIP、其验证输出与验证脚本，未改本子审旧资产。

本报告只说明本子审实际完成的检查。没有把项目原有 `VALIDATION.md`、`compute-validation.json`、`validation.json` 或作者的通过结论当作独立正确性证据。被审项目未被本子审修改、构建或发布。自编脚本与审计输出仅写入本报告所在目录。

## 结论

初审确认的四项 P2 问题 L1–L4 在当前网站与公开对应下载包中均已关闭：Audio GQA 已说明保留历史 K/V；Audio 三维图明确仅含 28 层主干、另 6 层 MIMO 未展开；Linear 上下文恢复官方配置值 1,048,576；MiniCPM3 的 62 层已分别列出 QK=96、V=64，并给 V 维度附固定源码来源。在本子审负责范围内，最终重新检查为 0 项未关闭的确定错误。原始发现和证据仍保留在下文，避免抹去审计过程。

OpenBMB 88 个仓库附录已逐项完成具体叙述语义审查，包含类别、用途、输入输出、模块流程、与普通 LLM 的区别、依赖、场景/局限共 616 段；当前 Markdown 和 HTML 与被审索引中的这 616 段全部一致。87 项有相应官方原文/配置/源码支持；HerculesBench 1 项仍缺正式任务范围和发布意图证据，原附录已经公开说明这一缺口，没有虚构模型能力。没有把目录、哈希或作者验证结论代替语义审查。逐项证据与判定见 [REPOSITORY-SEMANTIC-REVIEW.md](REPOSITORY-SEMANTIC-REVIEW.md) 和 `repo-semantic-ledger.json`。

站外 `local-artifact/2026-09-23/openbmb-research/outputs/` 是保留的历史原件，其旧 `compute/OpenBMB-layer-config.csv` 与旧 ZIP 仍含未区分 QK/V 的 `head_dim=64`。本次没有倒写历史原件；“修正通过”指当前网站公开版本及其再生下载包，不表示站外历史档案已重写。

七个审计检查点全部 473 个原始 safetensors 文件头已按官方固定 revision 独立联网重取，473/473 与本地原件逐对象完全一致，0失败、0不匹配。继而独立复算全部 1,610,434 个张量，逻辑参数、文件载荷、张量数量、逐模板数量、index 键集合、revision，以及全部596个分组/层的参数、载荷、attention参数与激活线性参数代理都与旧站记录一致。没有下载权重数据区或运行模型。

OpenBMB 17 模型、31 配置组件、751 配置层、6108 计算步骤、6108 后端映射行、51 个分模型 CSV ZIP 成员的全量结构和数值引用一致性检查通过。3826 个声明了明确线性矩阵尺寸的步骤通过输入/权重/输出维度契约检查。这个结果不证明完整计算图、实际权重、实际后端调用或性能正确。

## 初审已确认问题、精确位置与关闭状态

所有项目相对路径均相对于：`local-artifact/2026-09-19/model-research-atlas/outputs/model-research-atlas`。

### L1 · P2 · Audio GQA 被写成不分配语言 KV 缓存

最终状态：**已关闭**。`dist/structure.js:46` 增加 GQA 分支；28 个 Audio 主干节点均落入“GQA 保存历史 K/V”的说明，并保留布局/dtype/TP 依赖运行实现的限制。以下为修正前发现。

- 位置：`dist/structure.js:46`；选择 Kimi-Audio-7B-Instruct 的任何语言层都会触发。
- 条件：`data/families/kimi/audio-7b-instruct-architecture.json:1` 的全部 28 个节点均为 `type=GQA`。检查器缓存说明仅分辨 KDA、vision、MLA，其余全部落到“非注意力组件不分配独立的语言 KV 缓存。”
- 原始证据：同版配置第 31–41 行为 28 层、4 KV heads、`use_cache=true`。官方 `MoonshotAttention.forward` 在第 317–322 行拼接并返回 K/V 缓存。
- 固定来源：[Audio 配置](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b/config.json)、[Audio 注意力实现](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b/modeling_moonshot_kimia.py#L317)。本次官方重定向解析到这个 revision，下载原文及哈希保存在 `primary/` 与 `primary-receipts.json`。
- 影响：把有 KV 缓存的 GQA 语言层误认为非注意力组件，误导内存与推理结构理解。
- 建议：为 GQA 提供独立说明，使用 4 KV heads、head width 128 的配置语义；明确布局和运行 dtype 未实测，其他未识别注意力类型使用“未核验”而非否定判断。

### L2 · P2 · Audio 三维图没有注明另六层 MIMO 未展开

最终状态：**已关闭**。source/dist 两份 Audio 架构 `/scope` 明确“仅展示28层文本主干，不包含额外6层MIMO、音频编码器或完整音频生成链”；保留 28 层节点并指向完整配置层下载。以下为修正前发现。

- 位置：`data/families/kimi/audio-7b-instruct-architecture.json:1` 的 `/nodes` 与 `/scope`，以及镜像 `dist/data/families/kimi/audio-7b-instruct-architecture.json:1`。
- 修正前内容：只有 28 个主干 decoder 节点；scope 只有“仅配置级结构；未审计的矩阵与参数保持为空。”，未告知另一个已知的六层语言分支被省略。
- 原始证据：`data/families/kimi/configs/Kimi-Audio-7B-Instruct-config.json:24` 为 `kimia_mimo_layers=6`；官方实现第 555–562 行分别构造主干 `layers` 和 `mimo_layers`，第 765 行起执行后者。
- 固定来源：[Audio MIMO 构造](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct/blob/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b/modeling_moonshot_kimia.py#L555)。历史报告 `dist/assets/kimi/Kimi-K2至K3深度研究报告.md:375` 已明确写出另有六层 MIMO，因此当前图与报告覆盖范围不同。
- 影响：读者会把 28 层示意误认作配置已知的完整 Audio 结构。
- 建议：至少明确图仅展开 28 层主干，额外六层 MIMO、音频适配和外部音频部件未展开；如增加节点，应给 MIMO 独立组件范围，不能把 `num_hidden_layers=28` 擅自改为 34。
- 本轮 156 模板由主审另行检查，主审已确认新模板的 Audio 两版本均含 root 28 + audio_mimo 6；本问题限定于旧站三维资产。

### L3 · P2 · Linear 的百万上下文已存在于配置，页面却显示未知

最终状态：**已关闭**。`family.json:2107` 的 context 与架构 config/context 均为 1048576，证据为 official，使用固定 revision URL，并注明只是配置上限。以下为修正前发现。

- 位置：`data/families/kimi/family.json:2107`，模型 `linear-48b-a3b-instruct` 的 `/facts/context/value=null`、`evidence=unknown`；`linear-48b-a3b-instruct-architecture.json:1` 的 `/config/context=null`；镜像 dist 同值。
- 原始证据：`data/families/kimi/configs/Kimi-Linear-48B-A3B-Instruct-config.json:57` 已保存 `model_max_length=1048576`。独立联网复取官方配置与本地文件 SHA256 一致。
- 固定来源：[Linear 配置](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json#L57)。
- 交叉矛盾：旧报告 `dist/assets/kimi/Kimi-K2至K3深度研究报告.md:373` 与图 01 已写 1M 上下文配置，而版本库/比较表显示未核验。
- 建议：按此模型的 `model_max_length` 填入官方配置值 1048576，并保留“配置上限不等于质量或运行实测上限”的口径。生成逻辑不能只取 `max_position_embeddings`。

### L4 · P2 · MiniCPM3 的统一 head_dim 列没有区分 QK 与 V

最终状态：**已关闭**。CSV 第 82 行起全部 62 层为 `head_dim=QK=96; V=64`、`qk_head_dim=96`、`v_head_dim=64`，V 来源固定至 `modeling_minicpm.py#L354`；751 层全量 QK/V 字段对照通过。分模型 ZIP 51 个表及总研究报告 ZIP 同步。以下为修正前发现。

- 位置：`scripts/openbmb_exports.py:10` 无条件用 hidden/heads 补 d，随后第 15 行识别 MLA；`dist/reports/openbmb/compute/OpenBMB-layer-config.csv:82` 开始的全部 62 个 MiniCPM3-4B 层均写 `head_dim=64`、依据 hidden_size/heads 推导。ZIP 中对应 `MiniCPM3-4B/layers.csv` 同值。
- 同包矛盾：`OpenBMB-layer-compute-shapes.csv:728` 已正确写 `Q:[B,40,T,96]`；配置是 `qk_nope_head_dim=64`、`qk_rope_head_dim=32`。
- 固定源码确认：官方 `modeling_minicpm.py:352–356` 分别设 `v_head_dim=hidden_size/num_heads=64` 与 `q_head_dim=qk_nope_head_dim+qk_rope_head_dim=96`。
- 固定来源：[MiniCPM3 配置](https://huggingface.co/openbmb/MiniCPM3-4B/blob/d6b14ddaefdb11c624dd75c3c779549bc90b08cb/config.json)、[MiniCPM3 实现](https://huggingface.co/openbmb/MiniCPM3-4B/blob/d6b14ddaefdb11c624dd75c3c779549bc90b08cb/modeling_minicpm.py#L352)。本次独立下载实现 SHA256 为 `a964ac7a8a2fc5a6192b9abbf4b78c548dd1c6097c83f60d57705b5f8da44846`。
- 这是字段语义问题，64 本身是正确的 V 维度，不应简单判为所有 head 都应改成 96。
- 建议：拆分 QK head dim、V head dim、RoPE/NoPE 与压缩 rank；若保留统一 `head_dim`，对 MLA 写不适用并指向分别命名的字段。

## 全量清单与检查范围

`inventory.json` 为逐文件清单，含绝对路径、范围、大小和 SHA256；`inventory-summary.json` 为分类数量。最终清单纳入 1586 个文件，排除 `.git`、环境/依赖、缓存、此次 156 模板的 `template-models` 与 `model-documents`。

|范围|纳入文件|检查层次|
|---|---:|---|
|当前旧站项目|201|文件清单、JSON source/dist 镜像、下载大小/哈希、架构与报表数据核对；含站点代码与全部旧资产|
|Atlas 上两级 work|185|源码/研究快照清单；Kimi compute/API/hardware 的语义复核由接口子审负责|
|Kimi 历史主输出|28|全部站内同路径副本字节对照、报告/图表数据口径|
|Kimi 历史第二输出|25|与主输出全部同名文件对照，无字节差异|
|Kimi 原始研究资料|543|清单；七个检查点全部473份头联网重取相等，全部张量和596个分组/层独立复算|
|OpenBMB 历史输出|20|与站内对应文件、报告包及逐模型表对照|
|OpenBMB 原始研究资料|584|清单、88仓库逐项语义、616段镜像、238份来源全量远端同文；4份追加固定源码核验|

旧站三维资产是 `structure.js` 从 JSON 节点实时生成的示意几何，没有独立 GLB/GLTF/OBJ 模型文件。全部 11 个 `*-architecture.json` 纳入结构检查；包括层号连续性、语言/视觉层数、节点参数及模块参数求和，以及与七份 `*-layers.json` 总参数/载荷一致性。Audio/VL/Linear 的参数留空是明确的配置级限制，不把空值补成零。K2-Base 借用同尺寸 Instruct 模板的推导性质已注明，未误算为 Base 头部审计。

### Kimi 七个检查点的独立复算

|检查点|全部分片头|全部张量|逻辑参数|载荷 bytes|结果|
|---|---:|---:|---:|---:|---|
|K2-Instruct|61|139644|1026408232448|1029173256720|通过|
|K2-Instruct-0905|62|139644|1026408232448|1029173265504|通过|
|K2-Thinking|62|208276|1026408232448|594205920512|通过|
|K2.5|64|208550|1026879376368|595148192736|通过|
|K2.6|64|208550|1026879376368|595148192736|通过|
|K2.7-Code|64|208550|1026879376368|595148192736|通过|
|K3|96|497220|2779931837184|1560860324864|通过|

计算规则独立写于 `inventory_audit.py`：按 dtype 和 shape 检查每个张量 data_offsets 字节数、检查分片连续区间与重复名称、INT4 I32 打包按 8 倍、MXFP4 U8打包按2倍还原逻辑元素，scale/shape/RoPE buffer 不计逻辑权重，保留全部文件载荷。再与固定 index 的键集合、total_size、逐张量模板数量及站内数字对账。逐层统计矩阵、按top-k/专家数折算路由矩阵，独立重算attention参数和线性计算代理。不是重新调用作者原计算函数。六个K2文本线性代理均为31,686,066,176，K3为104,175,432,704。

`verify_remote_headers.py` 对每个文件先请求0–7字节解出头长度，再请求精确的8至header_length+7区间。必须返回HTTP206与精确Content-Range，否则记失败；没有读取张量权重的数据区。全部473份成功，耗时94.48秒。收据保存每份固定URL、revision、文件名、文件总字节数、远端原始头SHA256、规范化JSON SHA256、本地原件SHA256及逐对象相等结果。K3的A_log文件布局确实按已发布头计数，但这不解决其与源码头数的语义差异；AttnRes、运行加载与算子语义仍由主审和接口子审专项裁定。

### OpenBMB 全量表与来源检查

全量检查脚本为 `openbmb_audit.py`，结果为 `openbmb-checks.json`。最终 0 errors、0 flags；已按新字段独立逐层验证 QK/V 尺寸，保留初审结果 `initial-openbmb-checks.json` 供前后对照。

- 17 配置的文件哈希、revision、固定 URL 一致；751 层的组件定位、编号连续性、hidden/heads/KV heads/FFN 字段逐行对照来源配置。
- 6108 步骤与后端表逐键一一对应；无重复行键、无未识别模型；所有 NVIDIA/AMD/Ascend 后端行均保留未核验声明，没有把通用 torch 数学接口标成已验证硬件 API。
- 3826 个能明确构成二维线性矩阵的步骤独立检查 out/in 维度；专用递归/声学组件仍未知，不用矩阵一致性替代其语义验证。
- 51 个分模型 CSV ZIP 成员与总表逐行相等；全局未知项与17模型覆盖表对应。
- 88 仓库索引名称集合、fork/archived/default_branch 与原始 API 分页一致。原始分页 88+0，组织记录 public_repos=88；86 非 fork、2 fork，无 archived，与历史报告范围声明一致。
- 140 README、64 附加源码/文档、34 HF 配置/模型卡，共238份文件的保存哈希与来源凭据一致，且本次从各官方固定 revision 独立联网重取 238/238 成功，238/238 SHA256 匹配，0失败、0不匹配（`openbmb-primary-receipts.json`）；重点模型配置摘录按嵌套子集逐字段核对一致。
- 5 个已核验接口的声称限定于仓库级 README 或具体构造/动作头，没有声称完整 checkpoint 的每条运行分派。3 个关联实现来源本次联网复取并与保存 SHA256 全等。

### 88 仓库逐项语义审查

每项人工查看站内全部七类叙述，针对具体主张读取官方 README、已保存的相关源码、模型配置或官方论文，记录原文文件与行号。涉及模型尺寸/骨干的 AgentCPM、Eurus/Eurux、MiniCPM 各代、V/o、Robot、VisRAG、VoxCPM、WorkflowLlama 等均分别核对，没有仅凭组织归属归为原创骨干。工具与评测 API 的调用链和模型本体区分也逐项核对。依赖关系的合理推广及场景/局限仍标为工程分析，未将其当作实测。

ChatDev 的动态边、循环和 Thinking，以及 VoxCPM-demopage 静态 audio 页面，另独立取得 4 份固定提交文件，全部 Git blob SHA 与保存的官方 tree 一致；见 `semantic-primary-receipts.json`。HerculesBench 正式数据、指标和维护意图不能由简短根 README 与 OlympiadBench 评测子文档证实，此缺口在逐项表明确保留。

## 原始来源独立核验

`primary-receipts.json` 记录 34 次独立官方 HTTP 获取，全成功；31 份带本地预期哈希的资料全部匹配：17 OpenBMB 配置、11 Kimi 配置、3 OpenBMB 实现来源。另取 K2.7 模型卡、MiniCPM3 实现和 Audio 实现用于专项核查。文件来自官方 Hugging Face 组织或官方 GitHub 原始仓库，没有使用第三方二手解读来裁定结构。

此外 `verify_openbmb_sources.py` 独立重取全部238份 OpenBMB 官方保存来源，`fetch_semantic_gaps.py` 新取4份针对性源码/页面。两组分别全部匹配来源凭据 SHA256 或 Git blob SHA。34份专项来源与238份全集存在交集，不能简单加总为互不重复来源数。

若输入 URL 为 main，receipt 同时保留实际重定向到的固定提交路径。因此 Audio 和 Linear 的上述问题有可复查的固定 revision，不能只依赖可变 main。

## 报告、图表与页面的口径

全家族总览及10张 Kimi 分析图均已看图，10图同时核对原始作图程序和显示数据。第2/6/7图的数据源取自相同审计摘要，七个检查点总量及全部596个分组/层的参数、载荷、线性代理已用远端核验后的原始头独立复算；第10图的全部六项 K2.6/K2.7分数与本次复取固定 K2.7 官方模型卡一致。对数轴、双轴、十进制GB、线性计算代理和非硬件实测限制均有标签。第8图明确说明 K2 向1M外推是反事实参照，不误标为原版支持的窗口。

Kimi 历史主输出和第二输出中全部共同文件字节相同；主输出复制到站内的对应研究文件也相同。OpenBMB HTML 相对历史包的差异集中于站内/离线导航和文档阅读脚本，正文没有发现由这一复制导致的数值分叉；ZIP的离线导航差异不计作数据错误。审计期间主协调者正在修改网站，新增加的下载导航也属于非正文差异，最终冻结已复验。新增 QK/V 列是本轮唯一历史导出数据修正，当前总表、分模型包与总研究包一致，站外原件保留旧口径。

报告通读覆盖 Kimi 主报告正文和 OpenBMB `01-深度分析.md`，重点复核主模型尺寸、量化/来源、结构与能力之间的推断边界。OpenBMB 报告把模型权重来源、团队贡献、工程推断与硬件测量分开；不能把它的官方成绩描述当作本审已复现。Kimi Linear 和 Audio 的旧站差异见 L1–L3，现已关闭。K3 具体加载、缓存及 AttnRes 语义由主审及接口子审裁定；本报告的头部计数通过只说明已发布文件布局和站内数量一致。

## 实际限制与未验证范围

1. 本子审没有运行模型、下载完整权重、复现benchmark、测试GPU/NPU、检查真正的设备分派与数值误差。文中性能与能力的真值不在本次验证能力范围内。
2. 已逐项审查88仓库附录的616段具体叙述，但没有逐行审计全部88个仓库的所有源码，也没有证实作者实验真值。HerculesBench 正式任务定义/发布意图证据不足；工程用途、成本和局限解释仍是分析。MiniCPM-V 的浅层压缩与 Omni-Flow 另对照保存官方论文摘要/正文，不计入238份 GitHub/HF 文件远端逐字节复核。
3. Kimi 11图通过静态图像与生成代码/数据核查。OpenBMB谱系SVG已核其全部文本与实线/虚线关系、模型骨干/初始化/转换口径；对应MiniCPM-SALA原始README第12行确认从MiniCPM-4初始化，o4.5原始论文第5.1节确认从V4.5预训练checkpoint初始化。未重新执行绘图，也未逐页渲染PDF；Kimi总览PDF/SVG仅额外完成资产与字节一致性，不作完整排版结论。
4. 三维检查覆盖11份结构JSON与页面渲染逻辑，没有在本子审中逐个浏览器实际操控所有3D状态。没有把JSON结构完整性说成WebGL交互或GPU兼容性通过。
5. Kimi compute/API、hardware、cache公式和K3专项有其他子审负责；本报告对相关文件做清单与交叉数据检查，不能替代那些独立结论。L1缓存文案发现已转交接口子审归并。
6. 此次156模板不属于本子审；只在发现Audio旧站缺口后与主审确认新模板没有同样缺口。没有据此评价156份文档的全部正确性。
7. 最终检查锚定本地提交 `6613c6f584ab41eec0eed8c4c3981032595b66e5`。主审负责此次156模板、新文档页及整体发布决定；本子审未发布。站外历史原件未倒写，不能将当前网站修正的结论扩大为旧归档已同步更新。

## 可复验产物

- `frozen-recheck.json`：最终提交、工作区状态、L1–L4关闭检查与修正资产哈希。
- `REPOSITORY-SEMANTIC-REVIEW.md` / `repo-semantic-ledger.json`：88项完整语义判定、616段镜像检查和逐项固定URL/原文行号。
- `openbmb-primary-receipts.json` / `openbmb-primary-summary.json`：238份来源全量独立远端哈希验证。
- `semantic-primary-receipts.json` / `semantic-primary/`：4份针对性追加原始证据。
- `inventory.json` / `inventory-summary.json`：全资产清单与哈希。
- `checks.json`：Kimi原始头复算、source/dist镜像、配置事实及下载检查。
- `openbmb-checks.json`：OpenBMB全量表、ZIP和来源检查。
- `primary-receipts.json` / `primary/`：34份本次独立官方获取与凭据。
- `remote-header-receipts.json` / `remote-header-summary.json`：473份官方固定版本文件头的全量远端核验。
- `inventory_audit.py`、`openbmb_audit.py`、`fetch_primary.py`、`verify_remote_headers.py`：只向审计目录写出的自编只读核查工具。
