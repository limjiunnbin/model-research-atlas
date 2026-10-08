# DeepSeek 系列调研路线与交付规划

本文记录 DeepSeek 系列的研究路线、证据规则与实际交付状态。2026-10-01 已完成选定范围的 P0–P5、MTP/DSpark 与存储/BPE 补齐，以及 CANN 开源来源和 package 核对。当前覆盖 21 个 checkpoint；第 13–16 节保留阶段记录，第 17 节记录 2026-10-02 完成的 A1–A5 静态分析扩展与最新验收。用户确认当前没有硬件，设备运行与性能验证已移出本次范围，本次按静态研究与网站整体验收。

- 规划日期：2026-10-01
- 目标仓库：https://github.com/ddddwee1/model-research-atlas
- 调研模板基线：2d2411058b61529e25d64c7c96c26d4ec9992b0d
- 当前交付范围：P0–P5 静态研究与网站；21 份逻辑结构、1,122 个主干层、76,502 个展开步骤、11 个实现专题、4 张 SVG 和七 sheet XLSX/CSV/ZIP；全部 21 版文件头、MTP/DSpark、公开完整 BPE，以及 30 个 CANN 源码家族与本地 SDK 已核对；设备实验不纳入本次验收
- 文档约定：正文段落不按固定列宽硬换行；标题、列表、表格和代码块按语义换行
- 使用方式：新会话先阅读第 17 节和当前复现说明，第 13–16 节是此前阶段记录；复核数据后再处理新增任务。设备实验仅保留为未来可选附录，不是本次待办或验收门槛

## 1 目标与默认范围

### 1.1 最终要回答的问题

1. DeepSeek 各版本之间真正改变了什么：模型主干、注意力、专家结构、训练方式、推理模式、权重精度，还是服务配置？
2. 一个具体 checkpoint 的参数、层、矩阵、缓存和计算步骤分别是什么，结论能否回到固定版本的配置、权重元数据或源码？
3. 数学计算在参考实现、推理框架、后端分派与设备 kernel 之间如何对应，哪些路径有源码证据，哪些仍未知？
4. 面向一个明确的 Ascend 型号和软件栈，哪些模块值得优先研究或实验，如何验证正确性和收益？
5. 如何把以上结果整理为可复核、可维护、可继续扩展的数据和报告，而不是仅有一份论文摘要或性能排行榜？

### 1.2 首版默认范围

首版以主线文本模型的架构到推理计算为主，交付研究说明、版本对比、参数表、代表层算子表与来源清单。V3 作为第一份完整样例，V3.2 和 V4 作为架构增量研究；R1 单独说明后训练与底座关系。V2 用于理解 MLA 与 DeepSeekMoE 的来源。版本盘点可以更广，但逐矩阵和逐后端深读先聚焦代表版本。

本轮已覆盖原始 R1 发布的六个 Distill checkpoint。Coder、Math、VL、OCR 等其他系列、所有历史小版本、后续蒸馏发布、第三方量化版本以及跨硬件实测，列为后续可选范围。正式版本清单必须通过官方目录逐项核实；本文给出的研究顺序不是截至某日的完整产品目录，也不是“最新版本”声明。

已确认首轮以原始 DeepSeek-V3 为对象，采用研究报告、机器可读 JSON 和 CSV 表格交付；主线之外的全量分支和原始 Excel 模板不作为首轮要求。以下保留首轮选择；设备范围已按第 16 节移出本次验收：

- [x] 首轮交付：版本关系概览、V3 参数表、两类代表层的 prefill/decode 算子表、来源与缺口、静态复算；后续先做 V3 基础网站接入
- [x] 首轮深读主线文本 V3；V2/V3.2/V4/R1 只建立关系与后续入口，不宣称已完成其结构研究
- [x] 首个目标 checkpoint：原始 deepseek-ai/DeepSeek-V3，具体 revision 在 P0 固定
- 设备运行/性能阶段已移出本次范围；未来有硬件并另行发起时，再固定设备和软件栈，不能将该阶段视为已执行
- [x] 原始 V3 只读取全分片 header；权重载荷下载为 0。真实 profiling 与权重加载仅属未来可选研究
- [x] 采用独立研究 XLSX，另提供全层和分模型 CSV；不冒称兼容未取得的私有 Excel 模板

### 1.3 首个最小交付

第一份可评审成果包括：版本与架构关系概览；V3 固定版本的配置参数表；一层 Dense 和一个典型 MoE 层的数据流；分别覆盖 prefill/decode 与展开式/吸收式 MLA 的算子和 shape 表；来源与缺口清单；独立 shape/参数复算结果。首轮后端与 Ascend kernel 映射保留未知，其研究在 P2 开始。静态研究使用配置和固定源码，记录实际读取范围。

最重要的验收标准是：评审者从表中的任意一个事实、shape 或算子映射出发，能找到相应配置字段、源码符号、条件分支或权重元数据；推导过程可以复算；未知项没有被默认值、相邻版本或经验猜测填满。

## 2 如何理解与使用现有仓库

### 2.1 仓库提供的模板能力

仓库包含家族与版本目录、章节报告、结构可视化、逐层数据、后端接口研究、硬件与部署说明、优化待办，以及 XLSX、CSV 和附件下载。动态模型家族当前包括 Kimi、GLM 与 DeepSeek，另有 OpenBMB 独立报告。数据和静态资源是主要资产，Python 脚本负责生成部分导出与校验，前端从 JSON 加载内容。

仓库已有数据应当作为组织方法和接口示例使用。Kimi、GLM 或第三方部署配方里的参数、缓存假设、量化格式和性能说明不能直接移植到 DeepSeek；这些示例本身也有不同的证据深度。README 与验证记录明确区分静态研究、权重文件头审计和设备实测，当前规划继续保持这种区分。

### 2.2 建议阅读顺序

| 顺序 | 已有文件 | 要提取的内容 |
| --- | --- | --- |
| 1 | README.md（仓库文件 `README.md`） | 仓库用途、交付形式、数据流和已声明的边界 |
| 2 | ADDING_MODELS.md（仓库文件 `ADDING_MODELS.md`） | 新增家族、版本、结构、硬件与计算数据的规则 |
| 3 | schema.json（仓库文件 `schema.json`） | fact、model、family、architecture、hardware、compute 的数据契约 |
| 4 | data/catalog.json（仓库文件 `data/catalog.json`） | 家族入口和独立报告入口的接入方式 |
| 5 | data/families/kimi/family.json（仓库文件 `data/families/kimi/family.json`） | 完整模型条目、事实来源、报告与结构引用 |
| 6 | data/families/kimi/k2-instruct-architecture.json（仓库文件 `data/families/kimi/k2-instruct-architecture.json`） | 层、模块、专家模板、逻辑矩阵与存储矩阵 |
| 7 | data/families/glm/hardware.json（仓库文件 `data/families/glm/hardware.json`） | 实现专题、平台条件、待验证优化与实验协议 |
| 8 | dist/downloads/model-documents/manifest.json（仓库文件 `dist/downloads/model-documents/manifest.json`） | Excel 原字段、扩展证据列、模型覆盖口径 |
| 9 | scripts/build_compute_atlas.py（仓库文件 `scripts/build_compute_atlas.py`） | 固定源码证据与计算步骤的组织方式，不直接运行作为 DeepSeek 生成器 |
| 10 | scripts/build.py（仓库文件 `scripts/build.py`）、scripts/validate_compute.py（仓库文件 `scripts/validate_compute.py`）、scripts/validate_model_documents.py（仓库文件 `scripts/validate_model_documents.py`） | 当前实际执行的检查，以及 Kimi 专用逻辑和导出限制 |
| 11 | VALIDATION.md（仓库文件 `VALIDATION.md`）、THIRD_PARTY_NOTICES.md（仓库文件 `THIRD_PARTY_NOTICES.md`） | 验证范围、来源与许可边界 |

### 2.3 不能直接照搬的地方

- 现有 compute 生成器与验证器包含 Kimi 专用路径、组件和断言。新增 DeepSeek 需要对应生成与验证逻辑，并调整 build 的计算验证分派，不能只替换 familyId。
- 原 schema 的 architecture.cache 要求 mla、kda、vision。当前已改为各模型实际适用的缓存键，并允许未知存储 shape/dtype；消费者支持 DSA、CSA、HCA、SWA、GQA，旧家族继续保留原字段。
- 结构页面可接受未知类型并使用中性色，但“模型/3D 模块跳转到实现专题”的映射仍可能需要扩展和回归检查。
- 原始私有 Excel 模板未公开。仓库已有独立导出和字段定义可参考；不能宣称可以重新生成未取得的原始模板，也不能修改原21列后仍声称保持模板兼容。
- 现有逐层字段说明与 Excel 使用不同符号口径：逐层说明里的 S 是注意力可见长度，Excel manifest 里的 S 是 verify 长度。新数据先统一符号，并为兼容输出建立显式转换。
- 普通 build 会生成导出、重写部分 family 数据并复制 data 到 dist/data。后续运行前先记录工作树状态，运行后检查 diff，防止把与 DeepSeek 无关的生成变更一并提交。
- 仓库未设置覆盖所有内容的统一开源许可证。参考方法、复用代码、转载材料和发布权重元数据分别核查相关权利；公开可读不等于所有材料都获得了再分发许可。

## 3 证据与记录规则

### 3.1 两条维度分别记录

第一条维度沿用现有 fact.evidence：official 表示可追溯的模型发布方披露；derived 表示由明确输入计算得到；interpretation 表示解释或判断；unknown 表示未知。fact.value 为 null 时，evidence 必须是 unknown，不能用 0、空字符串或估计日期伪装为已知。

第二条维度记录“核验到哪里”：论文或模型卡、配置、参考源码、推理框架、设备实现、权重文件头、运行正确性、性能实验。这些层级不是简单的高低替代关系。例如模型卡给出的参数规模和文件头统计的存储载荷回答不同问题；源码存在也不能证明运行环境可用。现有 schema 未直接规定的细分状态，可以先放在来源记录、scope、limit 或运行条件说明中；若未来扩展字段，应同步修改契约与消费者。

| 信息 | 最少需要的依据 | 不能由此推定 |
| --- | --- | --- |
| 官方公布的架构与规模 | 官方论文或模型卡，记录版本与位置 | 精确权重载荷、任意设备上的实际成本 |
| 配置值 | 对应 checkpoint 的固定 revision 配置文件 | 配置默认值一定等于已发布权重的真实取值 |
| 逻辑 shape 与参数量 | 配置加参数声明或可复算公式 | 文件存储布局、kernel 内部布局、实测内存 |
| 权重 dtype、打包和载荷 | 实际权重元数据或文件头，加完整覆盖范围 | 运行时计算精度、速度、数值质量 |
| 参考计算 | 固定版本的源码符号、输入输出与分支 | 高性能后端按相同步骤逐一执行 |
| 后端或 kernel 映射 | 固定代码版本、调用关系和分派条件 | 任意版本或型号兼容、整模型可运行 |
| 性能结论 | 可复现实验配置、原始结果、统计和正确性检查 | 超出负载、硬件、精度和版本条件的普遍收益 |

### 3.2 来源清单

每个结论使用稳定的 source_id。建议保存 url、repo 或 model_id、revision、path、symbol、line/end、accessed、source_kind、license、sha256、适用模型和证据说明。源码要固定到 commit，HF 配置和权重元数据要固定到 model revision；网页没有稳定版本时，记录访问日期和必要的合法快照。未完成固定的来源显式标为待固定，不能写一个未经核验的 SHA。

函数体哈希与 AST 调用检查可以辅助发现版本变化，但 AST 的调用集合不能当作执行顺序。并列 API 可能是互斥分支；一个 framework module 可能包含多个 kernel；多个数学步骤也可能融合成一个 kernel。shape 表要能表达这种多对多关系。

### 3.3 冲突与缺口

- [x] 同一指标存在论文、config、源码默认值和权重头差异时，保留各方原值与来源，不静默选择一个
- [x] 区分官方模型发布方与后端实现方；后端仓库对模型结构的复述不能替代模型官方配置核验
- [x] 区分检查点 revision、推理框架 commit、设备栈版本和导出脚本版本
- [x] 未确认的发布日期、设备型号、支持条件和性能数值保留未知
- [x] 参数口径必须标明是否包含 embedding、output head、共享参数、MTP 和量化元数据
- [x] 已检查范围与未检查范围写在报告和导出中，不能仅放在开发日志里

## 4 分阶段路线

各阶段以产物和验收门槛推进，不预设未经确认的完成日期。P0 和 P1 完成后即可形成第一份有价值的评审材料；网站和设备实验不是前两阶段的前置条件。

### 4.1 已确认的执行顺序与首轮验收

已按 P0 → P1 → P5a 的起点连续完成 P2 静态实现研究、P3 架构增量、P4 原始 R1/六版 Distill 与 P5b 增量接入。用户授权完成后统一验收，阶段内执行独立证据复核。最新范围按第 16 节执行，设备实验没有当前待办。

2026-10-01 用户已授权连续推进后续阶段，完成后统一验收；阶段内由执行者进行证据复核和回归检查，无需逐阶段请求审批。设备入口曾从本机私有配置读取，连接信息未进入仓库。用户先要求完成静态研究与网站、暂缓设备实验，随后确认当前没有硬件并提出去掉上机验证。因此 P2 按固定源码与条件平台研究交付，设备实验移出本次范围。

| 首轮成果 | 验收门槛 |
| --- | --- |
| 版本关系概览 | 每个代表条目有官方入口，checkpoint、底座关系与推理模式分开；只盘点选定范围，不冒称完整目录 |
| V3 配置与参数表 | HF revision、参考源码 commit、输入文件哈希固定；配置字段与 demo 字段逐项对应 |
| Dense/MoE 代表层数据流与算子表 | 主干层编号、MTP、共享参数、prefill/decode 和 naive/absorb 分支明确；每项关键 shape 有可复算关系与源码位置 |
| 来源与缺口清单 | 事实证据类别、已读取层级和未核验范围同时可见；设备接口与性能未知项保持未知 |
| 独立静态复算结果 | 配置/参数汇总、矩阵收缩维、专家选择、缓存元素数和小规模场景相容；不声称权重加载或设备数值验证 |

首轮研究数据先保存在 `data/families/deepseek/` 的研究文件中，报告与 CSV 放在 `research/deepseek/round1/`。这些中间文件不冒充已接入的 `family.json`、`architecture.json` 或全层审计结果。P5a 开始时再适配网站契约和构建分派，复用已核验事实源。

### P0 固定范围与来源

目标：确定研究对象和来源，不让版本名称或后端假设混入后续数据。

- [x] 阅读本地适用的 AGENTS.md、仓库规范及本文；检查当前分支、工作树和已有 DeepSeek 工作
- [x] 首版获取官方作者目录前 100 个条目的快照，选定 17 个代表 checkpoint；本轮独立固定 4 个 DSpark/后续发布，共 21 版。Base、后训练、蒸馏和推理模式分列，其他条目列入 deferred
- [x] 为 V2、V3、V3.2 与 V4 选定代表模型；R1 建立与底座的关系记录
- [x] 对每个代表记录固定 revision 的官方文件目录、模型卡、配置、索引、许可和代码/论文入口；当前 21 版实际读取范围由各 manifest 与 header 审计记录，未发现的入口保留空值
- [x] 固定首个 V3 checkpoint revision、官方推理源码 commit 和研究输入文件哈希
- [x] 建立 source_id 清单与缺口列表，记录本次读取范围
- [x] 目标硬件与软件栈在缺口列表标记 unknown，继续静态研究

产物：研究范围说明、版本关系清单、来源清单、首个 V3 的配置快照。

验收：每个纳入深读的模型有唯一名称和可核对的官方入口；不存在把推理档位作为独立 checkpoint、把不同底座的 Distill 当作同构模型，或把流动 main 链接冒充固定快照的情况。

### P1 完成 DeepSeek V3 的首个闭环

目标：把一个模型从配置一路拆到可解释的计算步骤和来源。

- [x] 从官方配置逐项提取层数、隐藏维、heads、q/kv 低秩维度、qk_nope/qk_rope/v 维度、Dense宽度、expert宽度、专家数、top-k、共享专家和 Dense/MoE 层位置
- [x] 阅读官方 README_WEIGHTS，单独记录主模型、MTP、共享 embedding/head 和 FP8 scale 的口径；MTP 精确 unique 数与存储别名保持未知
- [x] 阅读官方 inference/model.py 的模型参数、MLA、MLP、Gate、MoE、Block 和整体 forward，并对照 HF 参数声明与路由代码
- [x] 画出 embedding 到输出的整体数据流，标出一层 Dense 与一个典型 MoE 层，以及 MTP 的独立位置；MTP 依据论文 §2.2 与权重说明，不宣称 demo 执行了 MTP
- [x] 给 attention、Dense FFN、router、routed experts、shared experts、norm 和 residual 建立参数与操作记录
- [x] 对初始 prefill 和常规 decode 分别记录 query、KV/cache、中间张量和输出；naive/absorb 分支分别说明，带历史 chunked prefill/verify 列为缺口
- [x] 按第 7 节符号约定写出 shape；独立检查场景、跨步骤形状、路由与 cache，使用 FP64 合成小矩阵复算 MLA 代数关系
- [x] 给每个结论附 source_id、固定路径、符号和条件；无法定位的项保留未知
- [x] 输出研究说明、77 行参数 CSV、348 行代表层算子 CSV、来源与缺口清单，未填设备 kernel 或性能猜测

产物：可审阅的 V3 单模型研究包。建议先用可读 Markdown/CSV 和机器可读 JSON 表达，避免一开始就手工维护多份重复事实。

验收：两个代表层中的每项关键 shape 均能复算，模块参数能够按声明口径汇总，MTP不混进主干层数，prefill/decode分开，逻辑 shape 与存储格式分开。第一版即使没有任何设备 kernel 映射，也应明确完整的数学与参考源码链条。

### P2 建立推理框架与 Ascend 的证据映射

目标：从“模型如何计算”推进到“某个明确软件栈如何实现”。

- [x] 静态主路径选定 vLLM / vLLM-Ascend，TileKernels 为独立实现对照；A2/A3 与 Ascend 950 条件分列，现场兼容栈验证仅属未来可选研究
- [x] 固定模型实现、attention backend、MoE backend、量化实现和相关设备扩展的源码版本
- [x] 追踪模型 forward 到框架模块、分派器、后端函数与设备算子的调用关系
- [x] 分开记录 prefill、decode、KV/cache管理、MoE路由、dispatch/combine、专家GEMM和通信路径
- [x] 映射注明设备代际、dtype、head/cache/layout、TP/EP/CP、图模式和分派条件；未取得完整条件的设备调用保留模块上下文或 unknown
- [x] 阅读 V3 官方转换和量化参考 Linear，以及 Ascend FP8/MXFP8/BF16 fallback、W4A8/MXFP4 loader；记录 scale/打包/条件，未实际执行权重转换
- [x] 给未确认映射写具体缺口，例如“只定位到分派器”“未检查设备分支”“未验证转换权重”
- [x] 形成候选优化列表，按问题、观察、假设、实验、指标、风险组织

优先专题建议：MLA 的投影与缓存、attention prefill/decode路径、MoE路由与通信、专家GEMM、量化与反量化融合。是否继续研究 DSA、CSA/HCA、mHC 或 Engram，取决于所选模型的真实结构和任务目标。

验收：每条“支持”声明都带具体条件；同一模型不同权重精度和不同设备型号不混用；源码可用、测试通过和性能实测三个状态分别标记。没有 profiler 证据时，只提出待验证瓶颈。

### P3 扩展 V3.2 与 V4 的架构增量

目标：复用已核实的共同部分，集中解释新结构怎样改变计算、缓存、通信和后端要求。

V3.2 任务：

- [x] 从官方模型卡与固定配置确认 DSA、indexer 及其参数，核对与 V3.2-Exp 的结构关系
- [x] 对比 V3 与 V3.2 的配置、参考张量命名、精度声明与 forward；本轮补齐 V3.2/Exp 各自完整 header 与 MTP
- [x] 拆解索引打分、token选择、稀疏attention和缓存之间的数据关系，说明额外计算与可能节省的计算
- [x] 分开记录训练侧机制与推理侧实际执行；不把论文算法直接当作某个后端kernel
- [x] 在算子表中新增 indexer 和稀疏路径的输入输出、条件、来源与未知项

V4 任务：

- [x] 明确研究 V4-Pro、V4-Flash、Base 或后训练 checkpoint，并固定各自 revision
- [x] 从官方配置与参考实现读取真实层排布、压缩参数、head维度、专家配置和量化设置
- [x] 分别拆解 CSA、HCA、局部/压缩缓存及相关索引，禁止沿用 V3 的缓存公式
- [x] 拆解 mHC 四流、控制投影、Sinkhorn、hc_pre/hc_post 和流折叠；逻辑形状、参考计算 dtype 与各版文件头的实际存储 dtype 分列
- [x] FP4/FP8 按 Base/后训练、routed/shared 分列；逻辑形状、I8 打包容器、group32 UE8M0 scale 和后端输入条件分别记录，各版实际存储已由自身文件头核对
- [x] 核对 hash、路由或记忆相关字段的真实语义，不仅根据字段名推断是 Engram
- [x] 输出“共同部分、变化部分、待核验部分”的结构/算子差异说明，并说明对应测试需求
- [x] 纳入 DSpark 的实际三个 stage、目标特征、并行 backbone、顺序 Markov、confidence、cache 与服务协议，具体核对见第 14 节；CANN 来源见第 15 节

2026-10-01 阅读到的官方 V4-Pro 模型卡明确列出 V4-Pro、V4-Flash 和各自 Base，介绍 CSA/HCA、mHC 与 Muon；Pro-Max/Flash-Max 是推理档位。Engram 官方仓库提供论文和演示，其 README 明确说明演示中的 Attention/MoE/mHC 被 mock。仅凭这些材料，不能认定 Engram 已用于某个具体 V4 或其他版本的公开 checkpoint。未来版本名称及集成关系必须重新查官方证据。

验收：所有复用项都能说明复用依据；所有新模块都有独立shape与实现边界；版本对比不把训练优化器、后训练方法和推理kernel混为一类。

### P4 整理 R1 与 Distill 分支

目标：解释能力训练与模型结构的关系，避免为同一主干重复制作整份结构数据，也避免错误合并不同底座。

- [x] 记录 R1/R1-Zero 与 V3-Base 的官方关系，再核对具体 checkpoint 配置
- [x] 本轮选定原始 R1/R1-Zero，核对与 V3-Base 的配置与参考代码字节一致；后续 R1 发布与权重数值比较不在首版范围
- [x] 六版 Distill 按实际 Qwen/Llama 底座建立 GQA/Dense 结构，并审计各自完整 tokenizer/BPE；Qwen 四版比较 config/完整 vocab/merge/added tokens，Llama 原底座 gated 差异明确保留未知
- [x] 为后训练方法、推理模式、采样设置和评测条件建立独立说明，不把它们写成架构差异
- [x] 架构共享使用显式关系或模板引用；只有实际核对一致的部分才复用

验收：版本谱系能够回答“谁基于谁训练、谁共享结构、谁只是推理模式”；R1-Distill-Qwen/Llama 的结构按其具体底座研究，不套用 V3 的 MLA/MoE 表。

### P5 接入仓库数据与页面

目标：在内容经过审核后，将同一事实源接入现有展示和导出体系。

P5a 先接入经过审核的 V3 研究，提供家族、版本、结构、来源与基础下载；P5b 随 P2/P3/P4 的成果逐步补充实现专题、其他版本和需要的 Excel。全层静态展开与实际权重头审计分别计数。

- [x] 按第 6 节建立 DeepSeek 数据目录和报告，不修改已有家族事实
- [x] 对照 schema 与前端实际读取字段，确认所有引用、ID、section和model关联
- [x] 编写或适配 DeepSeek 数据生成与验证器，避免运行 Kimi 专用验证器得到虚假覆盖声明
- [x] 按需要扩展 cache、模块类型、实现专题映射和计算验证分派，并为已有家族保留兼容性
- [x] 从 canonical JSON 生成 Markdown/CSV等导出，避免手工编辑生成物与源数据发生漂移
- [x] 从 canonical JSON 导出独立七 sheet XLSX、76,502 行全层 CSV 和 21 份模型 CSV；未实测 duration 留空，未提供原模板兼容声明
- [x] 执行静态检查、构建与浏览器验收，检查所有生成diff和附件哈希
- [x] 已提供本地静态预览与可托管 dist；本轮未请求公开部署、提交或推送

验收：家族入口到版本、层、模块、矩阵和来源的路径可用；搜索、比较、下载、窄屏和二维回退正常；已有Kimi/GLM/OpenBMB主要路径无回归；构建成功的含义与未运行的模型实验明确分开。

### 可选附录：未来有硬件后的运行与性能研究

状态：已按用户要求移出本次范围，以下内容仅供未来另行开展时使用，不属于本次待办。本次没有执行设备实验，静态结论不作为设备正确性或性能结论。

- 固定完整环境和权重转换过程，先做启动与小规模正确性检查
- 先对目标模块做数值校验，再进行整模型或服务级实验
- 分开 prefill、decode、verify 和端到端服务指标
- 记录 warmup、重复次数、随机种子、输入输出长度、batch、并发、TP/EP、拓扑及精度
- 一次只改变一个变量，保存基线与优化版本的原始结果
- 同时报性能、显存和质量/误差；量化同时报告质量
- 给出统计分布、异常和失败条件，结果只覆盖实际测试条件

如未来开展，实验结果需能按记录复现，并明确测试条件。这些条件不参与本次静态研究与网站的完成判定。

## 5 V3 首个闭环的具体拆解

### 5.1 固定输入

先取得同一 checkpoint 的 config、模型卡、权重说明、权重索引及需要的元数据，再取得对应官方推理实现。配置、推理demo默认参数和发布权重之间可能存在命名或默认值差异，必须通过字段映射核对。固定到本文记录的V3源码commit可作为起点，但正式产物还需要固定HF checkpoint revision。

官方 V3 demo 的 config_671B.json 在本次核查快照中包含61个主干层、前3个Dense层、hidden 7168、128个attention heads、q_lora_rank 1536、kv_lora_rank 512、qk_nope维128、qk_rope维64、v维128，以及256个routed experts、每token选8个、1个shared expert。这些值用于说明应读取哪些字段；后续必须与选定checkpoint配置核对，不能把demo配置替代完整权重审计。

### 5.2 逐模块记录问题

| 模块 | 需要回答的问题 | 第一版产物 |
| --- | --- | --- |
| Embedding与输出 | 词表和hidden是什么，是否共享，是否分片，是否包含在激活参数口径 | 全局组件参数表 |
| RMSNorm与Residual | 输入输出shape、累加dtype、是否与前后操作融合 | 数学关系与参考源码定位 |
| MLA投影 | Q与KV的低秩分解、RoPE与NoPE分量、输出投影如何构成 | 矩阵shape及中间张量表 |
| MLA注意力与缓存 | naive/absorb路径存什么，prefill/decode如何处理历史与新增token | 两阶段数据流与缓存公式 |
| Dense FFN | gate/up/down矩阵、激活、输出维度与并行切分 | 第一层完整计算样例 |
| Router | 打分、分组、top-k、校正和归一化在哪里，哪些仅训练使用 | 路由步骤与条件 |
| Routed专家 | 每个专家的矩阵、动态token数、计算合并和通信边界 | 一个专家模板加总数/top-k |
| Shared专家 | 是否独立执行、如何与路由专家合并、后端是否存在重叠 | 参考路径与实现缺口 |
| MTP | 额外模块、共享参数、训练目标与推理用法有什么区别 | 单列结构及支持状态 |

### 5.3 一份算子记录应包含什么

建议每行表示一个逻辑步骤，并允许关联零个、一个或多个实现条目。最少包含model_id、component_id、layer或template、step_id、阶段、操作含义、input/output/intermediate/cache shape、dtype、权重逻辑shape、数学关系、参数口径、reference_source_id、backend_source_id、实现层级、分派条件、证据状态和缺口。设备kernel尚未核查时保留unknown，不从“矩阵乘法”猜测某个具体API。

初始示意流程可以按“归一化 → MLA投影及位置处理 → 缓存与attention → 输出与residual → 归一化 → Dense或MoE → residual”拆解。MoE内部再区分routing、token重排/通信、专家GEMM和激活、combine与共享专家。最终步骤顺序必须回到所选实现核实；生产后端可能融合或重排，不能将示意流程当作实际kernel时序。

### 5.4 先做哪些检查

- [x] 配置字段与源码参数声明一致，维度约束有显式断言
- [x] 输入、输出和矩阵乘法的收缩维度一致
- [x] Dense和MoE层映射完整，逻辑层编号与源码0-based编号明确转换
- [x] 专家模板的单份参数、总专家参数和top-k激活代理分别计数
- [x] 共享参数不重复计入unique参数，MTP计数口径单列
- [x] 量化scale等元数据保留存储记录，但不误算为逻辑模型权重参数
- [x] cache公式注明元素数、dtype、层数、batch/长度条件及是否包含额外工作区
- [x] 数学推导、源码静态验证、文件头审计和设备测试的完成状态分别列出

## 6 建议的数据与文件映射

以下包含首轮研究文件与后续接入建议；各文件的实际完成状态见第 13 节。首轮研究 JSON 使用独立的研究契约，不能直接当作网站 compute 或 architecture 数据；P5a 开始时同步修改契约、生成器与消费者。

| 建议路径 | 用途 | 现有契约或参考 |
| --- | --- | --- |
| data/families/deepseek/research/v3-round1.json | 首轮参数、代表层矩阵、算子、公式、冲突与缺口的事实源 | 独立研究契约，由首轮验证器检查 |
| research/deepseek/round1/ | 首轮研究报告、参数 CSV、代表层算子 CSV 与复算记录 | 从同一研究 JSON 导出 |
| data/families/deepseek/family.json | 家族概览、版本、技术主题、场景、边界与下载索引 | schema.family和Kimi family |
| data/families/deepseek/report.json | 已审核的章节内容与模型section引用 | schema.report |
| data/families/deepseek/configs/ | 固定revision的原始配置快照，保留来源与哈希关联 | 已有configs目录 |
| data/families/deepseek/sources.json | 统一来源记录；是否独立成文件需与硬件/计算消费者设计一致 | 建议新增，不是现有强制契约 |
| data/families/deepseek/versions.json | 更广的版本盘点与关系，若family足够则不另建重复事实源 | 可选研究中间数据 |
| data/families/deepseek/v3-architecture.json | V3层、模块、矩阵、参数和缓存说明 | schema.architecture |
| data/families/deepseek/v3-layers.json | 只有完成相应权重头审计后才提供的逐层审计结果 | model.auditPath及已有layers样例 |
| data/families/deepseek/compute.json | 全局组件、代表层模板、展开映射、操作与证明 | schema.compute |
| data/families/deepseek/hardware.json | 实现专题、平台条件、优化待办、协议与固定来源 | schema.hardware |
| dist/assets/deepseek/ | 报告、CSV、图、附件及按需下载资源 | 路径相对dist |
| scripts/build_deepseek_*.py | 从固定输入生成规范化数据与导出，按实际复杂度拆分 | 建议新增，不照搬Kimi硬编码 |
| scripts/validate_deepseek.py | DeepSeek特定结构、shape、证据和导出一致性检查 | 建议新增并接入build分派 |
| data/catalog.json | 内容达到接入门槛后新增deepseek家族入口 | 现有catalog |

### 6.1 family 与 model

沿用模型唯一id、准确名称、branch、summary、tags、facts、sections等字段。已知事实附可核对来源；releaseDate无可靠证据时为null；没有权重审计时auditPath为null。官方config、结构与计算数据的引用按真实完成程度设置，不能为了让页面有入口而创建空壳完成声明。

建议事实包括主干层数、hidden、heads、q/kv低秩维、attention类型、expert数量/top-k/shared数量、Dense与expert宽度、上下文配置、词表、精度与参数口径。当前前端已补充 q/kv 低秩维、Q/K 与 V 单头维、残差流数和 indexer top-k；后续扩展字段仍需检查详情/比较消费者。

### 6.2 architecture

nodes按真实组件组织，语言层使用group=decoder并从1开始编号，全局组件为0。modules区分attention、norm、dense/routed/shared/router及模型真实包含的其他模块。MoE采用代表专家模板加实际count和selectedCount，不展开成数万个浏览器对象。matrices保留tensor_template、逻辑shape、逻辑参数、存储dtype/shape和量化元数据。

只有配置时可以给出配置层型与范围；由配置和源码推导矩阵时明确标derived。只有权重头或同等可核对存储依据才填写实际payload。既有“激活线性参数代理”不应改写成完整FLOPs或实际性能，计算口径必须随数据保存。

### 6.3 compute

组件映射、模板和步骤分开。层模板中的{i}只表示0-based层索引，{e}只表示专家索引；导出全层时用显式映射展开。每个步骤保留数学关系与参考实现，后端条目注明module、dispatcher或device kernel等层级。proof至少能定位repo、revision、path、symbol、line/end及内容哈希，存在条件分派时保存条件。

如果现有schema不能自然表达prefill/decode不同路径、MTP、压缩cache或多对多kernel映射，应先写小样例和兼容设计，再改schema与消费者。不要为了通过旧校验器丢失真实语义。

### 6.4 hardware

modules记录适用model IDs、flow、meaning、code、hardware、limit、refs；platforms按具体型号和软件栈分列；optimizations记录priority、status、scope、observation、proposal、metric、risk、refs。未profiling的项统一标为待验证，不借用上游README性能数字充当本项目实测。

### 6.5 Excel 与 CSV

现有独立模型文档前21列包含模块、算子、npu kernel、算子类型、input/output shape、dim、shape范围、dtype、format、后续dtype、并行资源、duration、计算利用率、mte、scalar和融合分析。若要求兼容，保持列顺序和含义，并使用附加证据列解释来源、参考调用、后端条件、缺口和数学关系。duration等字段只有实际测量才填数值，未知用明确标记；mte、scalar等profiler字段的单位和含义应按目标设备工具确认。

canonical JSON应作为事实来源，CSV/XLSX为导出。校验行数、列名、BOM约定、各模型覆盖范围、ZIP完整性和文件哈希；“有一份Excel”与“完成逐层研究”分别统计。未取得原始模板时，不宣称完整继承其公式、样式或隐藏逻辑。

## 7 Shape 与口径约定

新研究尽量使用语义明确的符号，避免S/T在不同产物中含义变化。若必须兼容旧Excel符号，应在导出说明和公式转换中明确对应关系。

| 符号 | 建议含义 | 注意事项 |
| --- | --- | --- |
| B | request batch | 与连续批处理的有效请求数及padding口径区分 |
| L_q | 本次每请求query/new-token长度 | 常规decode为1；verify不一定为1 |
| L_kv | 本次attention可见的KV长度 | 含历史和本次可见部分，受mask/窗口/压缩影响 |
| N_tok | 本rank实际参与计算的有效token数 | ragged batch使用求和，不无条件写成B乘L_q |
| H | residual hidden width | mHC额外stream轴单列，不合并成同一个hidden |
| N_head | attention head总数 | 局部head数按实际分片规则推导 |
| D_* | Q/K/V、NoPE、RoPE等明确子维度 | 不将不同模型所有head_dim视作相同 |
| R_q、R_kv | 低秩投影维度 | 仅在相应模型真实包含时使用 |
| E、K、E_local | 总专家数、每token选中数、本rank专家数 | 由具体EP布局和路由定义决定 |
| N_e | 第e个专家实际接收token数 | 动态量，padding、capacity和丢弃策略会改变关系 |
| TP、EP、DP | 具体并行维度 | 记录哪一层/哪种张量按哪一维切分或复制 |
| C_* | 明确命名的cache维度或压缩参数 | KV cache、indexer cache、状态和工作区分开 |

无丢弃且不计padding的路由分配数可检查为N_tok乘K，但不能把所有实现都假设成无padding或无容量限制。权重逻辑shape通常按输出维、输入维表示；实际转置、分片、NZ等设备布局和量化存储shape必须另列。单卡局部参数、全模型unique参数、文件载荷、运行时显存、FLOPs和激活参数代理各自保留口径。

## 8 Ascend 研究与实验设计

### 8.1 先确认平台再选实现

不要把“Ascend支持”当作单一布尔值。至少记录设备型号、卡数/拓扑、驱动固件、CANN、PyTorch、torch_npu、推理框架、设备插件、自定义算子包、模型revision、量化格式与转换脚本、图模式及启动参数。不同型号对dtype、指令、kernel和图模式的支持可能不同，应以所选版本官方文档与源码为准。

TileKernels在已核查的2026-09-30快照中新增Ascend后端，README列出Ascend 950和CANN 9.2.0及以上要求，并包含MoE路由、量化、Engram和mHC相关功能。这是值得进一步研究的具体来源，不代表本项目已经验证该栈，也不能推广为其他Ascend型号或完整DeepSeek模型的支持结论。

### 8.2 候选优化卡片

每个候选只写一个可验证问题，例如“某个decode形状下，MLA投影、cache处理和attention之间是否存在可消除的中间读写”。记录现有路径、源码证据、疑似开销、拟议变更、正确性参考、测量指标、前置条件、风险和停止条件。未拿到trace之前，不先断言瓶颈位置或收益百分比。

建议按以下顺序选择：先可运行且可测的基线，再查attention/cache，再查MoE计算与通信，之后研究量化/融合和图模式。若目标是V4，则在真实trace基础上纳入CSA/HCA与mHC；Engram只有在所选checkpoint或独立专题明确需要时才纳入，不凭热门关键词扩大模型范围。

### 8.3 实验记录最少字段

- 模型与权重：名称、revision、转换步骤、精度、scale、文件校验信息
- 环境：硬件、拓扑、软件版本、容器digest、编译选项、自定义算子版本
- 负载：B、L_q、L_kv、输入输出长度、并发、padding/ragged、TP/EP/DP
- 正确性：参考路径、输入种子、绝对/相对误差、logits或任务质量评测
- 性能：阶段、warmup、重复次数、统计分布、时间单位、吞吐口径和峰值显存
- 证据：启动命令、日志、trace、结果文件、失败记录、实验脚本版本

## 9 验证与完成定义

### 9.1 静态研究完成

- [x] 选定模型列表、配置revision和源码commit均明确
- [x] 每个已知事实有来源；null与unknown一致；冲突和缺口可见
- [x] 配置、层型、逻辑矩阵、参数公式和计算shape一致
- [x] 专家数量与top-k、共享参数、MTP和量化元数据的口径正确
- [x] 参考实现、框架分派和设备kernel的证据层级没有混淆
- [x] 后端支持条件和未检查范围随导出保留
- [x] 报告、JSON、CSV中的数值来自同一事实源或有一致性校验

### 9.2 网站接入完成

- [x] 新家族及版本ID唯一，所有路径和section引用存在
- [x] 主干/其他组件层数与数据契约一致，模型特例由相应验证器处理
- [x] 已审计数据的模块参数汇总与审计口径一致；未审计数据不冒称审计
- [x] 新计算验证器明确覆盖DeepSeek，build按family或能力分派
- [x] 附件大小、哈希、CSV行数、ZIP和下载路径核对通过
- [x] 浏览器测试覆盖家族、版本、比较、结构、实现、来源、搜索和下载
- [x] 窄屏及二维回退可用，现有家族无明显回归

### 9.3 本次验收边界

- [x] 当前验收覆盖静态研究、来源、结构/shape/参数、导出与网站
- [x] 设备实验已移出当前范围，未执行不作为未完成项
- [x] 上游自述、静态计算与设备实测分别标注，未知测量保持空值
- [x] 未来实验需要另行发起，不自动恢复

### 9.4 后续验证命令

以下是当前已执行的离线检查入口。源码恢复与完整生成说明见 [当前研究复现说明](README.md)；浏览器记录见 [browser-validation.json](metadata/browser-validation.json)。运行前仍应检查工作树，因为 build 会更新生成物。

~~~sh
git status --short
python3 scripts/build.py
node --check dist/app.js
node --check dist/structure.js
node --check dist/hardware.js
node --check dist/compute.js
git diff --stat
git diff --check
python3 scripts/serve.py
~~~

scripts/validate_deepseek.py 保留首轮研究契约；scripts/validate_deepseek_atlas.py 覆盖当前网站、全层公式、CSV/XLSX/ZIP 与下载并接入 build。另用 Draft 2020-12 JSON Schema 验证 26 份 DeepSeek/目录文件；579 份来源字节、330 个 AST 函数体与开放后端 C++ 范围通过 restore_deepseek_sources.py 复验。现有 Kimi 验证器仍只覆盖 Kimi。

## 10 已核实的来源起点

本节保留路线建立时的官方导航，当前 579 份实际研究输入及读取范围以 sources、extended-sources、backend-sources、followup-sources、cann-sources 与 site-proofs 为准。访问日期为 2026-10-01；当前网站使用的 21 个 HF checkpoint 均已固定 revision。每个引用只支持明确核验的内容，不扩大为完整运行支持。

| 来源 | 固定版本或状态 | 已核实用途与下一步 |
| --- | --- | --- |
| [研究模板基线](https://github.com/ddddwee1/model-research-atlas/tree/2d2411058b61529e25d64c7c96c26d4ec9992b0d) | 2d2411058b61529e25d64c7c96c26d4ec9992b0d | 已读README、目录、schema、新增指南、示例与验证脚本；后续检查仓库变更 |
| [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2/tree/ec98ee3cbffc32104cd55dba8af884b3d772602a) | ec98ee3cbffc32104cd55dba8af884b3d772602a | 已确认仓库快照；作为MLA/MoE演进回溯入口，论文和实现需继续细读 |
| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3/tree/9b4e9788e4a3a731f7567338ed15d3ec549ce03b) | 9b4e9788e4a3a731f7567338ed15d3ec549ce03b | 已读README、权重说明、demo配置及部分模型代码；适合首个闭环 |
| [V3权重说明](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/README_WEIGHTS.md) | 与上述V3快照一致 | 主模型、MTP、共享参数和FP8 scale口径；正式统计仍需权重证据 |
| [V3 demo配置](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/configs/config_671B.json) | 与上述V3快照一致 | 初始参数读取样例，不代替checkpoint配置 |
| [V3参考模型](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py) | 与上述V3快照一致 | 继续逐函数追踪MLA、MoE、Block与forward |
| [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/tree/0cf78561f1d51c84a21b2190626b21116d5c68bb) | 0cf78561f1d51c84a21b2190626b21116d5c68bb | README明确R1与V3-Base、Distill与Qwen/Llama底座关系 |
| [DeepSeek-V3.2模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) | a7e62ac04ecb2c0a54d736dc46601c5606cf10a6 | 已读取固定 config/inference，核对与 Exp 字节一致；actual storage 未审计 |
| [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/tree/87e509a2e5a100d221c97df52c6e8be7835f0057) | 87e509a2e5a100d221c97df52c6e8be7835f0057 | 已确认远端快照；作为V3.2官方模型卡所指实现入口，需继续代码审阅 |
| [DeepSeek-V4-Pro模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | b5968e9190ef611bbf34a7229255be88a0e937c1 | 已读取固定 config/inference，生成实际主干层排布、逻辑矩阵、mHC 与 MTP；actual storage 未审计 |
| [V4-Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json) | b5968e9190ef611bbf34a7229255be88a0e937c1 | 结构字段已与参考构造和逐层 shape 核对；未替代权重存储审计 |
| [V4-Pro inference目录](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1/inference) | b5968e9190ef611bbf34a7229255be88a0e937c1 | 本轮固定参考入口；compressor/indexer/mHC/路由/MTP 已按函数定位 |
| [Engram](https://github.com/deepseek-ai/Engram/tree/fb7f84a21f91223715394a33a1dc24bbfb7f788e) | fb7f84a21f91223715394a33a1dc24bbfb7f788e | 已读README，演示mock边界明确；具体checkpoint集成关系另查 |
| [TileKernels](https://github.com/deepseek-ai/TileKernels/tree/66258df6175d2f630ffecb04c5ab66bff8a2ae6a) | 66258df6175d2f630ffecb04c5ab66bff8a2ae6a | 已读README的Ascend更新和平台条件；后续检查API、参考实现与测试 |

当前静态主路径已固定 vLLM `bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f`、vLLM-Ascend `a8fcedb03d93e60efceddbfc912406f7fa491d57`，TileKernels 为独立 Ascend 950 实现对照；Transformers v4.44.0 固定到 `984bc11b0882ff1e5b34ba717ea357e069ceced9` 供 Distill eager 参考。SGLang、FlashMLA、DeepGEMM、DeepEP 与更多 CANN 路径仍为可选扩展；这些固定提交没有被组成或测试成兼容运行栈。

## 11 风险与停止条件

| 风险 | 处理方式 | 何时暂停相关工作 |
| --- | --- | --- |
| 范围扩张到所有系列与后端 | 先交付V3样例，新增范围单列优先级 | 无法说明新增项对当前交付的价值时 |
| 官方配置、代码默认值或权重头冲突 | 保存双方证据，定位版本和加载路径 | 关键形状无法自洽时，不继续生成确定性结论 |
| 源码/API漂移 | 固定commit与哈希，记录更新差异 | 上游版本无法定位或证据已失效时 |
| 未知硬件与转换精度 | 先做静态数据流，平台与性能保持未知 | 需要设备专属结论却没有平台信息时 |
| 把静态检查当运行验证 | 报告明确检查类型和范围 | 发现支持或性能结论超出证据时 |
| 模板专用断言误用于DeepSeek | 新建验证器及分派，补回归测试 | 校验器未实际覆盖新数据时 |
| 自动构建污染其他家族 | 构建前后检查git状态和diff | 出现无法解释的非目标变更时 |
| 第三方许可或大文件问题 | 只保存必要、允许分发的材料与引用 | 许可不明、需要发布权重或大规模附件前 |

## 12 本地新会话启动说明

### 12.1 建议启动任务

后续会话从已完成数据出发，不重新开启 P0/P1。可使用以下内容：

~~~text
请先阅读 roadmap_deepseek.md 第 15 节、research/deepseek/README.md、README.md、ADDING_MODELS.md 与适用的 AGENTS.md，检查 git 状态并保留他人的修改。

DeepSeek 当前静态研究与网站覆盖 21 个 checkpoint，含完整文件头、独立 MTP/DSpark、公开完整 BPE 与开放后端追踪。请从 family/architecture/compute/hardware、当前来源证明与验证记录复核已有成果，不把首轮 17 版 versions.json 的历史读取状态误当作当前进度。

用户要求本次按静态研究与网站整体验收，已移除设备运行与性能阶段。不要开展设备连接、权重加载、kernel、profiling 或 benchmark；未来需要时由用户另行发起。私有机器配置、连接地址与凭据不得写入仓库。

新增研究需说明目标 checkpoint、固定来源、证据深度与验收范围。正文不要按固定列宽硬换行。现有网站构建按 family 分派校验，DeepSeek 导出来自 canonical JSON；提交、推送与公开部署按本次会话授权处理。
~~~

### 12.2 继续维护的检查清单

1. 先复核当前研究 JSON、来源哈希、静态校验和网站回归记录。
2. 对用户指出的验收问题修正 canonical 数据或生成器，再生成并校验相应附件。
3. 涉及新 checkpoint 或上游升级时，保留旧 revision 与差异，重新确定读取范围。
4. 设备实验不在本次范围，不能作为遗留或阻塞；未来另行请求时再开展。
5. 记录具体失败、未执行范围和新缺口，保持首轮历史包与当前交付分开。

### 12.3 后续会话应留下的交接信息

- 当前分支、基线与最新相关commit，以及尚未提交的目标文件
- 已确认的研究范围、checkpoint revisions和源码commits
- 已生成的数据/报告/导出路径及其canonical来源
- 本轮实际执行的静态检查、构建、浏览器检查或设备实验
- 失败与未执行项、尚未解决的事实冲突、需要用户决定的问题
- 下一项可以直接开始的具体任务及验收标准

## 13 上一轮首版完成状态（17 版历史）

本节保留首版记录；后续补齐见第 14–15 节，当前验收范围见第 16 节，不能由旧缺口判断最新进度。

| 阶段 | 状态 | 实际交付与边界 |
| --- | --- | --- |
| P0 | 完成 | 官方目录选定 17 个 checkpoint，固定 revision、底座与来源；不是完整产品目录 |
| P1 | 完成 | 原始 V3 的首轮参数/代表层、naive/absorb 代数复算与来源验证；历史包保留 |
| P2 | 静态研究完成 | vLLM / vLLM-Ascend / TileKernels 固定源码，9 专题、2 条件平台、7 候选实验；未组成或测试兼容运行栈，当时 CANN 公开来源未继续核对，现见第 15 节 |
| P3 | 完成所选增量 | V3.2/Exp、四版 V4 配置、参考逻辑矩阵、逐层计算、缓存、mHC、hash/score 与量化声明；实际存储未审计 |
| P4 | 完成所选分支 | 原始 R1/Zero 与六版 Distill；四个 Qwen 底座比较，两个 Llama 底座 gated 差异保留未知；未审计完整 tokenizer/BPE |
| P5a/P5b | 完成 | 17 份架构、914 主干层、60 全局/独立组件、96 计算模板、42,476 展开步骤；网站、图表、报告、XLSX/CSV/ZIP 与回归 |
| V3 文件头 | 完成 | 163 分片、91,991 张量，完整 offset/shape/dtype；读取元数据 11,888,393 字节，权重载荷读取 0 |
| P6 | 用户明确暂缓 | 未执行 kernel、checkpoint、profiling、吞吐/显存/质量实验；性能字段为空 |
| 发布 | 未执行 | 本轮未请求提交、推送或公开部署；本地静态预览可整体验收 |

当前入口为 [研究与复现说明](README.md)、[完整报告](DeepSeek-研究报告.md)、[模型与算子工作簿](DeepSeek-模型与算子.xlsx)、[全层 CSV](DeepSeek-all-layer-shapes.csv) 和 [完整研究包](DeepSeek-研究报告包.zip)。网站 hash 入口为 `#/family/deepseek`。工作分支仍为 `codex/deepseek-v3-round1`，名称沿用首轮。

### 13.1 首轮历史记录

P0/P1 入口为 [首轮报告](round1/report.md) 与 [首轮复现说明](round1/README.md)，事实源为 [v3-round1.json](round1/v3-round1.json)。39 条首轮来源与 versions.json 的 verificationDepth 保留当时读取范围；当前进度由 family、17 份 architecture、compute、hardware 与 site-proofs.json 记录。

V3 HF revision 为 `e815299b0bcbac849fa540c768ef21845365c9eb`，官方推理源码 commit 为 `9b4e9788e4a3a731f7567338ed15d3ec549ce03b`。首轮主干逻辑参数为 671,026,419,200，含 61 主干层、独立 embedding/head、norm 与 routing correction，排除 MTP 和量化 scale；当前完整文件头已核对该主干数量。首轮 77 行参数 CSV 与 348 行代表层算子 CSV 仍是历史代表层交付，不替代本轮 42,476 行全层表。

### 13.2 首版检查与当时剩余边界

- 参数和形状：17 个模型独立公式、914 主干层、18,087 个矩阵实例与 42,476 个展开步骤通过；MTP/共享别名、scale、非训练 hash 表单列。
- 导出：CSV 每行、17 个模型 CSV、XLSX 五个 sheet 的 52,498 个数据单元格与 ZIP 每个条目逐值对照 canonical JSON；下载按大小与 SHA-256 核验。
- 来源：130 份固定公开输入、50 份随仓库保存的公开配置/元数据快照（51 次来源引用）和 171 个 AST 函数体复核；从空缓存恢复同一来源后复验通过，无模型权重下载。
- 契约与网站：Draft 2020-12 JSON Schema 校验 22 份文件；全仓库 build 通过，当前 3 家族、1 独立组织报告、1,788 个结构组件。
- 浏览器：家族、17 版详情、比较、结构/矩阵、实现过滤、prefill/decode/path、来源、报告、在线 Markdown、HTTP 下载、390px 与 WebGL 禁用回退；保留 Kimi/GLM/OpenBMB 回归记录。

没有使用设备连接结果推断模型可运行。V3 之外的真实存储、V3 MTP shared_head.norm 加载别名、Llama 原底座 gated 差异、完整 tokenizer/BPE、当时尚未核对的 kernel 运行行为和全部性能结论仍保留未知。新增历史/后续系列、更多框架与量化发布需要新范围；不影响本次已限定的静态交付。

V3-Base、R1/Zero、V3.2/Exp 的配置声明 MTP，但独立 MTP 权重与 forward 未展开，参数保持 unknown；当前全层完成口径为主干与已单列组件，不借用原始 V3 的 MTP 存储。

## 14 静态缺口补齐与 DSpark（CANN 纠正前的阶段记录）

用户已授权继续补齐静态研究，并要求纳入 V4 的 DSpark。P6 仍暂缓；本轮不开展设备连接、模型执行或性能实验。首轮 17 模型快照保留，新增 checkpoint 与来源独立固定，完成后同步当前网站与导出。

- [x] 核对原始 V3-Base、R1/Zero、V3.2/Exp 的实际索引与 MTP 发布情况；分别建立独立结构/forward/loader/共享口径
- [x] 对全部 21 个 checkpoint 读取完整 safetensors 文件头，区分逻辑参数、量化容器、scale、非训练表、存储副本和共享别名，权重载荷读取 0
- [x] 从官方 DSpark 论文、DeepSpec 与 V4 checkpoint 固定新机制，拆分并行 draft backbone、轻量顺序模块、confidence head、verification/rejection 与调度
- [x] 核实 V4-Flash/Pro-DSpark 和 Flash-0731/Pro-0813 的 checkpoint 关系与 draft 配置，未把所有 V4 版本都标为包含 DSpark
- [x] 独立对照 MTP、DSpark 与普通 AR：参数模块、target feature、KV/cache、draft/verify/update、推测 token 数、接受/拒绝与负载感知长度；论文收益只标来源自述
- [x] 完整 tokenizer.json/BPE 比较；原始 Llama 底座 gated 项保留明确的访问边界
- [x] 开放后端对应 Python 分派、C++ 注册/wrapper、host/tiling 与入口；未观察到的 TileKernels 接线与设备验证保留未知；CANN 来源继续按第 15 节核对
- [x] 同步 roadmap、canonical 数据、报告、网站、XLSX/CSV/ZIP；来源/公式/存储/导出校验与 20 组浏览器回归通过

已补齐 21 个 checkpoint 全部文件头、五版 MTP 独立结构/forward/loader，以及 Flash/Pro-DSpark 挂接版、Flash-0731/Pro-0813 正式发布。当前 1,122 主干层、93 全局/独立组件、145 模板、76,502 展开步骤、330 函数证明与 246 来源；header 覆盖 1,457,766 张量和 172,146,958 字节捕获元数据，权重载荷读取 0。

DSpark 真实三个 stage 位于 mtp.*；HF nextn=1、native stage=3、block=5 和 vLLM 服务示例7分别记录。Flash/Pro 独立 draft 参考参数为 19,845,850,983 / 77,498,408,103，排除主干。完整 BPE 审计六个公开 Distill、四个 Qwen 底座；两个原始 Llama 仍 gated。开放后端进一步对应 C++ 注册/wrapper、host/tiling 与入口，范围哈希不冒充 AST。

当前复核：33,507 个矩阵实例、76,502 行全层 CSV、21 个分模型表、七 sheet XLSX 的 132,269 个数据单元格与 canonical JSON 一致；26 份 Draft 2020-12 schema、246 来源/330 AST 证明、空缓存恢复、ZIP 与 19 个下载通过。Chrome 20 组交互覆盖 DSpark 三 stage/Markov/confidence、全部 21 版详情、四张 SVG、390px、WebGL 禁用回退和旧家族路径，无控制台错误或失败资源。

本节当时将 CANN 未核对部分概括为闭源，这一判断已按第 15 节纠正。上游训练共享/权重值证明、未观察到的 TileKernels 接线、Llama 基线权限及 P6 设备实验继续保留。

## 15 CANN 开源算子与本地 package（设备范围调整前的记录）

用户指出 CANN 算子可在开源仓或 Multipass 安装包中核查。此前把未继续追踪的 CANN 范围笼统称为闭源不准确，现撤回这一判断，并将源码是否可查、库是否安装、框架实际分派、ABI、设备数值与性能分别记录。P6 没有恢复。

- [x] 只读核对本地 CANN 9.2.0-beta.2、aarch64，记录 36 个组件版本及 6 个版本/高级 API 文件哈希；未保存实例、账号、主机或网络信息
- [x] 固定 ops-transformer `5f33f1e23d41fe0a047d01b7c7ae278777cac1f5`、ops-nn `30ef7dd563c8a4b74c3161835c8e47d1d96f87b6`、ops-math `0a2ce5b57caec6068d9e5658b740c2d41482aa15`，对应 tag 均为 v9.2.0-beta.2
- [x] 核对 30 个相关源码家族的 API、host/tiling 与 kernel 文件，333 个新增公开输入同时检查 SHA-256 与 Git blob SHA-1；350 个源码范围独立于 Python AST
- [x] 固定 Ascend/op-plugin `d83570a35dfe0d8e9869c3ecfca6647cfccdd9c8`；核对 Python FIA v2 到 aclnn V4/V5 的条件映射、MoE/GMM 多版本选择和 npu_swiglu 的生成声明
- [x] 官方 MhcPre/Post 与框架自定义 HcPre/Post、TileKernels 保留独立接线证据，未因名称或数学相同声称直接调用
- [x] 新增 CANN 专题、报告第 10 章、独立核对 JSON 和来源/范围验证；更新工作簿与报告包
- [x] 完成 26 个契约、构建、21 组浏览器交互、14 个手机路径、20 个附件 HTTP/哈希与打包文档链接核对；本地 42 个文件哈希复验一致，未执行算子

本地所检查路径提供编译器/AscendC SDK 与 SwiGLU/RMSNorm/Matmul 高级 API 源码，未发现三个独立算子库的 libopapi 与单算子头文件。对应算子从匹配 tag 的开源实现继续追踪；SDK 版本与源代码可读不证明任意二进制同源、框架兼容或设备数值正确。本轮没有安装/更新虚拟机软件、编译、执行算子或仿真实验。

当前来源为 579 个固定公开输入、330 个 AST 函数证明、原后端 12 个 C++ 范围及 CANN 350 个范围；21 版/1,122 主干层/93 全局与独立组件/145 模板/76,502 步保持。七 sheet 工作簿含 134,933 个已逐值核对数据单元格，来源恢复支持 cann/。当前结果见研究复现说明、cann-operator-audit.json 和对应验证记录。

剩余边界是两个 Llama 原底座权限、上游权重值/共享证明、实际算子库与 ABI/运行分支验证、未观察到的 TileKernels 接线，以及暂缓的 P6；不能再把 CANN 来源概括成闭源或不可查。提交、推送与公开部署仍未执行，当前用于本地整体验收。

## 16 本次仅做静态研究与网站（当前验收范围）

用户确认当前没有硬件，并提出去掉真正上机验证这一环节。设备运行、模块数值、整模型加载、profiling、吞吐/显存/质量实测已移出本次范围，不再作为遗留、阻塞或完成门槛；未进行的实验也不会标为已通过。

本次验收覆盖 21 个 checkpoint 的结构/文件头/参数与 shape、MTP/DSpark、公开 BPE、固定源码与 CANN/package 静态核查、来源哈希和恢复、JSON/XLSX/CSV/ZIP 一致性及网站交互。已完成的静态数据与测试保持，性能字段继续为空或未知。

原实验路线仅保留为未来可选附录。网站的七个优化卡片与工作簿“实验与缺口”页改为可选后续研究及证据边界，状态为“不纳入本次验收”。未来有硬件且需要运行结论时，由用户单独发起，不自动恢复。

范围调整后，26 份 JSON 契约、全仓库构建、七 sheet 工作簿的 134,933 个数据单元格及 CSV/ZIP/20 项下载一致性校验通过。Chrome 21 组页面回归含新验收范围、14 个窄屏路径、WebGL 禁用回退及原有家族路径，未发现控制台或资源错误。

两个 Llama 原底座权限、权重值/训练共享关系，以及具体运行栈/ABI/设备数值仍是证据边界；本次公开静态范围按现有证据验收。提交、推送与公开部署继续按用户另行安排处理。

## 17 静态分析工具与系统研究扩展（2026-10-02 已完成）

用户已同意全部开展以下五项。第 16 节保存上一轮的完成范围；本轮复用固定模型结构、文件头、源码与来源哈希，新增分析资料与网站工具，不开展设备连接、权重加载、kernel、profiling 或 benchmark。

- [x] A1 静态计算与存储成本模型：覆盖 21 个 checkpoint，按 batch、query/history 长度和参考分支估算声明矩阵乘法、attention、cache 与主要中间张量；输出公式、条件、示例和网站计算器。逻辑读写量不作为物理 HBM 流量，张量容量不作为实际峰值显存。
- [x] A2 CANN 算子约束手册：对现有 30 个源码家族提取 API/shape/dtype/layout/量化/平台条件和版本差异；每条约束有固定源码位置。网页检查器只判定已编码的必要条件，不把通过静态检查当作设备支持或运行成功。
- [x] A3 并行与通信专题：按固定框架源码研究 TP/EP/DP、专家分发/合并、负载均衡与 prefill/decode 分离；推导每 rank 的逻辑权重和通信量，保留路由分布、算法、padding、拓扑和后端条件。
- [x] A4 跨家族统一比较：将现有 DeepSeek、Kimi、GLM 规范化为同一比较入口，保留各家族来源、统计范围和核验深度；未知值不补零，模型能力和实测速度不排名。
- [x] A5 机制交互演示：逐步展示 MLA 吸收、DSA 选择、V4 压缩、mHC 和 DSpark；演示输入与中间结果可检查，教学数值不冒充真实模型权重或运行结果。

交付包含规范化 JSON、静态研究说明、网站页面和分析工作簿/CSV。新增公式用独立推导和小规模合成数据核对；新增来源可按固定哈希恢复；浏览器覆盖参数变化、边界输入、来源跳转、导出、窄屏、机制步骤与已有家族回归。完成后更新本节的实际范围与验证记录，再由用户统一验收。

本轮实际交付：21 版/1,122 层的成本模型、30 家族/4,207 条 CANN 表行与条件、六个并行专题和 21 份新增固定来源、三个家族的 42 个比较条目，以及五个机制的 25 个步进阶段。新分析工作簿五个 sheet、55,129 个数据单元格和四份 CSV 共 4,396 行核对一致；成本与并行分别提供 84/63 个固定样例。原七 sheet 工作簿和全层数据保留。

独立核对通过 672 个成本场景和 1,630 项数学/边界检查，来源恢复与 4,607 个引用范围通过；28 份 schema、全仓库构建、28 组 Chrome 交互、23 个窄屏路径及 WebGL 禁用回退通过，28 个下载项经 HTTP 大小/SHA-256 核对。当前模型结构没有扩大到新增 checkpoint，设备实验仍不在范围。

网站入口：`#/family/deepseek/cost`、`#/family/deepseek/contracts`、`#/family/deepseek/parallel`、`#/compare`、`#/family/deepseek/mechanisms`。说明与导出见 [静态分析说明](DeepSeek-静态分析.md)、[分析工作簿](DeepSeek-静态分析.xlsx) 和完整研究包。CANN 检查器只判定必要条件，未发现矛盾不代表设备支持；理论容量/收缩、通信假设和教学数值保留各自范围。
