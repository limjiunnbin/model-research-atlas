# 验证记录

本文保留历次验收记录，最新 DeepSeek 记录见末节（2026-10-01）。验证对象为本地 HTTP 预览与保存资料；模型推理、设备正确性与性能另行记录。

## 数据与静态文件

- `scripts/build.py` 通过：1 个家族、18 个版本、11 份结构接口、769 个层/全局组件。
- 七个已审计主线版本的每一语言层：各模块逻辑参数之和等于原审计层总参数。
- K3：93 层，69 KDA + 24 MLA，第 93 层 MLA；四个多模态主线各有 27 视觉层。
- 30 个附件的文件大小、SHA-256 与 HTTP 200 均核对通过。原始资料未修改。
- JavaScript 两个模块均通过 `node --check`。
- 全文 14 个章节条目（导读、12 节正文、附录），浏览器实际显示 13 个 h2 与 10 张分析图；全家族总览另在画廊。
- 报告网页数据将嵌入图片改为引用原始 PNG，以保留正文并避免重复传输；原版 HTML 下载仍逐字保留。
- schema.json 已提供完整数据契约；构建执行自带的跨文件/核心结构检查，不声称运行了第三方 JSON Schema 校验器。

## 真实浏览器交互

在 Codex 内置浏览器完成：

- 顶层模型家族入口、家族研究、版本库、完整报告、来源页面正常显示。
- 搜索 INT4 得到 4 / 18；无结果状态可见；重置后主线筛选为 8 / 18。
- 选中 K2-Instruct 与 K3，比较表表头和数值正确联动。
- K3 WebGL 画布实际渲染，初始 125 个对象：93 语言层、27 视觉层、5 个全局组件。
- 第 3 层 KDA → 第 4 层 MLA，上一层/下一层与检查面板同步；跳至 93 层显示 MLA/MoE。
- 展开路由专家：896 选 16，单矩阵 3072×3584，打包 3072×1792 / U8；代表专家范围明确标注。
- K3 视觉第 27 层：QKV 4608×1024，输出 1024×1536。
- Thinking：384 选 8，INT4 的 I32 容器与 BF16 scale 分开显示。
- 使用屏幕坐标真实点击 3D 注意力模块，成功进入其矩阵表；并非仅用外部下拉列表替代画布交互。
- 方向键旋转和放大按钮使绘制次数由 5 增至 7；按需渲染生效。
- 二维结构表打开后选择第 93 层，三维检查面板同步显示 MLA/MoE。
- K1 显示结构未知、3D 不可用，不生成虚构层。
- 来源页搜索“逐层”得到正确 CSV，执行下载链接点击；另以 HTTP 与文件哈希核对实际内容。
- 控制台错误检查未发现错误。

## 响应式与修复

- 390×844 手机视口：版本搜索与比较页面宽度均为 390，无整页横向溢出。
- 发现完整报告网格最小宽导致溢出后，修改 min-width 与 minmax；复验 document.scrollWidth = innerWidth = 390。
- 发现 3D 窄屏取景与标签拥挤后，加入旋转包围盒自动取景、手机双列模块/矩阵排列；截图确认模块清晰分离并完成点击。
- 表格保留内部横向滚动以容纳精确维度；窄屏检查面板改为画布下方。
- 测试后恢复浏览器默认视口。

## 未声称完成的项目

未进行真实模型推理、巨型权重加载、硬件吞吐/显存测量、跨所有设备的性能基准或公众部署。WebGL 禁用时的回退路径已实现；本次浏览器未禁用 GPU，因此只实际验收了同数据二维操作，不声称验证了所有驱动失效场景。

## 实现与硬件研究扩展（2026-09-19）

- 8 个实现专题、7 类硬件平台、7 项 Ascend 优先任务、42 条来源；所有模型/专题/来源引用通过 build 检查。
- 固定 Moonshot、vLLM、vLLM-Ascend、SGLang、AITER 提交；3 份官方 HF 模型代码重新读取固定 revision，SHA-256 与所读文件一致。每个行号链接均核对在源码文件范围内。
- 报告 Markdown、来源 CSV 从 canonical hardware.json 生成；HTTP 下载成功且 SHA-256 与附件索引一致。
- 浏览器核验：实现页8专题；硬件表7行；Ascend页7任务；Kimi Linear 仅显示5个适用专题，不出现 K3-only LatentMoE/AttnRes。K3 3D KDA、MLA 与 LatentMoE 的动态专题链接正确。
- 390×844：实现页、硬件页、Ascend页正文宽度均390；硬件表在容器内横向滚动。修复长提交号引起的溢出；恢复默认浏览器视口。
- 三个 JS 模块语法检查通过；所测页面浏览器无 error 日志。
- 只完成资料、代码与网站验证，未运行任何 GPU/NPU 基准或量化质量复测；不存在可据此得出的跨厂商性能排名。
- A3 教程 vLLM0.27.1、构建默认0.28.0与 verified commit 分列；原始 MXFP4、转换 W4A8 及其他低精度部署明确区分。A5 仅作算子设计范围说明。
# 2026-09-23 逐层扩展验证

- 覆盖 8 版、660 组件、16,974 个展开步骤；K3 的 69 KDA / 24 MLA 层号与配置完全一致。
- 1,610,434 条原始张量记录（七个独立审计版本）逐张量名称、存储 dtype/shape、逻辑 shape、参数量及专家份数对齐。Base 明确为同配置模板推导。
- 7,798 项配置独立形状检查；2,004 项 FP8 缩放表和 996 项 INT4/MXFP4 打包及缩放表检查。
- 133 处证据重新读取固定源码文件，校验文件 SHA256、AST 函数边界、函数内调用、函数/步骤语句 SHA256。
- 两份总 CSV 与 16 份模型 CSV 均检验行数、BOM、字段和参数汇总；ZIP CRC 与所有六个下载的大小和 SHA256 均通过。
- 设备未知映射按展开步骤统计：NVIDIA 5,743；AMD 5,930；Ascend 4,816。其余也可能只核验模块/分派层，不代表逐步独立 kernel 或运行兼容性已验证。

机器记录：`compute-validation.json`。构建执行离线数据验证；增加 `--research-root` 可重复源码核验。没有 GPU/NPU 数值精度、整模型加载或性能实测。

逐层页面人工浏览器验证：版本/层/后端筛选、Ascend关键词、空结果、INT4打包显示、K3视觉末层及连接器、第93层3D跳转、390px与桌面无横向溢出、6个下载入口及无浏览器错误。六个文件经HTTP 200完整读取并匹配本地SHA256；具体时间见机器记录。

## 2026-10-01 DeepSeek 首版静态研究与网站（17 版历史）

本节保存 17 版首版交付时的检查与缺口；当前完成状态见下一节。P0–P5 当时覆盖所选 17 个 checkpoint，P6 按用户要求暂缓。[复现说明](research/deepseek/README.md)、[roadmap](roadmap_deepseek.md) 与 `#/family/deepseek` 已更新为本轮结果，首轮历史包仍保留原读取范围。

- 独立公式核对 17 个模型的主干逻辑参数，914 个主干层、60 个全局/独立组件、18,087 个矩阵实例、96 模板与 42,476 个展开步骤通过。每阶段的权重引用与 architecture 一致；MTP、共享副本、FP8 scale 和非训练 hash 表分列。
- 原始 V3 全 163 分片、91,991 张量通过索引名称、完整连续 data_offsets、dtype 字节大小、shape、全部层/专家模板一致性核对。主干 671,026,419,200 参数；完整发布载荷 688,574,839,360 字节，含 MTP。文件头元数据 11,888,393 字节，权重载荷读取 0；索引声明 total_size 差异保留。
- 130 份公开输入逐字节 SHA-256、50 个仓库内快照（51 次来源引用）及 171 个 AST 函数体、位置与哈希复核。空缓存恢复相同固定来源后复验，并用恢复缓存生成当前架构/计算数据，未下载权重或完整第三方代码到仓库。
- 全层 CSV 42,476 数据行/22 列与 17 个分模型 CSV 逐行对照；UTF-8 BOM/CRLF、空 duration 与来源一致。五 sheet XLSX 的 52,498 个数据单元格逐值复核，数字类型保留；Artifact Tool 复算、错误检查和渲染预览完成。
- 分模型 ZIP 与报告 ZIP 每个条目 CRC/内容核对，15 个下载的大小与 SHA-256 一致；报告包包含规范化元数据、配置/tokenizer_config、来源、图与验证记录，不包含权重或完整第三方实现。
- jsonschema Draft 2020-12 检查 22 个文件通过；全仓库 build 通过，3 个家族、1 个独立组织报告、1,788 个结构组件。新增矩阵 nullable 存储与架构 cache 键保持旧数据兼容；build 明确分派 Kimi/DeepSeek 验证器。
- Chrome/Playwright 实际交互核对家族入口、17 版详情、搜索/空结果/重置、四版比较、3D 层/矩阵点选与旋转、V3 文件头/scale/MTP、DSA 与 mHC/FFN 专题跳转、阶段/分支/后端筛选、V3 MTP 61 与 V4 mtp.0、在线 Markdown 相对图片、三张 SVG、实际 XLSX 下载与全部附件 HTTP 200/哈希。
- 390×844 的版本、比较、结构、计算、实现、硬件、优化、报告、图与来源页无整页横向溢出；原生 Chrome 禁用 WebGL 后二维表自动展开，仍可选择末层与专家矩阵。修复旧 CSS 覆盖 hidden 属性的问题，同时隐藏空画布并禁用旋转/缩放按钮。
- 回归 Kimi-K3 的 93 层、末层 MLA 与原计算页，GLM 版本/详情、OpenBMB 入口；未改变这些家族的事实数据。当前页面检查没有控制台错误或失败资源。
- 扫描交付文本、XLSX XML 与 ZIP 条目，未发现私有机器地址、账号、SSH 参数或凭据。机器配置仅用于此前只读探测，没有进入研究数据。

机器记录：`data/families/deepseek/research/atlas-validation.json`、`source-validation.json`、`schema-validation.json`，以及 [browser-validation.json](research/deepseek/validation/browser-validation.json)。表格导出、header 与来源恢复有独立脚本，具体入口见复现说明。

没有 checkpoint 加载、设备 kernel、GPU/NPU 数值或性能实验。V3 之外的实际存储、V3 MTP shared_head.norm 加载别名、Llama 原底座 gated 差异、完整 tokenizer/BPE 与当时未继续核对的内核运行范围仍保留未知；源码快照没有被组成兼容运行栈。首轮 FP64 合成矩阵仅验证 MLA 代数等价，不能当作低精度或设备正确性结果。当前交付可整体验收，P6 待用户恢复。

V3-Base、R1/Zero、V3.2/Exp 的配置声明 MTP，但独立 MTP 权重与 forward 未展开，参数保持 unknown；当前全层完成口径为主干与已单列组件，不借用原始 V3 的 MTP 存储。

## 2026-10-01 DeepSeek 静态补齐与 DSpark（CANN 纠正前）

完成选定范围的静态缺口补齐与网站同步。新增 Flash/Pro-DSpark 和 Flash-0731/Pro-0813，当前共 21 个 checkpoint。原始 17 版 versions.json 与上节验证结果保留为历史，不用于推断最新进度。P6 继续按用户要求暂缓。

- 独立公式核对 21 版主干、MTP 与 DSpark 参数；1,122 主干层、93 全局/独立组件、33,507 个矩阵实例、145 模板与 76,502 展开步骤通过。五版缺失的 MLA MTP 已加入独立 forward/loader；SharedHead.norm 在所选参考中独立构造/加载，上游训练别名与权重值相同不由文件头证明。
- 各版本使用自身完整 safetensors header，覆盖 1,457,766 张量、172,146,958 字节捕获元数据，权重载荷读取 0。校验名称、dtype 字节、连续 offsets、每个层/专家模板与所有矩阵完整覆盖；I8 中的 FP4 打包、F8_E8M0 scale、I64 hash 表、共享 embedding/head 存储副本均分列。
- DSpark 的三个 stage、目标层特征、并行 backbone、顺序 Markov、原始 confidence logits、context cache 和 prefill/decode 分支分别有固定来源。HF nextn=1、native stage=3、checkpoint block=5 与服务示例 speculative_tokens=7 分别保存。协议的 FP64 合成概率核对验证 rejection identity 与前缀概率乘积，没有生成模型 token 或测性能；论文收益保留作者环境与口径。
- 六个公开 Distill 和四个 Qwen 原底座读取完整 tokenizer.json。四对 Qwen 的 151,643 个基础 vocab ID 与 151,387 个 merges 逐条一致，added tokens 差异单列；六版特殊 ID 均在词表和 embedding 范围内。两个原始 Llama 底座返回 gated/401，完整基线差异仍未知。
- 246 份固定公开输入、58 个仓库内快照（59 次引用）、330 个 AST 函数体、位置与 SHA-256 复核；完整空缓存恢复后复验通过。开放 Ascend compressor、hc_pre 与 grouped_matmul_swiglu 追踪到注册/wrapper、host/tiling 或入口，C++ 范围哈希与 AST 证明分开。未观察到 TileKernels 与所选框架直接接线，没有编译或执行内核。
- 76,502 行/22 列全层 CSV、21 个分模型 CSV 和七 sheet XLSX 的 132,269 个数据单元格逐值对照 canonical JSON。UTF-8 BOM/CRLF、数值类型与未测 duration 空值保持；Artifact Tool 复算、错误检查与七 sheet 渲染预览完成。
- 分模型/报告 ZIP 每个条目 CRC/内容及 19 个下载大小/SHA-256 核对。报告包包含规范化数据、全部 21 版 header 摘要、完整 BPE 派生核对、来源、四张 SVG 和验证记录；完整 tokenizer、代码、论文输入与 header 二进制留在仓库外缓存。
- Draft 2020-12 检查 26 份文件通过，Python 与前端 JS 语法及 git diff 空白检查通过。全仓库 build 通过：3 家族、1 独立组织报告、2,029 个结构组件，156 份既有模型独立文档和 OpenBMB 导出验证保持通过。
- Chrome/Playwright 20 组实际交互覆盖全部 21 版详情、DSpark 三 stage/Markov/confidence 与协议专题、搜索/比较、结构矩阵、阶段/分支/后端筛选、MTP 命名空间、九章报告、四张 SVG、Markdown 阅读器、实际 XLSX 下载和 19 个 HTTP 附件。390px 页面与 WebGL 禁用二维回退通过，Kimi/GLM/OpenBMB 回归通过，无控制台错误或失败资源。
- 交付文本、XLSX XML 与 ZIP 内容中的已知私有机器标识和凭据标记扫描通过，记录在 privacy-validation.json；本轮没有设备访问。

当前记录：[atlas-validation.json](data/families/deepseek/research/atlas-validation.json)、[source-validation.json](data/families/deepseek/research/source-validation.json)、[schema-validation.json](data/families/deepseek/research/schema-validation.json) 与 [browser-validation.json](research/deepseek/validation/browser-validation.json)。当前入口为 [复现说明](research/deepseek/README.md)、[roadmap 第 15 节](roadmap_deepseek.md#15-cann-开源算子与本地-package当前完成状态) 和 `#/family/deepseek`。

本节当时将 CANN 未继续核查部分概括成闭源，这一判断已按下节纠正。两个 Llama 原底座权限、上游训练共享/权重值证明、未观察到的 TileKernels 接线及 P6 设备实验仍保留；没有 checkpoint 加载、设备数值或性能结论。

## 2026-10-01 CANN 来源纠正与匹配版本核对（设备范围调整前）

按用户提供的 Multipass 安装完成只读 package 核对，并按 CANN v9.2.0-beta.2 开源 tag 继续追踪算子来源，撤回此前笼统的闭源判断。

- 本地 aarch64、CANN 9.2.0-beta.2、36 个组件版本记录与 6 个版本/高级 API 文件哈希。所检查路径有 AscendC SDK 源码，未发现三个独立算子库的 libopapi 与单算子头文件；没有安装/更新环境、编译或执行算子。
- 固定三个官方算子仓和独立 op-plugin revision，读取 30 个相关源码家族、333 份新增公开输入和 350 个源码范围。CANN 原始字节同时核对 Git blob SHA-1 和 SHA-256；API/host/tiling/kernel 与 op-plugin 条件集合有逐项固定链接。
- Python FIA v2 的所选源码按条件调用 aclnn V4/V5，MoE/GMM 多版本选择保留分支；SwiGLU 的生成 schema 明确指定 aclnnSwiGlu。相似命名不证明官方 MhcPre/Post、框架 custom-op 与 TileKernels 直接接线。
- 当前 579 个固定公开输入、330 个 AST 证明、原后端 12 个 C++ 范围和 CANN 350 个范围经恢复缓存复核；原 21 版 header、33,507 个矩阵实例、145 模板和 76,502 步保持。
- 来源更新纳入七 sheet 工作簿，134,933 个数据单元格对照 canonical JSON；CANN JSON、源码选择与本地包派生证据纳入报告包，完整第三方源文件/SDK/二进制留在仓库外。
- 全仓库 build、26 个 Draft 2020-12 契约、语法/空白与导出校验通过。Chrome 21 组交互覆盖 CANN 专题、源码链接、FIA v2 与 aclnn V4/V5 的区分、十章报告、20 个 HTTP 下载及原有结构/DSpark/旧家族回归；14 个 390px 路径和 WebGL 禁用回退通过，没有控制台错误或失败资源。
- 本地包再次只读检查 36 个组件版本文件与 6 个版本/高级 API 文件，42 个哈希均一致，未执行算子。交付文本、XLSX XML 和 ZIP 内容扫描没有发现已知私有机器标识、Multipass 实例/账号/网络信息或凭据。

本阶段状态记录在 atlas/source/schema/browser-validation 与 roadmap 第 15 节。源文件可读和 SDK 安装分别有证据，实际算子库、ABI、运行分支与设备数值当时留待 P6；设备范围随后按下一节移出本次验收，不把源代码存在当作设备实验。

## 2026-10-01 静态研究与网站验收（当前范围）

用户确认当前没有硬件，并将真正设备运行与性能验证移出本次范围。本次验收截止于静态研究、模型结构/文件头/shape/参数、MTP/DSpark、公开 BPE、CANN/source/package 只读核查、来源哈希/恢复、导出一致性及网站交互。

机器状态由 paused_by_user 改为 excluded_from_current_scope_by_user；family.acceptanceScope.deviceExperimentsRequired 与 hardware.experimentScope.required 均为 false。可选卡片和工作簿状态为“未来可选 · 不纳入本次验收”，不再作为未完成事项。未测量字段保持空值或未知，未执行的设备测试不标为通过。

范围调整后重新校验通过：26 份 Draft 2020-12 契约、全仓库构建、21 版结构/文件头/shape/来源、76,502 行 CSV、七 sheet XLSX 的 134,933 个数据单元格、两个 ZIP 与 20 项下载。Chrome 21 组交互检查覆盖新范围文字及七个可选卡片、14 个 390px 路径、禁用 WebGL 后的二维操作和既有家族回归，没有控制台错误或失败资源。

已有静态数据、范围哈希和检查证据保持；当前机器结果见 atlas/source/schema/browser-validation。未来实验仅在用户另行发起后开展，不自动恢复，也不影响本次完成判定。

## 2026-10-02 A1–A5 静态分析工具与系统研究（最新验收）

五项扩展完成并接入网站、报告、JSON、XLSX/CSV 和研究包。主研究仍为 21 个 checkpoint、1,122 主干层、76,502 展开步骤；新增分析不加载模型或恢复设备阶段。

- 成本模型覆盖所有主干矩阵的 52 个层型组，排除位置偏置表与 embedding lookup 的矩阵乘法计数。672 个场景以 canonical 矩阵逐项计数和 query 位置枚举独立核对；1,630 项检查涵盖压缩边界、合法/非法输入、ring 通信、MLA 代数、DSA mask、Sinkhorn、DSpark 前缀与 rejection distribution identity。
- CANN 手册覆盖 30 家族、4,207 个 API 表行/条件及入口上下文；9 个自动谓词仅核对已编码必要条件，成功子集保持 incomplete_static_check。版本、平台、参数组合、合并单元格和原型证据保留，不推定运行支持。
- 并行专题固定额外 21 个公开输入、453,116 字节，原始字节及 Git blob/SHA-256 经复核，并从空缓存恢复。分析引用共 4,607 个范围检查包含 CANN 表行、产品声明、必要条件和并行来源；六个专题保留进程组、权重份额、路由、collective、EPLB runner/量化与 P/D 边界。
- 跨家族比较对三个家族的 42 个既有条目逐事实复核，并检查输入模型摘要哈希。历史 null 保持未知，各版参数、辅助模块和载荷范围保留。
- 新分析工作簿五个 sheet、55,129 个数据单元格，四份 CSV 共 4,396 行与 canonical JSON 对照；84 个成本样例和 63 个并行样例保持原始输入与空的实测字段。原工作簿七个 sheet、134,933 个单元格与全层 CSV 的验证继续通过。
- 28 份 Draft 2020-12 契约、全仓库构建及语法/空白检查通过。Chrome 28 组交互覆盖新增计算器、算子筛选与必要条件、并行参数、跨家族筛选/六条目限制、五个机制的 25 个步骤、动态 JSON 下载，以及原有全部模型/家族/报告路径。23 个 390px 路径、WebGL 禁用二维回退和 28 项 HTTP 大小/SHA-256 核对通过；没有控制台错误或失败资源。

机器记录为 analysis-validation、analysis-math-validation、analysis-export-validation、schema-validation、atlas-validation 和 browser-validation。所有时间、吞吐、运行数值与实际峰值显存仍未测；教学数值和静态公式不作为设备实验。此前验收节保留历史日期和计数，当前范围见 roadmap 第 17 节。
