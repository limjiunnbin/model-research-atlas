# 模型研究图谱

独立、本地优先的中文模型研究网站。主页并列展示 Kimi、GLM、DeepSeek 三个模型家族，以及 OpenBMB 组织研究，按各自范围浏览。无需安装 npm 包或下载权重，浏览器端不依赖 CDN。

## GLM 主线与昇腾 NPU 推理优化

研究日期：2026-09-30。覆盖 GLM-5、GLM-5.2 与 GLM-5.3-Flash 三个条目，全部通过华为 CANN 推理配方 [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 的固定快照 `96e5813` 研究。

- [家族研究主页](#/family/glm)（本地预览）
- `data/families/glm/family.json`：版本、技术主题与场景
- `data/families/glm/glm-5.3-flash-architecture.json`：GLM-5.3-Flash 45 层逐矩阵结构
- `data/families/glm/hardware.json`：8 个实现专题、2 类平台、7 项待验证优化、21 条固定来源

GLM-5.3-Flash 是 45 层混合注意力模型：34 层 KDA 线性注意力 + 11 层 DSA（全程 NoPE 的吸收式 MLA + k-pool 压缩 indexer），叠加 4 流 mHC 超连接与 288 选 8 MoE，昇腾 950DT 上按模块分配 HiF8 / MXFP8 / MXFP4 三档精度。

结构数据由实现源码中参数的声明形状推导，`scripts/build_glm53_architecture.py` 可重新生成，推导过程写在脚本里。**这不是权重文件头审计**：`payloadBytes` 全部为空。配方 README 的吞吐数字按原样引用并标注为配方自述，**本站未复现，也未运行任何 NPU 性能实验**；「Ascend 优化」页的全部条目都是待验证项，不是已测得的收益。

按配置默认值推导的参数总数为 313,326,811,966，与配方 README 自述的约 306B 相差约一层 MoE；若前 4 层为 Dense 则为 306,203,703,838，与自述一致。该差异在结构页与来源页明确保留，未替厂商选定其一。

## K3 定制训练、推理与 Runtime 调度

新增 [SGLang、vLLM、TorchTitan 对比与 K3 Runtime 方案](dist/assets/kimi/K3-SGLang-vLLM-TorchTitan与定制runtime方案.md)：围绕 KDA/MLA 混合状态、Latent MoE 专家通信和视觉路径，比较推理/训练扩展点，提出共享 K3 模型契约、上游 engine adapter、两级调度和分阶段验收。结论是复用 SGLang 或 vLLM 做推理、TorchTitan 做训练基线；不把源码支持误写为硬件实测或收敛验证。

## OpenBMB 主模型与技术演进

研究日期：2026-09-23。覆盖 88 个公开仓库，按主模型家族优先阅读，包含模型来源与团队贡献、昇腾 Ascend NPU 的算子和编译优化分析。

- [在线阅读与下载](https://limjiunnbin.github.io/model-research-atlas/reports/openbmb/)
- [仓库中的报告资料](dist/reports/openbmb/)
- [深度分析 Markdown](dist/reports/openbmb/01-深度分析.md)
- [完整报告包](dist/reports/openbmb/OpenBMB-研究报告包.zip)

单文件 HTML、总览、88 仓库附录与索引、配置摘录、来源凭据、校验结果和谱系图均在独立目录中。该报告不改变 Kimi 的动态模型结构与数据；工程分析不等于硬件实测。

## 运行

需要 Python 3；Node 仅用于可选的 JavaScript 语法检查。

```sh
python3 scripts/build.py
python3 scripts/serve.py
```

打开 http://127.0.0.1:8765 。可使用 `--port 8766` 指定其他端口。服务仅监听本机，不向局域网或公网开放。若本机没有 Python，可用任意静态 HTTP 服务托管 `dist/`。不要直接以 `file://` 打开，因为浏览器需要读取 JSON。

## 已实现

- 模型家族 → 家族研究主页 → 版本详情、历史与技术演进、研究资料。
- Kimi 18 个历史/公开版本条目，GLM 3 个配方研究条目，DeepSeek 22 个代表 checkpoint（21 版文件头基线 + V4.1 配置/源码研究）；支持搜索、分支/应用筛选与版本并列比较。
- 完整报告逐章/全文阅读，11 张图、30 份原始资料附件及 SHA-256，另有实现研究报告与来源 CSV，按需下载。
- 原生 WebGL 三维结构：旋转、缩放、选层、前后层、跳层、聚焦、整体/拆解、模块/矩阵逐级进入。
- 八个主线版本均支持配置结构；其中七个有完整权重头审计。K2-Base 使用同尺寸 Instruct 模板推导，明确注明不是 Base 文件头审计。
- 有已保存配置的 VL、Audio、Linear Instruct 提供配置级结构；未经核验的版本与 K1/K1.5 不伪造结构。
- 全部三维结构都有同数据二维明细；不开启 WebGL 也可选择层、模块和矩阵。

- 实现与算子、硬件与部署、Ascend 优化研究三个入口：8 个专题、7 类平台、7 项优先任务、42 条来源。
- 版本详情及3D模块直接跳转对应实现；当前没有 GPU/NPU 性能实测，不提供未经统一条件验证的速度/成本排名。

### 三维显示的边界

三维块表示逻辑层/模块，不是芯片、真实物理位置或真实权重值。MoE 展示一个代表专家及明确的专家总数/top-k，不绘制数万个独立专家。模块下的矩阵和量化元数据保留逻辑形状、打包形状、存储类型。总参数、激活线性参数代理、文件载荷分开列示。激活代理不是完整 FLOPs 或性能实测。仅在操作与调整尺寸时绘制，空闲不持续渲染；像素倍率上限 1.5。

## 目录与数据流

- `data/catalog.json`：顶层研究目录；`families` 提供动态模型家族，`reports` 提供独立组织研究报告入口。
- `scripts/build_glm_family.py` / `build_glm_report.py` / `build_glm_hardware.py` / `build_glm53_architecture.py`：GLM 家族数据的生成脚本，全部事实与固定快照来源写在脚本内，可重跑复核。
- `data/families/<id>/family.json`：家族文案、版本、技术主题、场景、图与附件。
- `data/families/<id>/report.json`：已审核的章节 HTML，原报告文字完整保留。
- `*-layers.json`：主线逐层权重审计。
- `*-architecture.json`：三维/二维共用结构接口，按版本进入详情时加载。
- `configs/`：原始配置快照。
- `hardware.json`：实现专题、硬件支持、优化任务、实验协议与固定来源；`scripts/research_exports.py` 从同一数据生成 Markdown/CSV 并更新下载哈希。
- `dist/`：可直接托管的静态网站。HTML/CSS/JS 为源文件，直接作为静态网站源文件跟踪。
- `scripts/build.py`：验证引用、证据状态、层数、视觉层数、模块参数求和，并复制 `data/` 到 `dist/data/`。
- `schema.json`：数据契约；`ADDING_MODELS.md`：新增家族与版本说明。

大型 `全部张量清单.csv.gz` 从不作为启动依赖，只通过下载链接访问。报告与各版架构也是按需读取。

## 来源与维护

各专题保留自己的快照日期：Kimi 2026-09-19 起，GLM 2026-09-30，DeepSeek 基线 2026-10-01、V4.1 Flash 2026-10-08。已有资料从原任务复制，未移动或修改原文件。所有原始下载文件含哈希，模型卡与固定 revision 在网站来源页及报告内保留。

证据枚举：`official` 官方披露；`derived` 计算推导；`interpretation` 解释判断；`unknown` 未知。未知值为 JSON `null`，不填零或估计日期。产品模式、思考档位不自动视为新检查点。能力定位不是统一评测排名。

公共仓库：https://github.com/limjiunnbin/model-research-atlas

在线预览：https://limjiunnbin.github.io/model-research-atlas/

GitHub Actions 验证数据后将 `dist/` 发布到 GitHub Pages；所有资源使用相对路径，支持项目子路径和 hash 深链接。公共仓库从清理后的完整交付快照开始，本地工作历史不随首次发布上传。
# 逐层计算与接口扩展

入口：`#/family/kimi/compute`。8 个主线模型的 520 个语言层、108 个视觉层与 32 个全局组件映射到 53 个模板，CSV 将每层完整展开为 16,974 行。可按版本、组件、后端和关键词筛选；3D 组件面板链接到相应层。

`dist/assets/kimi/compute/` 提供 shape 总表、后端 API 总表、按模型拆分的 16 份 CSV ZIP、完整 JSON、Markdown 报告和字段说明。CSV 使用 UTF-8 BOM。`{i}` 表示从零开始的层号，`{e}` 表示专家编号；份数列给出完整专家数量。

源码固定到 2026-09-21 快照，按文件与函数体 SHA256 保留证据。API 可能是模块、分派器或条件设备实现，未知项明确标记；不是已验证的兼容运行栈。K2 Base 形状来自同配置 Instruct 模板，其他七版与原始文件头审计对齐。未进行 GPU/NPU 运行测试。

普通构建会核对全部原始张量、参数总数、配置形状、量化打包与缩放表及下载文件。第三方研究源快照不随仓库分发，可额外运行 `scripts/validate_compute.py --research-root <snapshot-directory>` 复验 AST 函数、调用与语句证据。结果保存在 `compute-validation.json`。

## 第三方资料与授权

配置快照、模型名称、论文与实现来源链接保留其原始来源及权利归属。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本仓库公开可读不等于对全部内容授予开源许可；未为第三方材料另行授权。

## 文档阅读与 OpenBMB 逐层数据

站内 Markdown 链接提供在线阅读与原始文件下载，阅读器支持相对图片/链接、代码块与表格。OpenBMB 下载页提供17个模型的逐层配置、逻辑shape/计算步骤、后端映射与分模型ZIP；专用后端未核验项明确保留未知。构建使用 `scripts/openbmb_exports.py` 从固定配置重建，非权重头或硬件审计。


## 独立模型 Excel 下载中心

入口 `model-documents.html`。156个具体模型/版本各有独立XLSX、算子CSV、参数CSV及在线预览，支持搜索与家族/覆盖筛选。122个条目有配置层展开；34个资料缺口条目明确标识，不能算逐层完成。K3增加模板对照sheet、逐单元格CSV与可筛选差异预览。模板原文件未发布、未覆盖。

独立数据与来源审查完成；缺口与未实测项仍明确保留。审查基线 `6613c6f584ab41eec0eed8c4c3981032595b66e5`，已关闭A01–A30；34配置缺口、K3的1932项待核验、HerculesBench证据不足及设备未实测不在完成声明内。每份保留“算子表”“模型参数”以及原21列。S为verify长度，B/S/TP/EP是场景输入；配置层包括可选模块，不能直接等同实际运行次数。配置/数学推导、源码路径、权重存储与设备实测分别记录，未实测性能不填猜测值。运行 `python3 scripts/validate_model_documents.py` 检查文件哈希、CSV行数、ZIP及K3语言93层覆盖。

## DeepSeek 静态研究与网站

当前覆盖 22 个代表 checkpoint：[V4.1 Flash 独立研究](research/deepseek/v4.1-flash.md) 已接入版本、40 个语言层/32 个视觉层/3 个 DSpark stage 的结构、五个实现专题、报告、下载和跨家族比较。新增部分为固定配置/源码/论文研究，未审计权重存储；CED、CSA2、Engram、原生视觉、Single-Pass mHC、DSpark、缓存与阶段口径分开，旧 21-model 的 FLOPs/通信公式不套用到本版；新增独立缓存页按真实 NVFP4 变体、MXFP4 和 MXFP8 打包容量计算。

21-model 基线研究日期：2026-10-01。已完成选定范围的 P0–P5 及静态缺口补齐，入口为 `#/family/deepseek`、[完整报告](dist/assets/deepseek/DeepSeek-研究报告.md)、[复现说明](research/deepseek/README.md) 与 [roadmap](roadmap_deepseek.md)。覆盖 V2、V3/Base、原始 R1/Zero、六版 Distill、V3.2/Exp、V4 Flash/Pro 的 Base 与后训练版，以及 Flash/Pro-DSpark、Flash-0731/Pro-0813，共 21 个 checkpoint。

21 份架构包含 1,122 个主干层和 93 个全局/独立组件，compute 用 145 模板展开 76,502 个步骤；prefill/decode、参考分支、逻辑矩阵、cache、mHC、MTP、DSpark 与来源分开记录。实现研究提供 11 专题、2 条件平台和 7 候选实验，附四张 SVG、[七 sheet XLSX](dist/assets/deepseek/DeepSeek-模型与算子.xlsx)、[全层 CSV](dist/assets/deepseek/DeepSeek-all-layer-shapes.csv)、[21 份分模型表](dist/assets/deepseek/DeepSeek-per-model-tables.zip) 与 [完整研究包](dist/assets/deepseek/DeepSeek-研究报告包.zip)。

全部 21 版完成各自 safetensors header/offset 审计，覆盖 1,457,766 个张量、172,146,958 字节捕获元数据，权重载荷读取为 0。逻辑参数、FP4/FP8 容器、scale、非训练表与存储副本分别核对。六个 MLA checkpoint 的独立 MTP 和四个 DSpark checkpoint 的三个 stage 已展开；DSpark 的并行 backbone、顺序 Markov、confidence 与服务验证/调度分别记录。

六个公开 Distill 的完整 tokenizer/BPE 与特殊 ID 已核对，四个 Qwen 原底座完成逐条 vocab/merge 比较；两个 Llama 原底座 gated 差异保留未知。开放 Ascend 后端进一步追踪到 C++ 注册/wrapper、host/tiling 与入口。CANN 已按本地 9.2.0-beta.2 SDK 和对应开源 tag 核对 30 个算子源码家族、333 个源文件与 350 个范围；此前将其笼统记作闭源的判断已纠正。当前没有硬件，用户已移除设备验证环节，本次按静态研究与网站验收；运行/数值和权重共享证明保持证据边界，不作为当前待办。

`python3 scripts/validate_deepseek_atlas.py` 检查当前研究、全层公式及导出，普通 build 已按家族分派。579 份来源字节与 330 个 AST 函数体及独立 CANN 范围复核、空缓存恢复、26 份 Draft 2020-12 schema、CSV/XLSX/ZIP、Chrome 交互与 20 个 HTTP 下载均有记录，见 [VALIDATION.md](VALIDATION.md)。[首轮包](research/deepseek/round1/README.md) 保留原 P0/P1 读取范围，当前状态看 family/architecture/compute/hardware，不由旧 verificationDepth 推断。

2026-10-02 完成 A1–A5 扩展：21 版成本计算器、30 家族 CANN 约束手册、并行/通信与 P/D 研究、42 条目跨家族比较，以及 MLA/DSA/压缩/mHC/DSpark 五个机制步进演示。新增 [静态分析说明](dist/assets/deepseek/DeepSeek-静态分析.md)、[分析工作簿](dist/assets/deepseek/DeepSeek-静态分析.xlsx) 和四份 CSV，28 项下载与主研究包同步。理论成本、静态必要条件和教学数据继续保留明确口径；最新验收见 roadmap 第 17 节及 VALIDATION.md 末节。
