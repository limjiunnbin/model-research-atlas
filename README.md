# 模型研究图谱

独立、本地优先的中文模型研究网站。动态模型家族入口收录 Kimi；另有独立的 OpenBMB 组织研究报告。无需安装 npm 包或下载权重，浏览器端不依赖 CDN。

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
- 18 个历史/公开版本条目，搜索、分支和应用筛选、任意版本并列比较。
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

- `data/catalog.json`：顶层家族目录。
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

研究快照：2026-09-19。已有资料从原任务复制，未移动或修改原文件。所有原始下载文件含哈希，模型卡与固定 revision 在网站来源页及报告内保留。

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
