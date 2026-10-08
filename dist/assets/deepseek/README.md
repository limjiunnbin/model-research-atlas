# DeepSeek 当前研究与复现说明

当前资料覆盖 22 个选定代表版本：21 个 checkpoint 完成独立文件头、逻辑矩阵与计算流审计；DeepSeek-V4.1-Flash 完成固定配置、参考源码、论文和真实 FP4/FP8 部署缓存研究。V4.1 未做权重文件头审计，FLOPs 与通信也未套用旧版本公式。设备运行与性能实验已按用户要求移出当前范围，不作为遗留或验收门槛。

网站由 `data/families/deepseek` 的 family、architecture、compute、hardware 与 analysis 驱动。家族更新日期表示资料更新，各模型和附件保留自己的固定版本与核验深度。下载页按用途分类，并逐项说明覆盖范围、字节数和 SHA-256。

## 阅读入口与附件

在仓库根目录运行 `python3 scripts/serve.py --port 8768`，打开 `http://127.0.0.1:8768/index.html#/family/deepseek`。先查版本差异，再进入模型结构、逐层计算、实现专题和来源页。Markdown 附件可在线阅读，较大的 CSV/JSON 在主动打开或下载时读取。

| 资料 | 覆盖范围与用途 |
| --- | --- |
| [完整研究报告](DeepSeek-研究报告.md) | 前 11 章为 21 版基础结构与系统研究，第 12 章为 V4.1 Flash |
| [V4.1 Flash 专题](DeepSeek-V4.1-Flash-研究.md) | CED、CSA2、Engram、Single-Pass mHC、原生视觉、DSpark、缓存格式与实现边界 |
| [V4.1 逻辑矩阵 CSV](DeepSeek-V4.1-Flash-逻辑矩阵.csv) | 79 个组件、1,393 行参考逻辑矩阵；存储形状、dtype、权重载荷保持未审计 |
| [模型与算子 XLSX](DeepSeek-模型与算子.xlsx) | 21 版基础研究，7 个 sheet；不含 V4.1 专用结构 |
| [全层 CSV](DeepSeek-all-layer-shapes.csv)、[分模型表 ZIP](DeepSeek-per-model-tables.zip) | 21 版、76,502 个声明步骤；同一权重在不同阶段重复引用，不能逐行累加参数 |
| [完整文件头包](DeepSeek-header-audits.zip) | 21 版各自完整分片文件头；读取权重载荷为 0，不含 V4.1 |
| [静态分析说明](DeepSeek-静态分析.md)、[分析工作簿](DeepSeek-静态分析.xlsx) | 21 版成本/并行场景、43 个跨家族条目、机制与 CANN 条件；V4.1 部署缓存由网页独立计算 |
| [实现与 Ascend 研究](DeepSeek-implementation-hardware-ascend.md) | 16 个专题，其中 5 个对应 V4.1；固定源码与候选映射，不推定设备性能 |
| [CANN 源码核对](DeepSeek-CANN-operators.json)、[完整条件 CSV](DeepSeek-CANN-contracts.csv) | 30 个算子源码家族、4,207 条 API 条件；匹配 tag、桥接与本地 SDK 证据分开保存 |
| [完整研究包](DeepSeek-研究报告包.zip) | 当前说明、研究、表格、配置、固定来源与注明范围的基础数据；无权重或完整第三方实现代码 |
| [调研路线](DeepSeek-roadmap.md) | 开头为当前范围；已存在的旧阶段说明保留原始读取层级 |

V4.1 的真实部署缓存入口是 `#/family/deepseek/cost/v4.1-flash`，主页、详情页和旧成本工具都有直接链接。Main KV 为 NVFP4-like 变体（E2M1 + 每 16 元素 E4M3 scale，省略第二级全局 scale），sliding window 为 MXFP8（E4M3 + 每 32 元素 E8M0 scale），index 为 MXFP4（E2M1 + 每 32 元素 E8M0 scale）。容量包含数据与块 scale；参考代码的量化/反量化模拟另行说明。

## 当前核验深度

| 范围 | 结果 |
| --- | --- |
| V2、V3-Base、V3、R1-Zero、R1 | 5 个 checkpoint 的配置、参考构造与完整文件头；后四版另记 MTP |
| 原始 R1 六版 Distill | Qwen 1.5B/7B/14B/32B、Llama 8B/70B；各自 Dense GQA 构造、完整文件头与 tokenizer/BPE |
| V3.2-Exp、V3.2 | 2 版固定配置/源码，DSA indexer 与 prefill/decode 路径 |
| V4 Flash/Pro 的 Base 与后训练版 | 4 版实际层排布、CSA/HCA/SWA、四流 mHC、hash/score、混合存储及独立 MTP |
| Flash/Pro-DSpark、Flash-0731/Pro-0813 | 4 版三阶段草稿、Markov/confidence、目标特征、独立 cache 与服务协议 |
| 21 版基础结构与存储 | 1,122 个主干层、93 个全局/独立组件、33,507 个矩阵实例；1,457,766 个张量、172,146,958 字节捕获元数据 |
| V4.1 Flash | 40 个语言层、32 个视觉层、3 个 DSpark stage、两处 Engram；9 份固定输入、19 个 AST 函数证明；独立真实打包缓存公式 |
| 静态分析导出 | 5-sheet 工作簿、55,142 个数据单元格；4 份分析 CSV 共 4,397 行，包含 84 个成本与 63 个并行场景 |
| 来源与后端 | 基础 579 份输入、330 个 AST 证明；分析另固定 21 份并行资料；CANN 350 个范围另行核对 |

逻辑参数、激活参数、权重载荷、KV 缓存与残差轴分别统计。21 版的 stored shape/dtype/payload 均来自各自文件头；FP4 容器和 scale 不改变逻辑矩阵尺寸。MTP/DSpark 参数与主干分开，共享 embedding/head 副本不重复计入逻辑参数；文件头不能证明上游训练别名或权重值相同。V4.1 的约 552B 主干与 196B Engram 为官方舍入披露，参考构造的精确元素数另列。

六版公开 Distill 的 tokenizer/BPE 已读取，四个公开 Qwen 底座完成 vocab/merge 比较；两个 Llama 原底座 gated，差异保留未知。CANN 开源仓和 package 可查；本地 Multipass SDK 的只读结果保留版本、vendor-relative 路径与哈希，不保存机器连接信息。固定源码、二进制 ABI、设备数值与性能是不同核验层级，当前未执行模型加载或设备实验。

## 复核现有交付

以下命令只读取现有研究数据并输出校验结果，无需下载权重：

```sh
python3 scripts/validate_deepseek_atlas.py
python3 scripts/validate_deepseek_exports.py
python3 scripts/validate_deepseek_v41.py
python3 scripts/validate_deepseek_analysis.py --exports
node scripts/check_deepseek_analysis.mjs
python3 scripts/validate_deepseek_schema.py
python3 scripts/build.py
```

schema 检查需要所选 Python 环境安装 `jsonschema`，当前范围为 29 个 canonical 文件。核心校验默认不写新的历史记录；部分脚本提供 `--write-record`，只有显式指定才保存。浏览器校验截图默认使用系统临时目录。旧版本校验文件是其原有范围的快照，不能用其中的日期或模型数推断当前整包覆盖。

本地预览可用 `scripts/check_deepseek_site.mjs` 做真实 Chrome 交互检查；它需要 Playwright，可用 `ATLAS_PREVIEW_URL=http://127.0.0.1:8768/` 与 `ATLAS_CHROME_PATH` 指定环境。网站验收包含导航、结构/矩阵、版本和跨家族比较、各类筛选、文档图片、下载、手机宽度与 WebGL 回退；截图是网页 QA，不是模型或设备实验。

## 恢复固定输入与重新生成

基础五份 manifest 保存固定公共输入的获取地址、revision 与原始字节哈希。恢复优先复用仓库内同字节的配置/元数据，其余写入仓库外缓存；不刷新到上游 main、不下载权重，不用第三方镜像绕过 gated。CANN 原始字节另核对 Git blob SHA-1。V4.1 的 9 份输入和哈希在 `scripts/build_deepseek_v41.py` 中，缓存位于 `~/.cache/model-research-atlas/deepseek-v41`。

```sh
task_deepseek_source=/tmp/deepseek-public-inputs
python3 scripts/restore_deepseek_sources.py --research-root "$task_deepseek_source"
python3 scripts/restore_deepseek_sources.py --research-root "$task_deepseek_source" --verify-only
python3 scripts/build_deepseek_v41.py --fetch
python3 scripts/build_deepseek_atlas.py --research-root "$task_deepseek_source"
python3 scripts/collect_deepseek_analysis.py
python3 scripts/collect_deepseek_analysis.py --verify-only
python3 scripts/build_deepseek_analysis.py
python3 scripts/validate_deepseek_v41.py --sources
node scripts/check_deepseek_analysis.mjs --export-scenarios
```

`--export-scenarios` 生成当前工作簿需要的场景输入；普通数学校验不覆盖它。完整 header 和 tokenizer 元数据可分别由 `audit_deepseek_checkpoints.py` 与 `audit_deepseek_tokenizers.py` 重建。Header Range 必须精确返回 206/Content-Range，脚本拒绝服务器忽略 Range 后传输整份权重。

CSV/XLSX 从同一 canonical JSON 导出。使用 Codex `load_workspace_dependencies` 返回的 bundled Node/Artifact Tool 路径，在仓库外建立运行目录；工作簿编辑遵循 Spreadsheets 技能标记流程，同次迭代不重复标记。渲染预览保存在该临时目录。

```sh
# 填入实际仓库和 bundled runtime 路径。
task_deepseek_repo=/absolute/path/to/model-research-atlas
task_deepseek_node=/absolute/path/to/bundled/node
task_deepseek_modules=/absolute/path/to/bundled/node_modules
task_deepseek_export=/tmp/deepseek-atlas-export
mkdir -p "$task_deepseek_export"
ln -s "$task_deepseek_modules" "$task_deepseek_export/node_modules"
(cd "$task_deepseek_export" && "$task_deepseek_node" "$task_deepseek_repo/scripts/export_deepseek_atlas.mjs")
(cd "$task_deepseek_export" && "$task_deepseek_node" "$task_deepseek_repo/scripts/export_deepseek_analysis.mjs")
python3 scripts/package_deepseek.py
python3 scripts/build.py
python3 scripts/validate_deepseek_analysis.py --sources --exports
```

`export_deepseek_atlas.mjs` 只导出 21 版基础工作簿，V4.1 逻辑矩阵由本版 builder 独立生成。完整分析重建后需重新导出工作簿和 manifest；仅跨家族表更新时可用 `export_deepseek_analysis.mjs --comparison-only` 保留其他 sheet。`package_deepseek.py` 会接入当前 V4.1 资料，`build_deepseek_v41.py --package` 可只刷新已有整包与下载哈希。下载 JSON 省略自引用下载哈希，网站 canonical 数据保留完整清单。

第三方材料的许可与分发边界见仓库 `THIRD_PARTY_NOTICES.md`。源码、PDF、导出预览与中间文件放在仓库外；正常校验不增加仓库内历史台账。当前验收为静态研究、来源/package 只读核查、模型结构与存储、公式和导出一致性、网站功能，设备实验无需等待恢复。
