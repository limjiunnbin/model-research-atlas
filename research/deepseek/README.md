# DeepSeek 当前研究与复现说明

2026-10-01 完成选定范围的静态研究与网站接入，2026-10-02 继续完成 A1–A5 分析工具与系统研究。用户确认当前没有硬件，设备运行与性能验证已移出本次范围，不作为遗留或验收门槛。本目录的 round1 保留首轮历史证据；当前结果由 data/families/deepseek 的 family、21 份 architecture、compute、hardware、analysis、固定来源与验证记录驱动。

## 阅读与验收

先运行 `python3 scripts/serve.py --port 8766`，打开 `http://127.0.0.1:8766/#/family/deepseek`。可从版本库选择 V3、V3.2、V4 与 Distill 比较，再进入结构中的层、模块、矩阵和逐层计算。实现专题按版本过滤；来源页提供下载与哈希，Markdown 可在线阅读。

- [完整研究报告](../../dist/assets/deepseek/DeepSeek-研究报告.md)：版本、参数/缓存、V3 文件头、V3.2/V4、MTP/DSpark、R1/Distill、Ascend 条件与实验边界。
- [模型与算子 XLSX](../../dist/assets/deepseek/DeepSeek-模型与算子.xlsx)：版本概览、矩阵模板、算子模板、来源、实验与缺口、推测解码、Tokenizer核对七个 sheet；数字保留数值类型，未测 duration 留空。
- [全层 CSV](../../dist/assets/deepseek/DeepSeek-all-layer-shapes.csv) 与 [分模型表 ZIP](../../dist/assets/deepseek/DeepSeek-per-model-tables.zip)：22 列、76,502 个数据行，21 个模型各自展开。
- [Ascend 实现研究](../../dist/assets/deepseek/DeepSeek-implementation-hardware-ascend.md)：11 个专题、2 个条件平台、7 个未来可选研究方向。
- [CANN 算子核对](../../dist/assets/deepseek/DeepSeek-CANN-operators.json)：30 个匹配 tag 的算子源码家族、op-plugin bridge、350 个源码范围与本地 SDK 证据。
- [完整报告包](../../dist/assets/deepseek/DeepSeek-研究报告包.zip)：报告、表格、图、规范化 JSON、配置/tokenizer_config 快照、来源和验证记录；不包含权重或完整第三方实现代码。
- [静态分析说明](../../dist/assets/deepseek/DeepSeek-静态分析.md) 与 [分析工作簿](../../dist/assets/deepseek/DeepSeek-静态分析.xlsx)：成本、算子契约、并行/通信、跨家族比较与五个机制演示。
- [路线与阶段状态](../../roadmap_deepseek.md)：第 16 节保存基础静态交付，第 17 节记录当前 A1–A5 扩展与验收。

## 覆盖范围与计数

| 范围 | 当前结果 | 核验深度 |
| --- | --- | --- |
| V2、V3-Base、V3、R1-Zero、R1 | 5 个 checkpoint | 固定配置、官方 HF 参考类、逐层逻辑形状/计算和各自完整 header；后四版含独立 MTP |
| 原始 R1 六版 Distill | Qwen 1.5B/7B/14B/32B，Llama 8B/70B | 各版固定配置、Dense GQA 参考类、完整 header/tokenizer；Qwen 原底座 config/BPE 比较 |
| V3.2-Exp、V3.2 | 2 个 checkpoint | 配置与官方 inference 字节一致；DSA indexer、展开式 prefill、吸收式 decode |
| V4 Flash/Pro 的 Base 与后训练版 | 4 个 checkpoint | 配置、实际层排布、CSA/HCA/SWA、mHC、hash/score、实际混合存储与独立 MTP |
| Flash/Pro-DSpark 与 Flash-0731/Pro-0813 | 4 个 checkpoint | 三个 DSpark stage、Markov/confidence、目标特征、独立 cache 和服务协议 |
| 当前全层计算 | 1,122 主干层、93 个全局/独立组件、145 模板、76,502 展开步骤 | 每组件和每阶段权重声明与架构一致；不是实际 kernel 执行时间线 |
| 当前来源 | 579 份输入、330 个 AST 函数证明 | 固定 revision、文件/函数 SHA-256；58 个仓库内快照、59 次引用；CANN 原始字节还校验 Git blob SHA-1，350 个范围另记 |

这不是 DeepSeek 全产品目录。后续 R1/V4 发布、Coder/Math/VL/OCR、第三方量化和更多框架仍需新范围。原始 versions.json 的 verificationDepth 是 P0/P1 的历史读取状态，当前结构进度看 family/architecture/compute 与 site-proofs.json。

## 存储、逻辑与未知项

V3 审计全部 163 分片、91,991 个张量。BF16 316、F8_E4M3 45,808、F32 45,867；全部发布载荷为 688,574,839,360 字节，包含 MTP。记录的完整 header 元数据为 11,888,393 字节，权重载荷读取为 0。索引 total_size=1,369,062,772,000 字节与实际载荷分开保存。

V3 主干逻辑参数为 671,026,419,200，文件头已核对。MTP 存储包含共享 embedding/head 副本；当前参考参数口径排除明确共享副本并包含所选 vLLM 独立 SharedHead.norm，上游训练别名仍不由 header 证明。全部 21 版 stored shape/dtype/payload 均来自各自文件头；FP4 容器和量化 metadata 独立于逻辑参数。

普通 V4 Flash 主干 2 SWA + 21 CSA + 20 HCA，Pro 为 30 CSA + 31 HCA；其 compress_ratios 最后一个 ratio=0 是 MTP，不计入主干。四个 DSpark checkpoint 的末尾有三个 ratio=0，对应独立 draft stage。四流残差轴与 hidden 分开；非训练 tid2eid 表与量化 scale 排除逻辑权重参数。Base FP8 与后训练 routed FP4 的模块归属分别记录。

两个 Llama 原底座的公开 resolve 返回 gated/401，基线差异保留未知。六个公开 Distill 的完整 tokenizer/BPE 都已读取，四个公开 Qwen 原底座完成逐条 vocab/merge 比较。后端快照未组成兼容运行栈；CANN 匹配版本来源已可查，实际库/分支、ABI、权重值加载、设备数值和吞吐/显存/质量仍未验证。

## 复核已交付数据

仓库根目录运行以下命令，无需下载权重：

```sh
python3 scripts/validate_deepseek_atlas.py
python3 scripts/validate_deepseek_exports.py
python3 scripts/build.py
node --check dist/app.js
node --check dist/structure.js
node --check dist/compute.js
node --check dist/hardware.js
node --check dist/document.js
```

atlas 验证器检查 21 个独立参数公式、33,507 个矩阵实例、76,502 个展开步骤、MTP/DSpark、共享/元数据口径、各自 header 完整覆盖、CANN 固定范围、引用、CSV/XLSX/ZIP 与全部下载。CSV 逐行及 XLSX 134,933 个数据单元格都与 canonical JSON 比较。`--skip-downloads` 只用于生成中的临时检查，不能作为最终整包验收记录。普通 build 为 Kimi 与 DeepSeek 分派不同验证器，并核验已有模型下载和 OpenBMB。

完整 Draft 2020-12 契约检查使用 `python3 scripts/validate_deepseek_schema.py`，需当前 Python 环境安装 jsonschema；本次在独立 QA 环境检查 26 个文件通过，版本保存在 schema-validation.json。首轮历史检查另用 `scripts/validate_deepseek.py`，不能替代当前网站检查。

## 恢复固定来源与重新生成

五个来源 manifest 保存原始字节哈希和获取地址。restore 优先用仓库内相同字节的配置/元数据快照，其余只取固定公共文件；输入漂移报错，不刷新 main，不使用 gated 资源的第三方镜像，也不下载权重。CANN 输入放在 cann/ 缓存组，原始字节同时核对 Git blob SHA-1。

```sh
task_deepseek_source=/tmp/deepseek-public-inputs
python3 scripts/restore_deepseek_sources.py --research-root "$task_deepseek_source"
python3 scripts/restore_deepseek_sources.py --research-root "$task_deepseek_source" --verify-only
python3 scripts/build_deepseek_atlas.py --research-root "$task_deepseek_source"
```

V3 header 可单独重建；Range 返回必须是精确的 206/Content-Range，脚本拒绝服务器忽略 Range 后传输整个权重。已有缓存会重新检查每张量 dtype 大小与连续 data_offsets。重新请求 header 是公开元数据审计，仍不是权重加载或设备实验。

```sh
task_deepseek_header=/tmp/deepseek-v3-headers
python3 scripts/audit_deepseek_headers.py --cache "$task_deepseek_header" --index "$task_deepseek_source/round1/hf/DeepSeek-V3/model.safetensors.index.json"
```

CSV/XLSX 用 scripts/export_deepseek_atlas.mjs 从同一 JSON 生成。使用 Codex load_workspace_dependencies 返回的 bundled Node 与 node_modules，在独立临时目录建立 node_modules 链接，再用 Node 执行脚本绝对路径。Artifact Tool 写入、复算并逐值检查工作簿/range；CSV 从核验后的 range values 序列化为 UTF-8 BOM/CRLF。独立目录保存七个 sheet 的渲染预览；生成 XLSX 时遵循 Spreadsheets 技能的创建/编辑标记流程，同次导出迭代不重复标记。

```sh
# 变量填入实际仓库和 bundled runtime 路径。
task_deepseek_repo=/absolute/path/to/model-research-atlas
task_deepseek_node=/absolute/path/to/bundled/node
task_deepseek_modules=/absolute/path/to/bundled/node_modules
task_deepseek_export=/tmp/deepseek-atlas-export
mkdir -p "$task_deepseek_export"
ln -s "$task_deepseek_modules" "$task_deepseek_export/node_modules"
(cd "$task_deepseek_export" && "$task_deepseek_node" "$task_deepseek_repo/scripts/export_deepseek_atlas.mjs")
python3 scripts/package_deepseek.py
python3 scripts/validate_deepseek_atlas.py
python3 scripts/package_deepseek.py
python3 scripts/build.py
```

第二次 package 将最终验证记录纳入下载，避免发布生成中的 skip-downloads 记录。下载版 compute/family JSON 不包含自引用下载哈希；网站 canonical 数据保留下载清单。图表由 deepseek_figures.py 从相同配置生成 SVG，包内保留 figure-inputs.json。

## 浏览器复验与记录

启动本地 serve 后，使用安装了 Playwright 的 Node 执行 scripts/check_deepseek_site.mjs。本次使用 bundled Node/Playwright 与本机 Chrome；可用 `ATLAS_CHROME_PATH` 和 `ATLAS_PREVIEW_URL` 指定其他 Chrome/预览地址。脚本通过真实页面交互测试比较、3D 画布点选/旋转、阶段/分支/后端过滤、MTP 层号、下载、Markdown 相对图片、全部 21 个详情页、390px、原生禁用 WebGL 的二维回退和旧家族路径。

机器记录位于 data/families/deepseek/research 的 atlas-validation、source-validation、schema-validation 与 research/deepseek/validation/browser-validation.json；浏览器截图仅是页面 QA，不代表模型或设备实验。设备连接配置没有进入上述文件或报告包。当前验收包含静态研究、来源/公式/导出和网站，设备实验不在本次范围。hardware.protocol 仅供未来另行发起设备研究时参考，不自动启动。

V3-Base、R1/Zero、V3.2/Exp 已对应各自 MTP header、独立 vLLM forward 和 loader。SharedHead.norm 在所选参考中独立构造/加载；参数按文档共享 embedding/head 的参考口径统计，上游训练共享与权重值相同仍不由 header 证明。

## 当前轮：MTP、DSpark 与完整文件头

新增 Flash/Pro-DSpark 与 Flash-0731/Pro-0813，当前共 21 个 checkpoint、1,122 主干层、93 个全局/独立组件、145 模板和 76,502 展开步骤。全部 checkpoint 都有各自完整 header，覆盖 1,457,766 张量、172,146,958 字节捕获元数据（含先前 V3 记录），权重载荷读取 0；未再借用原始 V3 的存储事实。

DSpark 的实际三个 stage 位于 mtp.0/1/2 namespace；首层 main_proj/main_norm 融合目标特征，末层 Markov 两个 [V,R] 矩阵和 [1,H+R] confidence，各 stage 有独立 SWA context cache。普通 MTP 的 eh_proj/enorm/hnorm 不套给 DSpark。HF nextn=1、native n_mtp_layers=3、checkpoint block=5 和模型卡 vLLM speculative_tokens=7 分别保留。

Flash 的 Markov rank=256、target layers=[40,41,42]，Pro 为 rank=512、layers=[58,59,60]；独立 draft 参考参数分别为 19,845,850,983 与 77,498,408,103。native backbone 并行，Markov 修正按前 token 顺序采样；confidence 返回 logits，服务侧 sigmoid、校准、前缀选择与负载感知调度独立。协议、源码证明和冲突见 [MTP/DSpark 数据](../../dist/assets/deepseek/DeepSeek-MTP-DSpark.json)。论文性能只作为作者服务条件下的自述，不用于本站或 Ascend 收益推定。

[全文件头包](../../dist/assets/deepseek/DeepSeek-header-audits.zip) 含全部 21 版。六个公开 Distill 的完整词表/特殊 ID 已审计，四个 Qwen 底座与蒸馏的 151,643 个基础 token ID、151,387 个 merges 一致，added tokens 不同；两个原始 Llama 底座仍 gated，见 [完整 BPE 记录](../../dist/assets/deepseek/R1-full-tokenizer-audit.json)。早前 tokenizer_config 记录是历史读取层级，本轮并未停在该层级。

[开放后端追踪](../../dist/assets/deepseek/DeepSeek-backend-trace.json) 进一步对应 Python 分派、C++ 注册/wrapper、host/tiling 与入口；Cpp 范围哈希独立于 AST，未编译或运行。TileKernels 与 vLLM-Ascend 是独立证据，未观察到直接接线；CANN 匹配源码已可读取，实际分派、ABI 与设备数值仍未验证。

五个 manifest（sources、extended-sources、backend-sources、followup-sources、cann-sources）包含 579 份固定公开输入、58 个仓库内快照（59 次引用）与 330 个函数证明。restore_deepseek_sources.py 支持 followup/ 和 cann/；完整 header 使用 audit_deepseek_checkpoints.py，完整 BPE 使用 audit_deepseek_tokenizers.py。当前工作簿共七个 sheet；标准 schema 检查为 26 文件。旧 17 模型快照保留在 versions.json 和 roadmap 的首版记录，不能用来推断当前读取范围。

## CANN 来源纠正与本地包

CANN 相关缺口此前被笼统写为闭源，这一判断已撤回。现固定 ops-transformer、ops-nn、ops-math 的 v9.2.0-beta.2，对 30 个相关源码家族读取 API/host/tiling/kernel 文件，另外固定 Ascend/op-plugin 桥接源码。333 个新增公开输入核对 SHA-256 和 CANN Git blob SHA-1；350 个源码范围与 Python AST 分开验证。

本地 Multipass 只读检查确认 aarch64、CANN 9.2.0-beta.2、36 个组件版本记录和 6 个版本/高级 API 文件哈希。所检查 SDK 路径包含 AscendC SwiGLU/RMSNorm/Matmul 源码，未发现三个独立算子库的 libopapi 与单算子头文件；包内高级 API 不等于完整 aclnn 算子库。检查结果仅保存 vendor-relative 路径与哈希，不保存实例名、账号、主机或网络信息。

Python npu_fused_infer_attention_score_v2 在所选 op-plugin 按条件调用 aclnn V4/V5；MoE 与 grouped_matmul 也有版本/格式分支。官方 MhcPre/Post 与框架 custom HcPre/Post 是独立接线证据。实际库/分支、ABI、设备数值与性能仍未实测，设备研究不在本次验收范围；没有安装/更新虚拟机软件、编译或执行算子。

`python3 scripts/collect_deepseek_cann.py --sources-only` 恢复对应 tag 的来源；本地包可用 `--multipass-instance` 指定已授权实例做只读检查。当前网站提供 CANN 实现专题、十章报告和独立核对 JSON；roadmap 第 15 节记录 CANN 纠正，第 16 节定义最新验收范围。

## 当前验收范围

本次仅验收静态研究、源码/package 只读核查、模型结构与文件头、参数/shape 公式、公开 tokenizer/BPE、来源恢复、导出一致性和网站功能。设备运行、模型加载、ABI/数值与性能实测已按用户要求移出当前范围，不再列为等待恢复的阶段或未完成项。

七个优化卡片和工作簿“实验与缺口”页保留为未来可选研究与证据边界。未测 duration、吞吐等保持空值/未知；移出范围不代表已通过设备测试。未来有硬件且需要时，由用户另行发起。

## A1–A5 分析工具与系统研究（2026-10-02）

网站入口为 `#/family/deepseek/cost`、`#/family/deepseek/contracts`、`#/family/deepseek/parallel`、`#/compare` 和 `#/family/deepseek/mechanisms`。成本页支持版本、阶段、batch、query/history、参考缓存路径、logits、精度和单层范围，并下载当前假设的 JSON。并行页支持 TP/DP、冗余专家、权重容量策略、远端路由和 collective 次数。机制页覆盖 MLA、DSA、V4 压缩、mHC、DSpark 的 25 个教学步骤。

analysis.json 复用 21 份配置和全部主干矩阵，分为 52 个层型组，明确排除 APE 位置表与 embedding lookup 的矩阵乘法计数。有效数学位置与参考稠密路径/槽位上界分别计算；主 KV、index cache、两个 FP32 压缩状态、声明中间张量与输出分开。算子手册包含 30 个家族的 4,207 条 API 表行、补充条件和入口上下文；保留章节与合并单元格语义，当前 9 条自动谓词仅用于 RMSNorm、SwiGLU 和二维 Matmul 的必要条件检查，其余条件按原文核对。

579 份主研究输入保持，analysis-sources.json 另固定 21 份 vLLM/vLLM-Ascend 并行资料，已核对 453,116 字节并从空缓存恢复。六个专题覆盖进程组、每 rank 权重、dispatch/combine、collective、EPLB 和 P/D；MRv1/MRv2、量化格式和平台文档声明分开。跨家族比较保存三个家族的 42 个既有条目，逐事实保留 evidence/source 与参数范围，不为未知填零。

新工作簿包含成本场景、并行场景、跨家族比较、机制要点、算子条件五个 sheet；55,129 个数据单元格与 JSON 逐值对照，四份 CSV 共 4,396 行。成本/并行分别提供 84/63 个固定示例；场景编辑与重新计算使用网页工具。原七 sheet 工作簿与 76,502 行全层 CSV 保留。

```sh
python3 scripts/collect_deepseek_analysis.py
python3 scripts/collect_deepseek_analysis.py --verify-only
python3 scripts/build_deepseek_analysis.py
node scripts/check_deepseek_analysis.mjs
node scripts/export_deepseek_analysis.mjs
python3 scripts/validate_deepseek_analysis.py --sources --exports
python3 scripts/package_deepseek.py
python3 scripts/build.py
```

导出仍使用前述 bundled Node/Artifact Tool 工作目录和 Spreadsheets 标记流程。重建分析 JSON 后，需要重新生成数学场景和导出，不能把旧工作簿的值或 manifest 默认当作新结果。validate_deepseek_analysis.py 默认核对 canonical 引用；--sources 逐字节与范围核对仓库外的固定输入，--exports 对照每个 CSV/XLSX 值。正常 build 不要求完整第三方源码缓存，也不执行这些源码。

最新验证为 28 个 JSON 契约、672 个独立成本场景、1,630 项数学/边界检查，以及 28 组 Chrome 交互、23 个 390px 路径和 WebGL 禁用回退。28 个下载项经 HTTP 大小与 SHA-256 核对；设备时间、吞吐、数值与实际峰值显存仍未测。
