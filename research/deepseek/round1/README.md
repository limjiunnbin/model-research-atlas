# DeepSeek 首轮研究包

本目录保留 P0/P1 首轮完成时的证据与读取范围。当前 17 版静态研究、V3 文件头和网站已另行完成，入口见 [当前研究复现说明](../README.md)。下文“未接入/未审计”等状态只指首轮快照。

[研究报告](report.md) 是阅读入口，[参数 CSV](v3-parameters.csv) 与 [代表层算子 CSV](v3-representative-operators.csv) 提供表格明细，[validation.json](validation.json) 保留实际检查记录。研究范围和后续顺序见 [roadmap_deepseek.md](../../../roadmap_deepseek.md)。

首轮唯一研究事实源为 [v3-round1.json](../../../data/families/deepseek/research/v3-round1.json)。[versions.json](../../../data/families/deepseek/versions.json) 记录代表 checkpoint、底座关系、材料入口与推理模式，[sources.json](../../../data/families/deepseek/sources.json) 记录固定 revision、访问日、原始内容哈希和读取范围。`tables.json` 是生成后的 CSV 数据中间文件，不手工维护。

这份数据使用首轮研究契约。首轮完成时未在网站 catalog 注册 DeepSeek，也未修改现有 Kimi 的计算验证分派；网站数据适配和回归验收在 P5a 开始。权重文件头、MTP 实际加载、生产后端与 GPU/NPU 实验分别保留未核验状态。

## 检查现有研究包

从仓库根目录运行：

```sh
python3 scripts/validate_deepseek.py
```

这会检查已保存输入、配置、参数、shape、版本关系、CSV 与独立代数复算。仓库外参考源码缺席时，记录 `sourceCodeReverified=false`，不冒称重新检查了源码内容。完整源码复核使用下一节恢复出的缓存。

## 恢复固定来源并重新生成

原始参考源码、模型卡、完整权重索引和论文 HTML 只保存在研究缓存中；仓库内保留配置、API 元数据快照、派生索引摘要、引用与哈希。`--restore` 使用已记录来源，逐文件核对哈希，不刷新为上游 main，也不下载权重分片。固定页面发生内容漂移时会报错，不静默接受新内容。

```sh
task_deepseek_cache=/tmp/deepseek-round1-sources
python3 scripts/collect_deepseek_round1.py --restore --cache "$task_deepseek_cache"
python3 scripts/build_deepseek_round1.py --research-root "$task_deepseek_cache"
```

CSV 由 `scripts/export_deepseek_round1.mjs` 从同一 JSON 生成。使用 Codex 的 `load_workspace_dependencies` 获取 bundled Node 与 node_modules，在一个独立的可写目录建立指向该 node_modules 的链接；从该目录用 bundled Node 调用导出脚本的绝对路径。脚本通过 Artifact Tool 写入并检查表格值，再输出 UTF-8 BOM/CRLF CSV；当前库没有文档化的 CSV 导出 API，CSV 序列化使用其已核验的 range values。无需为本仓库安装 npm 依赖。

```sh
# 以下路径变量使用 load_workspace_dependencies 返回的真实路径。
task_deepseek_repo=/absolute/path/to/model-research-atlas
task_deepseek_node=/absolute/path/to/bundled/node
task_deepseek_modules=/absolute/path/to/bundled/node_modules
task_deepseek_export=/tmp/deepseek-round1-export
mkdir -p "$task_deepseek_export"
ln -s "$task_deepseek_modules" "$task_deepseek_export/node_modules"
(cd "$task_deepseek_export" && "$task_deepseek_node" "$task_deepseek_repo/scripts/export_deepseek_round1.mjs")
python3 scripts/validate_deepseek.py --research-root "$task_deepseek_cache"
```

首次创建表格时还应遵循 Spreadsheets 技能的 artifact operation 标记流程；重复生成不重复发送创建标记。导出会在独立工作目录留下预览供核对，预览不属于交付数据。如果只复核研究 JSON，可以先运行验证器的 `--skip-exports`，但该记录的 `exportsChecked=false`，不能作为整包完成依据。

## 口径与已知缺口

参数来自配置加参考源码的逻辑推导，索引只核对张量名称和分片。存储 shape/dtype/payload 保持未知；FP8 block scale 的预期形状单列为推导。MTP 的论文数据流、官方近似参数与 demo 跳过行为分别记录，精确 unique 参数和共享 norm 别名尚未验证。

代表层表按层型、阶段和 attention 分支展开。初始 prefill 使用 start_pos=0，常规 decode 使用 L_q=1；每专家 N_e 是动态量，CSV 的 N_e=1 只是相容场景例子。包含共享位置 Key 的 unsqueeze/广播、按专家选择路由权重与 scatter 等中间步骤；表中的步骤顺序用于解释参考数据流，不是实测 kernel 时间线。

独立 MLA 复算使用固定种子的 FP64 合成小矩阵，只检查展开与吸收两种公式的代数关系，不验证 RoPE 实现、低精度舍入、checkpoint 或设备 kernel。具体检查数量和误差始终以 validation.json 为准。
