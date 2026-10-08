# 来源与第三方权利

此仓库包含模型研究说明、派生统计、可视化与浏览器界面，以及从公开来源保存的模型配置和来源信息。未包含模型权重或下载的完整第三方实现代码。

- Kimi 模型、配置与官方报告：Moonshot AI；逐版本来源、检查点 revision 保留在 `data/families/kimi/`、报告及下载资料中。
- PyTorch、vLLM、vLLM Ascend、SGLang、FlashInfer、Flash Linear Attention、AITER 等实现：仅保存研究引用、接口名称、函数位置与校验摘要；各自的权利与许可由相应上游项目规定。
- 论文、模型卡、商标与引用资料仍归各自权利人所有。引用链接不表示原作者认可本项目。

- DeepSeek 当前研究：21 个代表 checkpoint 的配置、公开 tokenizer_config/元数据快照、派生结构/shape 和各自完整文件头摘要保存在 `data/families/deepseek/`，各版本来源、revision、SHA-256 与读取范围保留。V3 代码采用上游 [LICENSE-CODE](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/LICENSE-CODE)，模型使用条件见独立 [LICENSE-MODEL](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/LICENSE-MODEL)。V4、DSpark 挂接版与后续发布分别保留官方许可入口，不据配置相同或 V3 条件推广授权范围。R1-Distill 的 DeepSeek 许可与 Qwen/Llama 底座条件分别保留；gated 原底座未复制。
- DSpark 论文与 [DeepSpec 固定快照](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2)、Transformers v4.44.0 的 Qwen/Llama 参考类、vLLM/vLLM-Ascend 与 TileKernels 仅在仓库保存引用、接口、函数位置或 C++ 范围与哈希摘要，具体权利和许可见各自固定版本。完整参考源码、论文页面及公开 tokenizer.json 只作为仓库外的研究输入；仓库保存派生 BPE/特殊 ID 核对结果，没有再分发完整词表、权重或第三方实现。
- CANN ops-transformer、ops-nn、ops-math 的 v9.2.0-beta.2 源码与独立固定 Ascend/op-plugin 仅作为仓库外的研究输入；仓库保留目录、API、范围、revision、Git blob/SHA-256 和本地 SDK 文件哈希。具体许可见各自固定版本 LICENSE；没有再分发完整 CANN source、安装包、SDK 源码或二进制。来源可查不作为任意发布/再分发授权或设备兼容声明。
- A1–A5 算子手册保存所选 CANN API 文档的参数表行、条件摘录和 API 原型上下文，均带固定来源、章节和范围哈希。并行专题引用 vLLM 与 vLLM-Ascend 的固定文档和源码，额外 21 个输入见 analysis-sources.json；上游 [vLLM LICENSE](https://github.com/vllm-project/vllm/blob/bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f/LICENSE) 与 [vLLM-Ascend LICENSE](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/LICENSE) 独立于模型权重条件。教学数值由本项目生成，不包含训练样本、真实权重或运行结果。

本项目未对不拥有的材料授予新许可，亦未设置覆盖整个仓库的统一开源许可证。使用、修改或再分发第三方材料前，请查阅对应固定版本的上游许可及模型使用条件。

研究数据有明确快照日期，属于静态分析。接口可用性、许可证和产品规则可能随后变化。未核验的信息在资料内保留为未知；没有 GPU/NPU 运行、数值精度或性能实测结论。


## Markdown 阅读组件

站内文档阅读使用本地分发的 Marked 与 DOMPurify；版本、包完整性凭据见 `dist/vendor/versions.json`，对应原始许可保留在同目录 `*-LICENSE.txt`。它们的许可仅适用于各自代码，不改变报告或其他第三方资料的权利归属。
