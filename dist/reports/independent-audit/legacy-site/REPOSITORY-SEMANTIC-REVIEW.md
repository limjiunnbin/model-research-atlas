# OpenBMB 88 仓库逐项语义审查

被审提交：`6613c6f584ab41eec0eed8c4c3981032595b66e5`。88/88 条目均人工对照具体原始 README、已保存源码或针对性追加的固定提交文件。索引 7 字段×88 = 616 段对应叙述均核查，Markdown/HTML 镜像全文对照另行程序验证。

“已审查”表示站内具体主张有相应官方原文/配置/源码并检查其边界；不等于官方性能、能力或安全宣传已经复现。“原始证据不足”表示条目含明确尚无充分原文的事实缺口。每项的普通 LLM 对比、场景/局限及依赖关系中的合理推广保留为工程分析，不将其标成实验事实。

组织 API 的仓库归属不代表 OpenBMB 是唯一作者；fork 来源、基础模型归属、工具/数据/应用与模型本体的区别是本次逐项重点。HerculesBench 的不足已经由原附录公开说明，本审未发现这一说明构成误导。

结果：{'已审查': 87, '原始证据不足': 1}；镜像/证据定位检查错误 0。

## 逐项结果

### 1. AceBench · 已审查

128 个可执行任务、100 个隐私标注任务、OpenClaw harness、utility/cost/privacy 及 Pass³ 三次一致性均有原文。属于基准，不是新模型。与 ClawXRouter 的关系仅作目标相近的工程分析，没有声称集成。

站内定位：[索引第 7 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L7)；类别：评测。

原始证据：[README.md L15–55](https://github.com/OpenBMB/AceBench/blob/9a17bc2c7ee3fab9ca023036b82a81898512a001/README.md#L15-L55)。

### 2. AgentCPM · 已审查

Explore 的 Qwen3-4B-Thinking-2507 和 Report 的 MiniCPM4.1-8B 归属区分正确；AgentDock/AgentRL/AgentToLeaP 是配套工具，不是模型。UltraRAG2.0/vLLM/Milvus/llama.cpp 来自部署原文；性能及隐私只是官方主张，附录已保留条件。

站内定位：[索引第 33 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L33)；类别：模型与Agent。

原始证据：[README.md L22–55](https://github.com/OpenBMB/AgentCPM/blob/4a43561e790c154292798b3edd50171f71241cec/README.md#L22-L55)；[README.md L147–164](https://github.com/OpenBMB/AgentCPM/blob/4a43561e790c154292798b3edd50171f71241cec/README.md#L147-L164)；[README.md L34–34](https://github.com/OpenBMB/AgentCPM/blob/4a43561e790c154292798b3edd50171f71241cec/AgentCPM-Explore/README.md#L34-L34)；[README.md L259–429](https://github.com/OpenBMB/AgentCPM/blob/4a43561e790c154292798b3edd50171f71241cec/AgentCPM-Explore/README.md#L259-L429)；[README.md L7–7](https://github.com/OpenBMB/AgentCPM/blob/4a43561e790c154292798b3edd50171f71241cec/AgentCPM-Explore/AgentRL/README.md#L7-L7)；[README.md L37–54](https://github.com/OpenBMB/AgentCPM/blob/4a43561e790c154292798b3edd50171f71241cec/AgentCPM-Report/README.md#L37-L54)。

### 3. AgentCPM-GUI · 已审查

8B MiniCPM-V 骨干、SFT/RFT、JSON 动作及 0–1000 坐标有官方依据。CAGUI 被 README 明确列为评测数据；附录只写“上游…CAGUI 数据”，不能进一步据此推定它是训练集。界面变化和误差传播属工程判断。

站内定位：[索引第 87 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L87)；类别：模型与Agent。

原始证据：[README.md L23–30](https://github.com/OpenBMB/AgentCPM-GUI/blob/2168ae21b1bed1cdb88736d422934825795a9fd7/README.md#L23-L30)；[README.md L106–139](https://github.com/OpenBMB/AgentCPM-GUI/blob/2168ae21b1bed1cdb88736d422934825795a9fd7/README.md#L106-L139)；[README.md L231–293](https://github.com/OpenBMB/AgentCPM-GUI/blob/2168ae21b1bed1cdb88736d422934825795a9fd7/README.md#L231-L293)。

### 4. AgentVerse · 已审查

task-solving/simulation 两框架、release-0.1 稳定 simulation 分支、模型 API/本地服务与任务环境对应。多智能体协作框架而非新基础模型；增加成本与不保证准确率属于合理工程推断。

站内定位：[索引第 117 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L117)；类别：Agent框架。

原始证据：[README.md L47–56](https://github.com/OpenBMB/AgentVerse/blob/f90c4bd9680fdd3bcff8c52c9170911a59b23478/README.md#L47-L56)；[README.md L241–398](https://github.com/OpenBMB/AgentVerse/blob/f90c4bd9680fdd3bcff8c52c9170911a59b23478/README.md#L241-L398)。

### 5. AppCopilot · 已审查

多模态、多 Agent、分层规划、跨应用跨设备流程来自概述；ADB、Android 和 vLLM 服务在安装运行节明确。附录正确说明设备客户端不能证明全部本地推理。可靠性限制为工程推断。

站内定位：[索引第 155 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L155)；类别：Agent应用。

原始证据：[README.md L13–15](https://github.com/OpenBMB/AppCopilot/blob/c23ab94bbcda95e82a7bc0e2798d5640cdc9b373/README.md#L13-L15)；[README.md L101–249](https://github.com/OpenBMB/AppCopilot/blob/c23ab94bbcda95e82a7bc0e2798d5640cdc9b373/README.md#L101-L249)。

### 6. ArcLight · 已审查

C/C++、统一内存、CPU ARM/x86、GGUF、NUMA 与跨节点 TP 均有原文；MiniCPM5 模型实现有代码。未误将尚在 TODO 的跨 NUMA pipeline parallelism 写成已支持。速度条件明确归为官方环境。

站内定位：[索引第 181 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L181)；类别：推理基础设施。

原始证据：[README.md L16–28](https://github.com/OpenBMB/ArcLight/blob/558ef06dede9668c9560b87fe62b8eb2a8710d0f/README.md#L16-L28)；[README.md L72–105](https://github.com/OpenBMB/ArcLight/blob/558ef06dede9668c9560b87fe62b8eb2a8710d0f/README.md#L72-L105)；[minicpm5.cpp L4–54](https://github.com/OpenBMB/ArcLight/blob/558ef06dede9668c9560b87fe62b8eb2a8710d0f/models/minicpm5.cpp#L4-L54)；[scheduler.h L1–10](https://github.com/OpenBMB/ArcLight/blob/558ef06dede9668c9560b87fe62b8eb2a8710d0f/nnml/include/scheduler.h#L1-L10)。

### 7. BMCook · 已审查

蒸馏/剪枝/量化/MoEfication 与 CookTrainer 钩子有对应；结构/非结构剪枝和 BMTrain/ModelCenter 关系有源码及文档依据。README 136 原文是不建议同时使用 MoE 与 distilling，附录表述为组合限制，未声称硬性代码禁止。

站内定位：[索引第 215 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L215)；类别：训练与压缩。

原始证据：[README.md L30–53](https://github.com/OpenBMB/BMCook/blob/9f32ec7e2250ce0deea96db9b2e753d2d7145b82/README.md#L30-L53)；[README.md L126–172](https://github.com/OpenBMB/BMCook/blob/9f32ec7e2250ce0deea96db9b2e753d2d7145b82/README.md#L126-L172)；[trainer.py L33–136](https://github.com/OpenBMB/BMCook/blob/9f32ec7e2250ce0deea96db9b2e753d2d7145b82/bmcook/trainer.py#L33-L136)；[README.md L9–17](https://github.com/OpenBMB/BMCook/blob/9f32ec7e2250ce0deea96db9b2e753d2d7145b82/bmcook/pruning/sprune/README.md#L9-L17)；[README.md L2–4](https://github.com/OpenBMB/BMCook/blob/9f32ec7e2250ce0deea96db9b2e753d2d7145b82/bmcook/moefication/README.md#L2-L4)。

### 8. BMInf · 已审查

wrapper 替换 Linear/ModuleList、CPU 保留参数与拒绝嵌套 ModuleList 均可直接从源码核对。属于旧推理及内存管理工具；兼容、传输成本与精度限制是工程推断，未假设支持全部现模型。

站内定位：[索引第 257 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L257)；类别：推理基础设施。

原始证据：[README.md L27–40](https://github.com/OpenBMB/BMInf/blob/a9aec55b1d019069762420b002d39ccccb54f577/README.md#L27-L40)；[README.md L76–115](https://github.com/OpenBMB/BMInf/blob/a9aec55b1d019069762420b002d39ccccb54f577/README.md#L76-L115)；[wrapper.py L1–42](https://github.com/OpenBMB/BMInf/blob/a9aec55b1d019069762420b002d39ccccb54f577/bminf/wrapper.py#L1-L42)。

### 9. BMInf-demos · 已审查

CPM2.1 填空、CPM1 故事、EVA 对话及 Docker/Web 使用方式全部与原文一致；属于示例，没有被误写成新模型。兼容性及生产能力限制为工程判断。

站内定位：[索引第 291 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L291)；类别：示例与文档。

原始证据：[README.md L3–23](https://github.com/OpenBMB/BMInf-demos/blob/0a55d22905515c87656d1adb790a8d54017b828e/README.md#L3-L23)。

### 10. BMList · 已审查

公开模型目录、至少 1B 门槛、按机构/语言/领域组织有原文；没有把清单说成性能排名。历史覆盖限制是工程判断。

站内定位：[索引第 317 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L317)；类别：资料与治理。

原始证据：[README.md L11–31](https://github.com/OpenBMB/BMList/blob/491a9c392463949d398bb75e8d64a30914fcb94d/README.md#L11-L31)。

### 11. BMPrinciples · 已审查

原文定位为汇集训练/规模现象和研究条目，附录将它归为资料而非模型或普遍定律，范围准确。论文结论的适用限制是审慎分析。

站内定位：[索引第 343 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L343)；类别：资料与治理。

原始证据：[README.md L4–27](https://github.com/OpenBMB/BMPrinciples/blob/b8c8826e5ea400ac496a377740225d9809a3e320/README.md#L4-L27)。

### 12. BMTools · 已审查

LangChain/ChatGPT-Plugins 来源、Python 工具构建和 XAgent 后续开发入口均准确；工具控制流程未当作模型架构。当前协议兼容性的限制属于工程判断。

站内定位：[索引第 373 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L373)；类别：Agent框架。

原始证据：[README.md L31–36](https://github.com/OpenBMB/BMTools/blob/85869fc6e26616508f5dfafa551546993ab35f52/README.md#L31-L36)；[README.md L79–190](https://github.com/OpenBMB/BMTools/blob/85869fc6e26616508f5dfafa551546993ab35f52/README.md#L79-L190)。

### 13. BMTrain · 已审查

DistributedParameter、Block/TransformerBlockList、ZeRO、计算通信重叠及 1.0 张量并行均有文档/代码。训练库与模型分开，性能没有复测；不随意混用通信机制属使用边界。

站内定位：[索引第 403 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L403)；类别：训练与压缩。

原始证据：[README.md L31–43](https://github.com/OpenBMB/BMTrain/blob/36a44fe525342f84b082682aaabbe30ae5e22a14/README.md#L31-L43)；[README.md L79–191](https://github.com/OpenBMB/BMTrain/blob/36a44fe525342f84b082682aaabbe30ae5e22a14/README.md#L79-L191)；[UPDATE_1.0.0.md L1–50](https://github.com/OpenBMB/BMTrain/blob/36a44fe525342f84b082682aaabbe30ae5e22a14/docs/UPDATE_1.0.0.md#L1-L50)；[block_layer.py L1–121](https://github.com/OpenBMB/BMTrain/blob/36a44fe525342f84b082682aaabbe30ae5e22a14/bmtrain/block_layer.py#L1-L121)。

### 14. ChatDev · 已审查

2.0 DevAll 当前主线与 1.0 legacy 分支的区别正确；DAG 层内并行有原保存源码。另独立读取相同固定提交的 dynamic edge/cycle executor 与 Thinking 文档（4 个补充文件之一组）并核 Git blob SHA，动态边、循环、memory/thinking/tool/provider 组件均得到源码或官方文档支持。没有将框架称为新模型，运行效果未实测。

站内定位：[索引第 441 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L441)；类别：Agent框架。

原始证据：[README.md L19–23](https://github.com/OpenBMB/ChatDev/blob/4fb2db0ea90375ce1059f44fe03ffbd191a7a169/README.md#L19-L23)；[README.md L267–280](https://github.com/OpenBMB/ChatDev/blob/4fb2db0ea90375ce1059f44fe03ffbd191a7a169/README.md#L267-L280)；[dag_executor.py L1–55](https://github.com/OpenBMB/ChatDev/blob/4fb2db0ea90375ce1059f44fe03ffbd191a7a169/workflow/executor/dag_executor.py#L1-L55)；[dynamic_edge_executor.py L1–86](https://github.com/OpenBMB/ChatDev/blob/4fb2db0ea90375ce1059f44fe03ffbd191a7a169/workflow/executor/dynamic_edge_executor.py#L1-L86)；[cycle_executor.py L14–130](https://github.com/OpenBMB/ChatDev/blob/4fb2db0ea90375ce1059f44fe03ffbd191a7a169/workflow/executor/cycle_executor.py#L14-L130)；[thinking.md L1–9](https://github.com/OpenBMB/ChatDev/blob/4fb2db0ea90375ce1059f44fe03ffbd191a7a169/docs/user_guide/en/modules/thinking.md#L1-L9)。

### 15. ClawXMemory · 已审查

Markdown 持久记忆、SQLite 运行状态、index/Dream/recall 与原文一致；模型外记忆区别于 KV cache 正确。EdgeClaw 兼容关系另由其 README 交叉佐证；长期效果及误召回为工程推断。

站内定位：[索引第 475 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L475)；类别：Agent框架。

原始证据：[README.md L32–47](https://github.com/OpenBMB/ClawXMemory/blob/bcd66c5d8611413ad29354819b448e20dd51d480/README.md#L32-L47)；[file-memory.ts L110–174](https://github.com/OpenBMB/ClawXMemory/blob/bcd66c5d8611413ad29354819b448e20dd51d480/clawxmemory/src/core/file-memory.ts#L110-L174)。

### 16. ClawXRouter · 已审查

OpenClaw/EdgeClaw 来源、S1 云/S2 脱敏云/S3 本地、规则+LLM 分类及复杂度路由有原文。未将 README 安全口号当实证；误判和完整轨迹核验为工程分析。

站内定位：[索引第 505 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L505)；类别：Agent框架。

原始证据：[README.md L49–59](https://github.com/OpenBMB/ClawXRouter/blob/6acac9f8ae1a8c2c236b84deb68b5df4d2d6336f/README.md#L49-L59)；[README.md L141–175](https://github.com/OpenBMB/ClawXRouter/blob/6acac9f8ae1a8c2c236b84deb68b5df4d2d6336f/README.md#L141-L175)；[README.md L344–409](https://github.com/OpenBMB/ClawXRouter/blob/6acac9f8ae1a8c2c236b84deb68b5df4d2d6336f/README.md#L344-L409)。

### 17. ConsJudge · 已审查

多选评判、自改进 judge、MiniCPM-Embedding 数据构造、LLaMA-Factory 及后续 RAG DPO 流程均有原文；一致性不等于正确性的解释合理。无新增模型尺寸或 API 主张。

站内定位：[索引第 535 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L535)；类别：检索与知识。

原始证据：[README.md L2–6](https://github.com/OpenBMB/ConsJudge/blob/951d1748a94268b4eb83f1455fd2849a31d3ad35/README.md#L2-L6)；[README.md L27–91](https://github.com/OpenBMB/ConsJudge/blob/951d1748a94268b4eb83f1455fd2849a31d3ad35/README.md#L27-L91)。

### 18. CPM-Bee · 已审查

10B、中英、Transformer 自回归基座、CPM-Live 里程碑、结构化输入以及 VisCPM 后继关系直接有官方资料；并未将其写成聊天对齐版，GML 只提示按版本阅读。

站内定位：[索引第 561 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L561)；类别：文本模型。

原始证据：[README.md L6–49](https://github.com/OpenBMB/CPM-Bee/blob/be2b6a904f596bfd0328cb5e9020a84db44f9629/README.md#L6-L49)；[README.md L82–172](https://github.com/OpenBMB/CPM-Bee/blob/be2b6a904f596bfd0328cb5e9020a84db44f9629/README.md#L82-L172)；[README.md L706–711](https://github.com/OpenBMB/CPM-Bee/blob/be2b6a904f596bfd0328cb5e9020a84db44f9629/README.md#L706-L711)。

### 19. CPM-Live · 已审查

CPM-Ant/Ant+/Bee、训练计划及过程更新均有原文。附录将其作为训练计划入口，没有新增第四种统一模型架构；阶段与仓库数量分开。

站内定位：[索引第 591 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L591)；类别：训练与压缩。

原始证据：[README.md L16–45](https://github.com/OpenBMB/CPM-Live/blob/3b27173f67827faf329ab649ee1d0cb2a873e56b/README.md#L16-L45)。

### 20. CPM.cu · 已审查

CUDA Graph、chunked prefill、InfLLMv2、MTP、EAGLE 与缓存代码对应；属于模型执行引擎。推测解码原理和条件性加速为工程说明，无声称全部模型/格式通用。

站内定位：[索引第 621 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L621)；类别：推理基础设施。

原始证据：[README.md L5–25](https://github.com/OpenBMB/CPM.cu/blob/23aa7b7fefc537113166691cec63d5baa5209ebe/README.md#L5-L25)；[README.md L191–277](https://github.com/OpenBMB/CPM.cu/blob/23aa7b7fefc537113166691cec63d5baa5209ebe/README.md#L191-L277)；[llm.py L238–257](https://github.com/OpenBMB/CPM.cu/blob/23aa7b7fefc537113166691cec63d5baa5209ebe/cpmcu/llm.py#L238-L257)；[eagle.py L1–54](https://github.com/OpenBMB/CPM.cu/blob/23aa7b7fefc537113166691cec63d5baa5209ebe/cpmcu/speculative/eagle.py#L1-L54)。

### 21. cpm_kernels · 已审查

README 仅定位 CUDA kernels；保存 GEMM Torch 绑定与 layernorm 源码支持算子库定位，其余算子名可从目录确认。dtype/shape 范围未外推。非语言模型或独立推理服务。

站内定位：[索引第 659 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L659)；类别：推理基础设施。

原始证据：[README.md L1–3](https://github.com/OpenBMB/cpm_kernels/blob/277a0eddd6d264c6b9b9477fa47cc00880d581c0/README.md#L1-L3)；[gemm.py L1–104](https://github.com/OpenBMB/cpm_kernels/blob/277a0eddd6d264c6b9b9477fa47cc00880d581c0/cpm_kernels/torch/gemm.py#L1-L104)。

### 22. CPO · 已审查

Controllable Preference Optimization 名称、CPSFT/CDPO/UltraSafety 数据及偏好分数控制直接有原文。能力达标限制是工程分析，没有混淆其他 CPO。

站内定位：[索引第 693 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L693)；类别：训练与压缩。

原始证据：[README.md L1–46](https://github.com/OpenBMB/CPO/blob/10c4480326cd02a35e0221ccf40ed63eb79bef7f/README.md#L1-L46)；[README.md L51–143](https://github.com/OpenBMB/CPO/blob/10c4480326cd02a35e0221ccf40ed63eb79bef7f/README.md#L51-L143)。

### 23. DEBATER · 已审查

CoD 与 self-distillation、MiniCPM2-2B/3-4B、MTEB/LLM2Vec 有文档；forward 一次调用骨干后 pooling，last_token 或 M_token 分支可核。向量检索与聊天区分准确；效率收益未实测。

站内定位：[索引第 719 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L719)；类别：检索与知识。

原始证据：[README.md L6–6](https://github.com/OpenBMB/DEBATER/blob/377a1421a094976009fa1c59dda1cb3659548355/README.md#L6-L6)；[README.md L26–43](https://github.com/OpenBMB/DEBATER/blob/377a1421a094976009fa1c59dda1cb3659548355/README.md#L26-L43)；[README.md L77–96](https://github.com/OpenBMB/DEBATER/blob/377a1421a094976009fa1c59dda1cb3659548355/README.md#L77-L96)；[llm2vec.py L160–185](https://github.com/OpenBMB/DEBATER/blob/377a1421a094976009fa1c59dda1cb3659548355/llm2vec/llm2vec.py#L160-L185)；[llm2vec.py L216–245](https://github.com/OpenBMB/DEBATER/blob/377a1421a094976009fa1c59dda1cb3659548355/llm2vec/llm2vec.py#L216-L245)。

### 24. DecorateLM · 已审查

Rate/Tag/Edit、成对 GPT 标注和 Bradley–Terry 评分直接有原文；项目为数据工程工具，附录没有误称新通用模型。数据偏差与真实性限制为工程推断。

站内定位：[索引第 749 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L749)；类别：数据与对齐。

原始证据：[README.md L3–13](https://github.com/OpenBMB/DecorateLM/blob/357555cac4409a56aac42f69d6a8e49f361c4e31/README.md#L3-L13)；[README.md L27–62](https://github.com/OpenBMB/DecorateLM/blob/357555cac4409a56aac42f69d6a8e49f361c4e31/README.md#L27-L62)；[README.md L79–111](https://github.com/OpenBMB/DecorateLM/blob/357555cac4409a56aac42f69d6a8e49f361c4e31/README.md#L79-L111)。

### 25. DecT · 已审查

MLM/LM/chat、OpenPrompt 及 LLaMA/Alpaca/Vicuna 模板直接有 README；no_grad 提取表征后训练任务 verbalizer 可由源码确认。没有把 Decoder Tuning 当新自回归基础模型。

站内定位：[索引第 775 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L775)；类别：训练与压缩。

原始证据：[README.md L3–52](https://github.com/OpenBMB/DecT/blob/a2aa29570c19d066aaa4b01fac278bf65ac37afb/README.md#L3-L52)；[dect_trainer.py L61–90](https://github.com/OpenBMB/DecT/blob/a2aa29570c19d066aaa4b01fac278bf65ac37afb/src/dect_trainer.py#L61-L90)；[dect_verbalizer.py L19–68](https://github.com/OpenBMB/DecT/blob/a2aa29570c19d066aaa4b01fac278bf65ac37afb/src/dect_verbalizer.py#L19-L68)；[dect_verbalizer.py L251–328](https://github.com/OpenBMB/DecT/blob/a2aa29570c19d066aaa4b01fac278bf65ac37afb/src/dect_verbalizer.py#L251-L328)。

### 26. DeepThinkVLA · 已审查

pi0-FAST 起点、先推理后并行动作、混合注意力和 CoT SFT/RL 与原文一致；PaliGemma 继承由源码确认。真机与 RobotWin TODO 被准确保留，未当作已完成验证。

站内定位：[索引第 809 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L809)；类别：具身与科学。

原始证据：[README.md L45–76](https://github.com/OpenBMB/DeepThinkVLA/blob/ff271e7c024422581c9875d96250f99dd0b743d1/README.md#L45-L76)；[modeling_deepthinkvla.py L3–11](https://github.com/OpenBMB/DeepThinkVLA/blob/ff271e7c024422581c9875d96250f99dd0b743d1/src/sft/modeling_deepthinkvla.py#L3-L11)；[modeling_deepthinkvla.py L79–90](https://github.com/OpenBMB/DeepThinkVLA/blob/ff271e7c024422581c9875d96250f99dd0b743d1/src/sft/modeling_deepthinkvla.py#L79-L90)。

### 27. EdgeClaw · 已审查

ClawXMemory/Router、Kairos/Governor/Sandbox/Skill/Context 模块与原文一致；API/路由/工具平台未误当模型。GitHub 元数据全量核对已确认 fork/parent openclaw，原作者来源没有被组织名覆盖。

站内定位：[索引第 839 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L839)；类别：Agent应用。

原始证据：[README.md L22–26](https://github.com/OpenBMB/EdgeClaw/blob/5e461861b370f5677d2eb6b35499764632989279/README.md#L22-L26)；[README.md L136–252](https://github.com/OpenBMB/EdgeClaw/blob/5e461861b370f5677d2eb6b35499764632989279/README.md#L136-L252)。

### 28. Eurus · 已审查

Eurus 7B Mistral、70B CodeLLaMA 与 Eurux Mixtral-8x22B 及 NCA/KTO 有明确对应，UltraInteract 轨迹树、UltraFeedback 也有来源。附录正确要求版本分别解释，没有混算模型归属。

站内定位：[索引第 870 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L870)；类别：模型与Agent。

原始证据：[README.md L16–17](https://github.com/OpenBMB/Eurus/blob/fd72db1861ef009fede0dbe5bcc9893c9012dacf/README.md#L16-L17)；[README.md L28–54](https://github.com/OpenBMB/Eurus/blob/fd72db1861ef009fede0dbe5bcc9893c9012dacf/README.md#L28-L54)。

### 29. ForgeStencil · 已审查

双 Agent 计划/编写/profile 与应用集成/校验、交错计时、A100/H100/B200 条件有官方说明；项目面向 stencil 优化而非 LLM 权重。性能仅归官方报告，未作本次复测结论。

站内定位：[索引第 896 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L896)；类别：编程与优化。

原始证据：[README.md L17–39](https://github.com/OpenBMB/ForgeStencil/blob/1c8311c8a79abaedd40fc6609d888a4f88dea3a2/README.md#L17-L39)；[README.md L209–287](https://github.com/OpenBMB/ForgeStencil/blob/1c8311c8a79abaedd40fc6609d888a4f88dea3a2/README.md#L209-L287)；[README.md L345–368](https://github.com/OpenBMB/ForgeStencil/blob/1c8311c8a79abaedd40fc6609d888a4f88dea3a2/README.md#L345-L368)。

### 30. ForgeTrain · 已审查

MiniCPM4-0.5B/8B、H100、44.13% MFU 及待开放 harness 都保留边界；子项目明确排除移植 FA4/CuTeDSL 代码的声明被正确转述。开发流程没有被误当模型架构创新或独立实验事实。

站内定位：[索引第 926 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L926)；类别：训练与压缩。

原始证据：[README.md L19–31](https://github.com/OpenBMB/ForgeTrain/blob/3242a5cd74851a14a7e0c1ebbfb19decdafd2cde/README.md#L19-L31)；[README.md L43–55](https://github.com/OpenBMB/ForgeTrain/blob/3242a5cd74851a14a7e0c1ebbfb19decdafd2cde/README.md#L43-L55)；[README.md L208–238](https://github.com/OpenBMB/ForgeTrain/blob/3242a5cd74851a14a7e0c1ebbfb19decdafd2cde/README.md#L208-L238)；[README.md L27–45](https://github.com/OpenBMB/ForgeTrain/blob/3242a5cd74851a14a7e0c1ebbfb19decdafd2cde/exports/train_engine_0.5B/README.md#L27-L45)。

### 31. General-Model-License · 已审查

原文是 GML 许可系列及可选限制说明，没有模型尺寸、推理 API 或可用模型之主张。本审仅核文档定位与来源，不作法律效力或具体许可合规判断。

站内定位：[索引第 960 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L960)；类别：资料与治理。

原始证据：[README.md L1–45](https://github.com/OpenBMB/General-Model-License/blob/e4eb9be48f2cfe8367205208516440fa186e8ae3/README.md#L1-L45)。

### 32. HerculesBench · 原始证据不足

根 README 仅 OlympiadBench-；补充文档仅能确认 OlympiadBench 模型调用/评分代码。附录已明确正式数据范围、指标和维护意图不足，未虚构 Hercules 新模型或正式新版关系；本审保留同一缺口。

站内定位：[索引第 986 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L986)；类别：评测。

原始证据：[README.md L1–1](https://github.com/OpenBMB/HerculesBench/blob/5efa6bc7d1cdff270934034bf16c93f86d226ba1/README.md#L1-L1)；[readme.md L1–50](https://github.com/OpenBMB/HerculesBench/blob/5efa6bc7d1cdff270934034bf16c93f86d226ba1/inference/readme.md#L1-L50)。

### 33. InfiniteBench · 已审查

100K+、12 项任务、真实/合成混合、书籍/对话/检索/代码/数学对应原文，属于基准。100K+ 是整体设计定位，不是逐个任务都超过 100K，原始表含更短数学/代码任务；附录未声称每项都超过。截断/MapReduce 等影响属工程推断。

站内定位：[索引第 1016 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1016)；类别：评测。

原始证据：[README.md L18–43](https://github.com/OpenBMB/InfiniteBench/blob/51d9b37b0f1790ead936df2243abbf7f0420e439/README.md#L18-L43)；[README.md L91–94](https://github.com/OpenBMB/InfiniteBench/blob/51d9b37b0f1790ead936df2243abbf7f0420e439/README.md#L91-L94)。

### 34. infllmv2_cuda_impl · 已审查

两阶段块打分聚合和稀疏注意力、TopK 外部执行、main 与 feature_infer 示例边界都与 README 一致；未混淆 FFN MoE 路由。梯度支持是官方声明，未编译运行验证。

站内定位：[索引第 1046 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1046)；类别：推理基础设施。

原始证据：[README.md L1–17](https://github.com/OpenBMB/infllmv2_cuda_impl/blob/93cf2ec28e5a7acebe3f0bb7329b6c73a1be91f6/README.md#L1-L17)；[README.md L79–140](https://github.com/OpenBMB/infllmv2_cuda_impl/blob/93cf2ec28e5a7acebe3f0bb7329b6c73a1be91f6/README.md#L79-L140)。

### 35. IoA · 已审查

异构 Agent、嵌套团队、异步任务及 AutoGPT/OpenInterpreter 示例、Docker/Milvus 均有对应。框架并非权重共享 MoE 或模型内部 attention，归类准确。

站内定位：[索引第 1072 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1072)；类别：Agent框架。

原始证据：[README.md L23–31](https://github.com/OpenBMB/IoA/blob/9d78e88e47160eb42ee1ee8538c05e1c2c47877a/README.md#L23-L31)；[README.md L98–151](https://github.com/OpenBMB/IoA/blob/9d78e88e47160eb42ee1ee8538c05e1c2c47877a/README.md#L98-L151)。

### 36. Locret · 已审查

Phi-3-mini-128K、Llama3.1-8B 当前支持及 MiniCPM 待办区分准确；源码分块 prefill、stabilizers 保护和 topk 保留可确认。缓存淘汰与文档检索没有混淆；压缩倍数未被写为普遍保证。

站内定位：[索引第 1098 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1098)；类别：推理基础设施。

原始证据：[README.md L15–38](https://github.com/OpenBMB/Locret/blob/28995c5ad6e6cc263913eb73fb76a4942f3dbf75/README.md#L15-L38)；[README.md L74–89](https://github.com/OpenBMB/Locret/blob/28995c5ad6e6cc263913eb73fb76a4942f3dbf75/README.md#L74-L89)；[infer.py L3–32](https://github.com/OpenBMB/Locret/blob/28995c5ad6e6cc263913eb73fb76a4942f3dbf75/locret/inference/infer.py#L3-L32)。

### 37. MA-ProofBench · 已审查

200 题、两档难度、Lean v4.28.0 及 Kimina/Mathlib 环境均有原文。形式化证明基准与自由文本证明区分正确。超时/版本对通过率影响为工程推断。

站内定位：[索引第 1128 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1128)；类别：评测。

原始证据：[README.md L12–19](https://github.com/OpenBMB/MA-ProofBench/blob/31bab4c78fd1b0c464e767d62e9202163d1e9029/README.md#L12-L19)；[README.md L49–65](https://github.com/OpenBMB/MA-ProofBench/blob/31bab4c78fd1b0c464e767d62e9202163d1e9029/README.md#L49-L65)。

### 38. MathForm · 已审查

Mathlib 检索、编译/语义反馈、FormalVerse、MathForm-8B SFT/RL 与 Lean4.21.0 均有来源。正确区分自动形式化 statement 与证明；语法不等于语义及跨版本需适配是工程分析。

站内定位：[索引第 1158 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1158)；类别：模型与Agent。

原始证据：[README.md L19–31](https://github.com/OpenBMB/MathForm/blob/360487f264e801ee08de0f49b12e57fcc600e5ca/README.md#L19-L31)；[README.md L70–70](https://github.com/OpenBMB/MathForm/blob/360487f264e801ee08de0f49b12e57fcc600e5ca/README.md#L70-L70)。

### 39. Meshy · 已审查

SGLang/Titan/CPU rollout、TransferQueue 同时承载数据与门控、token ring GPU 交接及 on/off-policy 窗口有文档/代码支持。性能、陈旧样本和布局限制没有被声称已实测。

站内定位：[索引第 1188 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1188)；类别：训练与压缩。

原始证据：[README.md L10–38](https://github.com/OpenBMB/Meshy/blob/04e145c6282c9d8cbea99e692256a027c83a7673/README.md#L10-L38)；[architecture_services.md L1–52](https://github.com/OpenBMB/Meshy/blob/04e145c6282c9d8cbea99e692256a027c83a7673/docs/architecture_services.md#L1-L52)；[topology.py L103–110](https://github.com/OpenBMB/Meshy/blob/04e145c6282c9d8cbea99e692256a027c83a7673/meshy/service/topology.py#L103-L110)。

### 40. MetaMem · 已审查

标题明确 Self-Reflective Symbolic Optimization；反思/反馈、LightMem 事实记忆和各外部模型依赖有 README。附录正确将“训练”限定为符号元记忆优化，不泛化为权重训练；泛化收益只归官方实验。

站内定位：[索引第 1226 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1226)；类别：检索与知识。

原始证据：[README.md L3–3](https://github.com/OpenBMB/MetaMem/blob/66232c1cb33d3d7939d323b431f21c741785142e/README.md#L3-L3)；[README.md L32–32](https://github.com/OpenBMB/MetaMem/blob/66232c1cb33d3d7939d323b431f21c741785142e/README.md#L32-L32)；[README.md L45–69](https://github.com/OpenBMB/MetaMem/blob/66232c1cb33d3d7939d323b431f21c741785142e/README.md#L45-L69)；[README.md L118–170](https://github.com/OpenBMB/MetaMem/blob/66232c1cb33d3d7939d323b431f21c741785142e/README.md#L118-L170)。

### 41. MiniCPM · 已审查

μP、MoE/upcycling、稀疏系列、3 的低秩注意力、4/4.1 InfLLMv2、SALA 和 5 的 LlamaForCausalLM 分代正确。17 模型配置全量核验另支撑结构；RAG Embedding/Reranker 由旧版文档确认。命名尺寸与全模型参数不等的提示正确。

站内定位：[索引第 1252 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1252)；类别：文本模型。

原始证据：[README.md L26–48](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/README.md#L26-L48)；[README.md L155–164](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/README.md#L155-L164)；[README.md L311–324](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/README.md#L311-L324)；[README-minicpm1-en.md L14–48](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/docs/README-minicpm1-en.md#L14-L48)；[README-minicpm2-en.md L16–32](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/docs/README-minicpm2-en.md#L16-L32)；[README-minicpm3-en.md L17–19](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/docs/README-minicpm3-en.md#L17-L19)；[README.md L5–9](https://github.com/OpenBMB/MiniCPM/blob/316cfb1cea39f39cfa16b4f5703b77495c2340be/minicpm_sala/README.md#L5-L9)。

### 42. MiniCPM-Desk-Pet · 已审查

桌面应用、角色适配、coding agent 事件、默认 MiniCPM5-1B GGUF 与平台状态均有原文。未把 Windows 安装包或 Linux validation 路线图概括为同等测试程度。拟人理解为分析边界。

站内定位：[索引第 1298 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1298)；类别：Agent应用。

原始证据：[README.md L17–32](https://github.com/OpenBMB/MiniCPM-Desk-Pet/blob/87dc4d7c39f30f5d9a2753cd60559a0e90287839/README.md#L17-L32)；[README.md L36–72](https://github.com/OpenBMB/MiniCPM-Desk-Pet/blob/87dc4d7c39f30f5d9a2753cd60559a0e90287839/README.md#L36-L72)；[README.md L103–113](https://github.com/OpenBMB/MiniCPM-Desk-Pet/blob/87dc4d7c39f30f5d9a2753cd60559a0e90287839/README.md#L103-L113)。

### 43. MiniCPM-o-Demo · 已审查

模型团队的 o4.5 PyTorch/CUDA 演示与实时音视频全双工有原文；C++/desktop 路线只是另入口。附录正确不把某条 demo 的 CUDA 要求推广到全部后端。

站内定位：[索引第 1328 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1328)；类别：示例与文档。

原始证据：[README.md L1–11](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/47709a9210dfd71afa76c058e017fc8c4db5c8d2/README.md#L1-L11)；[README.md L55–79](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/47709a9210dfd71afa76c058e017fc8c4db5c8d2/README.md#L55-L79)；[README.md L151–159](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/47709a9210dfd71afa76c058e017fc8c4db5c8d2/README.md#L151-L159)。

### 44. MiniCPM-Robot · 已审查

Manip 1.5B/V4.6 与 Track 0.9B/0.5B 骨干有原文；Track 8 路点、三维动作、control query 与 Funnel head 由配置源码核对。没有把路点当关节动作，模型总量与文本骨干大小区分正确。

站内定位：[索引第 1358 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1358)；类别：具身与科学。

原始证据：[README.md L19–23](https://github.com/OpenBMB/MiniCPM-Robot/blob/f666593066641d2c0cef06bc17bfb1e954547674/README.md#L19-L23)；[README.md L49–53](https://github.com/OpenBMB/MiniCPM-Robot/blob/f666593066641d2c0cef06bc17bfb1e954547674/README.md#L49-L53)；[README.md L148–153](https://github.com/OpenBMB/MiniCPM-Robot/blob/f666593066641d2c0cef06bc17bfb1e954547674/README.md#L148-L153)；[config.py L26–31](https://github.com/OpenBMB/MiniCPM-Robot/blob/f666593066641d2c0cef06bc17bfb1e954547674/MiniCPM-RobotTrack/minicpm_robot_track/config.py#L26-L31)；[modeling.py L62–103](https://github.com/OpenBMB/MiniCPM-Robot/blob/f666593066641d2c0cef06bc17bfb1e954547674/MiniCPM-RobotTrack/minicpm_robot_track/modeling.py#L62-L103)；[modeling.py L165–169](https://github.com/OpenBMB/MiniCPM-Robot/blob/f666593066641d2c0cef06bc17bfb1e954547674/MiniCPM-RobotTrack/minicpm_robot_track/modeling.py#L165-L169)；[GO2_DEPLOYMENT.md L1–13](https://github.com/OpenBMB/MiniCPM-Robot/blob/f666593066641d2c0cef06bc17bfb1e954547674/MiniCPM-RobotTrack/docs/GO2_DEPLOYMENT.md#L1-L13)。

### 45. MiniCPM-V · 已审查

V4.6 SigLIP2/Qwen3.5-0.8B、1.3B 与 o4.5 SigLip2/Whisper-medium/CosyVoice2/Qwen3-8B、9B 直接有 README；流式接口有原文。浅层 ViT 内压缩及 Omni-Flow 细节另对照保存的官方 arXiv 摘要（2605.08985、2604.27393），这些论文未列入 238 源文件远端字节复核统计。手机支持范围限定正确。

站内定位：[索引第 1404 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1404)；类别：多模态模型。

原始证据：[README.md L33–38](https://github.com/OpenBMB/MiniCPM-V/blob/6ada8e8ef5e2979670fc94406f02b87c3c7e7ee0/README.md#L33-L38)；[README.md L117–126](https://github.com/OpenBMB/MiniCPM-V/blob/6ada8e8ef5e2979670fc94406f02b87c3c7e7ee0/README.md#L117-L126)；[README.md L360–374](https://github.com/OpenBMB/MiniCPM-V/blob/6ada8e8ef5e2979670fc94406f02b87c3c7e7ee0/README.md#L360-L374)；[README.md L1581–1623](https://github.com/OpenBMB/MiniCPM-V/blob/6ada8e8ef5e2979670fc94406f02b87c3c7e7ee0/README.md#L1581-L1623)。

### 46. MiniCPM-V-Apps · 已审查

iOS/Android/Harmony、V/5/VoxCPM2 支持清单、tc-mb llama.cpp-omni 子模块和 GGUF 文档均直接对应。属于部署客户端而非模型创新；内存/热量/后台条件为工程推断。

站内定位：[索引第 1434 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1434)；类别：Agent应用。

原始证据：[README.md L5–22](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/d1aac05eecaac64b7fc35f1cc93a06b726ce10e3/README.md#L5-L22)；[README.md L50–157](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/d1aac05eecaac64b7fc35f1cc93a06b726ce10e3/README.md#L50-L157)；[README.md L196–237](https://github.com/OpenBMB/MiniCPM-V-Apps/blob/d1aac05eecaac64b7fc35f1cc93a06b726ce10e3/README.md#L196-L237)。

### 47. MobileCPM · 已审查

提示型 Agent、下载/替换本地模型和流式接口有原文；iOS 当前、Android coming soon 保留。没有借 V-Apps 新支持倒灌旧应用能力。

站内定位：[索引第 1464 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1464)；类别：Agent应用。

原始证据：[README.md L18–35](https://github.com/OpenBMB/MobileCPM/blob/3943591b2698ccb51f1d7e165d048d3ae68022d5/README.md#L18-L35)；[README.md L41–111](https://github.com/OpenBMB/MobileCPM/blob/3943591b2698ccb51f1d7e165d048d3ae68022d5/README.md#L41-L111)。

### 48. ModelCenter · 已审查

BMTrain 后端及 BERT/T5/LLaMA/ViT 目录和文档均可核；普通/SparseSelfAttention 类有源码。多架构库并非统一 decoder-only 模型，checkpoint 无缝兼容未被保证。

站内定位：[索引第 1494 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1494)；类别：训练与压缩。

原始证据：[README.md L35–47](https://github.com/OpenBMB/ModelCenter/blob/d0c176a20b5b0e241d591a374061aa8887116ae3/README.md#L35-L47)；[README.md L78–118](https://github.com/OpenBMB/ModelCenter/blob/d0c176a20b5b0e241d591a374061aa8887116ae3/README.md#L78-L118)；[README.md L261–335](https://github.com/OpenBMB/ModelCenter/blob/d0c176a20b5b0e241d591a374061aa8887116ae3/README.md#L261-L335)；[attention.py L25–27](https://github.com/OpenBMB/ModelCenter/blob/d0c176a20b5b0e241d591a374061aa8887116ae3/model_center/layer/attention.py#L25-L27)；[attention.py L238–281](https://github.com/OpenBMB/ModelCenter/blob/d0c176a20b5b0e241d591a374061aa8887116ae3/model_center/layer/attention.py#L238-L281)。

### 49. MoRE · 已审查

标题明确 Retrieval Experts；文本/UniIR 图像/Open-WikiTable、Step-GRPO 和 EasyR1 均有 README。外部检索器专家与 FFN MoE 区分准确，路由/召回/生成分开评价为工程分析。

站内定位：[索引第 1528 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1528)；类别：检索与知识。

原始证据：[README.md L3–3](https://github.com/OpenBMB/MoRE/blob/2823013408bd52a77ab7b4634f334cef0fbf8f84/README.md#L3-L3)；[README.md L47–80](https://github.com/OpenBMB/MoRE/blob/2823013408bd52a77ab7b4634f334cef0fbf8f84/README.md#L47-L80)。

### 50. OlympiadBench · 已审查

8,476 数学/物理题、中英多模态、题源、Mathpix OCR 和开放答案自动评分/证明题抽样评估有原文。附录没有把所有题说成统一自动评分，也未将旧榜当当前排名。

站内定位：[索引第 1554 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1554)；类别：评测。

原始证据：[README.md L53–74](https://github.com/OpenBMB/OlympiadBench/blob/ba5b26a7e2849940b598a9159c1190daa2b9175f/README.md#L53-L74)。

### 51. Omni-DuplexEval · 已审查

描述/主动提醒/纠正、时间戳预测 JSON、视频工具/judge 及不含模型预测生成代码均有原文，边界准确。评价时机不等同模型自带实时推理服务。

站内定位：[索引第 1580 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1580)；类别：评测。

原始证据：[README.md L8–30](https://github.com/OpenBMB/Omni-DuplexEval/blob/962cec448ac37a377ffb963b476778777d11d346/README.md#L8-L30)；[README.md L73–96](https://github.com/OpenBMB/Omni-DuplexEval/blob/962cec448ac37a377ffb963b476778777d11d346/README.md#L73-L96)；[README.md L181–192](https://github.com/OpenBMB/Omni-DuplexEval/blob/962cec448ac37a377ffb963b476778777d11d346/README.md#L181-L192)。

### 52. OmniEvalKit · 已审查

音频/音视频理解、分布式运行及 WER/CER/BLEU/VQA/LLM judge 是框架原文。附录没有以 omni 名称声称全双工，也没有未经解释混合不同指标。

站内定位：[索引第 1606 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1606)；类别：评测。

原始证据：[README.md L5–13](https://github.com/OpenBMB/OmniEvalKit/blob/1adac0577258d539efc03f16d8a0f4b3f7df6c19/README.md#L5-L13)；[README.md L62–125](https://github.com/OpenBMB/OmniEvalKit/blob/1adac0577258d539efc03f16d8a0f4b3f7df6c19/README.md#L62-L125)。

### 53. OpenAct · 已审查

339 题/7 域、GitHub 工具集成、OpenAgent 层级及双层经验机制直接有摘要，API/Docker 环境明确。区别任务基准 OpenAct 与 Agent 系统 OpenAgent，未误归为机器人动作模型。

站内定位：[索引第 1640 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1640)；类别：Agent框架。

原始证据：[README.md L9–22](https://github.com/OpenBMB/OpenAct/blob/a6ca7fd5ab2316c24cbee781fb3e15f77e11317b/README.md#L9-L22)。

### 54. openbmb.github.io · 已审查

根 README 极简，但保存 index.html 明确静态 HTML MiniCPM-o 4.5 展示页，目录素材/其他介绍对应站点定位。不能从站点断言模型结构，附录正好保留该边界；没有独立权重/模型发布主张。

站内定位：[索引第 1666 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1666)；类别：示例与文档。

原始证据：[README.md L1–1](https://github.com/OpenBMB/openbmb.github.io/blob/612263d57c4a80a98fbee801381b7ac86a44ecaa/README.md#L1-L1)；[index.html L1–9](https://github.com/OpenBMB/openbmb.github.io/blob/612263d57c4a80a98fbee801381b7ac86a44ecaa/index.html#L1-L9)。

### 55. PAGER · 已审查

认知提纲槽位、迭代检索填充知识页与最终生成有原文；Qwen3 Embedding/FAISS/vLLM/FlashRAG 依赖明确。并非 GPU paging/KV cache，功能归属正确。

站内定位：[索引第 1704 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1704)；类别：检索与知识。

原始证据：[README.md L31–31](https://github.com/OpenBMB/PAGER/blob/2d9d6355435debbe08f85a139c472eb6be992b4c/README.md#L31-L31)；[README.md L51–74](https://github.com/OpenBMB/PAGER/blob/2d9d6355435debbe08f85a139c472eb6be992b4c/README.md#L51-L74)。

### 56. ParamMute · 已审查

70–90% 深度 FFN 是该研究观察，抑制激活+适配训练、修改 Transformers 与 CoConflictQA 有原文。附录没有宣称全模型普遍规律；自身知识能力受损风险属工程推断。

站内定位：[索引第 1730 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1730)；类别：检索与知识。

原始证据：[README.md L63–63](https://github.com/OpenBMB/ParamMute/blob/cc81d30145c145b660f732da2bdc0556b1f6acaf/README.md#L63-L63)；[README.md L91–144](https://github.com/OpenBMB/ParamMute/blob/cc81d30145c145b660f732da2bdc0556b1f6acaf/README.md#L91-L144)。

### 57. PilotDeck · 已审查

WorkSpace、可检查记忆/路由/always-on 与 MCP、客户端入口有原文；会话 timeline 格式与传播有架构文档。不是新的模型，长期可靠性和副作用须实测的限制为工程分析。

站内定位：[索引第 1756 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1756)；类别：Agent应用。

原始证据：[README.md L42–53](https://github.com/OpenBMB/PilotDeck/blob/5a0efbc5106e7c80c3df8e20184c439ed3d0603b/README.md#L42-L53)；[README.md L55–106](https://github.com/OpenBMB/PilotDeck/blob/5a0efbc5106e7c80c3df8e20184c439ed3d0603b/README.md#L55-L106)；[README.md L293–477](https://github.com/OpenBMB/PilotDeck/blob/5a0efbc5106e7c80c3df8e20184c439ed3d0603b/README.md#L293-L477)；[session-timeline.md L1–42](https://github.com/OpenBMB/PilotDeck/blob/5a0efbc5106e7c80c3df8e20184c439ed3d0603b/docs/architecture/session-timeline.md#L1-L42)。

### 58. ProAgent · 已审查

APA/RPA 差别、专门化 Agent 构建/执行 n8n 工作流与 pre-Dev Day OpenAI 接口约束有原文。调用外部工作流引擎没有被当作模型结构，旧版本限制准确。

站内定位：[索引第 1810 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1810)；类别：Agent框架。

原始证据：[README.md L5–25](https://github.com/OpenBMB/ProAgent/blob/02a79ac62e7ad3ca2565e0e4fe5fc22824aaf079/README.md#L5-L25)；[README.md L29–52](https://github.com/OpenBMB/ProAgent/blob/02a79ac62e7ad3ca2565e0e4fe5fc22824aaf079/README.md#L29-L52)。

### 59. RaD-Agent · 已审查

Experience Exploration/Utility Learning、逐步骤 Elo 成对比较及 WebShop/RestGPT/Game24/ToolBench 环境一致。推理时搜索不等于权重更新的解释合理；成本及价值噪声属工程推断。

站内定位：[索引第 1840 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1840)；类别：Agent框架。

原始证据：[README.md L5–7](https://github.com/OpenBMB/RaD-Agent/blob/7bff04ffe6dc9ce537c891432762ef029afed7f8/README.md#L5-L7)；[README.md L29–125](https://github.com/OpenBMB/RaD-Agent/blob/7bff04ffe6dc9ce537c891432762ef029afed7f8/README.md#L29-L125)。

### 60. RAG-DDR · 已审查

MiniCPM/Llama 的生成及知识精炼模块、DPO 候选采样/偏好训练直接有流程和 DPOTrainer 代码。附录对普通 TopK 并非全程精确可微的边界正确，没有从题名过度外推。

站内定位：[索引第 1866 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1866)；类别：检索与知识。

原始证据：[README.md L38–38](https://github.com/OpenBMB/RAG-DDR/blob/3bd09e69cf8efe89025dcc7f40b964c54812c8e0/README.md#L38-L38)；[README.md L81–124](https://github.com/OpenBMB/RAG-DDR/blob/3bd09e69cf8efe89025dcc7f40b964c54812c8e0/README.md#L81-L124)；[train.py L5–5](https://github.com/OpenBMB/RAG-DDR/blob/3bd09e69cf8efe89025dcc7f40b964c54812c8e0/src/generator/train.py#L5-L5)；[train.py L206–206](https://github.com/OpenBMB/RAG-DDR/blob/3bd09e69cf8efe89025dcc7f40b964c54812c8e0/src/generator/train.py#L206-L206)。

### 61. RAGEval · 已审查

种子 schema、配置/文档生成、QAR、DragonBall 和 Completeness/Hallucination/Irrelevance 都有原文。合成数据不等真实业务事实、共享偏差属工程解释。

站内定位：[索引第 1896 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1896)；类别：评测。

原始证据：[README.md L18–41](https://github.com/OpenBMB/RAGEval/blob/c44aa98aea18362eb5e3751f11e13dbc8f6cab83/README.md#L18-L41)；[README.md L46–50](https://github.com/OpenBMB/RAGEval/blob/c44aa98aea18362eb5e3751f11e13dbc8f6cab83/README.md#L46-L50)。

### 62. RepoAgent · 已审查

Python 对象文档、AST、双向调用关系及 diff 增量更新有文档；不是另一个代码基础模型。动态调用覆盖和文本解释不能代替测试是合理工程边界。

站内定位：[索引第 1922 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1922)；类别：Agent应用。

原始证据：[README.md L37–49](https://github.com/OpenBMB/RepoAgent/blob/825d988127d7bfd757237d9c4e8678d9104030f0/README.md#L37-L49)；[README.md L115–140](https://github.com/OpenBMB/RepoAgent/blob/825d988127d7bfd757237d9c4e8678d9104030f0/README.md#L115-L140)。

### 63. RLPR · 已审查

参考答案平均解码概率作奖励，Qwen/Llama/Gemma 示例有原文；无外部 verifier 不等无参考答案区分准确。参考质量与概率校准影响为工程分析。

站内定位：[索引第 1952 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1952)；类别：训练与压缩。

原始证据：[README.md L29–37](https://github.com/OpenBMB/RLPR/blob/235870e1a934b52ed28902d5d68ec8a811063cdc/README.md#L29-L37)；[README.md L61–78](https://github.com/OpenBMB/RLPR/blob/235870e1a934b52ed28902d5d68ec8a811063cdc/README.md#L61-L78)；[README.md L120–145](https://github.com/OpenBMB/RLPR/blob/235870e1a934b52ed28902d5d68ec8a811063cdc/README.md#L120-L145)。

### 64. SciCore-Mol · 已审查

GVP、diffusion、Reaction Transformer、special token/hidden-state 融合和三阶段训练描述有原文及类依赖；不是只将 SMILES 当文本。有效性/属性/合成可行性区分合理，训练显存没有被推广成推理最低值。

站内定位：[索引第 1982 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L1982)；类别：具身与科学。

原始证据：[README.md L19–21](https://github.com/OpenBMB/SciCore-Mol/blob/37a7184af1c25d9854d8b027a16619b4b42a7c22/README.md#L19-L21)；[README.md L76–110](https://github.com/OpenBMB/SciCore-Mol/blob/37a7184af1c25d9854d8b027a16619b4b42a7c22/README.md#L76-L110)；[mol_aware_lm.py L10–22](https://github.com/OpenBMB/SciCore-Mol/blob/37a7184af1c25d9854d8b027a16619b4b42a7c22/modules/mol_aware_lm.py#L10-L22)。

### 65. Scicore-Omics · 已审查

组织图像/转录组/生物语言、NicheFormer→GeneQFormer→Projector 和 distill/CPT-SFT/RL 均有 README，交叉注意力实现可核。没有外推临床效用或因果结论。

站内定位：[索引第 2012 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2012)；类别：具身与科学。

原始证据：[README.md L35–59](https://github.com/OpenBMB/Scicore-Omics/blob/32b0ef71fad37db9bda07b6959b644d8048e5978/README.md#L35-L59)；[gene_qformer_module.py L23–61](https://github.com/OpenBMB/Scicore-Omics/blob/32b0ef71fad37db9bda07b6959b644d8048e5978/model/gene_qformer_module.py#L23-L61)；[modeling_minicpmv.py L22–64](https://github.com/OpenBMB/Scicore-Omics/blob/32b0ef71fad37db9bda07b6959b644d8048e5978/model/modeling_minicpmv.py#L22-L64)。

### 66. sglang · 已审查

OpenBMB fork、parent sgl-project/sglang 来自已核 GitHub 原始元数据；MiniCPM-SALA 专门分支由 SALA/sparse_kernel README 交叉确认。附录正确不把上游支持矩阵当 fork 每个分支的保证。

站内定位：[索引第 2050 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2050)；类别：推理基础设施。

原始证据：[README.md L42–50](https://github.com/OpenBMB/sglang/blob/7fc91fe148797c32b6e209d86b8dd50e0402815f/README.md#L42-L50)。

### 67. SHIFT · 已审查

冻结主模型+小门控、GRPO 与需修改 vLLM Qwen/Llama 实现均有原文；attention output×sigmoid(gate_score) 直接存在源码。与 ParamMute FFN 的区别准确。

站内定位：[索引第 2077 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2077)；类别：检索与知识。

原始证据：[README.md L46–46](https://github.com/OpenBMB/SHIFT/blob/0c647191b3925a64303727625f44c7aba1793d69/README.md#L46-L46)；[README.md L69–91](https://github.com/OpenBMB/SHIFT/blob/0c647191b3925a64303727625f44c7aba1793d69/README.md#L69-L91)；[modeling_qwen3.py L310–360](https://github.com/OpenBMB/SHIFT/blob/0c647191b3925a64303727625f44c7aba1793d69/models/Qwen-3-0.6B/modeling_qwen3.py#L310-L360)。

### 68. SimpleMemVLA · 已审查

采样时间戳视频直接作 native context、Qwen3.5-4B、subtask hidden states→DiT flow matching 与 shared-prefix 缓存有 README/类依赖。真机演示有成功/失败统计，附录没有将成功演示当全任务成功或无限记忆。

站内定位：[索引第 2107 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2107)；类别：具身与科学。

原始证据：[README.md L70–76](https://github.com/OpenBMB/SimpleMemVLA/blob/404215d752704fd8d8157fa6b7ccdfafdb2bd44e/README.md#L70-L76)；[README.md L128–145](https://github.com/OpenBMB/SimpleMemVLA/blob/404215d752704fd8d8157fa6b7ccdfafdb2bd44e/README.md#L128-L145)；[README.md L175–199](https://github.com/OpenBMB/SimpleMemVLA/blob/404215d752704fd8d8157fa6b7ccdfafdb2bd44e/README.md#L175-L199)；[modeling_simplememvla.py L13–39](https://github.com/OpenBMB/SimpleMemVLA/blob/404215d752704fd8d8157fa6b7ccdfafdb2bd44e/simplememvla/model/modeling_simplememvla.py#L13-L39)。

### 69. SimpleNav · 已审查

Qwen3.5-VL、BATS/TVI、DiT-B、LeRobot v3 和当前锚点机体系 H=8×4 路点有直接文档。历史 state/绝对 pose/未来 action 分别解释且 world model 是未来方向，准确。

站内定位：[索引第 2137 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2137)；类别：具身与科学。

原始证据：[README.md L29–29](https://github.com/OpenBMB/SimpleNav/blob/64925eab9136318ef90eddacf2aa7e9058311c0d/README.md#L29-L29)；[README.md L70–96](https://github.com/OpenBMB/SimpleNav/blob/64925eab9136318ef90eddacf2aa7e9058311c0d/README.md#L70-L96)；[DATA_STRUCTURE.md L26–61](https://github.com/OpenBMB/SimpleNav/blob/64925eab9136318ef90eddacf2aa7e9058311c0d/docs/guides/DATA_STRUCTURE.md#L26-L61)；[MODEL_ARCHITECTURE.md L22–85](https://github.com/OpenBMB/SimpleNav/blob/64925eab9136318ef90eddacf2aa7e9058311c0d/docs/guides/MODEL_ARCHITECTURE.md#L22-L85)。

### 70. SOAR-Toolkit · 已审查

SALA 推理优化比赛、公共/隐藏集合、指定硬件与 SGLang 测试均有文档。未把赛事资源约束当 SALA 普遍推理最低要求；公开集合不代表正式隐藏成绩的边界正确。

站内定位：[索引第 2175 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2175)；类别：评测。

原始证据：[README.md L11–17](https://github.com/OpenBMB/SOAR-Toolkit/blob/2ed4ade13a601f2f8879c2bc482ed7d0f11c61ef/README.md#L11-L17)；[README.md L85–136](https://github.com/OpenBMB/SOAR-Toolkit/blob/2ed4ade13a601f2f8879c2bc482ed7d0f11c61ef/README.md#L85-L136)。

### 71. sparse_kernel · 已审查

README 指向 sglang minicpm_sala 子模块；源码是 topk_idx/block_table 等到 out_block_table 的 wrapper/PyBind，定位为稀疏表构造准确。没有扩大为完整模型或全部 InfLLMv2 算子。

站内定位：[索引第 2205 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2205)；类别：推理基础设施。

原始证据：[README.md L1–4](https://github.com/OpenBMB/sparse_kernel/blob/bfa61186b2c87569185523a800780e997cbbf601/README.md#L1-L4)；[README.md L38–45](https://github.com/OpenBMB/sparse_kernel/blob/bfa61186b2c87569185523a800780e997cbbf601/README.md#L38-L45)；[get_table_kernel.cu L154–178](https://github.com/OpenBMB/sparse_kernel/blob/bfa61186b2c87569185523a800780e997cbbf601/get_table_kernel.cu#L154-L178)。

### 72. StaffDeck · 已审查

SOP 状态机、知识索引、技能版本、过程事件与人工交接有原文/后端组件；业务 Agent 平台不是独立员工模型。附录的权限、失败恢复和人工监督边界属工程判断。

站内定位：[索引第 2235 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2235)；类别：Agent应用。

原始证据：[README.md L30–36](https://github.com/OpenBMB/StaffDeck/blob/7adc7c84f61bd6cca13ff0380a811cbb3ae3c544/README.md#L30-L36)；[README.md L229–242](https://github.com/OpenBMB/StaffDeck/blob/7adc7c84f61bd6cca13ff0380a811cbb3ae3c544/README.md#L229-L242)；[README.md L342–349](https://github.com/OpenBMB/StaffDeck/blob/7adc7c84f61bd6cca13ff0380a811cbb3ae3c544/README.md#L342-L349)；[README.md L3–34](https://github.com/OpenBMB/StaffDeck/blob/7adc7c84f61bd6cca13ff0380a811cbb3ae3c544/backend/README.md#L3-L34)；[harness_v2_engine.py L15–35](https://github.com/OpenBMB/StaffDeck/blob/7adc7c84f61bd6cca13ff0380a811cbb3ae3c544/backend/app/core/harness_v2_engine.py#L15-L35)。

### 73. Tell_Me_More · 已审查

IN3、Mistral-Interact、Mistral-7B 适配及先澄清后执行有直接文档，没有新的输入模态主张。追问次数与收益是工程权衡。

站内定位：[索引第 2273 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2273)；类别：模型与Agent。

原始证据：[README.md L21–32](https://github.com/OpenBMB/Tell_Me_More/blob/654fcec4da3b3465536d57b0d1995e8699a5a8f0/README.md#L21-L32)；[README.md L56–59](https://github.com/OpenBMB/Tell_Me_More/blob/654fcec4da3b3465536d57b0d1995e8699a5a8f0/README.md#L56-L59)；[README.md L88–92](https://github.com/OpenBMB/Tell_Me_More/blob/654fcec4da3b3465536d57b0d1995e8699a5a8f0/README.md#L88-L92)。

### 74. ToolBench · 已审查

ToolLLM/ToolBench/ToolLLaMA/retriever/ToolEval 链条和 RapidAPI 外部执行均有原文；StableToolBench 指向 zhichengg 仓库，未错误计入 OpenBMB 88 个仓库或把工具 API 当模型内知识。

站内定位：[索引第 2299 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2299)；类别：模型与Agent。

原始证据：[README.md L32–41](https://github.com/OpenBMB/ToolBench/blob/d56fdd89faf8c91fa135090b212bb9057ee5cfc2/README.md#L32-L41)；[README.md L148–174](https://github.com/OpenBMB/ToolBench/blob/d56fdd89faf8c91fa135090b212bb9057ee5cfc2/README.md#L148-L174)；[README.md L272–383](https://github.com/OpenBMB/ToolBench/blob/d56fdd89faf8c91fa135090b212bb9057ee5cfc2/README.md#L272-L383)；[README.md L504–515](https://github.com/OpenBMB/ToolBench/blob/d56fdd89faf8c91fa135090b212bb9057ee5cfc2/README.md#L504-L515)。

### 75. UltraEval · 已审查

通用模型评测、统一提示及模块化数据/推理/指标，MiniCPM 使用关系有原文。工具支持范围不等于每项已经实测的边界正确。

站内定位：[索引第 2329 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2329)；类别：评测。

原始证据：[README.md L24–40](https://github.com/OpenBMB/UltraEval/blob/5d967b4ea5725ab1252904520bcaa87b40165b4b/README.md#L24-L40)；[README.md L64–153](https://github.com/OpenBMB/UltraEval/blob/5d967b4ea5725ab1252904520bcaa87b40165b4b/README.md#L64-L153)。

### 76. UltraEval-Audio · 已审查

34 benchmarks/12 tasks/10 languages 以及 v1.1 isolated process IPC、TTS/ASR/codec 均是官方当前 README 明示。没有将它说成音频模型，指标不可互相替代为工程分析。

站内定位：[索引第 2359 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2359)；类别：评测。

原始证据：[README.md L15–25](https://github.com/OpenBMB/UltraEval-Audio/blob/bead726925d43f526bed48a4a6a827595169429d/README.md#L15-L25)；[README.md L180–249](https://github.com/OpenBMB/UltraEval-Audio/blob/bead726925d43f526bed48a4a6a827595169429d/README.md#L180-L249)。

### 77. UltraFeedback · 已审查

约 64K prompts、GPT-4 多维评价、UltraRM/UltraCM 与 overall_score 修订均有原文。数据/奖励评论模型与从零聊天模型区分正确；自动反馈不等人工真值。

站内定位：[索引第 2393 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2393)；类别：数据与对齐。

原始证据：[README.md L19–29](https://github.com/OpenBMB/UltraFeedback/blob/bf80fd46a8c6ceecc86e8babb1ae8771f26a3cbb/README.md#L19-L29)；[README.md L40–46](https://github.com/OpenBMB/UltraFeedback/blob/bf80fd46a8c6ceecc86e8babb1ae8771f26a3cbb/README.md#L40-L46)；[README.md L106–122](https://github.com/OpenBMB/UltraFeedback/blob/bf80fd46a8c6ceecc86e8babb1ae8771f26a3cbb/README.md#L106-L122)。

### 78. UltraLink · 已审查

五语言（英中西俄法）、语言专有/无关知识、多轮数据和 UltraLink-LM 有原文，语言无关数据利用/减少冗余是论文方法定位。未虚构全文化覆盖；“约 5 种”略松但无数值错误。

站内定位：[索引第 2419 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2419)；类别：数据与对齐。

原始证据：[README.md L23–26](https://github.com/OpenBMB/UltraLink/blob/6f967d8376eaa8371519ce8d9dd737c040026e50/README.md#L23-L26)；[README.md L159–175](https://github.com/OpenBMB/UltraLink/blob/6f967d8376eaa8371519ce8d9dd737c040026e50/README.md#L159-L175)；[README.md L245–253](https://github.com/OpenBMB/UltraLink/blob/6f967d8376eaa8371519ce8d9dd737c040026e50/README.md#L245-L253)。

### 79. UltraRAG · 已审查

当前 3.0、MCP server/client、YAML 条件循环和模型后端有文档及接口源码。正确区别框架与单模型；AgentCPM-Report 的 2.0 实例版本差异已在两处说明。

站内定位：[索引第 2445 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2445)；类别：检索与知识。

原始证据：[README.md L33–40](https://github.com/OpenBMB/UltraRAG/blob/a763d34432007fcd1b261209f222bb10df907beb/README.md#L33-L40)；[README.md L53–55](https://github.com/OpenBMB/UltraRAG/blob/a763d34432007fcd1b261209f222bb10df907beb/README.md#L53-L55)；[README.md L87–94](https://github.com/OpenBMB/UltraRAG/blob/a763d34432007fcd1b261209f222bb10df907beb/README.md#L87-L94)；[retriever.py L15–25](https://github.com/OpenBMB/UltraRAG/blob/a763d34432007fcd1b261209f222bb10df907beb/servers/retriever/src/retriever.py#L15-L25)；[generation.py L8–49](https://github.com/OpenBMB/UltraRAG/blob/a763d34432007fcd1b261209f222bb10df907beb/servers/generation/src/generation.py#L8-L49)；[api.py L6–19](https://github.com/OpenBMB/UltraRAG/blob/a763d34432007fcd1b261209f222bb10df907beb/src/ultrarag/api.py#L6-L19)。

### 80. UltraX · 已审查

插入/删除/替换函数、LAM/DCR、滑窗聚合与执行流程有文档和解析代码；输出程序编辑指令并执行，不是逐字重写。格式正确不等事实正确及收益范围是工程分析。

站内定位：[索引第 2483 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2483)；类别：数据与对齐。

原始证据：[README.md L26–34](https://github.com/OpenBMB/UltraX/blob/cb0c2fae44c30b68e5e736a0255eef61e82f026b/README.md#L26-L34)；[README.md L45–68](https://github.com/OpenBMB/UltraX/blob/cb0c2fae44c30b68e5e736a0255eef61e82f026b/README.md#L45-L68)；[post_process_and_execute.py L197–310](https://github.com/OpenBMB/UltraX/blob/cb0c2fae44c30b68e5e736a0255eef61e82f026b/stage2_large_scale_execution/post_processing/post_process_and_execute.py#L197-L310)。

### 81. VisCPM · 已审查

CPM-Bee10B、Muffin、Diffusion-UNet、Chat 与 Paint 中英任务区分有原文；未把双语泛化夸成所有语言或单一全模态模型，权重/依赖要求分开。

站内定位：[索引第 2517 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2517)；类别：多模态模型。

原始证据：[README.md L23–28](https://github.com/OpenBMB/VisCPM/blob/d21f3ebda1d5195c49c9ca4cf6c1db8be50e0385/README.md#L23-L28)；[README.md L289–299](https://github.com/OpenBMB/VisCPM/blob/d21f3ebda1d5195c49c9ca4cf6c1db8be50e0385/README.md#L289-L299)。

### 82. VisRAG · 已审查

Ret 的 MiniCPM-V2.0 与 EVisRAG 的 Qwen2.5-VL3B/7B 区别明确；图像页面检索、逐图语言证据、RS-GRPO 有原文。没有将保留原图宣称为完全无损检索/理解。

站内定位：[索引第 2547 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2547)；类别：检索与知识。

原始证据：[README.md L28–33](https://github.com/OpenBMB/VisRAG/blob/7205787862e35d70cee7ebfd8f2b58d3eb5d563b/README.md#L28-L33)；[README.md L54–62](https://github.com/OpenBMB/VisRAG/blob/7205787862e35d70cee7ebfd8f2b58d3eb5d563b/README.md#L54-L62)；[README.md L92–112](https://github.com/OpenBMB/VisRAG/blob/7205787862e35d70cee7ebfd8f2b58d3eb5d563b/README.md#L92-L112)；[modeling_visrag_ret.py L5–30](https://github.com/OpenBMB/VisRAG/blob/7205787862e35d70cee7ebfd8f2b58d3eb5d563b/src/openmatch/modeling/modeling_visrag_ret/modeling_visrag_ret.py#L5-L30)。

### 83. VoxCPM · 已审查

0.5B/1.5/2 版本区分、VoxCPM2 官方 2B/30 语言/16k 编码48k解码正确；TSLM/RALM/LocDiT/AudioVAE 与源码对应。源码确有 ScalarQuantizationLayer，故 tokenizer-free 不等无文本 tokenizer/无量化的澄清准确。

站内定位：[索引第 2581 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2581)；类别：多模态模型。

原始证据：[README.md L41–51](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/README.md#L41-L51)；[README.md L364–386](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/README.md#L364-L386)；[voxcpm2.py L193–227](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/model/voxcpm2.py#L193-L227)；[audio_vae_v2.py L359–367](https://github.com/OpenBMB/VoxCPM/blob/f772e498a45fbb5fb8e13fbf9b9c48be9fe33e69/src/voxcpm/modules/audiovae/audio_vae_v2.py#L359-L367)。

### 84. VoxCPM-demopage · 已审查

为补强根 README 极简的证据，独立获取固定提交 index.html 并核 Git blob SHA；页面直接嵌入本地 audio 样例，另链接 Hugging Face playground，无自身模型推理 API。精选样例不等统计测试的边界准确。

站内定位：[索引第 2623 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2623)；类别：示例与文档。

原始证据：[README.md L1–1](https://github.com/OpenBMB/VoxCPM-demopage/blob/a08e4491b06136c198dd1c97ed0316393d54faca/README.md#L1-L1)；[index.html L53–56](https://github.com/OpenBMB/VoxCPM-demopage/blob/a08e4491b06136c198dd1c97ed0316393d54faca/index.html#L53-L56)；[index.html L122–135](https://github.com/OpenBMB/VoxCPM-demopage/blob/a08e4491b06136c198dd1c97ed0316393d54faca/index.html#L122-L135)。

### 85. voxcpm2-demopage · 已审查

原文就是 Markdown 到静态 demo 的生成工作区，audio/pics/css/md/output 与 render-demo.js 有明确说明，未当模型编码器或服务。样例与运行性能的区别为工程分析。

站内定位：[索引第 2649 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2649)；类别：示例与文档。

原始证据：[README.md L1–42](https://github.com/OpenBMB/voxcpm2-demopage/blob/59b843d1830ec0f9fa44270d89e9dda95c641389/README.md#L1-L42)。

### 86. WorkflowLLM · 已审查

106,763 样本、1,503 API、83 应用、Apple Shortcuts 到 Python 风格流程、Llama3.1-8B→WorkflowLlama 均有原文；数据与执行工具 API 没有混作新基础模型，生成不等成功执行的边界准确。

站内定位：[索引第 2675 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2675)；类别：模型与Agent。

原始证据：[README.MD L14–31](https://github.com/OpenBMB/WorkflowLLM/blob/2b5a2dee5fefbb6bf2fe68310393c3cf0c24cfcc/README.MD#L14-L31)。

### 87. XAgent · 已审查

Dispatcher/Planner/Actor、Docker ToolServer、Python/browser/file 及 human cooperation 有原文；解决任意任务只是目标，附录没有把愿景当保证。与 BMTools 关系由其原文交叉佐证。

站内定位：[索引第 2701 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2701)；类别：Agent框架。

原始证据：[README.md L28–31](https://github.com/OpenBMB/XAgent/blob/0dea79924347fad46e114621d7334d1136b30691/README.md#L28-L31)；[README.md L44–67](https://github.com/OpenBMB/XAgent/blob/0dea79924347fad46e114621d7334d1136b30691/README.md#L44-L67)。

### 88. XAgent-doc · 已审查

README 明确 XAgent 文档站点及 make html，附录正确归为文档而非第二套 Agent 模型。版本一致性限制为工程判断。

站内定位：[索引第 2735 行](https://github.com/limjiunnbin/model-research-atlas/blob/6613c6f584ab41eec0eed8c4c3981032595b66e5/dist/reports/openbmb/03-完整仓库索引.json#L2735)；类别：示例与文档。

原始证据：[README.md L1–31](https://github.com/OpenBMB/XAgent-doc/blob/dec88eb9cc261c978fe8bbde3997913c88815e57/README.md#L1-L31)。

## 精确缺口与额外来源

- HerculesBench：根 README 仅 OlympiadBench-；推理子文档不足以证实其正式发布意图、完整数据范围、官方指标与 OlympiadBench 正式版本关系。原附录明确披露，没有将推断写成确定事实。
- MiniCPM-V 条目的浅层 ViT 内压缩和 Omni-Flow 另对照保存的官方论文摘要 2605.08985、2604.27393；论文不计入 238 份 GitHub/HF 文件的远端逐字节一致性统计。
- ChatDev 的动态边、循环和 Thinking，以及 VoxCPM-demopage 的静态 audio 页面，追加独立获取 4 份固定提交原文件；四份 Git blob SHA 与已保存 tree 一致，见 semantic-primary-receipts.json。
- 没有完整执行仓库、调用模型 API、复现作者成绩、校验训练数据真值，或逐行审计全部源代码。用途/风险部分的工程分析仍是分析。
