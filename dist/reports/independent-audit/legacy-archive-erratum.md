# 历史归档勘误：MiniCPM3 的 Q/K 与 V 维度

更新日期：2026-09-26。

**本目录的历史 CSV 和 ZIP 保留原件，不代表现行修正版。请勿把旧 `head_dim=64` 当作 MiniCPM3-4B 的统一 Q/K/V 维度。**

MiniCPM3-4B 的 Q/K 每头维度为 **96（64+32）**；V 每头维度为 **64（2560/40）**。旧 `OpenBMB-layer-config.csv` 及 `OpenBMB-per-model-tables.zip` 中的单列 `head_dim=64` 未区分这两种含义，应按本勘误阅读。

- [现行逐层配置 CSV：QK/V 分列](https://limjiunnbin.github.io/model-research-atlas/reports/openbmb/compute/OpenBMB-layer-config.csv)
- [修正版固定提交记录](https://github.com/limjiunnbin/model-research-atlas/blob/083af0a41b1134b7424fe514c30120597df10a8f/dist/reports/openbmb/compute/OpenBMB-layer-config.csv)
- [独立模型下载中心](https://limjiunnbin.github.io/model-research-atlas/model-documents.html?model=openbmb-minicpm3-4b)
- [固定模型配置](https://huggingface.co/openbmb/MiniCPM3-4B/blob/d6b14ddaefdb11c624dd75c3c779549bc90b08cb/config.json)
- [固定实现：V 维度计算，L354](https://huggingface.co/openbmb/MiniCPM3-4B/blob/d6b14ddaefdb11c624dd75c3c779549bc90b08cb/modeling_minicpm.py#L354)

本勘误核对配置和源码含义，不代表完成权重加载、数值推理或硬件运行验证。MiniCPM3-RAG-LoRA 的自身实现尚未核验，V 维度仍标“未核验”，不套用本条源码确认。
