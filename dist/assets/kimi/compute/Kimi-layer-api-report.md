# Kimi K2—K3 逐层计算与后端接口对照

源码快照：2026-09-21；整理：2026-09-23

研究源码快照为2026-09-21，结果整理于2026-09-23；不同仓库固定提交是分别核对的研究快照，不声称它们组成已测试的兼容软件栈。

PyTorch参考计算与vLLM服务实现分开。接口列按模块/分派/设备算子分层；并列API可能是互斥分支，不能当作依次执行或1:1映射。

K2 Base没有独立完整文件头审计：尺寸模板来自同配置Instruct，保留推导证据；其余7版对齐既有检查点逐张量审计。

FP8权重scale_inv是F32二维block表，块大小128×128；INT4权重I32按输入维打包8元素/容器，group32 scale为BF16；K3路由MXFP4每U8打包2元素，group32 scale为U8。具体每张量shape在步骤里保留。

K3 A_log审计存储[128]，参考配置96 heads。vLLM a_log_weight_loader按param局部head数narrow取片；不把存储值修改成96。该差异并未通过整模型加载实测。

旧检查点rotary_emb.inv_freq存储长度可能不同于config旋转维度所需频率长度；生成运行时RoPE的具体加载/重建路径尚未逐版验证，不据此推算运行cache。

K3 mla_use_nope=true，生产构造use_rope=False：MLA为NoPE身份路径，RoPE不适用；64维共享key与512+64压缩缓存仍保留。

Ascend W4A8接口列是转换检查点/对应量化方法的条件路径，不等同直接支持原始INT4/MXFP4。MC2通信、A5特化与A3部署条件分别适用。

未执行GPU/NPU基准、权重加载或数值精度实验；未核验设备内核处明确留空。shape检查是静态相容性检查，不是运行测试。

## 符号与字段

- B：批大小；不同引擎可将序列打包

- T：本次新增token数；常规decode=1，prefill>=1

- S：注意力可见历史+本次token总长度

- N：本次有效token总数=ΣT_i；padded参考实现为B×T

- N_e：第e个专家收到的token数，动态；无丢弃/无padding时ΣN_e=N×TopK

- H：语言残差宽度7168

- heads：语言MLA/KDA头数；K2为64 MLA，K3为96

- R：AttnRes有效跨层残差块数；不是上下文长度

- Nv：视觉patch token总数，按图片/视频网格分段

- Nm：合并/池化后的视觉token数

- Np：输入patch数

- G：空间合并单元数=2×2=4

- TP：tensor-parallel分片数；局部head/投影维可能除以TP

- Vocab：词表163840

- weight_shape：逻辑权重按[输出,输入]；不是运行时NZ/转置布局

- packed_shape：safetensors文件存储形状；I32/U8是打包容器，非实际权重计算精度

- {i}：模板中的0-based层索引；逐层CSV已展开

- {e}：专家索引0≤e<experts；份数列表示该模板覆盖的全部专家

## 完整层映射

### Kimi-K2-Base

语言层 61；视觉层 0；全部组件 64。

- decoder-1 / 1-based 1 / MLA / Dense → k2-base-t1；参数 497500160；同尺寸Instruct模板推导，非Base文件头审计

- decoder-2 / 1-based 2 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-3 / 1-based 3 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-5 / 1-based 5 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-6 / 1-based 6 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-7 / 1-based 7 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-9 / 1-based 9 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-10 / 1-based 10 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-11 / 1-based 11 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-13 / 1-based 13 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-14 / 1-based 14 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-15 / 1-based 15 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-17 / 1-based 17 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-18 / 1-based 18 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-19 / 1-based 19 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-21 / 1-based 21 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-22 / 1-based 22 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-23 / 1-based 23 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-25 / 1-based 25 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-26 / 1-based 26 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-27 / 1-based 27 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-29 / 1-based 29 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-30 / 1-based 30 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-31 / 1-based 31 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-33 / 1-based 33 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-34 / 1-based 34 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-35 / 1-based 35 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-37 / 1-based 37 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-38 / 1-based 38 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-39 / 1-based 39 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-41 / 1-based 41 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-42 / 1-based 42 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-43 / 1-based 43 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-45 / 1-based 45 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-46 / 1-based 46 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-47 / 1-based 47 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-49 / 1-based 49 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-50 / 1-based 50 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-51 / 1-based 51 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-53 / 1-based 53 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-54 / 1-based 54 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-55 / 1-based 55 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-57 / 1-based 57 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-58 / 1-based 58 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-59 / 1-based 59 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- decoder-61 / 1-based 61 / MLA / MoE → k2-base-t2；参数 17059365248；同尺寸Instruct模板推导，非Base文件头审计

- embedding / 1-based 全局 / embedding / None → k2-base-t3；参数 1174405120；同尺寸Instruct模板推导，非Base文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k2-base-t4；参数 7168；同尺寸Instruct模板推导，非Base文件头审计

- output_head / 1-based 全局 / output_head / None → k2-base-t5；参数 1174405120；同尺寸Instruct模板推导，非Base文件头审计

### Kimi-K2-Instruct

语言层 61；视觉层 0；全部组件 64。

- decoder-1 / 1-based 1 / MLA / Dense → k2-instruct-t1；参数 497500160；固定检查点文件头审计

- decoder-2 / 1-based 2 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-3 / 1-based 3 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-5 / 1-based 5 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-6 / 1-based 6 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-7 / 1-based 7 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-9 / 1-based 9 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-10 / 1-based 10 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-11 / 1-based 11 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-13 / 1-based 13 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-14 / 1-based 14 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-15 / 1-based 15 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-17 / 1-based 17 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-18 / 1-based 18 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-19 / 1-based 19 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-21 / 1-based 21 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-22 / 1-based 22 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-23 / 1-based 23 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-25 / 1-based 25 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-26 / 1-based 26 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-27 / 1-based 27 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-29 / 1-based 29 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-30 / 1-based 30 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-31 / 1-based 31 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-33 / 1-based 33 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-34 / 1-based 34 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-35 / 1-based 35 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-37 / 1-based 37 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-38 / 1-based 38 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-39 / 1-based 39 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-41 / 1-based 41 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-42 / 1-based 42 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-43 / 1-based 43 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-45 / 1-based 45 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-46 / 1-based 46 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-47 / 1-based 47 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-49 / 1-based 49 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-50 / 1-based 50 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-51 / 1-based 51 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-53 / 1-based 53 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-54 / 1-based 54 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-55 / 1-based 55 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-57 / 1-based 57 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-58 / 1-based 58 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-59 / 1-based 59 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- decoder-61 / 1-based 61 / MLA / MoE → k2-instruct-t2；参数 17059365248；固定检查点文件头审计

- embedding / 1-based 全局 / embedding / None → k2-instruct-t3；参数 1174405120；固定检查点文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k2-instruct-t4；参数 7168；固定检查点文件头审计

- output_head / 1-based 全局 / output_head / None → k2-instruct-t5；参数 1174405120；固定检查点文件头审计

### Kimi-K2-Instruct-0905

语言层 61；视觉层 0；全部组件 64。

- decoder-1 / 1-based 1 / MLA / Dense → k2-instruct-0905-t1；参数 497500160；固定检查点文件头审计

- decoder-2 / 1-based 2 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-3 / 1-based 3 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-5 / 1-based 5 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-6 / 1-based 6 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-7 / 1-based 7 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-9 / 1-based 9 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-10 / 1-based 10 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-11 / 1-based 11 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-13 / 1-based 13 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-14 / 1-based 14 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-15 / 1-based 15 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-17 / 1-based 17 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-18 / 1-based 18 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-19 / 1-based 19 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-21 / 1-based 21 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-22 / 1-based 22 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-23 / 1-based 23 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-25 / 1-based 25 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-26 / 1-based 26 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-27 / 1-based 27 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-29 / 1-based 29 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-30 / 1-based 30 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-31 / 1-based 31 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-33 / 1-based 33 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-34 / 1-based 34 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-35 / 1-based 35 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-37 / 1-based 37 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-38 / 1-based 38 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-39 / 1-based 39 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-41 / 1-based 41 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-42 / 1-based 42 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-43 / 1-based 43 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-45 / 1-based 45 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-46 / 1-based 46 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-47 / 1-based 47 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-49 / 1-based 49 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-50 / 1-based 50 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-51 / 1-based 51 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-53 / 1-based 53 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-54 / 1-based 54 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-55 / 1-based 55 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-57 / 1-based 57 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-58 / 1-based 58 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-59 / 1-based 59 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- decoder-61 / 1-based 61 / MLA / MoE → k2-instruct-0905-t2；参数 17059365248；固定检查点文件头审计

- embedding / 1-based 全局 / embedding / None → k2-instruct-0905-t3；参数 1174405120；固定检查点文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k2-instruct-0905-t4；参数 7168；固定检查点文件头审计

- output_head / 1-based 全局 / output_head / None → k2-instruct-0905-t5；参数 1174405120；固定检查点文件头审计

### Kimi-K2-Thinking

语言层 61；视觉层 0；全部组件 64。

- decoder-1 / 1-based 1 / MLA / Dense → k2-thinking-t1；参数 497500160；固定检查点文件头审计

- decoder-2 / 1-based 2 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-3 / 1-based 3 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-5 / 1-based 5 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-6 / 1-based 6 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-7 / 1-based 7 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-9 / 1-based 9 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-10 / 1-based 10 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-11 / 1-based 11 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-13 / 1-based 13 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-14 / 1-based 14 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-15 / 1-based 15 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-17 / 1-based 17 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-18 / 1-based 18 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-19 / 1-based 19 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-21 / 1-based 21 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-22 / 1-based 22 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-23 / 1-based 23 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-25 / 1-based 25 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-26 / 1-based 26 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-27 / 1-based 27 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-29 / 1-based 29 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-30 / 1-based 30 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-31 / 1-based 31 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-33 / 1-based 33 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-34 / 1-based 34 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-35 / 1-based 35 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-37 / 1-based 37 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-38 / 1-based 38 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-39 / 1-based 39 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-41 / 1-based 41 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-42 / 1-based 42 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-43 / 1-based 43 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-45 / 1-based 45 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-46 / 1-based 46 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-47 / 1-based 47 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-49 / 1-based 49 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-50 / 1-based 50 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-51 / 1-based 51 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-53 / 1-based 53 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-54 / 1-based 54 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-55 / 1-based 55 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-57 / 1-based 57 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-58 / 1-based 58 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-59 / 1-based 59 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- decoder-61 / 1-based 61 / MLA / MoE → k2-thinking-t2；参数 17059365248；固定检查点文件头审计

- embedding / 1-based 全局 / embedding / None → k2-thinking-t3；参数 1174405120；固定检查点文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k2-thinking-t4；参数 7168；固定检查点文件头审计

- output_head / 1-based 全局 / output_head / None → k2-thinking-t5；参数 1174405120；固定检查点文件头审计

### Kimi-K2.5

语言层 61；视觉层 27；全部组件 93。

- decoder-1 / 1-based 1 / MLA / Dense → k2.5-t1；参数 497500160；固定检查点文件头审计

- decoder-2 / 1-based 2 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-3 / 1-based 3 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-5 / 1-based 5 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-6 / 1-based 6 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-7 / 1-based 7 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-9 / 1-based 9 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-10 / 1-based 10 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-11 / 1-based 11 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-13 / 1-based 13 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-14 / 1-based 14 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-15 / 1-based 15 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-17 / 1-based 17 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-18 / 1-based 18 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-19 / 1-based 19 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-21 / 1-based 21 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-22 / 1-based 22 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-23 / 1-based 23 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-25 / 1-based 25 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-26 / 1-based 26 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-27 / 1-based 27 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-29 / 1-based 29 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-30 / 1-based 30 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-31 / 1-based 31 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-33 / 1-based 33 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-34 / 1-based 34 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-35 / 1-based 35 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-37 / 1-based 37 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-38 / 1-based 38 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-39 / 1-based 39 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-41 / 1-based 41 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-42 / 1-based 42 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-43 / 1-based 43 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-45 / 1-based 45 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-46 / 1-based 46 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-47 / 1-based 47 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-49 / 1-based 49 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-50 / 1-based 50 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-51 / 1-based 51 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-53 / 1-based 53 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-54 / 1-based 54 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-55 / 1-based 55 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-57 / 1-based 57 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-58 / 1-based 58 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-59 / 1-based 59 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- decoder-61 / 1-based 61 / MLA / MoE → k2.5-t2；参数 17059365248；固定检查点文件头审计

- vision-1 / 1-based 1 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-2 / 1-based 2 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-3 / 1-based 3 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-4 / 1-based 4 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-5 / 1-based 5 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-6 / 1-based 6 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-7 / 1-based 7 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-8 / 1-based 8 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-9 / 1-based 9 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-10 / 1-based 10 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-11 / 1-based 11 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-12 / 1-based 12 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-13 / 1-based 13 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-14 / 1-based 14 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-15 / 1-based 15 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-16 / 1-based 16 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-17 / 1-based 17 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-18 / 1-based 18 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-19 / 1-based 19 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-20 / 1-based 20 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-21 / 1-based 21 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-22 / 1-based 22 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-23 / 1-based 23 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-24 / 1-based 24 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-25 / 1-based 25 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-26 / 1-based 26 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- vision-27 / 1-based 27 / Vision / MLP → k2.5-t3；参数 15239504；固定检查点文件头审计

- embedding / 1-based 全局 / embedding / None → k2.5-t4；参数 1174405120；固定检查点文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k2.5-t5；参数 7168；固定检查点文件头审计

- output_head / 1-based 全局 / output_head / None → k2.5-t6；参数 1174405120；固定检查点文件头审计

- projector / 1-based 全局 / projector / None → k2.5-t7；参数 54277888；固定检查点文件头审计

- vision_other / 1-based 全局 / vision_other / None → k2.5-t8；参数 5399424；固定检查点文件头审计

### Kimi-K2.6

语言层 61；视觉层 27；全部组件 93。

- decoder-1 / 1-based 1 / MLA / Dense → k2.6-t1；参数 497500160；固定检查点文件头审计

- decoder-2 / 1-based 2 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-3 / 1-based 3 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-5 / 1-based 5 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-6 / 1-based 6 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-7 / 1-based 7 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-9 / 1-based 9 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-10 / 1-based 10 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-11 / 1-based 11 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-13 / 1-based 13 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-14 / 1-based 14 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-15 / 1-based 15 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-17 / 1-based 17 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-18 / 1-based 18 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-19 / 1-based 19 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-21 / 1-based 21 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-22 / 1-based 22 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-23 / 1-based 23 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-25 / 1-based 25 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-26 / 1-based 26 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-27 / 1-based 27 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-29 / 1-based 29 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-30 / 1-based 30 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-31 / 1-based 31 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-33 / 1-based 33 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-34 / 1-based 34 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-35 / 1-based 35 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-37 / 1-based 37 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-38 / 1-based 38 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-39 / 1-based 39 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-41 / 1-based 41 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-42 / 1-based 42 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-43 / 1-based 43 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-45 / 1-based 45 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-46 / 1-based 46 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-47 / 1-based 47 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-49 / 1-based 49 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-50 / 1-based 50 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-51 / 1-based 51 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-53 / 1-based 53 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-54 / 1-based 54 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-55 / 1-based 55 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-57 / 1-based 57 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-58 / 1-based 58 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-59 / 1-based 59 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- decoder-61 / 1-based 61 / MLA / MoE → k2.6-t2；参数 17059365248；固定检查点文件头审计

- vision-1 / 1-based 1 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-2 / 1-based 2 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-3 / 1-based 3 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-4 / 1-based 4 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-5 / 1-based 5 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-6 / 1-based 6 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-7 / 1-based 7 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-8 / 1-based 8 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-9 / 1-based 9 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-10 / 1-based 10 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-11 / 1-based 11 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-12 / 1-based 12 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-13 / 1-based 13 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-14 / 1-based 14 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-15 / 1-based 15 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-16 / 1-based 16 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-17 / 1-based 17 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-18 / 1-based 18 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-19 / 1-based 19 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-20 / 1-based 20 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-21 / 1-based 21 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-22 / 1-based 22 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-23 / 1-based 23 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-24 / 1-based 24 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-25 / 1-based 25 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-26 / 1-based 26 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- vision-27 / 1-based 27 / Vision / MLP → k2.6-t3；参数 15239504；固定检查点文件头审计

- embedding / 1-based 全局 / embedding / None → k2.6-t4；参数 1174405120；固定检查点文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k2.6-t5；参数 7168；固定检查点文件头审计

- output_head / 1-based 全局 / output_head / None → k2.6-t6；参数 1174405120；固定检查点文件头审计

- projector / 1-based 全局 / projector / None → k2.6-t7；参数 54277888；固定检查点文件头审计

- vision_other / 1-based 全局 / vision_other / None → k2.6-t8；参数 5399424；固定检查点文件头审计

### Kimi-K2.7-Code

语言层 61；视觉层 27；全部组件 93。

- decoder-1 / 1-based 1 / MLA / Dense → k2.7-code-t1；参数 497500160；固定检查点文件头审计

- decoder-2 / 1-based 2 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-3 / 1-based 3 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-5 / 1-based 5 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-6 / 1-based 6 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-7 / 1-based 7 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-9 / 1-based 9 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-10 / 1-based 10 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-11 / 1-based 11 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-13 / 1-based 13 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-14 / 1-based 14 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-15 / 1-based 15 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-17 / 1-based 17 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-18 / 1-based 18 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-19 / 1-based 19 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-21 / 1-based 21 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-22 / 1-based 22 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-23 / 1-based 23 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-25 / 1-based 25 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-26 / 1-based 26 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-27 / 1-based 27 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-29 / 1-based 29 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-30 / 1-based 30 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-31 / 1-based 31 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-33 / 1-based 33 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-34 / 1-based 34 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-35 / 1-based 35 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-37 / 1-based 37 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-38 / 1-based 38 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-39 / 1-based 39 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-41 / 1-based 41 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-42 / 1-based 42 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-43 / 1-based 43 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-45 / 1-based 45 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-46 / 1-based 46 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-47 / 1-based 47 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-49 / 1-based 49 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-50 / 1-based 50 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-51 / 1-based 51 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-53 / 1-based 53 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-54 / 1-based 54 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-55 / 1-based 55 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-57 / 1-based 57 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-58 / 1-based 58 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-59 / 1-based 59 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- decoder-61 / 1-based 61 / MLA / MoE → k2.7-code-t2；参数 17059365248；固定检查点文件头审计

- vision-1 / 1-based 1 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-2 / 1-based 2 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-3 / 1-based 3 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-4 / 1-based 4 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-5 / 1-based 5 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-6 / 1-based 6 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-7 / 1-based 7 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-8 / 1-based 8 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-9 / 1-based 9 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-10 / 1-based 10 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-11 / 1-based 11 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-12 / 1-based 12 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-13 / 1-based 13 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-14 / 1-based 14 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-15 / 1-based 15 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-16 / 1-based 16 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-17 / 1-based 17 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-18 / 1-based 18 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-19 / 1-based 19 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-20 / 1-based 20 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-21 / 1-based 21 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-22 / 1-based 22 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-23 / 1-based 23 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-24 / 1-based 24 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-25 / 1-based 25 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-26 / 1-based 26 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- vision-27 / 1-based 27 / Vision / MLP → k2.7-code-t3；参数 15239504；固定检查点文件头审计

- embedding / 1-based 全局 / embedding / None → k2.7-code-t4；参数 1174405120；固定检查点文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k2.7-code-t5；参数 7168；固定检查点文件头审计

- output_head / 1-based 全局 / output_head / None → k2.7-code-t6；参数 1174405120；固定检查点文件头审计

- projector / 1-based 全局 / projector / None → k2.7-code-t7；参数 54277888；固定检查点文件头审计

- vision_other / 1-based 全局 / vision_other / None → k2.7-code-t8；参数 5399424；固定检查点文件头审计

### Kimi-K3

语言层 93；视觉层 27；全部组件 125。

- decoder-1 / 1-based 1 / KDA / Dense → k3-t1；参数 1170446592；固定检查点文件头审计

- decoder-2 / 1-based 2 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-3 / 1-based 3 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-4 / 1-based 4 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-5 / 1-based 5 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-6 / 1-based 6 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-7 / 1-based 7 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-8 / 1-based 8 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-9 / 1-based 9 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-10 / 1-based 10 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-11 / 1-based 11 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-12 / 1-based 12 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-13 / 1-based 13 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-14 / 1-based 14 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-15 / 1-based 15 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-16 / 1-based 16 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-17 / 1-based 17 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-18 / 1-based 18 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-19 / 1-based 19 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-20 / 1-based 20 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-21 / 1-based 21 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-22 / 1-based 22 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-23 / 1-based 23 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-24 / 1-based 24 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-25 / 1-based 25 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-26 / 1-based 26 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-27 / 1-based 27 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-28 / 1-based 28 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-29 / 1-based 29 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-30 / 1-based 30 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-31 / 1-based 31 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-32 / 1-based 32 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-33 / 1-based 33 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-34 / 1-based 34 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-35 / 1-based 35 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-36 / 1-based 36 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-37 / 1-based 37 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-38 / 1-based 38 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-39 / 1-based 39 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-40 / 1-based 40 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-41 / 1-based 41 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-42 / 1-based 42 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-43 / 1-based 43 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-44 / 1-based 44 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-45 / 1-based 45 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-46 / 1-based 46 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-47 / 1-based 47 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-48 / 1-based 48 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-49 / 1-based 49 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-50 / 1-based 50 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-51 / 1-based 51 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-52 / 1-based 52 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-53 / 1-based 53 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-54 / 1-based 54 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-55 / 1-based 55 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-56 / 1-based 56 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-57 / 1-based 57 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-58 / 1-based 58 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-59 / 1-based 59 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-60 / 1-based 60 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-61 / 1-based 61 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-62 / 1-based 62 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-63 / 1-based 63 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-64 / 1-based 64 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-65 / 1-based 65 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-66 / 1-based 66 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-67 / 1-based 67 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-68 / 1-based 68 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-69 / 1-based 69 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-70 / 1-based 70 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-71 / 1-based 71 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-72 / 1-based 72 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-73 / 1-based 73 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-74 / 1-based 74 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-75 / 1-based 75 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-76 / 1-based 76 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-77 / 1-based 77 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-78 / 1-based 78 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-79 / 1-based 79 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-80 / 1-based 80 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-81 / 1-based 81 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-82 / 1-based 82 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-83 / 1-based 83 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-84 / 1-based 84 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-85 / 1-based 85 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-86 / 1-based 86 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-87 / 1-based 87 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-88 / 1-based 88 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-89 / 1-based 89 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-90 / 1-based 90 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-91 / 1-based 91 / KDA / MoE → k3-t2；参数 30228720256；固定检查点文件头审计

- decoder-92 / 1-based 92 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- decoder-93 / 1-based 93 / MLA / MoE → k3-t3；参数 30017175936；固定检查点文件头审计

- vision-1 / 1-based 1 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-2 / 1-based 2 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-3 / 1-based 3 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-4 / 1-based 4 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-5 / 1-based 5 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-6 / 1-based 6 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-7 / 1-based 7 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-8 / 1-based 8 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-9 / 1-based 9 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-10 / 1-based 10 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-11 / 1-based 11 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-12 / 1-based 12 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-13 / 1-based 13 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-14 / 1-based 14 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-15 / 1-based 15 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-16 / 1-based 16 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-17 / 1-based 17 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-18 / 1-based 18 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-19 / 1-based 19 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-20 / 1-based 20 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-21 / 1-based 21 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-22 / 1-based 22 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-23 / 1-based 23 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-24 / 1-based 24 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-25 / 1-based 25 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-26 / 1-based 26 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- vision-27 / 1-based 27 / Vision / MLP → k3-t4；参数 14682112；固定检查点文件头审计

- embedding / 1-based 全局 / embedding / None → k3-t5；参数 1174405120；固定检查点文件头审计

- final_norm_and_residual / 1-based 全局 / final_norm_and_residual / None → k3-t6；参数 21504；固定检查点文件头审计

- output_head / 1-based 全局 / output_head / None → k3-t7；参数 1174405120；固定检查点文件头审计

- projector / 1-based 全局 / projector / None → k3-t8；参数 46144512；固定检查点文件头审计

- vision_other / 1-based 全局 / vision_other / None → k3-t9；参数 4797440；固定检查点文件头审计

## 步骤模板（逐层CSV已完整展开）

### k2-base-t1

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight_scale_inv：logical metadata；stored 12x56 F32；multiplicity 1

- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight_scale_inv：logical metadata；stored 96x12 F32；multiplicity 1

- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv：logical metadata；stored 5x56 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c5baa6bc5b14

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv：logical metadata；stored 128x4 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c5baa6bc5b14

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 56 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：260cc1915f9b

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：c5baa6bc5b14-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：c5baa6bc5b14-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight_scale_inv：logical metadata；stored 56x64 F32；multiplicity 1

- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：cca6a392eb8c

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.gate_proj.weight_scale_inv：logical metadata；stored 144x56 F32；multiplicity 1

- model.layers.{i}.mlp.gate_proj.weight：logical 18432x7168；stored 18432x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s17 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.up_proj.weight_scale_inv：logical metadata；stored 144x56 F32；multiplicity 1

- model.layers.{i}.mlp.up_proj.weight：logical 18432x7168；stored 18432x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s18 门控激活

gate/up:[N,18432] → SiLU(gate)×up → [N,18432]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ed2d079af130

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s19 Dense下投影

[N,18432] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.down_proj.weight_scale_inv：logical metadata；stored 56x144 F32；multiplicity 1

- model.layers.{i}.mlp.down_proj.weight：logical 7168x18432；stored 7168x18432 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s20 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：cca6a392eb8c

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-base-t2

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight_scale_inv：logical metadata；stored 12x56 F32；multiplicity 1

- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight_scale_inv：logical metadata；stored 96x12 F32；multiplicity 1

- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv：logical metadata；stored 5x56 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c5baa6bc5b14

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv：logical metadata；stored 128x4 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c5baa6bc5b14

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 56 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：260cc1915f9b

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：c5baa6bc5b14-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：c5baa6bc5b14-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight_scale_inv：logical metadata；stored 56x64 F32；multiplicity 1

- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：cca6a392eb8c

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,384]

无持久缓存



- model.layers.{i}.mlp.gate.e_score_correction_bias：logical 384；stored 384 F32；multiplicity 1

- model.layers.{i}.mlp.gate.weight：logical 384x7168；stored 384x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b6d3839075c1

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s17 Top-K专家选择

scores:[N,384] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,8]

无持久缓存



- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b6d3839075c1

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s18 专家分发与token重排

X:[N,7168]; ids:[N,8] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,7168]

Σ_e N_e=N×8（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：de2e790cec87

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s19 路由专家门控GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.gate_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：de2e790cec87

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s20 路由专家上投影GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.up_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：de2e790cec87

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s21 门控激活

gate/up:[N_e,2048] → SiLU(gate)×up → [N_e,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ed2d079af130

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s22 路由专家下投影GEMM

[N_e,2048] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,7168]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale_inv：logical metadata；stored 56x16 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.down_proj.weight：logical 7168x2048；stored 7168x2048 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：de2e790cec87

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s23 专家加权合并

Y_e:[N_e,7168]; weights:[N,8] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,7168]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：de2e790cec87-601

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s24 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.gate_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.gate_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s25 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.up_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.up_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s26 共享专家门控激活

gate/up:[N,2048] → SiLU(gate)×up → [N,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ed2d079af130

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s27 共享专家下投影

[N,2048] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.shared_experts.down_proj.weight_scale_inv：logical metadata；stored 56x16 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.down_proj.weight：logical 7168x2048；stored 7168x2048 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s28 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：cca6a392eb8c

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s29 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：cca6a392eb8c

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-base-t3

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k2-base-t4

#### s1 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：DeepseekV3RMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e37fe49f1452

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k2-base-t5

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k2-instruct-t1

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight_scale_inv：logical metadata；stored 12x56 F32；multiplicity 1

- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight_scale_inv：logical metadata；stored 96x12 F32；multiplicity 1

- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv：logical metadata；stored 5x56 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a3ec1b8b4763

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv：logical metadata；stored 128x4 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a3ec1b8b4763

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 56 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：04b9cf8f5c4a

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a3ec1b8b4763-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a3ec1b8b4763-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight_scale_inv：logical metadata；stored 56x64 F32；multiplicity 1

- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：872642b6c3e2

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.gate_proj.weight_scale_inv：logical metadata；stored 144x56 F32；multiplicity 1

- model.layers.{i}.mlp.gate_proj.weight：logical 18432x7168；stored 18432x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s17 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.up_proj.weight_scale_inv：logical metadata；stored 144x56 F32；multiplicity 1

- model.layers.{i}.mlp.up_proj.weight：logical 18432x7168；stored 18432x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s18 门控激活

gate/up:[N,18432] → SiLU(gate)×up → [N,18432]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：96c39d65fc6f

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s19 Dense下投影

[N,18432] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.down_proj.weight_scale_inv：logical metadata；stored 56x144 F32；multiplicity 1

- model.layers.{i}.mlp.down_proj.weight：logical 7168x18432；stored 7168x18432 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s20 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：872642b6c3e2

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-instruct-t2

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight_scale_inv：logical metadata；stored 12x56 F32；multiplicity 1

- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight_scale_inv：logical metadata；stored 96x12 F32；multiplicity 1

- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv：logical metadata；stored 5x56 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a3ec1b8b4763

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv：logical metadata；stored 128x4 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a3ec1b8b4763

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 56 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：04b9cf8f5c4a

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a3ec1b8b4763-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a3ec1b8b4763-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight_scale_inv：logical metadata；stored 56x64 F32；multiplicity 1

- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：872642b6c3e2

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,384]

无持久缓存



- model.layers.{i}.mlp.gate.e_score_correction_bias：logical 384；stored 384 F32；multiplicity 1

- model.layers.{i}.mlp.gate.weight：logical 384x7168；stored 384x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a396e8fd85da

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s17 Top-K专家选择

scores:[N,384] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,8]

无持久缓存



- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a396e8fd85da

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s18 专家分发与token重排

X:[N,7168]; ids:[N,8] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,7168]

Σ_e N_e=N×8（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5da897a9f506

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s19 路由专家门控GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.gate_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5da897a9f506

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s20 路由专家上投影GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.up_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5da897a9f506

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s21 门控激活

gate/up:[N_e,2048] → SiLU(gate)×up → [N_e,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：96c39d65fc6f

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s22 路由专家下投影GEMM

[N_e,2048] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,7168]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale_inv：logical metadata；stored 56x16 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.down_proj.weight：logical 7168x2048；stored 7168x2048 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5da897a9f506

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s23 专家加权合并

Y_e:[N_e,7168]; weights:[N,8] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,7168]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：5da897a9f506-601

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s24 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.gate_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.gate_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s25 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.up_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.up_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s26 共享专家门控激活

gate/up:[N,2048] → SiLU(gate)×up → [N,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：96c39d65fc6f

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s27 共享专家下投影

[N,2048] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.shared_experts.down_proj.weight_scale_inv：logical metadata；stored 56x16 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.down_proj.weight：logical 7168x2048；stored 7168x2048 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s28 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：872642b6c3e2

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s29 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：872642b6c3e2

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-instruct-t3

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k2-instruct-t4

#### s1 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：DeepseekV3RMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：966cf86d3cd6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k2-instruct-t5

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k2-instruct-0905-t1

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight_scale_inv：logical metadata；stored 12x56 F32；multiplicity 1

- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight_scale_inv：logical metadata；stored 96x12 F32；multiplicity 1

- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv：logical metadata；stored 5x56 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：49f3797d886b

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv：logical metadata；stored 128x4 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：49f3797d886b

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 128 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：8b9d444c7c4f

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：49f3797d886b-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：49f3797d886b-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight_scale_inv：logical metadata；stored 56x64 F32；multiplicity 1

- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：04227bc06df5

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.gate_proj.weight_scale_inv：logical metadata；stored 144x56 F32；multiplicity 1

- model.layers.{i}.mlp.gate_proj.weight：logical 18432x7168；stored 18432x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s17 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.up_proj.weight_scale_inv：logical metadata；stored 144x56 F32；multiplicity 1

- model.layers.{i}.mlp.up_proj.weight：logical 18432x7168；stored 18432x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s18 门控激活

gate/up:[N,18432] → SiLU(gate)×up → [N,18432]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5eb11ce8a99d

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s19 Dense下投影

[N,18432] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.down_proj.weight_scale_inv：logical metadata；stored 56x144 F32；multiplicity 1

- model.layers.{i}.mlp.down_proj.weight：logical 7168x18432；stored 7168x18432 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s20 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：04227bc06df5

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-instruct-0905-t2

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight_scale_inv：logical metadata；stored 12x56 F32；multiplicity 1

- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight_scale_inv：logical metadata；stored 96x12 F32；multiplicity 1

- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight_scale_inv：logical metadata；stored 5x56 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：49f3797d886b

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight_scale_inv：logical metadata；stored 128x4 F32；multiplicity 1

- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：49f3797d886b

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 128 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：8b9d444c7c4f

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：49f3797d886b-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：49f3797d886b-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight_scale_inv：logical metadata；stored 56x64 F32；multiplicity 1

- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：04227bc06df5

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,384]

无持久缓存



- model.layers.{i}.mlp.gate.e_score_correction_bias：logical 384；stored 384 F32；multiplicity 1

- model.layers.{i}.mlp.gate.weight：logical 384x7168；stored 384x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：6976ecff6eb7

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s17 Top-K专家选择

scores:[N,384] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,8]

无持久缓存



- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：6976ecff6eb7

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s18 专家分发与token重排

X:[N,7168]; ids:[N,8] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,7168]

Σ_e N_e=N×8（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e3f1a60d4575

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s19 路由专家门控GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.gate_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e3f1a60d4575

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s20 路由专家上投影GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.up_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e3f1a60d4575

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s21 门控激活

gate/up:[N_e,2048] → SiLU(gate)×up → [N_e,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5eb11ce8a99d

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s22 路由专家下投影GEMM

[N_e,2048] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,7168]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale_inv：logical metadata；stored 56x16 F32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.down_proj.weight：logical 7168x2048；stored 7168x2048 F8_E4M3；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e3f1a60d4575

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s23 专家加权合并

Y_e:[N_e,7168]; weights:[N,8] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,7168]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：e3f1a60d4575-601

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s24 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.gate_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.gate_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s25 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.up_proj.weight_scale_inv：logical metadata；stored 16x56 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.up_proj.weight：logical 2048x7168；stored 2048x7168 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s26 共享专家门控激活

gate/up:[N,2048] → SiLU(gate)×up → [N,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5eb11ce8a99d

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s27 共享专家下投影

[N,2048] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.shared_experts.down_proj.weight_scale_inv：logical metadata；stored 56x16 F32；multiplicity 1

- model.layers.{i}.mlp.shared_experts.down_proj.weight：logical 7168x2048；stored 7168x2048 F8_E4M3；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- amd / 框架函数 / 条件分派：Fp8LinearMethod.apply → self.fp8_linear.apply_weights。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3f8018165c8c

- ascend / 未知：未核验。条件：检查点低精度格式需要匹配量化方法；不能由存储dtype直接指定一个Ascend算子。。来源：

#### s28 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：04227bc06df5

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s29 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：04227bc06df5

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-instruct-0905-t3

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k2-instruct-0905-t4

#### s1 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：DeepseekV3RMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：373d2ad688c6

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k2-instruct-0905-t5

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k2-thinking-t1

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a84d4ae48f26

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a84d4ae48f26

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 128 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：932106e78793

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a84d4ae48f26-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a84d4ae48f26-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a9a94eb60491

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.gate_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s17 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- model.layers.{i}.mlp.up_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s18 门控激活

gate/up:[N,18432] → SiLU(gate)×up → [N,18432]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7418dca0e99a

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s19 Dense下投影

[N,18432] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.down_proj.weight：logical 7168x18432；stored 7168x18432 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s20 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a9a94eb60491

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-thinking-t2

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a84d4ae48f26

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a84d4ae48f26

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- model.layers.{i}.self_attn.rotary_emb.inv_freq：logical metadata；stored 128 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：932106e78793

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a84d4ae48f26-816

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a84d4ae48f26-840

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a9a94eb60491

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,384]

无持久缓存



- model.layers.{i}.mlp.gate.e_score_correction_bias：logical 384；stored 384 F32；multiplicity 1

- model.layers.{i}.mlp.gate.weight：logical 384x7168；stored 384x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7deaea179cf8

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s17 Top-K专家选择

scores:[N,384] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,8]

无持久缓存



- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7deaea179cf8

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s18 专家分发与token重排

X:[N,7168]; ids:[N,8] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,7168]

Σ_e N_e=N×8（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a69383bbaecd

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s19 路由专家门控GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.gate_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.gate_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a69383bbaecd

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s20 路由专家上投影GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.up_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.up_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a69383bbaecd

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s21 门控激活

gate/up:[N_e,2048] → SiLU(gate)×up → [N_e,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7418dca0e99a

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s22 路由专家下投影GEMM

[N_e,2048] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,7168]

无持久缓存



- model.layers.{i}.mlp.experts.{e}.down_proj.weight_packed：logical 7168x2048；stored 7168x256 I32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.down_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale：logical metadata；stored 7168x64 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a69383bbaecd

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s23 专家加权合并

Y_e:[N_e,7168]; weights:[N,8] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,7168]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a69383bbaecd-601

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s24 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.gate_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s25 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- model.layers.{i}.mlp.shared_experts.up_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s26 共享专家门控激活

gate/up:[N,2048] → SiLU(gate)×up → [N,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7418dca0e99a

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s27 共享专家下投影

[N,2048] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- model.layers.{i}.mlp.shared_experts.down_proj.weight：logical 7168x2048；stored 7168x2048 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s28 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a9a94eb60491

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s29 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a9a94eb60491

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2-thinking-t3

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k2-thinking-t4

#### s1 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：DeepseekV3RMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4772e70c5fef

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k2-thinking-t5

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k2.5-t1

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a5794f7773fa

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a5794f7773fa

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ef5f0a39e0de

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a5794f7773fa-812

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a5794f7773fa-836

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：43a137f02277

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- language_model.model.layers.{i}.mlp.gate_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s17 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- language_model.model.layers.{i}.mlp.up_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s18 门控激活

gate/up:[N,18432] → SiLU(gate)×up → [N,18432]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c6983d2868ff

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s19 Dense下投影

[N,18432] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.down_proj.weight：logical 7168x18432；stored 7168x18432 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s20 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：43a137f02277

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.5-t2

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a5794f7773fa

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a5794f7773fa

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ef5f0a39e0de

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a5794f7773fa-812

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：a5794f7773fa-836

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：43a137f02277

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,384]

无持久缓存



- language_model.model.layers.{i}.mlp.gate.e_score_correction_bias：logical 384；stored 384 F32；multiplicity 1

- language_model.model.layers.{i}.mlp.gate.weight：logical 384x7168；stored 384x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d5e9baafcc2d

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s17 Top-K专家选择

scores:[N,384] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,8]

无持久缓存



- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d5e9baafcc2d

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s18 专家分发与token重排

X:[N,7168]; ids:[N,8] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,7168]

Σ_e N_e=N×8（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c9bacd41e46b

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s19 路由专家门控GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c9bacd41e46b

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s20 路由专家上投影GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c9bacd41e46b

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s21 门控激活

gate/up:[N_e,2048] → SiLU(gate)×up → [N_e,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c6983d2868ff

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s22 路由专家下投影GEMM

[N_e,2048] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_packed：logical 7168x2048；stored 7168x256 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale：logical metadata；stored 7168x64 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c9bacd41e46b

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s23 专家加权合并

Y_e:[N_e,7168]; weights:[N,8] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,7168]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：c9bacd41e46b-605

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s24 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.gate_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s25 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.up_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s26 共享专家门控激活

gate/up:[N,2048] → SiLU(gate)×up → [N,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c6983d2868ff

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s27 共享专家下投影

[N,2048] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.down_proj.weight：logical 7168x2048；stored 7168x2048 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s28 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：43a137f02277

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s29 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：43a137f02277

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.5-t3

#### s1 视觉注意力前归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm0.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.norm0.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ab36b61a75be

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s2 视觉融合QKV投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,3456]

无持久缓存



- vision_tower.encoder.blocks.{i}.wqkv.bias：logical 3456；stored 3456 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.wqkv.weight：logical 3456x1152；stored 3456x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 视觉QKV拆分 / RoPE

[Nv,3456] → view(...,3,heads,d)→unbind→二维RoPE → Q,K,V:[Nv,16,72]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ac58dedbdf7c

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s4 视觉注意力核心

Q,K,V:[Nv,16,72] → 按cu_seqlens分段计算非因果视觉attention；不跨独立图像任意注意 → [Nv,1152]

逻辑P:[heads,Lv,Lv]按每视觉段；Flash实现不必物化；无文本自回归KV缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ac58dedbdf7c

- nvidia / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- amd / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- ascend / 未知：未核验。条件：Ascend视觉塔集成已知，但MMEncoderAttention到具体CANN内核未追踪；不借用语言MLA接口。。来源：

#### s5 视觉注意力输出投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.wo.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.wo.weight：logical 1152x1152；stored 1152x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 视觉注意力残差

branch,residual:[Nv,1152] → 相加 → [Nv,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ab36b61a75be

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 视觉MLP前归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm1.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.norm1.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ab36b61a75be

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s8 视觉MLP上投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,4304]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc0.bias：logical 4304；stored 4304 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.mlp.fc0.weight：logical 4304x1152；stored 4304x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 视觉GELU

[Nv,4304] → GELU；具体tanh近似由vision config决定 → [Nv,4304]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MLP2.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d2fd11ae84f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 视觉MLP下投影

[Nv,4304] → Y=X @ Wᵀ + bias → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc1.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.mlp.fc1.weight：logical 1152x4304；stored 1152x4304 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s11 视觉MLP残差

branch,residual:[Nv,1152] → 相加 → [Nv,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ab36b61a75be

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.5-t4

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- language_model.model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k2.5-t5

#### s1 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：DeepseekV3RMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：343f4ac2b717

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k2.5-t6

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- language_model.lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k2.5-t7

#### s1 空间合并与时间池化

视觉片段:[frames,h,w,1152] → 视觉塔tpool_patch_merger先空间2×2打包并按时间组均值；随后进入projector；Nm依赖processor网格 → grouped:[Nm,4,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：tpool_patch_merger（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：025c90ee058b

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 时间池化后、展平前LayerNorm

[Nm,G,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nm,G,1152]

无持久缓存



- mm_projector.pre_norm.bias：logical 1152；stored 1152 BF16；multiplicity 1

- mm_projector.pre_norm.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a0129ccce050

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 连接器输入展平

[Nm,4,1152] → projector内部view；前序pre_norm仅适用于K2.5/2.6/2.7 → [Nm,4608]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a0129ccce050

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s4 连接器第一线性

[Nm,4608] → Y=X @ Wᵀ + bias → [Nm,4608]

无持久缓存



- mm_projector.proj.0.bias：logical 4608；stored 4608 BF16；multiplicity 1

- mm_projector.proj.0.weight：logical 4608x4608；stored 4608x4608 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 连接器GELU

[Nm,4608] → nn.GELU → [Nm,4608]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a0129ccce050

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s6 投影到语言宽度

[Nm,4608] → Y=X @ Wᵀ + bias → [Nm,7168]

无持久缓存



- mm_projector.proj.2.bias：logical 7168；stored 7168 BF16；multiplicity 1

- mm_projector.proj.2.weight：logical 7168x4608；stored 7168x4608 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s7 插入多模态embedding

text embedding:[B,T,7168]; visual:[Nm,7168] → 按media placeholder位置填入视觉特征；数量由processor与网格校验 → [B,T,7168]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：KimiK25ForConditionalGeneration._merge_input_ids_with_image_features（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d15b277cb06d

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.5-t8

#### s1 Patch嵌入卷积

[Np,3,14,14] → Conv2d(patch14×14,stride14)，展平视觉patch序列 → [Np,1152]

无持久缓存



- vision_tower.patch_embed.proj.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.patch_embed.proj.weight：logical 1152x3x14x14；stored 1152x3x14x14 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f12177b1227e

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 插值位置嵌入

[Nv,1152] → 按图像网格插值位置表，并加到patch特征；不是GEMM → [Nv,1152]

无持久缓存



- vision_tower.patch_embed.pos_emb.weight：logical 64x64x1152；stored 64x64x1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f12177b1227e

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 视觉最终归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.final_layernorm.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.final_layernorm.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViT3dEncoder.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5f7e9167e6b3

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.6-t1

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ad7a465c7aa3

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ad7a465c7aa3

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：8a172ed6bb67

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：ad7a465c7aa3-812

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：ad7a465c7aa3-836

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5bcfed8d4432

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- language_model.model.layers.{i}.mlp.gate_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s17 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- language_model.model.layers.{i}.mlp.up_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s18 门控激活

gate/up:[N,18432] → SiLU(gate)×up → [N,18432]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ffca734f0ce0

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s19 Dense下投影

[N,18432] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.down_proj.weight：logical 7168x18432；stored 7168x18432 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s20 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5bcfed8d4432

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.6-t2

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ad7a465c7aa3

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ad7a465c7aa3

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：8a172ed6bb67

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：ad7a465c7aa3-812

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：ad7a465c7aa3-836

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5bcfed8d4432

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,384]

无持久缓存



- language_model.model.layers.{i}.mlp.gate.e_score_correction_bias：logical 384；stored 384 F32；multiplicity 1

- language_model.model.layers.{i}.mlp.gate.weight：logical 384x7168；stored 384x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：104cdf65ee0e

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s17 Top-K专家选择

scores:[N,384] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,8]

无持久缓存



- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：104cdf65ee0e

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s18 专家分发与token重排

X:[N,7168]; ids:[N,8] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,7168]

Σ_e N_e=N×8（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ee53519c67f5

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s19 路由专家门控GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ee53519c67f5

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s20 路由专家上投影GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ee53519c67f5

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s21 门控激活

gate/up:[N_e,2048] → SiLU(gate)×up → [N_e,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ffca734f0ce0

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s22 路由专家下投影GEMM

[N_e,2048] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_packed：logical 7168x2048；stored 7168x256 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale：logical metadata；stored 7168x64 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ee53519c67f5

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s23 专家加权合并

Y_e:[N_e,7168]; weights:[N,8] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,7168]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：ee53519c67f5-605

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s24 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.gate_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s25 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.up_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s26 共享专家门控激活

gate/up:[N,2048] → SiLU(gate)×up → [N,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ffca734f0ce0

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s27 共享专家下投影

[N,2048] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.down_proj.weight：logical 7168x2048；stored 7168x2048 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s28 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5bcfed8d4432

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s29 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5bcfed8d4432

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.6-t3

#### s1 视觉注意力前归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm0.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.norm0.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ece6b3d2836b

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s2 视觉融合QKV投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,3456]

无持久缓存



- vision_tower.encoder.blocks.{i}.wqkv.bias：logical 3456；stored 3456 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.wqkv.weight：logical 3456x1152；stored 3456x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 视觉QKV拆分 / RoPE

[Nv,3456] → view(...,3,heads,d)→unbind→二维RoPE → Q,K,V:[Nv,16,72]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：33c11b5edf97

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s4 视觉注意力核心

Q,K,V:[Nv,16,72] → 按cu_seqlens分段计算非因果视觉attention；不跨独立图像任意注意 → [Nv,1152]

逻辑P:[heads,Lv,Lv]按每视觉段；Flash实现不必物化；无文本自回归KV缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：33c11b5edf97

- nvidia / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- amd / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- ascend / 未知：未核验。条件：Ascend视觉塔集成已知，但MMEncoderAttention到具体CANN内核未追踪；不借用语言MLA接口。。来源：

#### s5 视觉注意力输出投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.wo.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.wo.weight：logical 1152x1152；stored 1152x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 视觉注意力残差

branch,residual:[Nv,1152] → 相加 → [Nv,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ece6b3d2836b

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 视觉MLP前归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm1.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.norm1.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ece6b3d2836b

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s8 视觉MLP上投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,4304]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc0.bias：logical 4304；stored 4304 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.mlp.fc0.weight：logical 4304x1152；stored 4304x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 视觉GELU

[Nv,4304] → GELU；具体tanh近似由vision config决定 → [Nv,4304]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MLP2.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f45c08461917

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 视觉MLP下投影

[Nv,4304] → Y=X @ Wᵀ + bias → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc1.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.mlp.fc1.weight：logical 1152x4304；stored 1152x4304 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s11 视觉MLP残差

branch,residual:[Nv,1152] → 相加 → [Nv,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ece6b3d2836b

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.6-t4

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- language_model.model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k2.6-t5

#### s1 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：DeepseekV3RMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c3cf0c431a1

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k2.6-t6

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- language_model.lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k2.6-t7

#### s1 空间合并与时间池化

视觉片段:[frames,h,w,1152] → 视觉塔tpool_patch_merger先空间2×2打包并按时间组均值；随后进入projector；Nm依赖processor网格 → grouped:[Nm,4,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：tpool_patch_merger（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：8433345fb702

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 时间池化后、展平前LayerNorm

[Nm,G,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nm,G,1152]

无持久缓存



- mm_projector.pre_norm.bias：logical 1152；stored 1152 BF16；multiplicity 1

- mm_projector.pre_norm.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c718cfcc69de

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 连接器输入展平

[Nm,4,1152] → projector内部view；前序pre_norm仅适用于K2.5/2.6/2.7 → [Nm,4608]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c718cfcc69de

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s4 连接器第一线性

[Nm,4608] → Y=X @ Wᵀ + bias → [Nm,4608]

无持久缓存



- mm_projector.proj.0.bias：logical 4608；stored 4608 BF16；multiplicity 1

- mm_projector.proj.0.weight：logical 4608x4608；stored 4608x4608 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 连接器GELU

[Nm,4608] → nn.GELU → [Nm,4608]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c718cfcc69de

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s6 投影到语言宽度

[Nm,4608] → Y=X @ Wᵀ + bias → [Nm,7168]

无持久缓存



- mm_projector.proj.2.bias：logical 7168；stored 7168 BF16；multiplicity 1

- mm_projector.proj.2.weight：logical 7168x4608；stored 7168x4608 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s7 插入多模态embedding

text embedding:[B,T,7168]; visual:[Nm,7168] → 按media placeholder位置填入视觉特征；数量由processor与网格校验 → [B,T,7168]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：KimiK25ForConditionalGeneration._merge_input_ids_with_image_features（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：41a11dbb404a

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.6-t8

#### s1 Patch嵌入卷积

[Np,3,14,14] → Conv2d(patch14×14,stride14)，展平视觉patch序列 → [Np,1152]

无持久缓存



- vision_tower.patch_embed.proj.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.patch_embed.proj.weight：logical 1152x3x14x14；stored 1152x3x14x14 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：bc375878d3f5

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 插值位置嵌入

[Nv,1152] → 按图像网格插值位置表，并加到patch特征；不是GEMM → [Nv,1152]

无持久缓存



- vision_tower.patch_embed.pos_emb.weight：logical 64x64x1152；stored 64x64x1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：bc375878d3f5

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 视觉最终归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.final_layernorm.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.final_layernorm.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViT3dEncoder.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c16fea60f6d7

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.7-code-t1

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：6e450223e5d9

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：6e450223e5d9

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e33615f374ab

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：6e450223e5d9-819

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：6e450223e5d9-843

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f438445d3c2e

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- language_model.model.layers.{i}.mlp.gate_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s17 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- language_model.model.layers.{i}.mlp.up_proj.weight：logical 18432x7168；stored 18432x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s18 门控激活

gate/up:[N,18432] → SiLU(gate)×up → [N,18432]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ca6e7760d3d2

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s19 Dense下投影

[N,18432] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.down_proj.weight：logical 7168x18432；stored 7168x18432 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s20 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f438445d3c2e

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.7-code-t2

#### s1 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s2 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s4 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_b_proj.weight：logical 12288x1536；stored 12288x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：6e450223e5d9

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s8 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,16384]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_b_proj.weight：logical 16384x512；stored 16384x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 拆分Q/K/V与布局变换

Q:[N,12288]; KV:[N,16384]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,64,T,192]; V:[B,64,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3Attention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：6e450223e5d9

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 RoPE / 位置处理

Q共享子空间:[B,64,T,64]; K共享子空间:[B,1,T,64] → K2参考调用apply_rotary_pos_emb → 旋转后的同shape

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- torch / 框架函数 / 条件分派：apply_rotary_pos_emb（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：e33615f374ab

- nvidia / 未知：未核验。条件：参考RoPE函数已定位；CUDA具体融合内核受MLA后端决定。。来源：

- amd / 未知：未核验。条件：参考RoPE函数已定位；ROCm具体融合内核未逐一确认。。来源：

- ascend / 框架函数 / 条件分派：AscendDeepseekScalingRotaryEmbedding.forward → torch.ops.vllm.npu_rotary_embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：498708f97f63

#### s11 注意力得分与归一化

Q:[B,64,T,192]; K:[B,64,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,64,T,S]（数学逻辑）

参考展开K/V:[B,64,S,192] / [B,64,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(Q,K.transpose(2,3)) × scale → 加mask → torch.nn.functional.softmax(FP32).to(Q.dtype) → torch.nn.functional.dropout（按training开关）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：6e450223e5d9-819

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s12 对V加权汇聚

P:[B,64,T,S]; V:[B,64,S,128] → O=P @ V，转置/合并heads → [B,T,8192]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.matmul(P,V) → Tensor.transpose(1,2).contiguous() → Tensor.reshape(B,T,heads×Dv)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：6e450223e5d9-843

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 MLA输出投影

[N,8192] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x8192；stored 7168x8192 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s14 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f438445d3c2e

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s15 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：DeepseekV3RMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s16 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,384]

无持久缓存



- language_model.model.layers.{i}.mlp.gate.e_score_correction_bias：logical 384；stored 384 F32；multiplicity 1

- language_model.model.layers.{i}.mlp.gate.weight：logical 384x7168；stored 384x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：44b7683f6e46

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s17 Top-K专家选择

scores:[N,384] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,8]

无持久缓存



- torch / 框架函数 / 条件分派：MoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：44b7683f6e46

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s18 专家分发与token重排

X:[N,7168]; ids:[N,8] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,7168]

Σ_e N_e=N×8（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d35446fc19f

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s19 路由专家门控GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.gate_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d35446fc19f

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s20 路由专家上投影GEMM

[N_e,7168] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_packed：logical 2048x7168；stored 2048x896 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.up_proj.weight_scale：logical metadata；stored 2048x224 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d35446fc19f

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s21 门控激活

gate/up:[N_e,2048] → SiLU(gate)×up → [N_e,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ca6e7760d3d2

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s22 路由专家下投影GEMM

[N_e,2048] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_packed：logical 7168x2048；stored 7168x256 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_shape：logical metadata；stored 2 I32；multiplicity 384

- language_model.model.layers.{i}.mlp.experts.{e}.down_proj.weight_scale：logical metadata；stored 7168x64 BF16；multiplicity 384

- torch / 框架函数 / 条件分派：DeepseekV3MoE.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d35446fc19f

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s23 专家加权合并

Y_e:[N_e,7168]; weights:[N,8] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,7168]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：5d35446fc19f-612

- nvidia / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- amd / 框架函数 / 条件分派：fused_experts_impl → dispatch_fused_moe_kernel / ops.moe_sum。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd793c2e912

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s24 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.gate_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s25 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,2048]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.up_proj.weight：logical 2048x7168；stored 2048x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s26 共享专家门控激活

gate/up:[N,2048] → SiLU(gate)×up → [N,2048]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：DeepseekV3MLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ca6e7760d3d2

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendSiluAndMul.forward_oot → torch_npu.npu_swiglu。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：57620477d2d7

#### s27 共享专家下投影

[N,2048] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.shared_experts.down_proj.weight：logical 7168x2048；stored 7168x2048 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s28 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f438445d3c2e

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s29 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：DeepseekV3DecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：f438445d3c2e

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.7-code-t3

#### s1 视觉注意力前归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm0.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.norm0.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5ebd0f659b0d

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s2 视觉融合QKV投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,3456]

无持久缓存



- vision_tower.encoder.blocks.{i}.wqkv.bias：logical 3456；stored 3456 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.wqkv.weight：logical 3456x1152；stored 3456x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 视觉QKV拆分 / RoPE

[Nv,3456] → view(...,3,heads,d)→unbind→二维RoPE → Q,K,V:[Nv,16,72]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a6d5074fcb82

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s4 视觉注意力核心

Q,K,V:[Nv,16,72] → 按cu_seqlens分段计算非因果视觉attention；不跨独立图像任意注意 → [Nv,1152]

逻辑P:[heads,Lv,Lv]按每视觉段；Flash实现不必物化；无文本自回归KV缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：a6d5074fcb82

- nvidia / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- amd / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- ascend / 未知：未核验。条件：Ascend视觉塔集成已知，但MMEncoderAttention到具体CANN内核未追踪；不借用语言MLA接口。。来源：

#### s5 视觉注意力输出投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.wo.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.wo.weight：logical 1152x1152；stored 1152x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 视觉注意力残差

branch,residual:[Nv,1152] → 相加 → [Nv,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5ebd0f659b0d

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 视觉MLP前归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm1.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.norm1.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5ebd0f659b0d

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s8 视觉MLP上投影

[Nv,1152] → Y=X @ Wᵀ + bias → [Nv,4304]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc0.bias：logical 4304；stored 4304 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.mlp.fc0.weight：logical 4304x1152；stored 4304x1152 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 视觉GELU

[Nv,4304] → GELU；具体tanh近似由vision config决定 → [Nv,4304]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MLP2.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9d63ae8d370a

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 视觉MLP下投影

[Nv,4304] → Y=X @ Wᵀ + bias → [Nv,1152]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc1.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.blocks.{i}.mlp.fc1.weight：logical 1152x4304；stored 1152x4304 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s11 视觉MLP残差

branch,residual:[Nv,1152] → 相加 → [Nv,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5ebd0f659b0d

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.7-code-t4

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- language_model.model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k2.7-code-t5

#### s1 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：DeepseekV3RMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：4b713095a999

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k2.7-code-t6

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- language_model.lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k2.7-code-t7

#### s1 空间合并与时间池化

视觉片段:[frames,h,w,1152] → 视觉塔tpool_patch_merger先空间2×2打包并按时间组均值；随后进入projector；Nm依赖processor网格 → grouped:[Nm,4,1152]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：tpool_patch_merger（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b3331effb242

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 时间池化后、展平前LayerNorm

[Nm,G,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nm,G,1152]

无持久缓存



- mm_projector.pre_norm.bias：logical 1152；stored 1152 BF16；multiplicity 1

- mm_projector.pre_norm.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：fc186d292a78

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 连接器输入展平

[Nm,4,1152] → projector内部view；前序pre_norm仅适用于K2.5/2.6/2.7 → [Nm,4608]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：fc186d292a78

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s4 连接器第一线性

[Nm,4608] → Y=X @ Wᵀ + bias → [Nm,4608]

无持久缓存



- mm_projector.proj.0.bias：logical 4608；stored 4608 BF16；multiplicity 1

- mm_projector.proj.0.weight：logical 4608x4608；stored 4608x4608 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 连接器GELU

[Nm,4608] → nn.GELU → [Nm,4608]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLP.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：fc186d292a78

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s6 投影到语言宽度

[Nm,4608] → Y=X @ Wᵀ + bias → [Nm,7168]

无持久缓存



- mm_projector.proj.2.bias：logical 7168；stored 7168 BF16；multiplicity 1

- mm_projector.proj.2.weight：logical 7168x4608；stored 7168x4608 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s7 插入多模态embedding

text embedding:[B,T,7168]; visual:[Nm,7168] → 按media placeholder位置填入视觉特征；数量由processor与网格校验 → [B,T,7168]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：KimiK25ForConditionalGeneration._merge_input_ids_with_image_features（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：13d695cbda05

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k2.7-code-t8

#### s1 Patch嵌入卷积

[Np,3,14,14] → Conv2d(patch14×14,stride14)，展平视觉patch序列 → [Np,1152]

无持久缓存



- vision_tower.patch_embed.proj.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.patch_embed.proj.weight：logical 1152x3x14x14；stored 1152x3x14x14 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：0c052c5d1778

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 插值位置嵌入

[Nv,1152] → 按图像网格插值位置表，并加到patch特征；不是GEMM → [Nv,1152]

无持久缓存



- vision_tower.patch_embed.pos_emb.weight：logical 64x64x1152；stored 64x64x1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：0c052c5d1778

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 视觉最终归一化

[Nv,1152] → LayerNorm: (X-mean)/sqrt(var+eps)×γ+β → [Nv,1152]

无持久缓存



- vision_tower.encoder.final_layernorm.bias：logical 1152；stored 1152 BF16；multiplicity 1

- vision_tower.encoder.final_layernorm.weight：logical 1152；stored 1152 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViT3dEncoder.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ba6c21ad95c8

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k3-t1

#### s1 self_attention 跨层残差读取

prefix:[N,7168]; blocks:[N,R,7168] → 沿残差块维R归一化评分→softmax→对未归一化values加权和 → [N,7168]

values:[N,R+1,7168]; scores:[N,R+1]; R由0-based层索引和block_size12决定

这里R是网络深度残差块数，不是历史token长度。首层无先前块时可跳过。

- language_model.model.layers.{i}.self_attention_res_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- language_model.model.layers.{i}.self_attention_res_proj.weight：logical 1x7168；stored 1x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：_apply_attn_res（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff4715a1dd17

- nvidia / 框架函数 / 条件分派：attn_res → ops.kimi_k3_attn_res / _attn_res_kernel[num_tokens,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd6325436a6

- amd / 未知：未核验。条件：本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。。来源：

- ascend / 框架函数 / 条件分派：apply_attn_res → _apply_attn_res_kernel[num_vectorcore,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：70bb9dc817ba

#### s2 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s3 q_proj

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s4 k_proj

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.k_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 v_proj

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.v_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 q_conv1d

[B,T,12288] → 逐通道因果conv1d(width=4)后SiLU；无跨通道混合 → [B,T,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_conv1d.weight：logical 12288x1x4；stored 12288x1x4 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：AscendKimiK3DeltaAttention._run_causal_conv1d（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：71a26fedb404

#### s7 k_conv1d

[B,T,12288] → 逐通道因果conv1d(width=4)后SiLU；无跨通道混合 → [B,T,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.k_conv1d.weight：logical 12288x1x4；stored 12288x1x4 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：AscendKimiK3DeltaAttention._run_causal_conv1d（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：71a26fedb404

#### s8 v_conv1d

[B,T,12288] → 逐通道因果conv1d(width=4)后SiLU；无跨通道混合 → [B,T,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.v_conv1d.weight：logical 12288x1x4；stored 12288x1x4 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：AscendKimiK3DeltaAttention._run_causal_conv1d（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：71a26fedb404

#### s9 f_a_proj

[N,7168] → Y=X @ Wᵀ → [N,128]

无持久缓存



- language_model.model.layers.{i}.self_attn.f_a_proj.weight：logical 128x7168；stored 128x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s10 f_b_proj

[N,128] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.f_b_proj.weight：logical 12288x128；stored 12288x128 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s11 b_proj

[N,7168] → Y=X @ Wᵀ → [N,96]

无持久缓存



- language_model.model.layers.{i}.self_attn.b_proj.weight：logical 96x7168；stored 96x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s12 KDA衰减与更新门

raw_gate:[B,T,96,128]; beta:[B,T,96] → 融合gate转换与beta sigmoid；Q/K逐head做L2归一化 → g:[B,T,96,128]; β:[B,T,96]

无持久缓存

A_log存储[128]，config/参考参数预期[96]；vLLM加载器narrow按local heads取片。存储128保持审计值，不改写为96；未实测该检查点完整加载。

- language_model.model.layers.{i}.self_attn.A_log：logical 128；stored 128 F32；multiplicity 1

- language_model.model.layers.{i}.self_attn.dt_bias：logical 12288；stored 12288 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0 64a721ce536f

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s13 KDA chunk / recurrence核心

Q,K,V,g:[B,T,96,128]; beta:[B,T,96] → 按key维衰减H_state_bar=diag(exp(g))H_state_prev；δ=v−kᵀH_state_bar；H_state=H_state_bar+βkδᵀ；o=qᵀH_state（含配置scale） → O:[B,T,96,128]

逻辑状态H_state:[B,96,128,128]；FLA transpose_state_layout=True和Ascend state_v_first=True采用V,K末轴；分片为heads/TP；vLLM合并卷积state为[B,3×12288/TP,3+num_spec]或转轴布局，窗口长度4不等于状态存储长度；FLA参考库的具体conv缓存长度未逐实现核验

prefill分块与decode递归数学对应但算子不同；本配置K=V=128时shape相同仍必须注明轴含义。H_state是矩阵状态，不是序列长度。

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）；自定义库fla.ops.kda.chunk_kda / fused_recurrent_kda，无单个标准torch API直达。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0 5f9a935af53a bf6a8a0cf8c1

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：run_chunk_kda → torch.ops._C_ascend.chunk_kda_fwd / l2norm_fwd；decode: run_recurrent_kda → torch.ops._C_ascend.recurrent_kda。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5c543dac1049 a49777194c1a

#### s14 输出门控投影

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.g_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s15 每head输出RMSNorm

[B,T,96,128] → 每head FusedRMSNormGated：RMSNorm(o)×sigmoid(g)；下一行仅展开同一调用的门控语义，不是再执行一次 → [B,T,96,128]

无持久缓存

构造FusedRMSNormGated，forward self.o_norm(o,g)；普通KimiRMSNorm不是此处调用。

- language_model.model.layers.{i}.self_attn.o_norm.weight：logical 128；stored 128 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward → self.o_norm(o,g)：FusedRMSNormGated(activation=sigmoid)；与下一逻辑门控行同一融合调用。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 未知：未核验。条件：带sigmoid门的FusedRMSNormGated；本步骤最终设备分派未核验，不能套用普通RMSNorm接口。。来源：

- amd / 未知：未核验。条件：带sigmoid门的FusedRMSNormGated；本步骤最终设备分派未核验，不能套用普通RMSNorm接口。。来源：

- ascend / 未知：未核验。条件：带sigmoid门的FusedRMSNormGated；本步骤最终设备分派未核验，不能套用普通RMSNorm接口。。来源：

#### s16 KDA门控输出

normalized_O,g:[B,T,96,128] → O←RMSNorm(O)×sigmoid(g)，再合并heads → [B,T,12288]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s17 KDA输出投影

[N,12288] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x12288；stored 7168x12288 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：RowParallelLinear.forward → self.quant_method.apply / tensor_model_parallel_all_reduce。条件：固定K3构造代码o_proj=RowParallelLinear；reduce_results、TP及可选融合gemm_rs_ar决定实际通信，非无条件all-reduce。。来源：306689932e3b ebcb6575be7b

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s18 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s19 mlp 跨层残差读取

prefix:[N,7168]; blocks:[N,R,7168] → 沿残差块维R归一化评分→softmax→对未归一化values加权和 → [N,7168]

values:[N,R+1,7168]; scores:[N,R+1]; R由0-based层索引和block_size12决定

这里R是网络深度残差块数，不是历史token长度。首层无先前块时可跳过。

- language_model.model.layers.{i}.mlp_res_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- language_model.model.layers.{i}.mlp_res_proj.weight：logical 1x7168；stored 1x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：_apply_attn_res（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff4715a1dd17

- nvidia / 框架函数 / 条件分派：attn_res → ops.kimi_k3_attn_res / _attn_res_kernel[num_tokens,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd6325436a6

- amd / 未知：未核验。条件：本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。。来源：

- ascend / 框架函数 / 条件分派：apply_attn_res → _apply_attn_res_kernel[num_vectorcore,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：70bb9dc817ba

#### s20 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s21 Dense门控投影

[N,7168] → Y=X @ Wᵀ → [N,33792]

无持久缓存



- language_model.model.layers.{i}.mlp.gate_proj.weight：logical 33792x7168；stored 33792x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s22 Dense上投影

[N,7168] → Y=X @ Wᵀ → [N,33792]

无持久缓存



- language_model.model.layers.{i}.mlp.up_proj.weight：logical 33792x7168；stored 33792x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s23 门控激活

gate/up:[N,33792] → SiTU = 4*tanh(gate/4)*sigmoid(gate) * (25*tanh(up/25))；beta=4, linear_beta=25（固定配置） → [N,33792]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：SituAndMul.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d33ac8f6ae9

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 未知：未核验。条件：Dense/共享专家SiTU不等于路由专家W4A8 apply_gmm1_act_quant；本步骤设备融合路径未核验。。来源：

#### s24 Dense下投影

[N,33792] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.mlp.down_proj.weight：logical 7168x33792；stored 7168x33792 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s25 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k3-t2

#### s1 self_attention 跨层残差读取

prefix:[N,7168]; blocks:[N,R,7168] → 沿残差块维R归一化评分→softmax→对未归一化values加权和 → [N,7168]

values:[N,R+1,7168]; scores:[N,R+1]; R由0-based层索引和block_size12决定

这里R是网络深度残差块数，不是历史token长度。首层无先前块时可跳过。

- language_model.model.layers.{i}.self_attention_res_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- language_model.model.layers.{i}.self_attention_res_proj.weight：logical 1x7168；stored 1x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：_apply_attn_res（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff4715a1dd17

- nvidia / 框架函数 / 条件分派：attn_res → ops.kimi_k3_attn_res / _attn_res_kernel[num_tokens,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd6325436a6

- amd / 未知：未核验。条件：本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。。来源：

- ascend / 框架函数 / 条件分派：apply_attn_res → _apply_attn_res_kernel[num_vectorcore,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：70bb9dc817ba

#### s2 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s3 q_proj

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s4 k_proj

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.k_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s5 v_proj

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.v_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 q_conv1d

[B,T,12288] → 逐通道因果conv1d(width=4)后SiLU；无跨通道混合 → [B,T,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_conv1d.weight：logical 12288x1x4；stored 12288x1x4 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：AscendKimiK3DeltaAttention._run_causal_conv1d（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：71a26fedb404

#### s7 k_conv1d

[B,T,12288] → 逐通道因果conv1d(width=4)后SiLU；无跨通道混合 → [B,T,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.k_conv1d.weight：logical 12288x1x4；stored 12288x1x4 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：AscendKimiK3DeltaAttention._run_causal_conv1d（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：71a26fedb404

#### s8 v_conv1d

[B,T,12288] → 逐通道因果conv1d(width=4)后SiLU；无跨通道混合 → [B,T,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.v_conv1d.weight：logical 12288x1x4；stored 12288x1x4 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：AscendKimiK3DeltaAttention._run_causal_conv1d（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：71a26fedb404

#### s9 f_a_proj

[N,7168] → Y=X @ Wᵀ → [N,128]

无持久缓存



- language_model.model.layers.{i}.self_attn.f_a_proj.weight：logical 128x7168；stored 128x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s10 f_b_proj

[N,128] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.f_b_proj.weight：logical 12288x128；stored 12288x128 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s11 b_proj

[N,7168] → Y=X @ Wᵀ → [N,96]

无持久缓存



- language_model.model.layers.{i}.self_attn.b_proj.weight：logical 96x7168；stored 96x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s12 KDA衰减与更新门

raw_gate:[B,T,96,128]; beta:[B,T,96] → 融合gate转换与beta sigmoid；Q/K逐head做L2归一化 → g:[B,T,96,128]; β:[B,T,96]

无持久缓存

A_log存储[128]，config/参考参数预期[96]；vLLM加载器narrow按local heads取片。存储128保持审计值，不改写为96；未实测该检查点完整加载。

- language_model.model.layers.{i}.self_attn.A_log：logical 128；stored 128 F32；multiplicity 1

- language_model.model.layers.{i}.self_attn.dt_bias：logical 12288；stored 12288 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0 64a721ce536f

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s13 KDA chunk / recurrence核心

Q,K,V,g:[B,T,96,128]; beta:[B,T,96] → 按key维衰减H_state_bar=diag(exp(g))H_state_prev；δ=v−kᵀH_state_bar；H_state=H_state_bar+βkδᵀ；o=qᵀH_state（含配置scale） → O:[B,T,96,128]

逻辑状态H_state:[B,96,128,128]；FLA transpose_state_layout=True和Ascend state_v_first=True采用V,K末轴；分片为heads/TP；vLLM合并卷积state为[B,3×12288/TP,3+num_spec]或转轴布局，窗口长度4不等于状态存储长度；FLA参考库的具体conv缓存长度未逐实现核验

prefill分块与decode递归数学对应但算子不同；本配置K=V=128时shape相同仍必须注明轴含义。H_state是矩阵状态，不是序列长度。

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）；自定义库fla.ops.kda.chunk_kda / fused_recurrent_kda，无单个标准torch API直达。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0 5f9a935af53a bf6a8a0cf8c1

- nvidia / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → flashinfer_fused_kda_decode / _flashkda_prefill / fused_recurrent_kda。条件：FlashInfer / FlashKDA / recurrence 是可选分支；按prefill/decode、可用库、batch、状态和capture条件分派，不是三者依次执行。。来源：c443d40c09b0

- amd / 框架函数 / 条件分派：KimiK3DeltaAttention._forward → ops.fused_kda_decode / chunk_kda_prefill / fused_recurrent_kda。条件：融合分支：gfx942/gfx950，local heads∈{12,24,48,96}，head_dim128，conv4，input/conv_state BF16，num_spec=0，特定状态布局；否则走其他分支。。来源：4c432db2035a 5d932a1c853c

- ascend / 框架函数 / 条件分派：run_chunk_kda → torch.ops._C_ascend.chunk_kda_fwd / l2norm_fwd；decode: run_recurrent_kda → torch.ops._C_ascend.recurrent_kda。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5c543dac1049 a49777194c1a

#### s14 输出门控投影

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.g_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s15 每head输出RMSNorm

[B,T,96,128] → 每head FusedRMSNormGated：RMSNorm(o)×sigmoid(g)；下一行仅展开同一调用的门控语义，不是再执行一次 → [B,T,96,128]

无持久缓存

构造FusedRMSNormGated，forward self.o_norm(o,g)；普通KimiRMSNorm不是此处调用。

- language_model.model.layers.{i}.self_attn.o_norm.weight：logical 128；stored 128 F32；multiplicity 1

- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward → self.o_norm(o,g)：FusedRMSNormGated(activation=sigmoid)；与下一逻辑门控行同一融合调用。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 未知：未核验。条件：带sigmoid门的FusedRMSNormGated；本步骤最终设备分派未核验，不能套用普通RMSNorm接口。。来源：

- amd / 未知：未核验。条件：带sigmoid门的FusedRMSNormGated；本步骤最终设备分派未核验，不能套用普通RMSNorm接口。。来源：

- ascend / 未知：未核验。条件：带sigmoid门的FusedRMSNormGated；本步骤最终设备分派未核验，不能套用普通RMSNorm接口。。来源：

#### s16 KDA门控输出

normalized_O,g:[B,T,96,128] → O←RMSNorm(O)×sigmoid(g)，再合并heads → [B,T,12288]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDeltaAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：810b468803e0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s17 KDA输出投影

[N,12288] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x12288；stored 7168x12288 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：RowParallelLinear.forward → self.quant_method.apply / tensor_model_parallel_all_reduce。条件：固定K3构造代码o_proj=RowParallelLinear；reduce_results、TP及可选融合gemm_rs_ar决定实际通信，非无条件all-reduce。。来源：306689932e3b ebcb6575be7b

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s18 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s19 mlp 跨层残差读取

prefix:[N,7168]; blocks:[N,R,7168] → 沿残差块维R归一化评分→softmax→对未归一化values加权和 → [N,7168]

values:[N,R+1,7168]; scores:[N,R+1]; R由0-based层索引和block_size12决定

这里R是网络深度残差块数，不是历史token长度。首层无先前块时可跳过。

- language_model.model.layers.{i}.mlp_res_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- language_model.model.layers.{i}.mlp_res_proj.weight：logical 1x7168；stored 1x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：_apply_attn_res（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff4715a1dd17

- nvidia / 框架函数 / 条件分派：attn_res → ops.kimi_k3_attn_res / _attn_res_kernel[num_tokens,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd6325436a6

- amd / 未知：未核验。条件：本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。。来源：

- ascend / 框架函数 / 条件分派：apply_attn_res → _apply_attn_res_kernel[num_vectorcore,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：70bb9dc817ba

#### s20 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s21 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,896]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.gate.e_score_correction_bias：logical 896；stored 896 F32；multiplicity 1

- language_model.model.layers.{i}.block_sparse_moe.gate.weight：logical 896x7168；stored 896x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：KimiMoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：81e5edcfb3a4

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s22 Top-K专家选择

scores:[N,896] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,16]

无持久缓存



- torch / 框架函数 / 条件分派：KimiMoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：81e5edcfb3a4

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s23 潜空间下投影

[N,7168] → Y=X @ Wᵀ → [N,3584]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.routed_expert_down_proj.weight：logical 3584x7168；stored 3584x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s24 专家分发与token重排

X:[N,3584]; ids:[N,16] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,3584]

Σ_e N_e=N×16（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s25 路由专家门控GEMM

[N_e,3584] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,3072]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight_packed：logical 3072x3584；stored 3072x1792 U8；multiplicity 896

- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight_scale：logical metadata；stored 3072x112 U8；multiplicity 896

- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s26 路由专家上投影GEMM

[N_e,3584] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,3072]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight_packed：logical 3072x3584；stored 3072x1792 U8；multiplicity 896

- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight_scale：logical metadata；stored 3072x112 U8；multiplicity 896

- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s27 门控激活

gate/up:[N_e,3072] → SiTU = 4*tanh(gate/4)*sigmoid(gate) * (25*tanh(up/25))；beta=4, linear_beta=25（固定配置） → [N_e,3072]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：SituAndMul.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d33ac8f6ae9

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1_act_quant → torch.ops._C_ascend.dequant_situ_quant。条件：W4A8 SiTU分支；不是所有BF16 SiTU步骤均落到此算子。。来源：05163dde1e66

#### s28 路由专家下投影GEMM

[N_e,3072] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,3584]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight_packed：logical 3584x3072；stored 3584x1536 U8；multiplicity 896

- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight_scale：logical metadata；stored 3584x96 U8；multiplicity 896

- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s29 专家加权合并

Y_e:[N_e,3584]; weights:[N,16] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,3584]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：18868abba78e-866

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s30 潜空间输出归一化

[N,3584] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,3584]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.routed_expert_norm.weight：logical 3584；stored 3584 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s31 潜空间上投影

[N,3584] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.routed_expert_up_proj.weight：logical 7168x3584；stored 7168x3584 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s32 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,6144]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.shared_experts.gate_proj.weight：logical 6144x7168；stored 6144x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s33 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,6144]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.shared_experts.up_proj.weight：logical 6144x7168；stored 6144x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s34 共享专家门控激活

gate/up:[N,6144] → SiTU = 4*tanh(gate/4)*sigmoid(gate) * (25*tanh(up/25))；beta=4, linear_beta=25（固定配置） → [N,6144]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：SituAndMul.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d33ac8f6ae9

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 未知：未核验。条件：Dense/共享专家SiTU不等于路由专家W4A8 apply_gmm1_act_quant；本步骤设备融合路径未核验。。来源：

#### s35 共享专家下投影

[N,6144] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.shared_experts.down_proj.weight：logical 7168x6144；stored 7168x6144 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s36 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s37 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k3-t3

#### s1 self_attention 跨层残差读取

prefix:[N,7168]; blocks:[N,R,7168] → 沿残差块维R归一化评分→softmax→对未归一化values加权和 → [N,7168]

values:[N,R+1,7168]; scores:[N,R+1]; R由0-based层索引和block_size12决定

这里R是网络深度残差块数，不是历史token长度。首层无先前块时可跳过。

- language_model.model.layers.{i}.self_attention_res_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- language_model.model.layers.{i}.self_attention_res_proj.weight：logical 1x7168；stored 1x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：_apply_attn_res（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff4715a1dd17

- nvidia / 框架函数 / 条件分派：attn_res → ops.kimi_k3_attn_res / _attn_res_kernel[num_tokens,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd6325436a6

- amd / 未知：未核验。条件：本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。。来源：

- ascend / 框架函数 / 条件分派：apply_attn_res → _apply_attn_res_kernel[num_vectorcore,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：70bb9dc817ba

#### s2 输入归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.input_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s3 q_a_proj

[N,7168] → Y=X @ Wᵀ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_proj.weight：logical 1536x7168；stored 1536x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s4 q_a_layernorm

[N,1536] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,1536]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_a_layernorm.weight：logical 1536；stored 1536 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：KimiRMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s5 q_b_proj

[N,1536] → Y=X @ Wᵀ → [N,18432]

无持久缓存



- language_model.model.layers.{i}.self_attn.q_b_proj.weight：logical 18432x1536；stored 18432x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 KV压缩投影

[N,7168] → Y=X @ Wᵀ → [N,576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight：logical 576x7168；stored 576x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s7 拆分KV潜变量与旋转键

[N,576] → split([kv_rank512,rope_dim64]) → KV_c:[N,512]; Krope:[N,64]

无持久缓存



- torch / 框架函数 / 条件分派：KimiMLAAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ce998ab6e49a

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s8 kv_a_layernorm

[N,512] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,512]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_a_layernorm.weight：logical 512；stored 512 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：KimiRMSNorm.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s9 kv_b_proj

[N,512] → Y=X @ Wᵀ → [N,24576]

无持久缓存



- language_model.model.layers.{i}.self_attn.kv_b_proj.weight：logical 24576x512；stored 24576x512 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s10 拆分Q/K/V与布局变换

Q:[N,18432]; KV:[N,24576]; Krope:[N,64] → view/transpose/split；Krope在head维广播，不新增权重 → Q/K:[B,96,T,192]; V:[B,96,T,128]

无持久缓存



- torch / 框架函数 / 条件分派：KimiMLAAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ce998ab6e49a

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s11 NoPE身份路径（RoPE不适用）

Q共享子空间:[B,96,T,64]; K共享子空间:[B,1,T,64] → K3 mla_use_nope=true，use_rope=False；64维共享key仍存在，不能据此把cache改为512 → 同shape，不旋转

无持久缓存

历史inv_freq缓冲存储不证明运行时执行旋转；cache与head宽度独立记录。

- torch / 框架函数 / 条件分派：KimiMLAAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ce998ab6e49a

- nvidia / 配置条件排除：不适用：NoPE身份路径，无RoPE旋转。条件：mla_use_nope=true；共享64维key与512+64缓存仍保留。来源：2cf4ee46912b 1bc2e6502ffe

- amd / 配置条件排除：不适用：NoPE身份路径，无RoPE旋转。条件：mla_use_nope=true；共享64维key与512+64缓存仍保留。来源：2cf4ee46912b 1bc2e6502ffe

- ascend / 配置条件排除：不适用：NoPE身份路径，无RoPE旋转。条件：mla_use_nope=true；共享64维key与512+64缓存仍保留。来源：2cf4ee46912b 1bc2e6502ffe

#### s12 注意力得分与归一化

Q:[B,96,T,192]; K:[B,96,S,192] → P=softmax(QKᵀ×scale+causal_mask)；Flash/MLA内核无需物化完整P → P:[B,96,T,S]（数学逻辑）

参考展开K/V:[B,96,S,192] / [B,96,S,128]; 压缩MLA语义cache:[Nblocks,block_size,512+64]；具体生产布局/量化另见API分支

prefill S=history+T；decode T=1。不同backend的分页轴、KV dtype、DCP分片不能统一假定。

- torch / 参考实现本步骤；按源码语句顺序：torch.einsum("bhqd,bhkd->bhqk",Q,K) × scale → 加mask → torch.nn.functional.softmax(FP32).to(query.dtype) → torch.nn.functional.dropout（推理p=0）。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：84d2a5c76c4a-324

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s13 对V加权汇聚

P:[B,96,T,S]; V:[B,96,S,128] → O=P @ V，转置/合并heads → [B,T,12288]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：torch.einsum("bhqk,bhkd->bhqd",P,V) → Tensor.transpose(1,2).contiguous()；上层合并heads。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：84d2a5c76c4a-330

- nvidia / 框架函数 / 条件分派：FlashInferMLAImpl.forward_mqa → trtllm_batch_decode_with_kv_cache_mla。条件：FlashInfer MLA decode分支；prefill及其他backend另选；实际后端、KV精度和SM支持需运行配置确认。。来源：afb1f6a360fb

- amd / 框架函数 / 条件分派：AiterMLAImpl.forward_mqa → rocm_aiter_ops.mla_decode_fwd / mla_gluon。条件：AITER MLA decode；mla_gluon与mla_decode_fwd是不同条件分支，受head数、缓存dtype、DCP影响。。来源：a90d9a90c565

- ascend / 框架函数 / 条件分派：AscendMLAImpl._forward_prefill → torch_npu.npu_fused_infer_attention_score；decode: AscendMLAImpl._forward_decode → torch_npu.npu_fused_infer_attention_score_v2。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c003100048bb cf3b4807068e 460613656cec

#### s14 MLA输出门控投影

[N,7168] → Y=X @ Wᵀ → [N,12288]

无持久缓存



- language_model.model.layers.{i}.self_attn.g_proj.weight：logical 12288x7168；stored 12288x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s15 MLA输出门控

O,g:[N,12288] → O←O×sigmoid(g) → [N,12288]

无持久缓存



- torch / 框架函数 / 条件分派：KimiMLAAttention.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ce998ab6e49a

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s16 MLA输出投影

[N,12288] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.self_attn.o_proj.weight：logical 7168x12288；stored 7168x12288 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 框架函数 / 条件分派：RowParallelLinear.forward → self.quant_method.apply / tensor_model_parallel_all_reduce。条件：固定K3构造代码o_proj=RowParallelLinear；reduce_results、TP及可选融合gemm_rs_ar决定实际通信，非无条件all-reduce。。来源：306689932e3b f7142733ff6a

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s17 注意力残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s18 mlp 跨层残差读取

prefix:[N,7168]; blocks:[N,R,7168] → 沿残差块维R归一化评分→softmax→对未归一化values加权和 → [N,7168]

values:[N,R+1,7168]; scores:[N,R+1]; R由0-based层索引和block_size12决定

这里R是网络深度残差块数，不是历史token长度。首层无先前块时可跳过。

- language_model.model.layers.{i}.mlp_res_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- language_model.model.layers.{i}.mlp_res_proj.weight：logical 1x7168；stored 1x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：_apply_attn_res（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff4715a1dd17

- nvidia / 框架函数 / 条件分派：attn_res → ops.kimi_k3_attn_res / _attn_res_kernel[num_tokens,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd6325436a6

- amd / 未知：未核验。条件：本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。。来源：

- ascend / 框架函数 / 条件分派：apply_attn_res → _apply_attn_res_kernel[num_vectorcore,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：70bb9dc817ba

#### s19 前馈前归一化

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.post_attention_layernorm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s20 MoE路由评分

[N,7168] → FP32路由线性→sigmoid；选择分数可加correction_bias，最终权重取原分数并归一化/缩放 → logits:[N,896]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.gate.e_score_correction_bias：logical 896；stored 896 F32；multiplicity 1

- language_model.model.layers.{i}.block_sparse_moe.gate.weight：logical 896x7168；stored 896x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：KimiMoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：81e5edcfb3a4

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s21 Top-K专家选择

scores:[N,896] → grouped/noaux_tc路由规则；总专家E与每token选中K严格区分 → ids,weights:[N,16]

无持久缓存



- torch / 框架函数 / 条件分派：KimiMoEGate.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：81e5edcfb3a4

- nvidia / 未知：未核验。条件：路由是MoE分派的一部分；本轮未固定CUDA TopK内核选择。。来源：

- amd / 未知：未核验。条件：本轮未固定ROCm TopK内核选择。。来源：

- ascend / 框架函数 / 条件分派：AscendGroupedTopKRouter._compute_routing → torch.topk。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：25751b89cb7c

#### s22 潜空间下投影

[N,7168] → Y=X @ Wᵀ → [N,3584]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.routed_expert_down_proj.weight：logical 3584x7168；stored 3584x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s23 专家分发与token重排

X:[N,3584]; ids:[N,16] → argsort/gather或EP dispatch；N_e由实际路由确定 → 每专家X_e:[N_e,3584]

Σ_e N_e=N×16（忽略padding且无token丢弃）；EP后是本地接收量，容量padding/通信缓冲大小由后端决定



- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_dispatch → torch_npu.npu_moe_distribute_dispatch_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：116a6619588f

#### s24 路由专家门控GEMM

[N_e,3584] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,3072]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight_packed：logical 3072x3584；stored 3072x1792 U8；multiplicity 896

- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight_scale：logical metadata；stored 3072x112 U8；multiplicity 896

- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s25 路由专家上投影GEMM

[N_e,3584] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,3072]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight_packed：logical 3072x3584；stored 3072x1792 U8；multiplicity 896

- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight_scale：logical metadata；stored 3072x112 U8；multiplicity 896

- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。。来源：241791015955

#### s26 门控激活

gate/up:[N_e,3072] → SiTU = 4*tanh(gate/4)*sigmoid(gate) * (25*tanh(up/25))；beta=4, linear_beta=25（固定配置） → [N_e,3072]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：SituAndMul.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d33ac8f6ae9

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm1_act_quant → torch.ops._C_ascend.dequant_situ_quant。条件：W4A8 SiTU分支；不是所有BF16 SiTU步骤均落到此算子。。来源：05163dde1e66

#### s27 路由专家下投影GEMM

[N_e,3072] → Y=X @ Wᵀ；每专家独立，N_e动态，ΣN_e=N×TopK（无丢弃时） → [N_e,3584]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight_packed：logical 3584x3072；stored 3584x1536 U8；multiplicity 896

- language_model.model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight_scale：logical metadata；stored 3584x96 U8；multiplicity 896

- torch / 框架函数 / 条件分派：KimiSparseMoeBlock.moe_infer（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：18868abba78e

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：AscendW4A8DynamicFusedMoEMethod.apply_gmm2 → torch_npu.npu_grouped_matmul。条件：仅转换W4A8量化方法被选中时；不是原始INT4/MXFP4检查点的直接等价接口；硬件与CANN须匹配。 本步骤使用w2下投影；apply_gmm1只对应w1。。来源：69f409e02d16

#### s28 专家加权合并

Y_e:[N_e,3584]; weights:[N,16] → 逆重排、乘路由权重、对TopK求和；跨rank可能有combine/约简 → [N,3584]

无持久缓存



- torch / 参考实现本步骤；按源码语句顺序：索引回填 new_x[idxs]=outs → Tensor.view(N,TopK,H_e) → Tensor.type(weight.dtype) → Tensor.mul_(topk_weight.unsqueeze(-1)) → Tensor.sum(dim=1) → Tensor.type(output dtype)。条件：与模块整体设备融合/分派分开；参考表达式并不表示逐个调用独立设备kernel。。来源：18868abba78e-866

- nvidia / 框架函数 / 条件分派：TrtLlmMxfp4ExpertsMonolithic.apply → trtllm_fp4_block_scale_moe。条件：已核验可选MXFP4实现；仅当量化oracle、架构capability、激活函数、输入布局均匹配时选择。没有确认每一张设备都会走此路径。。来源：9acc9260bde1

- amd / 未知：未核验。条件：所查AITER Triton W4A16仅接受SWIGLUOAI/SILU，不满足K3 SiTU；不能列为K3兼容候选。其他SiTU兼容分支与实际选择尚未完整追踪。。来源：0379a0e38be2

- ascend / 框架函数 / 条件分派：TokenDispatcherWithMC2.token_combine → torch_npu.npu_moe_distribute_combine_v2。条件：仅MC2通信路径；另有All2AllV路径，不能认定所有EP规模均使用该API。。来源：f3a3b132f63f

#### s29 潜空间输出归一化

[N,3584] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,3584]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.routed_expert_norm.weight：logical 3584；stored 3584 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s30 潜空间上投影

[N,3584] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.routed_expert_up_proj.weight：logical 7168x3584；stored 7168x3584 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s31 共享专家gate_proj

[N,7168] → Y=X @ Wᵀ → [N,6144]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.shared_experts.gate_proj.weight：logical 6144x7168；stored 6144x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s32 共享专家up_proj

[N,7168] → Y=X @ Wᵀ → [N,6144]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.shared_experts.up_proj.weight：logical 6144x7168；stored 6144x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s33 共享专家门控激活

gate/up:[N,6144] → SiTU = 4*tanh(gate/4)*sigmoid(gate) * (25*tanh(up/25))；beta=4, linear_beta=25（固定配置） → [N,6144]

无持久缓存

激活先计算FP32再回写的分支见来源；不是把I32/U8容器直接相乘。

- torch / 框架函数 / 条件分派：SituAndMul.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：5d33ac8f6ae9

- nvidia / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- amd / 未知：未核验。条件：参考SiLU/SiTU表达式已定位；可能被MoE/MLP融合，未单独指定kernel。。来源：

- ascend / 未知：未核验。条件：Dense/共享专家SiTU不等于路由专家W4A8 apply_gmm1_act_quant；本步骤设备融合路径未核验。。来源：

#### s34 共享专家下投影

[N,6144] → Y=X @ Wᵀ → [N,7168]

无持久缓存



- language_model.model.layers.{i}.block_sparse_moe.shared_experts.down_proj.weight：logical 7168x6144；stored 7168x6144 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s35 共享/路由分支合并

shared,routed:[N,7168] → Y=Y_routed+Y_shared；共享专家使用原残差宽度输入 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s36 前馈残差合并

residual,branch:[N,7168] → Y=residual+branch；K3块边界prefix_sum可能为空并新开残差块 → [N,7168]

无持久缓存



- torch / 框架函数 / 条件分派：KimiDecoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：d86a747142f0

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k3-t4

#### s1 视觉注意力前归一化

[Nv,1024] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [Nv,1024]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm0.weight：logical 1024；stored 1024 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：db18af94f22d

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s2 视觉融合QKV投影

[Nv,1024] → Y=X @ Wᵀ → [Nv,4608]

无持久缓存



- vision_tower.encoder.blocks.{i}.wqkv.weight：logical 4608x1024；stored 4608x1024 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s3 视觉QKV拆分 / RoPE

[Nv,4608] → view(...,3,heads,d)→unbind→二维RoPE → Q,K,V:[Nv,12,128]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：edcaf149bc62

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s4 视觉注意力核心

Q,K,V:[Nv,12,128] → 按cu_seqlens分段计算非因果视觉attention；不跨独立图像任意注意 → [Nv,1536]

逻辑P:[heads,Lv,Lv]按每视觉段；Flash实现不必物化；无文本自回归KV缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.attention_qkvpacked（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：edcaf149bc62

- nvidia / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- amd / 框架函数 / 条件分派：MoonViTEncoderLayer.attention_qkvpacked → self.wqkv / self.attn / self.wo。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：762483cb9d6c

- ascend / 未知：未核验。条件：Ascend视觉塔集成已知，但MMEncoderAttention到具体CANN内核未追踪；不借用语言MLA接口。。来源：

#### s5 视觉注意力输出投影

[Nv,1536] → Y=X @ Wᵀ → [Nv,1024]

无持久缓存



- vision_tower.encoder.blocks.{i}.wo.weight：logical 1024x1536；stored 1024x1536 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 视觉注意力残差

branch,residual:[Nv,1024] → 相加 → [Nv,1024]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：db18af94f22d

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s7 视觉MLP前归一化

[Nv,1024] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [Nv,1024]

无持久缓存



- vision_tower.encoder.blocks.{i}.norm1.weight：logical 1024；stored 1024 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：db18af94f22d

- nvidia / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- amd / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

- ascend / 未知：未核验。条件：视觉层归一化模块的设备分派未逐项核验，不能套用语言RMSNorm。。来源：

#### s8 视觉MLP上投影

[Nv,1024] → Y=X @ Wᵀ → [Nv,4096]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc0.weight：logical 4096x1024；stored 4096x1024 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s9 视觉GELU

[Nv,4096] → GELU；具体tanh近似由vision config决定 → [Nv,4096]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MLP2.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：9c268ae3cc41

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s10 视觉MLP下投影

[Nv,4096] → Y=X @ Wᵀ → [Nv,1024]

无持久缓存



- vision_tower.encoder.blocks.{i}.mlp.fc1.weight：logical 1024x4096；stored 1024x4096 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s11 视觉MLP残差

branch,residual:[Nv,1024] → 相加 → [Nv,1024]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViTEncoderLayer.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：db18af94f22d

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k3-t5

#### s1 Token嵌入查表

[B,T] int token IDs → Y[b,t,:]=W[token_id[b,t],:] → [B,T,7168]

无持久缓存



- language_model.model.embed_tokens.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Embedding → torch.nn.functional.embedding：Embedding.forward → torch.nn.functional.embedding。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c395ace2f975

- nvidia / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- amd / 框架函数 / 条件分派：VocabParallelEmbedding.forward → self.quant_method.embedding / tensor_model_parallel_all_reduce。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：b488af82b00d

- ascend / 未知：未核验。条件：参考Embedding语义已明确；Ascend此嵌入的最终设备接口未追踪。。来源：

### k3-t6

#### s1 最终AttnRes

prefix/blocks:[N,R,7168] → 最终跨块加权混合 → [N,7168]

无持久缓存



- language_model.model.output_attn_res_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- language_model.model.output_attn_res_proj.weight：logical 1x7168；stored 1x7168 BF16；multiplicity 1

- torch / 框架函数 / 条件分派：_apply_attn_res（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff4715a1dd17

- nvidia / 框架函数 / 条件分派：attn_res → ops.kimi_k3_attn_res / _attn_res_kernel[num_tokens,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：7dd6325436a6

- amd / 未知：未核验。条件：本轮未将AMD AttnRes逐层追至专用内核；保留参考torch计算定义。。来源：

- ascend / 框架函数 / 条件分派：apply_attn_res → _apply_attn_res_kernel[num_vectorcore,]。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：70bb9dc817ba

#### s2 最终RMSNorm

[N,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [N,7168]

无持久缓存



- language_model.model.norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / 参考RMSNorm源码步骤；不是单独设备API：KimiRMSNorm.forward：Tensor.float/to(FP32) → Tensor.pow(2).mean(-1,keepdim=True) → torch.rsqrt(var+eps) → 输入乘归一化因子 → 转输入dtype并乘gamma。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：06cde78b0e97

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

### k3-t7

#### s1 词表输出头

[N,7168] → Y=X @ Wᵀ → [N,163840]

无持久缓存

可只选待生成位置再计算；输出logits为[N_selected,Vocab]，不一定保留全T。

- language_model.lm_head.weight：logical 163840x7168；stored 163840x7168 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

### k3-t8

#### s1 空间合并与时间池化

视觉片段:[frames,h,w,1024] → 视觉塔tpool_patch_merger先空间2×2打包并按时间组均值；随后进入projector；Nm依赖processor网格 → grouped:[Nm,4,1024]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：tpool_patch_merger（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：521f6594eb83

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 连接器输入展平

[Nm,4,1024] → projector内部view；前序pre_norm仅适用于K2.5/2.6/2.7 → [Nm,4096]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLPV2.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：0cc97b41f6f7

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 连接器第一线性

[Nm,4096] → Y=X @ Wᵀ → [Nm,4096]

无持久缓存



- mm_projector.proj.0.weight：logical 4096x4096；stored 4096x4096 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s4 连接器GELU

[Nm,4096] → nn.GELU → [Nm,4096]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLPV2.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：0cc97b41f6f7

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s5 投影到语言宽度

[Nm,4096] → Y=X @ Wᵀ → [Nm,7168]

无持久缓存



- mm_projector.proj.2.weight：logical 7168x4096；stored 7168x4096 BF16；multiplicity 1

- torch / torch.nn.Linear → torch.nn.functional.linear：Linear.forward → torch.nn.functional.linear。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：54ee794d774c

- nvidia / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- amd / 未知：未核验。条件：此线性层的Column/Row/Replicated封装需按具体模块核验，不能只由linear类别推定。。来源：

- ascend / 框架函数 / 条件分派：AscendUnquantizedLinearMethod.apply → torch.ops.vllm.unquantized_gemm → unquantized_gemm → torch.nn.functional.linear。条件：仅未量化线性分支；最终CANN设备算子未核验，不猜测MatMul接口。。来源：df7860e6c4c6 1c155f529c07

#### s6 连接器输出RMSNorm

[Nm,7168] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [Nm,7168]

无持久缓存



- mm_projector.post_norm.weight：logical 7168；stored 7168 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：PatchMergerMLPV2.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：0cc97b41f6f7

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

#### s7 插入多模态embedding

text embedding:[B,T,7168]; visual:[Nm,7168] → 按media placeholder位置填入视觉特征；数量由processor与网格校验 → [B,T,7168]

无持久缓存



- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：KimiK3ForConditionalGeneration._merge_input_ids_with_image_features（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：c47424276a48

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

### k3-t9

#### s1 Patch嵌入卷积

[Np,3,14,14] → Conv2d(patch14×14,stride14)，展平视觉patch序列 → [Np,1024]

无持久缓存



- vision_tower.patch_embed.proj.weight：logical 1024x3x14x14；stored 1024x3x14x14 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：34810eb33520

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s2 插值位置嵌入

[Nv,1024] → 按图像网格插值位置表，并加到patch特征；不是GEMM → [Nv,1024]

无持久缓存



- vision_tower.patch_embed.pos_emb.weight：logical 64x64x1024；stored 64x64x1024 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonVision3dPatchEmbed.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：34810eb33520

- nvidia / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认CUDA底层映射。。来源：

- amd / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认ROCm底层映射。。来源：

- ascend / 未知：未核验。条件：此步骤仅核验参考实现/shape；未确认Ascend底层映射。。来源：

#### s3 视觉最终归一化

[Nv,1024] → RMSNorm: X×rsqrt(mean(X²)+eps)×γ → [Nv,1024]

无持久缓存



- vision_tower.encoder.final_layernorm.weight：logical 1024；stored 1024 BF16；multiplicity 1

- torch / Moonshot参考模块forward；含多个步骤，不代表1:1设备调用：MoonViT3dEncoder.forward（函数上下文，不是本步骤的有序调用链）。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：3ec798fd8f7c

- nvidia / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- amd / 已核验native/IR分支；非已确认设备kernel：RMSNorm.forward_native → ir.ops.rms_norm / ir.ops.fused_add_rms_norm.maybe_inplace。条件：所列forward_native进入IR注册算子；不能等同最终CUDA/ROCm kernel。CUDA forward另有BATCH_INVARIANT分支；本轮未追完编译/IR后端注册选择。。来源：4efff8a09679

- ascend / 框架函数 / 条件分派：AscendRMSNorm.forward_oot → torch_npu.npu_rms_norm / torch.ops._C_ascend.npu_add_rms_norm_bias。条件：仅证实所列源码分支；并列调用可能互斥或覆盖整个模块，不是本步骤1:1设备映射。未执行本机设备测试。。来源：ff22f8d895b3

## 精确函数来源

- 54ee794d774c [Linear.forward](https://github.com/pytorch/pytorch/blob/9b6e45278f06dc46c14d32220063f01d249fc21d/torch/nn/modules/linear.py#L130) — pytorch/pytorch@9b6e45278f06dc46c14d32220063f01d249fc21d，torch/nn/modules/linear.py L130-134，函数体SHA256 1ecdd2f30160d869cf64caee74303715e22395e4c173240cc8051963fb551cc9

- c395ace2f975 [Embedding.forward](https://github.com/pytorch/pytorch/blob/9b6e45278f06dc46c14d32220063f01d249fc21d/torch/nn/modules/sparse.py#L188) — pytorch/pytorch@9b6e45278f06dc46c14d32220063f01d249fc21d，torch/nn/modules/sparse.py L188-197，函数体SHA256 2b3991e5db4731ccdab086c9405bd1a4a788c3496597308d27d56c6066a127fa

- 5f2a0ec64e01 [ColumnParallelLinear.forward](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/linear.py#L603) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/linear.py L603-621，函数体SHA256 9d97b2911506fa36fccc973d8355c315dcf41947d08508dee4bf55b44cb8acb0

- 306689932e3b [RowParallelLinear.forward](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/linear.py#L1762) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/linear.py L1762-1788，函数体SHA256 110cb650730bbdb34dc3acfdda779ef9e69dea9cfd4a62eb543488dfba6d526d

- 3f8018165c8c [Fp8LinearMethod.apply](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/quantization/fp8.py#L432) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/quantization/fp8.py L432-475，函数体SHA256 897fa162c88032a9bdd91a74916d2bff357ed97f7cd27f4f6b72a4e1811ba594

- 02a481161b17 [CompressedTensorsWNA16.apply_weights](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L236) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py L236-239，函数体SHA256 9f69af93546513dddb70495057bd00386371cabfb01ea01e37c8e6182e53ef80

- 4efff8a09679 [RMSNorm.forward_native](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/layernorm.py#L74) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/layernorm.py L74-94，函数体SHA256 9f60c671c5a576be166f039dbebad0b88494d0d9c9d8345dd71be812c15bc137

- b488af82b00d [VocabParallelEmbedding.forward](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/vocab_parallel_embedding.py#L511) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/vocab_parallel_embedding.py L511-556，函数体SHA256 83d4943ee623515d0c989f37e35e59a27dadafdeb1745d82799428340bbc4948

- 7dd793c2e912 [fused_experts_impl](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/fused_moe/fused_moe.py#L1652) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/fused_moe/fused_moe.py L1652-1855，函数体SHA256 ab1dd99cef56b353548679f65a7d79951c6e4ff0f2b179a699ffbb9d90b95beb

- 9acc9260bde1 [TrtLlmMxfp4ExpertsMonolithic.apply](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py#L202) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py L202-283，函数体SHA256 47889b4084f751496155db935549c427b84158cb78f7695f1a8a348e79e8aa6a

- cf2d178545ba [aiter_triton_kernel_w4a16_moe_forward](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py#L116) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py L116-275，函数体SHA256 04605c07298f546ad9f409833a06153893ffdf98e5a7387afb50969d41fa2cbd

- afb1f6a360fb [FlashInferMLAImpl.forward_mqa](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/v1/attention/backends/mla/flashinfer_mla.py#L275) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/v1/attention/backends/mla/flashinfer_mla.py L275-388，函数体SHA256 43f8f684a08f36ab5d0f864234986e1ec3a05c4a172bcbe4c7ea620cad4c8b5f

- a90d9a90c565 [AiterMLAImpl.forward_mqa](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/v1/attention/backends/mla/rocm_aiter_mla.py#L1903) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/v1/attention/backends/mla/rocm_aiter_mla.py L1903-2125，函数体SHA256 fac72dbc1fd415e8cdcc060c1b477652b948cba0093b9aac6239a296690a2e5e

- c443d40c09b0 [KimiK3DeltaAttention._forward](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/models/kimi_k3/nvidia/kda.py#L816) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/models/kimi_k3/nvidia/kda.py L816-1251，函数体SHA256 864d3813292e4e4051396bbcd4f6c101252a3b90a86730e7a23c06cfaafe50e6

- 4c432db2035a [KimiK3DeltaAttention._forward](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/models/kimi_k3/amd/kda.py#L312) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/models/kimi_k3/amd/kda.py L312-640，函数体SHA256 50e884d60c13f8490759e572ebdb0f8226ec0f2739285825904e1cae7164c514

- 7dd6325436a6 [attn_res](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/models/kimi_k3/nvidia/ops/attn_res.py#L170) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/models/kimi_k3/nvidia/ops/attn_res.py L170-249，函数体SHA256 4304f5c6332325507d0177042a136d993d7087b4a90f548896f329f6cb1be85b

- 762483cb9d6c [MoonViTEncoderLayer.attention_qkvpacked](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/models/kimi_k25_vit.py#L438) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/models/kimi_k25_vit.py L438-491，函数体SHA256 1d664b799f2f2b54f3ff8e6cb05bfc918a185171e44bd9983f00c1b9f1629951

- 0a075517313d [KimiK25MultiModalProjector.forward](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/models/kimi_k25_vit.py#L938) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/models/kimi_k25_vit.py L938-950，函数体SHA256 1aed65671a7669afc8b9d9751e190134bd31df22e204bab722dd1eff879f6b12

- df7860e6c4c6 [AscendUnquantizedLinearMethod.apply](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/linear.py#L135) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/linear.py L135-141，函数体SHA256 f4a047728d7e4ab9b39ad5a0ee27028b1971709396fb54680509fb78567fd504

- 1c155f529c07 [unquantized_gemm](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/linear.py#L49) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/linear.py L49-54，函数体SHA256 d160422a52b7e814df35006a617feb7341124b81b6109f11da39000db2f24fc7

- ff22f8d895b3 [AscendRMSNorm.forward_oot](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/layernorm.py#L64) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/layernorm.py L64-84，函数体SHA256 465e34f310f3736cee19ae87f90c96af6dc016bd7834e8793809c838a0004fec

- 57620477d2d7 [AscendSiluAndMul.forward_oot](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/activation.py#L36) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/activation.py L36-38，函数体SHA256 ed406005fe0d872246abe48c213e9171afc20a8ffd8404c52d8a4c3c8ce3685c

- 498708f97f63 [AscendDeepseekScalingRotaryEmbedding.forward](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/rotary_embedding.py#L509) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/rotary_embedding.py L509-526，函数体SHA256 072e762815777c9e5c131694e66e8a1d9b950c10503f610fece1845b9d36e048

- c003100048bb [AscendMLAImpl._forward_prefill](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/attention/mla_v1.py#L1323) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/attention/mla_v1.py L1323-1394，函数体SHA256 a5ad9e882430fba6f64a86972ac40a90be441acfeb7575e07aed3a9780e23d14

- cf3b4807068e [AscendMLAImpl._forward_decode](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/attention/mla_v1.py#L1598) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/attention/mla_v1.py L1598-1838，函数体SHA256 b6e718d1ebf55e26f61e80839d827ef9af28bf1d0794c4dab69aeb3c50bc2031

- 460613656cec [AscendMLAImpl.exec_kv_decode](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/attention/mla_v1.py#L1453) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/attention/mla_v1.py L1453-1498，函数体SHA256 b3087186902f042a5e83eca7e587dba24e2dc453a525dbfa1d1a3b7c8ffa209b

- 5c543dac1049 [run_chunk_kda](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/kda.py#L52) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/kda.py L52-89，函数体SHA256 d79103b1376eae0b6076fb7d784150bf07f5ec7cf94ef4f5d291407749fe1ae6

- a49777194c1a [run_recurrent_kda](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/kda.py#L13) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/kda.py L13-49，函数体SHA256 2c1cc6e92ee5d0bc9de63041384410a0bce2ce3f171ba1d684aadddcbc0f3f76

- 71a26fedb404 [AscendKimiK3DeltaAttention._run_causal_conv1d](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/kimi_kda.py#L370) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/kimi_kda.py L370-398，函数体SHA256 b452cb8f346e245071c776aaad6725188bcde4b044442b1ef86fb542ec1e3984

- 25751b89cb7c [AscendGroupedTopKRouter._compute_routing](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/fused_moe/router/grouped_topk_router.py#L80) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/fused_moe/router/grouped_topk_router.py L80-128，函数体SHA256 dcaeb3cfc778868c40e225275f40bcbfb2cdebb36876422331d535ead90a04c6

- 116a6619588f [TokenDispatcherWithMC2.token_dispatch](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/fused_moe/token_dispatcher.py#L227) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/fused_moe/token_dispatcher.py L227-261，函数体SHA256 1c43ed9d14436028cf4f316924ef266829c2c8c68c2e22c1a8ff324a8df94695

- f3a3b132f63f [TokenDispatcherWithMC2.token_combine](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/fused_moe/token_dispatcher.py#L327) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/fused_moe/token_dispatcher.py L327-333，函数体SHA256 316fd4b3a72e91e7182860073e7fba7e72e743d60a3f98ee669fb22a4cf8316c

- 241791015955 [AscendW4A8DynamicFusedMoEMethod.apply_gmm1](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/quantization/methods/w4a8/w4a8.py#L506) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/quantization/methods/w4a8/w4a8.py L506-527，函数体SHA256 eef60158a0b63fd082414c94e739b68340b6e1174faef09de1ab14d875c72ef0

- 69f409e02d16 [AscendW4A8DynamicFusedMoEMethod.apply_gmm2](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/quantization/methods/w4a8/w4a8.py#L532) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/quantization/methods/w4a8/w4a8.py L532-547，函数体SHA256 4ba07e1dba1160ea0dcc26c0cf5a4b3f9e655ab2af3670973dba3c76a4fbfd6e

- 0379a0e38be2 [AiterW4A16ExpertsMonolithic._supports_activation](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py#L316) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py L316-317，函数体SHA256 55ea8244e7444c57b54ce410efc30053e5e66aea17e719ddfc5c78183ea34486

- 05163dde1e66 [AscendW4A8DynamicFusedMoEMethod.apply_gmm1_act_quant](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/quantization/methods/w4a8/w4a8.py#L447) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/quantization/methods/w4a8/w4a8.py L447-504，函数体SHA256 11c2b05ac9c7784b8db3fc9fe5bd338cf367b248dadf9937856d345b822cd6bf

- 70bb9dc817ba [apply_attn_res](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/ops/triton/kimi_k3/attention_residual.py#L78) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/ops/triton/kimi_k3/attention_residual.py L78-116，函数体SHA256 52ede996de30d06fcef4a773e03de0bb8595173f5c9a7bc9429d90a2f82fb6e5

- 5f9a935af53a [chunk_kda](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/kda/chunk.py#L197) — fla-org/flash-linear-attention@954438d1fcb5e1bb05c22f9908de9c5c2df74ae5，fla/ops/kda/chunk.py L197-494，函数体SHA256 7a87ecc79f518a5377f40af289641499448270644dd7d67d66303b555ac7782f

- bf6a8a0cf8c1 [fused_recurrent_kda](https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/ops/kda/fused_recurrent.py#L339) — fla-org/flash-linear-attention@954438d1fcb5e1bb05c22f9908de9c5c2df74ae5，fla/ops/kda/fused_recurrent.py L339-491，函数体SHA256 22476e217feaa056c27cd2ed0ace24e65058c427b0dbe533d04d4e7a67f41d5a

- 64a721ce536f [a_log_weight_loader](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/models/kimi_k3/nvidia/kda.py#L71) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/models/kimi_k3/nvidia/kda.py L71-93，函数体SHA256 882bf281b1d4cc707f215e0f79ebd11f530e075691d589161442f76e4a6e7cb5

- 5d932a1c853c [is_fused_kda_decode_supported](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/models/kimi_k3/amd/ops/kda_decode.py#L29) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/models/kimi_k3/amd/ops/kda_decode.py L29-53，函数体SHA256 12b50e0329b86b987412ade77a37ece4668ab59375825cf6d7f25600e85ac240

- 91fc0c610bbc [MambaStateShapeCalculator.kda_state_shape](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/model_executor/layers/mamba/mamba_utils.py#L303) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/model_executor/layers/mamba/mamba_utils.py L303-326，函数体SHA256 6ddcdff63883c5a838e13ab82c2e628b086af816d278c893adb38ea50aa909f7

- e37fe49f1452 [DeepseekV3RMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L103) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L103-108，函数体SHA256 9a60087840825e1853db6aaf071c649ce5f0ea3cacb15e2367e89c7c35da4ab6

- cca6a392eb8c [DeepseekV3DecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L1168) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L1168-1228，函数体SHA256 92bcd5947049b3756b90d5e58bba7c2478840bff50f05cee711a72b3330778ee

- c5baa6bc5b14 [DeepseekV3Attention.forward](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L751) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L751-857，函数体SHA256 0d830741ac25e203e842a3ae654c6724874695180c30fa63a3c2da7a8c739bea

- 260cc1915f9b [apply_rotary_pos_emb](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L339) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L339-371，函数体SHA256 39851acdc7fb26b777566c3136d987cb9ca6597a2164b6d49bee4c96b9de9ca6

- c5baa6bc5b14-816 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L816) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L816-840，函数体SHA256 4a3e0a866efbb51f8e09efe62853b97ac5354f17ee09ddfddc3446aa657cd92b

- c5baa6bc5b14-840 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L840) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L840-852，函数体SHA256 159d7b207301503b767bc6263722a2e5f450ba90fff3b87521da7ab15e205cfa

- ed2d079af130 [DeepseekV3MLP.forward](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L388) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L388-390，函数体SHA256 c175739048a2dbe3d01102f0229b1c5d9159a7cfb5a4a72bea816801658f8f78

- b6d3839075c1 [MoEGate.forward](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L423) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L423-474，函数体SHA256 3e7518c4d08cafc73f7155115d733fa9823ef7f51040499ff89506490fb9573b

- de2e790cec87 [DeepseekV3MoE.moe_infer](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L536) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L536-609，函数体SHA256 0a3d7df256f1ead71010efcf90bf06032b904d44bb9a603464ca5976c6e05b1f

- de2e790cec87-601 [DeepseekV3MoE.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Base/blob/ce72df012259dcc55d945e890f815fe7ef69159c/modeling_deepseek.py#L601) — moonshotai/Kimi-K2-Base@ce72df012259dcc55d945e890f815fe7ef69159c，modeling_deepseek.py L601-609，函数体SHA256 49cda8061fc33569516f4ffe43c8495f5dc03cd94acb57b46893ca058cd3dac7

- 966cf86d3cd6 [DeepseekV3RMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L103) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L103-108，函数体SHA256 9a60087840825e1853db6aaf071c649ce5f0ea3cacb15e2367e89c7c35da4ab6

- 872642b6c3e2 [DeepseekV3DecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L1168) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L1168-1228，函数体SHA256 92bcd5947049b3756b90d5e58bba7c2478840bff50f05cee711a72b3330778ee

- a3ec1b8b4763 [DeepseekV3Attention.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L751) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L751-857，函数体SHA256 0d830741ac25e203e842a3ae654c6724874695180c30fa63a3c2da7a8c739bea

- 04b9cf8f5c4a [apply_rotary_pos_emb](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L339) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L339-371，函数体SHA256 39851acdc7fb26b777566c3136d987cb9ca6597a2164b6d49bee4c96b9de9ca6

- a3ec1b8b4763-816 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L816) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L816-840，函数体SHA256 4a3e0a866efbb51f8e09efe62853b97ac5354f17ee09ddfddc3446aa657cd92b

- a3ec1b8b4763-840 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L840) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L840-852，函数体SHA256 159d7b207301503b767bc6263722a2e5f450ba90fff3b87521da7ab15e205cfa

- 96c39d65fc6f [DeepseekV3MLP.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L388) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L388-390，函数体SHA256 c175739048a2dbe3d01102f0229b1c5d9159a7cfb5a4a72bea816801658f8f78

- a396e8fd85da [MoEGate.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L423) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L423-474，函数体SHA256 3e7518c4d08cafc73f7155115d733fa9823ef7f51040499ff89506490fb9573b

- 5da897a9f506 [DeepseekV3MoE.moe_infer](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L536) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L536-609，函数体SHA256 0a3d7df256f1ead71010efcf90bf06032b904d44bb9a603464ca5976c6e05b1f

- 5da897a9f506-601 [DeepseekV3MoE.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py#L601) — moonshotai/Kimi-K2-Instruct@fd1984e2b7a3350dbf7305fe73a4ede25c14de50，modeling_deepseek.py L601-609，函数体SHA256 49cda8061fc33569516f4ffe43c8495f5dc03cd94acb57b46893ca058cd3dac7

- 373d2ad688c6 [DeepseekV3RMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L103) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L103-108，函数体SHA256 9a60087840825e1853db6aaf071c649ce5f0ea3cacb15e2367e89c7c35da4ab6

- 04227bc06df5 [DeepseekV3DecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L1168) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L1168-1228，函数体SHA256 92bcd5947049b3756b90d5e58bba7c2478840bff50f05cee711a72b3330778ee

- 49f3797d886b [DeepseekV3Attention.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L751) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L751-857，函数体SHA256 0d830741ac25e203e842a3ae654c6724874695180c30fa63a3c2da7a8c739bea

- 8b9d444c7c4f [apply_rotary_pos_emb](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L339) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L339-371，函数体SHA256 39851acdc7fb26b777566c3136d987cb9ca6597a2164b6d49bee4c96b9de9ca6

- 49f3797d886b-816 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L816) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L816-840，函数体SHA256 4a3e0a866efbb51f8e09efe62853b97ac5354f17ee09ddfddc3446aa657cd92b

- 49f3797d886b-840 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L840) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L840-852，函数体SHA256 159d7b207301503b767bc6263722a2e5f450ba90fff3b87521da7ab15e205cfa

- 5eb11ce8a99d [DeepseekV3MLP.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L388) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L388-390，函数体SHA256 c175739048a2dbe3d01102f0229b1c5d9159a7cfb5a4a72bea816801658f8f78

- 6976ecff6eb7 [MoEGate.forward](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L423) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L423-474，函数体SHA256 3e7518c4d08cafc73f7155115d733fa9823ef7f51040499ff89506490fb9573b

- e3f1a60d4575 [DeepseekV3MoE.moe_infer](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L536) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L536-609，函数体SHA256 0a3d7df256f1ead71010efcf90bf06032b904d44bb9a603464ca5976c6e05b1f

- e3f1a60d4575-601 [DeepseekV3MoE.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905/blob/ac6c49f04883bd0a0598b790693a72061c676629/modeling_deepseek.py#L601) — moonshotai/Kimi-K2-Instruct-0905@ac6c49f04883bd0a0598b790693a72061c676629，modeling_deepseek.py L601-609，函数体SHA256 49cda8061fc33569516f4ffe43c8495f5dc03cd94acb57b46893ca058cd3dac7

- 4772e70c5fef [DeepseekV3RMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L103) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L103-108，函数体SHA256 9a60087840825e1853db6aaf071c649ce5f0ea3cacb15e2367e89c7c35da4ab6

- a9a94eb60491 [DeepseekV3DecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L1168) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L1168-1228，函数体SHA256 92bcd5947049b3756b90d5e58bba7c2478840bff50f05cee711a72b3330778ee

- a84d4ae48f26 [DeepseekV3Attention.forward](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L751) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L751-857，函数体SHA256 0d830741ac25e203e842a3ae654c6724874695180c30fa63a3c2da7a8c739bea

- 932106e78793 [apply_rotary_pos_emb](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L339) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L339-371，函数体SHA256 39851acdc7fb26b777566c3136d987cb9ca6597a2164b6d49bee4c96b9de9ca6

- a84d4ae48f26-816 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L816) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L816-840，函数体SHA256 4a3e0a866efbb51f8e09efe62853b97ac5354f17ee09ddfddc3446aa657cd92b

- a84d4ae48f26-840 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L840) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L840-852，函数体SHA256 159d7b207301503b767bc6263722a2e5f450ba90fff3b87521da7ab15e205cfa

- 7418dca0e99a [DeepseekV3MLP.forward](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L388) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L388-390，函数体SHA256 c175739048a2dbe3d01102f0229b1c5d9159a7cfb5a4a72bea816801658f8f78

- 7deaea179cf8 [MoEGate.forward](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L423) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L423-474，函数体SHA256 3e7518c4d08cafc73f7155115d733fa9823ef7f51040499ff89506490fb9573b

- a69383bbaecd [DeepseekV3MoE.moe_infer](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L536) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L536-609，函数体SHA256 0a3d7df256f1ead71010efcf90bf06032b904d44bb9a603464ca5976c6e05b1f

- a69383bbaecd-601 [DeepseekV3MoE.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/modeling_deepseek.py#L601) — moonshotai/Kimi-K2-Thinking@a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55，modeling_deepseek.py L601-609，函数体SHA256 49cda8061fc33569516f4ffe43c8495f5dc03cd94acb57b46893ca058cd3dac7

- 343f4ac2b717 [DeepseekV3RMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L104) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L104-110，函数体SHA256 e745856c65e1b0345a5e3e18d91d6a46e49e2b6432d3681d370d7e741e96a39d

- 43a137f02277 [DeepseekV3DecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L1153) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L1153-1212，函数体SHA256 604fd898a1af50a99082190468ec461ab7ba2277d52fb454f9509d1ed88d9ac7

- a5794f7773fa [DeepseekV3Attention.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L750) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L750-853，函数体SHA256 6137c58e8cb3ef3bd7ce21a9dd70b48c32b8c5389063bfcec53c8c32c5074534

- ef5f0a39e0de [apply_rotary_pos_emb](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L351) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L351-383，函数体SHA256 39851acdc7fb26b777566c3136d987cb9ca6597a2164b6d49bee4c96b9de9ca6

- a5794f7773fa-812 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L812) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L812-836，函数体SHA256 5f3f3f377f6c21d4785d39794e8ce46590317054675e61ae1a41df3d360207b3

- a5794f7773fa-836 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L836) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L836-848，函数体SHA256 1f91b8eba0e7b69e3462e02a7fb672c6fce39ccf99931d904de5099befb1f5b7

- c6983d2868ff [DeepseekV3MLP.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L406) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L406-409，函数体SHA256 bc1a02778a10656379f8395a7fe717e2a958d2db99ab4c1eac312985760194e5

- d5e9baafcc2d [MoEGate.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L441) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L441-490，函数体SHA256 29e689204a82aca52a96c2b58f00c4cf0e39a5b958fa7c946ec7cbc8b90104c4

- c9bacd41e46b [DeepseekV3MoE.moe_infer](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L544) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L544-609，函数体SHA256 a1fa6a7cbbeeff84316204dda23178c63980e50984c5b4caafab489632209a5d

- c9bacd41e46b-605 [DeepseekV3MoE.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_deepseek.py#L605) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_deepseek.py L605-609，函数体SHA256 40e92a0e04b79038e82b1d36f475369eab86405751ef94effc72c3d270c29b98

- ab36b61a75be [MoonViTEncoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L537) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L537-556，函数体SHA256 dafd875879e3d35dad4e9319dfd8c37008773eb7b7ed63eefa345eccfbd329aa

- ac58dedbdf7c [MoonViTEncoderLayer.attention_qkvpacked](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L499) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L499-535，函数体SHA256 090f3c33ae71aed08035aba0021e91cdadb7ca92b4b9dc3b8e34a73fd9208fb6

- d2fd11ae84f0 [MLP2.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L467) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L467-470，函数体SHA256 cf645d95ca97020a78e8a05e97e005598e25153c7c0e2f256a784583c6e7fdbb

- 025c90ee058b [tpool_patch_merger](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L606) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L606-631，函数体SHA256 6cc0d6a2a85aed1e49d33a5f842801468e650d884fed2def1eca68ebafa7d84c

- a0129ccce050 [PatchMergerMLP.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L751) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L751-761，函数体SHA256 abd57a7bec8c74195a203698a602f091d35ac626229e743490bf120a98acc06d

- d15b277cb06d [KimiK25ForConditionalGeneration._merge_input_ids_with_image_features](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L889) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L889-1021，函数体SHA256 8fb3ba94bb0334600038d27ec4a9a3729fa15760cb6c04d1b79334ba01617111

- f12177b1227e [MoonVision3dPatchEmbed.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L337) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L337-350，函数体SHA256 4db4950007387977b838d66000a3b9c534220478c719a76371f53922899d7cf9

- 5f7e9167e6b3 [MoonViT3dEncoder.forward](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/modeling_kimi_k25.py#L580) — moonshotai/Kimi-K2.5@4d01dfe0332d63057c186e0b262165819efb6611，modeling_kimi_k25.py L580-603，函数体SHA256 0e73b20a0153a60d4bbad28ebf3df4eaa684684f83b40b62b64ac212c438f5da

- 9c3cf0c431a1 [DeepseekV3RMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L104) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L104-110，函数体SHA256 e745856c65e1b0345a5e3e18d91d6a46e49e2b6432d3681d370d7e741e96a39d

- 5bcfed8d4432 [DeepseekV3DecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L1153) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L1153-1212，函数体SHA256 604fd898a1af50a99082190468ec461ab7ba2277d52fb454f9509d1ed88d9ac7

- ad7a465c7aa3 [DeepseekV3Attention.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L750) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L750-853，函数体SHA256 6137c58e8cb3ef3bd7ce21a9dd70b48c32b8c5389063bfcec53c8c32c5074534

- 8a172ed6bb67 [apply_rotary_pos_emb](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L351) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L351-383，函数体SHA256 39851acdc7fb26b777566c3136d987cb9ca6597a2164b6d49bee4c96b9de9ca6

- ad7a465c7aa3-812 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L812) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L812-836，函数体SHA256 5f3f3f377f6c21d4785d39794e8ce46590317054675e61ae1a41df3d360207b3

- ad7a465c7aa3-836 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L836) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L836-848，函数体SHA256 1f91b8eba0e7b69e3462e02a7fb672c6fce39ccf99931d904de5099befb1f5b7

- ffca734f0ce0 [DeepseekV3MLP.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L406) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L406-409，函数体SHA256 bc1a02778a10656379f8395a7fe717e2a958d2db99ab4c1eac312985760194e5

- 104cdf65ee0e [MoEGate.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L441) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L441-490，函数体SHA256 29e689204a82aca52a96c2b58f00c4cf0e39a5b958fa7c946ec7cbc8b90104c4

- ee53519c67f5 [DeepseekV3MoE.moe_infer](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L544) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L544-609，函数体SHA256 a1fa6a7cbbeeff84316204dda23178c63980e50984c5b4caafab489632209a5d

- ee53519c67f5-605 [DeepseekV3MoE.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_deepseek.py#L605) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_deepseek.py L605-609，函数体SHA256 40e92a0e04b79038e82b1d36f475369eab86405751ef94effc72c3d270c29b98

- ece6b3d2836b [MoonViTEncoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L537) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L537-556，函数体SHA256 dafd875879e3d35dad4e9319dfd8c37008773eb7b7ed63eefa345eccfbd329aa

- 33c11b5edf97 [MoonViTEncoderLayer.attention_qkvpacked](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L499) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L499-535，函数体SHA256 090f3c33ae71aed08035aba0021e91cdadb7ca92b4b9dc3b8e34a73fd9208fb6

- f45c08461917 [MLP2.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L467) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L467-470，函数体SHA256 cf645d95ca97020a78e8a05e97e005598e25153c7c0e2f256a784583c6e7fdbb

- 8433345fb702 [tpool_patch_merger](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L606) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L606-631，函数体SHA256 6cc0d6a2a85aed1e49d33a5f842801468e650d884fed2def1eca68ebafa7d84c

- c718cfcc69de [PatchMergerMLP.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L751) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L751-761，函数体SHA256 abd57a7bec8c74195a203698a602f091d35ac626229e743490bf120a98acc06d

- 41a11dbb404a [KimiK25ForConditionalGeneration._merge_input_ids_with_image_features](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L889) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L889-1021，函数体SHA256 8fb3ba94bb0334600038d27ec4a9a3729fa15760cb6c04d1b79334ba01617111

- bc375878d3f5 [MoonVision3dPatchEmbed.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L337) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L337-350，函数体SHA256 4db4950007387977b838d66000a3b9c534220478c719a76371f53922899d7cf9

- c16fea60f6d7 [MoonViT3dEncoder.forward](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/modeling_kimi_k25.py#L580) — moonshotai/Kimi-K2.6@7eb5002f6aadc958aed6a9177b7ed26bb94011bb，modeling_kimi_k25.py L580-603，函数体SHA256 0e73b20a0153a60d4bbad28ebf3df4eaa684684f83b40b62b64ac212c438f5da

- 4b713095a999 [DeepseekV3RMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L111) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L111-117，函数体SHA256 e745856c65e1b0345a5e3e18d91d6a46e49e2b6432d3681d370d7e741e96a39d

- f438445d3c2e [DeepseekV3DecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L1160) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L1160-1219，函数体SHA256 604fd898a1af50a99082190468ec461ab7ba2277d52fb454f9509d1ed88d9ac7

- 6e450223e5d9 [DeepseekV3Attention.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L757) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L757-860，函数体SHA256 6137c58e8cb3ef3bd7ce21a9dd70b48c32b8c5389063bfcec53c8c32c5074534

- e33615f374ab [apply_rotary_pos_emb](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L358) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L358-390，函数体SHA256 39851acdc7fb26b777566c3136d987cb9ca6597a2164b6d49bee4c96b9de9ca6

- 6e450223e5d9-819 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L819) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L819-843，函数体SHA256 5f3f3f377f6c21d4785d39794e8ce46590317054675e61ae1a41df3d360207b3

- 6e450223e5d9-843 [DeepseekV3Attention.forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L843) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L843-855，函数体SHA256 1f91b8eba0e7b69e3462e02a7fb672c6fce39ccf99931d904de5099befb1f5b7

- ca6e7760d3d2 [DeepseekV3MLP.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L413) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L413-416，函数体SHA256 bc1a02778a10656379f8395a7fe717e2a958d2db99ab4c1eac312985760194e5

- 44b7683f6e46 [MoEGate.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L448) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L448-497，函数体SHA256 29e689204a82aca52a96c2b58f00c4cf0e39a5b958fa7c946ec7cbc8b90104c4

- 5d35446fc19f [DeepseekV3MoE.moe_infer](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L551) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L551-616，函数体SHA256 a1fa6a7cbbeeff84316204dda23178c63980e50984c5b4caafab489632209a5d

- 5d35446fc19f-612 [DeepseekV3MoE.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_deepseek.py#L612) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_deepseek.py L612-616，函数体SHA256 40e92a0e04b79038e82b1d36f475369eab86405751ef94effc72c3d270c29b98

- 5ebd0f659b0d [MoonViTEncoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L571) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L571-590，函数体SHA256 dafd875879e3d35dad4e9319dfd8c37008773eb7b7ed63eefa345eccfbd329aa

- a6d5074fcb82 [MoonViTEncoderLayer.attention_qkvpacked](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L533) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L533-569，函数体SHA256 090f3c33ae71aed08035aba0021e91cdadb7ca92b4b9dc3b8e34a73fd9208fb6

- 9d63ae8d370a [MLP2.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L501) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L501-504，函数体SHA256 cf645d95ca97020a78e8a05e97e005598e25153c7c0e2f256a784583c6e7fdbb

- b3331effb242 [tpool_patch_merger](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L640) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L640-665，函数体SHA256 6cc0d6a2a85aed1e49d33a5f842801468e650d884fed2def1eca68ebafa7d84c

- fc186d292a78 [PatchMergerMLP.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L786) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L786-796，函数体SHA256 abd57a7bec8c74195a203698a602f091d35ac626229e743490bf120a98acc06d

- 13d695cbda05 [KimiK25ForConditionalGeneration._merge_input_ids_with_image_features](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L926) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L926-1058，函数体SHA256 8fb3ba94bb0334600038d27ec4a9a3729fa15760cb6c04d1b79334ba01617111

- 0c052c5d1778 [MoonVision3dPatchEmbed.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L371) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L371-384，函数体SHA256 4db4950007387977b838d66000a3b9c534220478c719a76371f53922899d7cf9

- ba6c21ad95c8 [MoonViT3dEncoder.forward](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/modeling_kimi_k25.py#L614) — moonshotai/Kimi-K2.7-Code@74797c9c62378b951a1f6fcf5c4631024e9b8bef，modeling_kimi_k25.py L614-637，函数体SHA256 0e73b20a0153a60d4bbad28ebf3df4eaa684684f83b40b62b64ac212c438f5da

- ff4715a1dd17 [_apply_attn_res](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L1075) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L1075-1088，函数体SHA256 c277f44ce2862f3379a1d1073134f701cd08c76143198cb64d36de32056b83d7

- 06cde78b0e97 [KimiRMSNorm.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L232) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L232-236，函数体SHA256 72e7fad629f3a75b66aa4cfe64bbd8470fc962fa99ccb343520f3366450ea258

- d86a747142f0 [KimiDecoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L919) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L919-971，函数体SHA256 ff75584ddb6e9cb9a71f3783406f01bded8ecd7937190601e322e8dfda91ed2f

- 810b468803e0 [KimiDeltaAttention.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L543) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L543-663，函数体SHA256 ab5c019f76ae9f36acd07f0bdee07212c8be2341d2eb7b19c522748c2e25da30

- ebcb6575be7b [KimiK3DeltaAttention.__init__](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/models/kimi_k3/nvidia/kda.py#L515) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/models/kimi_k3/nvidia/kda.py L515-739，函数体SHA256 92072a81b07c73e6b34ce4e44c193394872fa9b6f972a6c7bcf37ae5bef867cc

- 5d33ac8f6ae9 [SituAndMul.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L75) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L75-82，函数体SHA256 833a9db24a7dc5a52add755f6c62154c15fd684496289a3329f31928366c3ac9

- 81e5edcfb3a4 [KimiMoEGate.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L703) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L703-759，函数体SHA256 8da7330ba5a9c446c0fdda92ca30743215ee6293722d8766ea095237b30aea98

- 18868abba78e [KimiSparseMoeBlock.moe_infer](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L841) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L841-874，函数体SHA256 ef19d9e76bb1b1773487303e04f91f2834de717ca6040def18557a3605586f79

- 18868abba78e-866 [KimiSparseMoeBlock.moe_infer / 步骤语句](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L866) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L866-874，函数体SHA256 49cda8061fc33569516f4ffe43c8495f5dc03cd94acb57b46893ca058cd3dac7

- ce998ab6e49a [KimiMLAAttention.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L405) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L405-474，函数体SHA256 bc6eddbce0e078ab536121c3c9aaaf7342f225e783cd63daf780e5b54da54cc6

- 2cf4ee46912b [AscendKimiDecoderLayer.__init__](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/models/kimi_k3.py#L395) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/models/kimi_k3.py L395-509，函数体SHA256 1f62bd6af0beadfb6c2b93f158bd182588bb5f944058e2dc5ec1acbea66e77cb

- 1bc2e6502ffe [AscendKimiMLAAttention.__init__](https://github.com/vllm-project/vllm-ascend/blob/0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1/vllm_ascend/models/kimi_k3.py#L285) — vllm-project/vllm-ascend@0003e1b75b8ff4d35a5871fee1b2e9e7860bc3f1，vllm_ascend/models/kimi_k3.py L285-354，函数体SHA256 9b815b17b4d9ecd904353fe17f22e4490bac762f75629bf3394daa09e150c035

- 84d2a5c76c4a [eager_attention_forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L311) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L311-332，函数体SHA256 ffe6d605a728a60381f78b8e060f9c15f14890c27a120af72d6d8c5a859b6681

- 84d2a5c76c4a-324 [eager_attention_forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L324) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L324-330，函数体SHA256 692889f2a2d5d27150bb981c57bb0a59677e1bd71a8a1cf81e48581cef1b9402

- 84d2a5c76c4a-330 [eager_attention_forward / 步骤语句](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py#L330) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_linear.py L330-332，函数体SHA256 78054544e99199531158b80426e1526e0570835e2590f66ddf6f3102091056eb

- f7142733ff6a [MultiHeadLatentAttention.__init__](https://github.com/vllm-project/vllm/blob/97dc6b19d2fe92b794b49e684677a2e1a8b2c540/vllm/models/kimi_k3/nvidia/mla.py#L128) — vllm-project/vllm@97dc6b19d2fe92b794b49e684677a2e1a8b2c540，vllm/models/kimi_k3/nvidia/mla.py L128-407，函数体SHA256 db213a2b3b6a3e661ad192ff1e002f43c626d5695a1a0650b48aa8981b4590ed

- db18af94f22d [MoonViTEncoderLayer.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L545) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L545-564，函数体SHA256 dafd875879e3d35dad4e9319dfd8c37008773eb7b7ed63eefa345eccfbd329aa

- edcaf149bc62 [MoonViTEncoderLayer.attention_qkvpacked](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L507) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L507-543，函数体SHA256 090f3c33ae71aed08035aba0021e91cdadb7ca92b4b9dc3b8e34a73fd9208fb6

- 9c268ae3cc41 [MLP2.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L455) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L455-458，函数体SHA256 cf645d95ca97020a78e8a05e97e005598e25153c7c0e2f256a784583c6e7fdbb

- 521f6594eb83 [tpool_patch_merger](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L621) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L621-646，函数体SHA256 6cc0d6a2a85aed1e49d33a5f842801468e650d884fed2def1eca68ebafa7d84c

- 0cc97b41f6f7 [PatchMergerMLPV2.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L803) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L803-815，函数体SHA256 f312226687d52923fcde272fc32f26f591889679153743cfb6a9b7fb34ad49ec

- c47424276a48 [KimiK3ForConditionalGeneration._merge_input_ids_with_image_features](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L958) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L958-1090，函数体SHA256 8fb3ba94bb0334600038d27ec4a9a3729fa15760cb6c04d1b79334ba01617111

- 34810eb33520 [MoonVision3dPatchEmbed.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L325) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L325-338，函数体SHA256 4db4950007387977b838d66000a3b9c534220478c719a76371f53922899d7cf9

- 3ec798fd8f7c [MoonViT3dEncoder.forward](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_k3.py#L595) — moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06，modeling_kimi_k3.py L595-618，函数体SHA256 0e73b20a0153a60d4bbad28ebf3df4eaa684684f83b40b62b64ac212c438f5da