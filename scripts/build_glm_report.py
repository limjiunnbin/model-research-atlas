"""Author the GLM research report sections (data/families/glm/report.json).

Section HTML is authored here, not scraped, so every number can be traced to the pinned
snapshot cited in hardware.json. No external scripts or event attributes are emitted.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/families/glm/report.json'
COMMIT = '3b36eeac4cde5e608f00cdd0560a715ff8925e15'
REPO = 'https://gitcode.com/cann/cann-recipes-infer'


def table(headers, rows):
    head = ''.join(f'<th>{h}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


S0 = f"""<h2>GLM-5 → GLM-5.3-Flash：结构演进、混合注意力与昇腾部署研究</h2>
<p>本研究基于华为 CANN 推理配方仓库 <a href="{REPO}" target="_blank" rel="noopener">cann-recipes-infer</a>
的固定快照 <code>{COMMIT}</code>，整理 GLM-5、GLM-5.2 与 GLM-5.3-Flash 三个版本在昇腾 NPU 上的实现方式。</p>
<p>研究材料全部是公开源码与配方文档。<strong>本站没有加载任何权重、没有做文件头审计、没有运行任何 NPU 性能实验。</strong>
所有结构数据由实现源码中参数的声明形状推导，所有性能数字均标注为配方自述而非本站复现。</p>"""

S1 = """<h2>1. 研究口径与证据强度</h2>
<p>本家族的证据链只有一条：Huawei 的推理配方仓库。它不是智谱的官方模型卡，因此所有事实的证据等级统一记为<strong>计算推导</strong>，
而不是官方披露——即使某个数值在配方里被直接写出。</p>
<p>三类信息严格分开：</p>
<ul>
<li><strong>源码可核验</strong>：参数声明形状、层排布规则、量化分档的判定分支、并行约束。这些可以逐行定位。</li>
<li><strong>配方自述</strong>：README 给出的吞吐表、算子文档里"性能较高/较低"这类定性描述。本站原样引用并标注来源，不作为已验证结论。</li>
<li><strong>本站实测</strong>：<strong>没有</strong>。本研究不包含任何运行结果。</li>
</ul>
<p>凡是源码里没有、配方里也没写的，一律记为未知，不做估算。</p>"""

S2 = """<h2>2. 主线演进：改变在哪里</h2>
""" + table(
    ['阶段', '结构变化', '部署侧变化', '不能据此声称'],
    [['GLM-5', '78 层 DSA（MLA + Lightning Indexer）+ MoE + MTP；hidden 6144，256 专家选 8 + 1 共享',
      'Atlas A3；发布 BF16 与 FP8 两种权重；配方转 W8A8(int8) 或 MXFP8',
      '配方自述"结构与 DeepSeek-V3.2-Exp 基本一致"等于两者权重同源'],
     ['GLM-5 → GLM-5.2', '主干尺寸不变；核心新增 IndexShare：按 indexer_types 每层取 full 或 shared，shared 层复用上一个 full 层的 top-k 且不持有 indexer 权重',
      'Atlas A3；发布 BF16；真实注意力维度需 CANN 9.1 的 mla_prolog_v3 内核',
      'IndexShare 是参数量变化，或对质量无代价'],
     ['GLM-5.2 → GLM-5.3-Flash', '层数 78 → 45，hidden 6144 → 4096；34 层换成 KDA 线性注意力，11 层 DSA 改为全程 NoPE + k-pool 压缩 indexer；新增 4 流 mHC 超连接；专家 256 → 288；上下文上限 202,752 → 1,048,576',
      '昇腾 950DT；发布 FP8；配方转 Hybrid HiF8-MXFP8-MXFP4；依赖 mHC 与 causal_conv1d 两个自定义算子包；暂不支持 MTP',
      '5.3 是 5.2 的缩小版，或层数减少等于能力下降']]) + """
<p>GLM-5.3-Flash 不是把 GLM-5.2 等比缩小。它同时改了注意力的组成方式（混合）、残差通路的结构（mHC 四流）、
以及长上下文的代价模型（KDA 递推状态 + 压缩潜在缓存）。名字里的 Flash 对应的是部署形态的变化，不只是参数规模。</p>"""

S3 = """<h2>3. 总体规格并列</h2>
<p>下表只放可从配方配置类读到的值。参数总数一栏留空，因为本站没有做权重审计。</p>
""" + table(
    ['项', 'GLM-5', 'GLM-5.2', 'GLM-5.3-Flash'],
    [['语言层数', '78', '78', '45'],
     ['注意力组成', '全部 DSA', '全部 DSA（部分层 IndexShare）', '34 KDA + 11 DSA'],
     ['残差宽度 hidden', '6144', '6144', '4096'],
     ['注意力头数', '64', '64', '64'],
     ['路由专家 / top-k / 共享', '256 / 8 / 1', '256 / 8 / 1', '288 / 8 / 1'],
     ['专家中间宽度', '2048', '2048', '2048'],
     ['Dense 中间宽度', '12288', '12288', '12288'],
     ['q_lora_rank', '2048', '2048', '1536'],
     ['qk_nope / qk_rope', '192 / 64', '192 / 64', '256 / 0（全程 NoPE）'],
     ['kv_lora_rank', '512', '512', '512'],
     ['上下文配置上限', '202,752', '202,752', '1,048,576'],
     ['词表', '154,880', '154,880', '154,880'],
     ['发布精度', 'BF16 与 FP8', 'BF16', 'FP8'],
     ['参数总数', '未审计', '未审计', '未审计（配方自述约 306B）']]) + """
<p><strong>关于参数总数的一处口径差异。</strong>GLM-5.3-Flash 的配置已与智谱发布的
<code>config.json</code> 逐项核对：<code>first_k_dense_replace</code> 确为 <strong>3</strong>，
<code>layer_types</code> 为 34 个 <code>linear_attention</code> 加 11 个
<code>deepseek_sparse_attention</code>，<code>linear_attn_config.full_attn_layers</code> 为
<code>[3,7,11,15,19,23,27,31,35,39,43]</code>，<code>mlp_layer_types</code> 为前 3 层 dense 加 42 层
sparse。实现默认值没有被覆盖。</p>
<p>据此逐矩阵推导，45 层解码主干加 embed / lm_head / 末层归一为
<code>313,326,811,966</code>（不含视觉塔与 MTP 层）。配方 README 自述的"约 306B"与这个口径<strong>不符</strong>，
而仓库没有给出它的推导过程。本站如实保留两个数字，不替任何一方消解差异。</p>
<p>需要说明的是，早先曾推测"约 306B 对应前 4 层 Dense"——该推测已被官方配置<strong>证伪</strong>，
相关表述已全部更正。</p>
<p>发布检查点还包含两部分本配方不构建的内容：<strong>24 层视觉塔</strong>
（<code>vision_config.depth=24</code>、<code>hidden_size=1024</code>、<code>out_hidden_size=4096</code>）
与 <strong>1 个 MTP 层</strong>（<code>num_nextn_predict_layers=1</code>，位于第 45 层，同为 DSA + MoE）。
它们在本站只有配置级记录，没有逐矩阵形状。</p>"""

S4 = """<h2>4. 混合注意力：KDA 与 DSA 各自承担什么</h2>
<p>GLM-5.3-Flash 的 45 层按 <code>i % 4 == 3</code> 划分：第 4、8、12……层（0 起计为 3、7、11……）是 DSA，
共 11 层；其余 34 层是 KDA。这个排布来自配置类默认值。</p>
<h3>4.1 KDA：用固定大小的状态代替 KV</h3>
<p>KDA 是线性注意力：短卷积加门控 delta 递推。它的每层缓存由两部分构成，都与序列长度无关：</p>
<ul>
<li>短卷积状态：<code>[kernel-1, 8192] = [3, 8192]</code>，q/k/v 各一份；</li>
<li>递推矩阵状态：<code>[64, 128, 128]</code>，融合路径下为 float32。</li>
</ul>
<p>投影方面，q/k/v 是满秩 <code>[8192, 4096]</code>；衰减与输出门都走低秩对（rank 128）：
<code>f_a_proj [128,4096] → f_b_proj [8192,128]</code>、<code>g_a_proj → g_b_proj</code> 同形；
另有每头一个标量的 <code>b_proj [64, 4096]</code>、<code>A_log [64]</code> 与 <code>dt_bias [8192]</code>。</p>
<p>prefill 走 <code>flash_kda</code>，decode 走 <code>fused_recurrent_kda</code>；两者都直接消费<strong>未激活</strong>的原始
decay，把 l2norm、门控激活与 beta sigmoid 融合进内核。短卷积与 SiLU 由 <code>causal_conv1d</code> 一并完成。</p>
<h3>4.2 DSA：吸收式 MLA，全程不做 RoPE</h3>
<p>DSA 层的 <code>qk_rope_head_dim = 0</code>，配置类会显式校验这一点。注意力全程在压缩潜在空间进行：
加载后把 <code>kv_b_proj [32768, 512]</code> 拆成 <code>w_k [64, 256, 512]</code> 与 <code>w_v [64, 512, 256]</code>
并释放原权重，查询经 einsum 吸收进潜在空间，直接在 512 维潜在缓存上做稀疏注意力，输出侧再展开。</p>
<p>因此 <strong>每 token 每 DSA 层只缓存 512 个元素</strong>，缓存里没有任何位置分量。稀疏注意力内核仍要求 rope 操作数，
由常零张量合成——这是接口适配，不是模型里存在被置零的位置编码参数。</p>
<h3>4.3 k-pool 压缩 indexer</h3>
<p>Lightning Indexer 要对全部历史打分，长上下文下这一步本身会成为瓶颈。GLM-5.3 把每 4 个连续 token
用 <code>softmax(门控 + 位置嵌入)</code> 在组内加权合并成一个池化 key，打分对象从 token 数降到 token 数 / 4。</p>
<p>关键性质是：一个池在它的 4 个 token 齐全后就不再变化，所以只池化一次、之后从缓存读回。
decode 用 <code>npu_lightning_indexer</code> 在池化缓存上选 top-k；<strong>prefill 没有对应算子</strong>，
仍走 torch 的 matmul 与 topk，按 1024 分块控制显存。选择宽度 <code>index_topk / kpool = 512</code>。</p>"""

S5 = """<h2>5. mHC：把残差通路从一条变成四条</h2>
<p>GLM-5.3-Flash 的层间隐藏态不是一个向量，而是 4 条并行流 <code>[T, 4, D]</code>。
embedding 之后复制成 4 份，45 层之后取均值收回单流——收敛处是无参数的算术平均，没有学习到的加权头。</p>
<p>每个子层前后各有一次流的折叠与展开，<strong>注意力侧与 FFN 侧各有一组独立参数</strong>：</p>
<ul>
<li><code>hc_attn_fn</code> / <code>hc_ffn_fn</code>：<code>[24, 16384]</code>，float32；</li>
<li><code>hc_attn_base</code> / <code>hc_ffn_base</code>：<code>[24]</code>；</li>
<li><code>hc_attn_scale</code> / <code>hc_ffn_scale</code>：<code>[3]</code>。</li>
</ul>
<p>其中 <code>24 = (2 + 4) × 4</code>，拆成 <code>pre[4] | post[4] | comb[4×4]</code>；<code>16384 = 4 × 4096</code>
是展平后的四流向量。<code>npu_hc_pre</code> 把"展平 → RMS 归一 → 与 hc_fn 矩阵乘 → 拆分 → Sinkhorn 归一（20 轮）→ 按 pre 折叠"
融合成一个算子；<code>npu_hc_post</code> 把"按 post 与 comb 将子层输出与残差重新展开回四流"融合成另一个。</p>
<p>代价是每层多两次全流量的矩阵混合，调用次数与层数同阶（45 层 × 2 对 = 90 次）。这也是为什么这两步必须是融合算子：
拆开写会直接变成访存瓶颈。算子文档注明一个<strong>形状相关的分支</strong>——当 <code>T/bs ≤ 128</code> 且能被 16 整除时
启用融合 kernel，否则回退到两个拆分 kernel，文档分别标注"性能较高/较低"，但<strong>没有给出任何数字</strong>。</p>
<p>这两个算子由独立的 <code>opp/vendors/customize</code> 包提供，不在主仓库内；<code>set_env.sh</code> 负责载入。</p>"""

S6 = """<h2>6. 混合精度与昇腾侧优化</h2>
<h3>6.1 Hybrid HiF8-MXFP8-MXFP4：按模块分档</h3>
<p>转换脚本读入发布的 FP8 权重（e4m3，128×128 块 scale），反量化到 BF16，再按张量名判定归属。
这不是"全模型降到某个位宽"，而是按参数量与数值敏感度分三档：</p>
""" + table(
    ['张量', '格式', '落盘形态'],
    [['<code>mlp.experts.{e}.{gate,up,down}_proj</code>（路由专家）', 'MXFP4 W4A8',
      'uint8 <code>[N, K/2]</code>，2 值/字节；scale uint8 <code>[N, K/32]</code>（E8M0）'],
     ['<code>mlp.shared_experts.*</code>（共享专家）', 'MXFP8 W8A8',
      'float8_e4m3fn <code>[N, K]</code>；scale uint8 <code>[N, K/32]</code>（E8M0）'],
     ['DSA 的 <code>q_a_proj / q_b_proj / kv_a_proj_with_mqa / o_proj</code>，以及前几层 dense MLP', 'HiF8 W8A8',
      'uint8 <code>[N, K]</code>；scale float32 <code>[N, 1]</code>，<strong>逐输出通道</strong>'],
     ['KDA 层全部 self_attn、<code>kv_b_proj</code>、整个 indexer、<code>mlp.gate</code>、各 norm、embed / lm_head',
      '保持 BF16 / FP32', '转换时跳过']]) + """
<p>几处值得注意的判定：<code>kv_b_proj</code> 必须留 BF16，因为它要被拆成吸收式 MLA 的 w_k / w_v；
路由 <code>mlp.gate</code> 不量化，运行时是 fp32；<strong>KDA 层的全部权重都不量化</strong>——
这意味着 34/45 的注意力权重仍是 BF16，显存收益主要来自专家而不是注意力。</p>
<p>HiF8 的编码由 <code>torch_npu.npu_dynamic_quant(dst_type=torch_npu.hifloat8)</code> 完成，
算子自己产出逐行 scale；文档说明该 dst_type 需要 SoC 原生支持，910B/A2 会拒绝。
MX 系列块大小固定 32（沿 K 方向），共享指数为 E8M0 存成 uint8。</p>
<p>运行时靠 compressed-tensors 的 target 匹配还原成三个 config group，<strong>匹配顺序有意义</strong>：
共享专家的正则排在最前，其余 Linear 才落到 HiF8。模型侧还会断言量化模式必须恰好是
<code>("w8a8hifloat8", "w4a8mxfloat4")</code>，否则直接报错。</p>
<h3>6.2 并行：并行度全部压在专家维</h3>
<p><code>world_size=8</code>、<code>moe_tp_size=1</code>，框架据此派生 <code>moe_ep_size = 8</code>，288 个专家按 Expert 维分到 8 卡。
<code>cp / attn_tp / dense_tp / moe_tp / o_proj_tp / shared_tp</code> 必须全为 1，否则直接抛错。</p>
<p>也就是说注意力与 dense 部分每卡各算各的（DP），只有 MoE 做 EP 切分。好处是注意力侧没有 TP 通信，
代价是每卡都要完整持有全部注意力与 dense 权重，外加 288/8 = 36 个专家。</p>
<h3>6.3 重叠与图模式</h3>
<p><code>enable_multi_streams</code> 默认开启：decode 且 EP &gt; 1 时，共享专家切到侧流执行，
与路由专家的 dispatch / GMM / combine 重叠，combine 前汇合。共享专家对每个 token 都要算，
但与路由专家无数据依赖，正好用来填住 EP 全局通信的空窗。prefill 与单卡 EP 下改为内联。</p>
<p>decode 默认 <code>npugraph_ex</code> 图模式，<strong>只捕获 decode step</strong>，prefill 仍是 eager。
仓库里多处写法是为满足图模式约束而存在：decode 分支不做 host 侧读取以便 fullgraph 追踪；
常零 rope 缓冲按 is_prefill 分键，避免穿插的 prefill 重新分配已被捕获的 decode 缓冲；
indexer 的池聚合只在 prefill 使用，因为它在 host 上读 <code>.max()</code>。</p>
<h3>6.4 配方自述的吞吐</h3>
<p>README 给出一张表，条件为昇腾 950DT、8 卡、8K 输入（InfiniteBench）、npugraph_ex、多流开启、
Hybrid HiF8-MXFP8-MXFP4：</p>
""" + table(
    ['每卡 BS', 'Decode 每步耗时 (ms)', '每卡 TPS', '8 卡总 TPS'],
    [['1', '16.60', '60.24', '481.93'],
     ['16', '24.07', '664.73', '5,317.82'],
     ['64', '35.69', '1,793.22', '14,345.76'],
     ['128', '49.88', '2,566.16', '20,529.27']]) + """
<p><strong>这是配方自述，本站未复现。</strong>该表没有写明 CANN、torch_npu、固件驱动版本、重复次数与预热方式，
也没有给出精度结果，且只覆盖 decode。仓库内<strong>没有任何针对单项优化的 A/B 数据</strong>——
mHC 融合算子、多流、图模式、k-pool、吸收式 MLA、HiF8 与 BF16 的对比全都没有实测数字。</p>
<p>另需注意：根 README 里的困惑度 3.5888 属于另一条 SGLang 单卡 MoE offload 路径（注意力与常驻专家走 INT8 W8A8、
offload 专家走 MXFP4），与本配方是不同代码路径，不能归到这里。</p>
<h3>6.5 一处代码一致性疑点</h3>
<p>模型向路由专家的量化配置传入 <code>{"swiglu_limit": ..., "enable_custom_ops": True}</code>，
但仓库内除该行外没有任何位置读取 <code>enable_custom_ops</code>。MXFP4 的 GMM 因此走默认的
<code>npu_swiglu_mx_quant</code> 分支，<code>clamp_limit</code> 不参与；而共享专家的 MXFP8 路径是显式传了 clamp 的。
这是源码阅读得出的疑点，未经运行确认，也可能由算子包内部默认处理。</p>"""

S7 = """<h2>7. 能确定什么，哪些还需要实测</h2>
<h3>可以确定（源码逐行可核验）</h3>
<ul>
<li>45 层的排布规则、每个矩阵的声明形状、KDA 与 DSA 的模块构成。</li>
<li>量化分档的判定分支与每一档的落盘格式、scale 粒度与 dtype。</li>
<li>并行约束：EP = world_size，其余并行维必须为 1，且有集中校验。</li>
<li>mHC 每层两组独立参数，以及两个融合算子的 I/O 契约与形状约束。</li>
<li>k-pool 的压缩数学、decode 与 prefill 走不同实现这一事实。</li>
</ul>
<h3>还不能确定（需要实测或厂商确认）</h3>
<ul>
<li>发布 config.json 的实际 <code>first_k_dense_replace</code>，以及由此确定的真实参数总数。</li>
<li>任何一项优化的量化收益——仓库没有 A/B 数据，本站也没有运行实验。</li>
<li>分模块量化对任务质量的影响：无精度评测可引用。</li>
<li>mHC 融合算子在目标负载下是否命中融合分支还是回退分支。</li>
<li>prefill 阶段 indexer 的 torch 路径是否构成长序列瓶颈。</li>
<li>Sinkhorn 归一化、flash_kda、稀疏注意力等内核的内部数值实现（在算子包内）。</li>
</ul>
<p>这些条目在「Ascend 优化」页按 P0/P1/P2 列成了待验证清单，每条都写明观察、方案、指标与风险。</p>"""

S8 = f"""<h2>8. 来源与复核方式</h2>
<p>全部材料来自 <a href="{REPO}" target="_blank" rel="noopener">cann-recipes-infer</a> 固定快照
<code>{COMMIT}</code>。来源页提供逐文件的 URL、revision 与 SHA-256，可逐项复核。</p>
<p>复核路径：克隆该仓库并检出上述 commit，对照来源 CSV 里的 SHA-256 校验文件，
再按各条结论的引用行号定位源码。结构数据可用 <code>scripts/build_glm53_architecture.py</code> 重新生成，
该脚本把形状与逻辑参数的推导过程完整写在代码里。</p>
<p>本站未与智谱发布的 config.json 或权重文件逐项复核，也未运行任何 NPU 实验。</p>"""

S9 = """<h2>9. Profiling 与实测数据：分清是哪条路径</h2>
<p>GLM-5.3-Flash 在这个仓库里有<strong>两条互不相同的部署路径</strong>，实测数据的分布极不均匀。
把它们混在一起看会得出完全错误的结论，所以先分开。</p>
""" + table(
    ['', '路径 A：多卡 HiF8 配方', '路径 B：单卡 CPU-MoE offload'],
    [['代码位置', '<code>models/glm_5_3/</code>（本站主要研究对象）',
      '<code>integration/sglang/glm53-flash-single-npu-moe-offload/</code>'],
     ['栈', 'cann-recipes-infer 自带 executor', 'SGLang + KTransformers 补丁'],
     ['量化', 'Hybrid <strong>HiF8</strong>-MXFP8-MXFP4', '注意力/dense/常驻专家 <strong>INT8 W8A8</strong>；offload 专家 MXFP4'],
     ['硬件', '昇腾 950DT × 8，EP8', '单张 910C A3 die + 鲲鹏 CPU，61.3 GiB HBM'],
     ['实测深度', '<strong>仅一张 decode 吞吐表</strong>', '<strong>profiler 导出的逐组件分解 + 多项 A/B + 精度账</strong>']]) + """
<h3>9.1 路径 A（HiF8 配方）：没有 profiling 数据</h3>
<p>这正是本站研究的那条路径，它的实测材料非常薄：</p>
<ul>
<li>只有 README 的 decode 吞吐表（见 6.4），属配方自述，未写明 CANN / torch_npu / 固件驱动版本、重复次数与预热方式，只覆盖 decode，不含精度。</li>
<li><strong>没有提交任何 profiling trace 或 timeline 产物。</strong></li>
<li><strong>没有任何单项优化的 A/B</strong>：mHC 融合算子、共享专家多流、npugraph_ex、k-pool indexer、吸收式 MLA、causal_conv1d 全部没有开关前后的数字。</li>
<li><strong>HiF8 本身没有任何 profiling 或与其他精度档的算子级耗时对照。</strong></li>
<li>没有针对本配方的精度或困惑度结果。</li>
</ul>
<p>所以「Ascend 优化」页的每一条都是<strong>待验证项</strong>，指标栏写的是"应该采集什么"，不是"已经测到多少"。
本站不替这些格子填推测值。</p>
<h3>9.2 路径 B（单卡 offload）：有 profiler 导出的实测，但不能挪用</h3>
<p>这条路径的设计文档是整个仓库里最深的性能分析。它<strong>不是</strong> HiF8 配方的数据，
量化方案、卡数与框架都不同，<strong>不可用于推断路径 A</strong>。但它对理解 GLM-5.3 的结构开销很有参考价值。</p>
<p><strong>NPU 关键路径按模型结构分解</strong>（16K 上下文，单步 33.3 ms）：</p>
""" + table(
    ['组件', '耗时 (ms)', '占 NPU 关键路径'],
    [['KDA（34 层线性注意力）', '10.4', '约 31%'],
     ['MoE', '9.6', '约 29%'],
     ['DSA（11 层稀疏注意力）', '5.4', '约 16%'],
     ['<strong>mHC</strong>（45 层 × 2 对）', '<strong>3.3</strong>', '<strong>约 10%</strong>'],
     ['合计', '33.3', '100%']]) + """
<p>文档注明该分解<strong>由 trace 定位，不是用减法反推</strong>；同时说明 profiler 本身会抬高单步耗时
（profiling 下 20.89–28.72 ms，生产 20.57 ms），所以绝对值不可直接当生产指标。</p>
<p><strong>带置信区间的隔离 A/B</strong>（这条路径上）：</p>
<ul>
<li>动态热专家常驻 开/关：19.28 → 23.13 tok/s，<strong>1.200× ± 0.017</strong>（n=8 交错，95% CI [1.166, 1.233]）。</li>
<li>侧流重叠 开/关：decode 增益约 <strong>12%</strong>；A5 上为 −5.97 ms/token。结构上限由 trace 数出
（每步 44.1 个 &gt;50 µs 的空隙，合计 21,038 µs）。</li>
<li>流式 prefill 开/关：13,933 token 的 TTFT 208.36 → 63.93 s（<strong>3.26×</strong>）；
但 630 token 的短输入<strong>反而变慢</strong> 11.4 → 18.3 s，因此设了 512 的阈值。</li>
<li>NUMA 绑核：双节点达成 230–250 GB/s（上限 343，67–73%）；配错成 <code>NODES=0</code> 加 <code>tp=1</code>
会掉到 <strong>38.5 GB/s</strong>。A5 上按 L3 域轮转分配使每层 116.2 → 94.6 µs（<strong>−18.6%</strong>）。</li>
</ul>
<p><strong>精度账</strong>（路径 B）：GSM8K 单卡 97.35%（多卡 TP8 97.42–97.65%，官方 BF16 97.50%，n=1319）；
GPQA-Diamond 单卡 <strong>84.85%</strong>（168/198，多卡 85.35%）；wikitext-2 困惑度
<strong>3.5888</strong>（32 窗）、相对基线 <strong>+1.13%</strong>。
注意困惑度的窗数不可跨设置比较——同一配置 3 窗是 2.2445、12 窗是 3.1449、32 窗才是 3.5888。</p>
<p>这也更正了本站此前的一处表述：根 README 的 3.5888 确属这条单卡 offload 路径，
对应 INT8 W8A8 加 MXFP4，<strong>不是</strong> HiF8 配方的结果。</p>
<h3>9.3 同家族其他版本的实测</h3>
<ul>
<li><strong>GLM-5</strong>：Atlas A3、W8A8、全局 BS 256、序列 65536、64 卡 → TPOT <strong>22.54 ms</strong>、
单卡吞吐 <strong>177.46 tokens/p/s</strong>；GSM8K <strong>96.59%</strong>。
条件注明"基于 MTP3 与 force eplb 采集，平均 3 个 draft token 中 accept 1.44 个"。</li>
<li><strong>GLM-5.2</strong>：<strong>仓库内没有任何性能或精度数字</strong>，是一处明确空缺。</li>
</ul>
<h3>9.4 怎么自己采集</h3>
<p>框架的 profiler 是 <code>torch_npu.profiler</code>，统一走 <code>executor/utils/profiler_context.py</code>。
YAML 只暴露 <code>enable_profiler</code> 一个开关，采集参数（Level1、PipeUtilization、
<code>active=3, repeat=1, skip_first=3</code>）全部硬编码在该文件里。产物落在
<code>&lt;output_path&gt;/prof/{prefill,decode,mm_encode}</code>。</p>
<p>两个已知坑：decode 阶段<strong>无法冷启动单独采集</strong>（必须先经过 prefill，且一旦进入 decode 不再切回）；
离线 msprof 解析常只产出截断的 <code>trace_view.json</code> 而 CSV 全缺，仓库把根因归到输出目录所在的文件系统。</p>
<h3>9.5 仍然没有的</h3>
<ul>
<li>任何可供再分析的原始 trace / CSV（profiler 是开关，产物不入库）。</li>
<li><code>ops/ascendc/docs/</code> 下的 20 份算子文档<strong>全部没有实测数字</strong>，只有定性描述
（如 hc_pre 的"性能较高 / 较低"分支）。</li>
<li><code>benchmark/</code> 目录只做精度评测，不含吞吐或时延，也没有 GLM 条目。</li>
<li>路径 A 上 HiF8 与 BF16 的任何对照。</li>
</ul>"""


S10 = """<h2>10. 未来结构预判</h2>
<p><strong>本节是解释判断，不是官方路线图。</strong>依据只有一类：<em>代码与配置里已经存在、但在
GLM-5.3-Flash 上尚未启用的钩子</em>。这类"预留但未用"的痕迹比趋势外推可靠，但仍然可能永远不落地。
每条都给出它的实际依据。</p>
<h3>10.1 有明确预留痕迹的（依据较强）</h3>
""" + table(
    ['预判', '已存在的痕迹', '为什么是预留而非遗漏'],
    [['<strong>IndexShare 进入混合注意力线</strong>：部分 DSA 层复用上一个 full 层的 top-k，不再持有 indexer 权重',
      'GLM-5.3 的 <code>indexer_types</code> 全为 <code>full</code>（官方 config 的 45 项亦全为 full），但配置类完整保留了该调度字段的解析逻辑',
      '配置类注释明说保留该调度"以备未来检查点"；GLM-5.2 已在非混合主干上验证过 IndexShare'],
     ['<strong>MTP 推理支持</strong>：第 45 层参与投机解码',
      'MTP 层已在发布检查点中（<code>num_nextn_predict_layers=1</code>，排除列表含 <code>eh_proj</code> / <code>enorm</code> / <code>hnorm</code> / <code>shared_head.norm</code>），转换脚本会处理它，但运行时拒绝 <code>next_n&gt;0</code>，README 写"暂不支持 MTP"',
      '权重已发布、转换已支持，缺的只是运行时；"暂不支持"是明确的待办表述'],
     ['<strong>多模态通路</strong>：24 层视觉塔接入',
      '发布 config 是多模态的（<code>Glm5NextForConditionalGeneration</code>，含 <code>vision_config</code> 与 image/video token id），但配方丢弃 <code>vision_config</code> 并跳过 <code>model.visual.*</code>',
      '视觉权重与配置都已发布，配方侧是明确的"仅文本"取舍'],
     ['<strong>hc_pre 的 v2 变体</strong>：复用 <code>pre_mix</code> 降低 mHC 开销',
      '算子包内已有 <code>npu_hc_pre_v2</code>，但 GLM-5.3 不调用它',
      '算子已随包发布，调用侧未接'],
     ['<strong>W4A4 激活档</strong>：路由专家激活也降到 4 bit',
      '转换脚本注明 W4A8 与 W4A4 的落盘权重<strong>完全相同</strong>，切换只改 config 的 <code>input_activations.num_bits</code>，无需重新转换',
      '落盘格式已兼容，纯配置开关']]) + """
<h3>10.2 趋势外推（依据较弱，仅供对照）</h3>
<ul>
<li><strong>线性注意力占比继续上升。</strong>GLM-5 / 5.2 是 0/78，GLM-5.3-Flash 是 34/45（约 76%）。
若长上下文仍是主要目标，提高 KDA 比例是最直接的缓存压降手段。但这同时意味着更依赖递推状态的数值稳定性，
而目前 KDA 的全部权重都不量化——这两件事会互相拉扯。</li>
<li><strong>mHC 的宽度泛化。</strong>当前融合算子把 <code>hc_mult=4</code>、<code>d=4096</code>、
<code>hc_mix=24</code> 与 Sinkhorn 迭代 20 都写死，换宽度或换流数就需要新 kernel。
若 mHC 要用到其他尺寸的模型上，算子必须先泛化。</li>
<li><strong>prefill 侧 indexer 的融合实现。</strong>decode 已有 <code>npu_lightning_indexer</code>，
prefill 仍是 torch matmul 加 topk 并按 1024 分块。长输入场景若成为瓶颈，这是最明显的缺口。</li>
</ul>
<h3>10.3 一处可能是遗漏而非预留</h3>
<p><code>enable_custom_ops</code> 被传入路由专家的量化配置，但仓库内没有任何位置读取它，
导致路由专家 GMM 的 <code>swiglu_limit</code> clamp 实际不参与（共享专家的 MXFP8 路径则显式传了）。
这更像接线未完成而不是有意预留，但未经运行确认。</p>
<p class="small">以上全部为研究推断。没有任何一条来自厂商或 CANN 团队的公开路线图声明。</p>"""


SECTIONS = [
    dict(id='s0', title='GLM-5 → GLM-5.3-Flash：结构演进、混合注意力与昇腾部署研究', html=S0),
    dict(id='s1', title='1. 研究口径与证据强度', html=S1),
    dict(id='s2', title='2. 主线演进：改变在哪里', html=S2),
    dict(id='s3', title='3. 总体规格并列', html=S3),
    dict(id='s4', title='4. 混合注意力：KDA 与 DSA 各自承担什么', html=S4),
    dict(id='s5', title='5. mHC：把残差通路从一条变成四条', html=S5),
    dict(id='s6', title='6. 混合精度与昇腾侧优化', html=S6),
    dict(id='s7', title='7. 能确定什么，哪些还需要实测', html=S7),
    dict(id='s8', title='8. 来源与复核方式', html=S8),
    dict(id='s9', title='9. Profiling 与实测数据：分清是哪条路径', html=S9),
    dict(id='s10', title='10. 未来结构预判', html=S10),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(SECTIONS, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'GLM report: {len(SECTIONS)} sections.')


if __name__ == '__main__':
    main()
