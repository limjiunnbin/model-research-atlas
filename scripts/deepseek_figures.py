"""Draw portable SVG research figures from the canonical configs and architecture data."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist/assets/deepseek/figures'
COLORS = {'MLA': '#335d9a', 'DSA': '#426eaf', 'SWA': '#718092', 'CSA': '#008a95', 'HCA': '#7556ad'}


class Figure:
    def __init__(self, title, subtitle, height=850):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 {height}" role="img" aria-labelledby="title desc">', f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(subtitle)}</desc>', '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>', f'<rect width="1600" height="{height}" fill="#f5f7fb"/>', '<g font-family="Arial, PingFang SC, Microsoft YaHei, sans-serif" fill="#24344a">']
        self.text(60, 66, [title], size=34, bold=True)
        self.text(60, 106, [subtitle], size=22, color='#54647b')

    def text(self, x, y, lines, size=23, bold=False, color='#24344a', anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">')
        for i, line in enumerate(lines):
            self.parts.append(f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.5}">{html.escape(str(line))}</tspan>')
        self.parts.append('</text>')

    def box(self, x, y, w, h, title, lines, color='#335d9a'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="#d0d9e6" stroke-width="2"/><rect x="{x}" y="{y}" width="7" height="{h}" rx="3" fill="{color}"/>')
        self.text(x + 24, y + 37, [title], size=25, bold=True, color=color)
        self.text(x + 24, y + 76, lines, size=22)

    def arrow(self, x1, y1, x2, y2, dashed=False):
        self.parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="#64748b" stroke-width="2.5" marker-end="url(#arrow)"' + (' stroke-dasharray="7 6"' if dashed else '') + '/>')

    def save(self, name):
        self.parts.extend(['</g>', '</svg>'])
        path = OUT / name
        path.write_text('\n'.join(self.parts) + '\n')
        return 'assets/deepseek/figures/' + name


def build_figures(family, architectures):
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {m['id']: json.loads((ROOT / m['configPath']).read_text()) for m in family['models']}
    figures = []
    f = Figure('DeepSeek：结构演进与 R1 底座关系', f'{len(family["models"])} 个代表 checkpoint；设计变化、后训练与蒸馏分开。虚线表示设计比较，不证明直接继承权重。', 840)
    for x, title, lines, color in [(60, 'V2', ['MLA + DeepSeekMoE', f"{configs['v2']['num_hidden_layers']} 个主干层"], 'MLA'), (445, 'V3 / V3-Base', ['3 Dense + 58 MoE', '主干 61 层；MTP 单列'], 'MLA'), (830, 'V3.2 / Exp', ['DSA + Lightning Indexer', '61 层；配置/代码一致'], 'DSA'), (1215, 'V4 Flash / Pro', ['窗口 + CSA/HCA + mHC', '主干 43 / 61 层'], 'CSA')]:
        f.box(x, 150, 325, 155, title, lines, COLORS[color])
    for x in (385, 770, 1155):
        f.arrow(x, 220, x + 53, 220, dashed=True)
    f.text(60, 365, ['R1 / R1-Zero：官方基于 V3-Base；相同配置与参考代码不等于相同权重或能力。'], bold=True)
    f.box(60, 398, 410, 155, 'V3-Base', ['61 层 MLA / MoE', '固定 checkpoint 配置'], COLORS['MLA'])
    f.box(565, 398, 435, 155, 'R1-Zero', ['直接进行大规模 RL', '未以 SFT 作前置'], COLORS['DSA'])
    f.box(1095, 398, 445, 155, 'R1', ['冷启动；两次 SFT / 两阶段 RL', '推理采样条件单列'], COLORS['DSA'])
    f.arrow(470, 450, 552, 450)
    f.parts.append('<path d="M 265 553 V 584 H 1040 V 465 H 1082" fill="none" stroke="#64748b" stroke-width="2.5" marker-end="url(#arrow)"/>')
    f.box(60, 618, 720, 145, 'R1-Distill-Qwen：1.5B / 7B / 14B / 32B', ['Qwen2 系 Dense GQA；配置与 tokenizer_config 比较', '1.5B 档位取消 embedding/head 共享，逻辑参数约 1.777B'], COLORS['HCA'])
    f.box(830, 618, 710, 145, 'R1-Distill-Llama：8B / 70B', ['Llama 系 Dense GQA；蒸馏 checkpoint 配置已读取', '原始底座 gated，配置/tokenizer 差异保留未知'], COLORS['HCA'])
    f.text(1260, 590, ['R1 生成数据 → 蒸馏'], size=20, color='#54647b', anchor='middle')
    f.text(60, 807, ['依据：固定官方目录、checkpoint config、R1 README 与逐版本底座关系；无设备性能结论。'], size=20, color='#54647b')
    figures.append({'id': 0, 'path': f.save('lineage.svg'), 'caption': '结构演进、R1 后训练与蒸馏底座', 'source': f'固定版本关系、官方 R1 README 与 {len(configs)} 份配置；虚线只表示设计比较。'})

    c = configs['v3']
    h, nq, rq, rkv = c['hidden_size'], c['num_attention_heads'], c['q_lora_rank'], c['kv_lora_rank']
    dr, dn, dv = c['qk_rope_head_dim'], c['qk_nope_head_dim'], c['v_head_dim']
    f = Figure('V3 MLA：投影、两条数学路径与 cache', 'MP=1 参考口径；B 为 padded batch，L_q 为新增长度，L_kv 为可见历史长度。', 840)
    f.box(60, 155, 340, 178, '输入 hidden', [f'x [B, L_q, {h}]', 'RMSNorm 与残差另计'], COLORS['MLA'])
    f.box(495, 155, 480, 178, 'Q 低秩投影与位置分量', [f'{h} → {rq} → {nq} × {dn + dr}', f'NoPE={dn}，RoPE={dr}'], COLORS['MLA'])
    f.box(1060, 155, 480, 178, 'KV 低秩与位置投影', [f'{h} → {rkv}+{dr}，低秩部分 RMSNorm', f'展开：{rkv} → {nq} × ({dn}+{dv})'], COLORS['MLA'])
    f.arrow(400, 235, 482, 235)
    f.parts.append('<path d="M 230 155 V 133 H 1300 V 155" fill="none" stroke="#64748b" stroke-width="2.5" marker-end="url(#arrow)"/>')
    f.box(60, 390, 720, 223, '展开式 / naive', [f'K cache [B, L_kv, {nq}, {dn + dr}]', f'V cache [B, L_kv, {nq}, {dv}]', f'每 token/层 {nq} × ({dn + dr}+{dv}) = {nq * (dn + dr + dv):,} 元素', '直接做 QKᵀ、softmax、加权 V'], COLORS['DSA'])
    f.box(830, 390, 710, 223, '吸收式 / absorb', [f'latent cache [B, L_kv, {rkv}]', f'RoPE cache [B, L_kv, {dr}]', f'每 token/层 {rkv}+{dr} = {rkv + dr} 元素', '吸收 K/V 投影：先潜变量打分，再还原输出'], COLORS['CSA'])
    f.box(60, 665, 1480, 107, '共同输出与边界', [f'输出投影 [{h}, {nq * dv}]；参考 score [B, {nq}, L_q, L_kv]；分页、TP 与 dtype 按框架另核验。'], '#54647b')
    f.text(60, 809, ['依据：V3 固定 config、MLA.forward 与首轮代数复算。cache 元素公式不代表设备显存或性能。'], size=20, color='#54647b')
    figures.append({'id': 1, 'path': f.save('v3-mla.svg'), 'caption': 'V3 MLA 两条参考数学路径与缓存口径', 'source': 'V3 config + 固定官方 MLA.forward；两条数学路径由首轮 FP64 合成矩阵复算。'})

    f = Figure('V4：主干层排布、压缩 cache 与四流残差', 'Flash / Pro 各自固定配置；末尾 ratio=0 的 MTP 条目从主干层排布中排除。', 900)
    for y, mid, title in [(165, 'v4-flash', 'Flash'), (315, 'v4-pro', 'Pro')]:
        cc = configs[mid];ratios = cc['compress_ratios'][:cc['num_hidden_layers']]
        counts = {r: ratios.count(r) for r in (0, 4, 128)}
        f.text(60, y, [f"{title}：{len(ratios)} 层 · SWA {counts[0]} / CSA {counts[4]} / HCA {counts[128]}"], bold=True)
        width = 1470 / len(ratios)
        for i, r in enumerate(ratios):
            color = COLORS[{0: 'SWA', 4: 'CSA', 128: 'HCA'}[r]]
            x = 60 + i * width
            f.parts.append(f'<rect x="{x:.2f}" y="{y+22}" width="{width-3:.2f}" height="50" rx="3" fill="{color}"/>')
            if i == 0 or (i + 1) % 5 == 0 or i == len(ratios) - 1:
                f.text(x + (width - 3) / 2, y + 97, [i + 1], size=16, anchor='middle')
    f.text(60, 480, ['SWA：窗口 KV', 'CSA：窗口 + ratio=4 overlap 压缩 + 压缩 indexer', 'HCA：窗口 + ratio=128 压缩前缀'], size=22)
    f.box(870, 461, 670, 152, 'cache 与压缩状态分别计数', ['C=floor(L_kv/r)；未满窗口不生成压缩 token', '主 KV、indexer key/scale、两个 FP32 状态分列'], COLORS['CSA'])
    f.box(60, 659, 330, 138, '四条残差流', ['[B, L_q, 4, H]', 'H 不合并为 4H'], COLORS['HCA'])
    f.box(450, 659, 330, 138, 'hc_pre → 子层', ['控制投影 [24, 4H]', '归一化 → attention / FFN'], COLORS['HCA'])
    f.box(840, 659, 330, 138, 'hc_post', ['pre/post/comb 分开', '返回 [B, L_q, 4, H]'], COLORS['HCA'])
    f.box(1230, 659, 310, 138, '输出前 hc_head', ['折叠四流 → H', 'MTP 单列'], COLORS['HCA'])
    for x in (390, 780, 1170):
        f.arrow(x, 720, x + 47, 720)
    f.text(60, 848, ['依据：Flash/Pro compress_ratios + Compressor、Block、hc_pre/hc_post 声明。示意不代表 kernel 顺序。'], size=20, color='#54647b')
    figures.append({'id': 2, 'path': f.save('v4-layers-mhc.svg'), 'caption': 'V4 实际主干层排布与 mHC 数据流', 'source': 'Flash/Pro 固定配置与参考模型；压缩率、MTP 边界、四流轴按 canonical 数据生成。'})
    d=configs['v4-flash-dspark'];H=d['hidden_size'];K=d['dspark_block_size'];R=d['dspark_markov_rank'];V=d['vocab_size']
    f=Figure('V4 DSpark：并行 backbone、顺序 Markov 与验证前缀', '此图按 Flash-DSpark 固定配置：3 stages / block=5 / rank=256；设备实验不在本次范围，箭头是协议示意。', 900)
    f.box(60,160,445,192,'目标特征与独立 context KV',[f'选层 {d["dspark_target_layer_ids"]}，每处折叠四流',f'concat 3H={3*H} → main_proj / norm',f'每 stage context cache [B,128,512]'],COLORS['MLA'])
    f.box(578,160,445,192,'一次并行 draft backbone',[f'已知 token + noise → [B,{K},4,{H}]','3 个 DSparkBlock；窗口 + block 非因果','mtp.0/1/2 是存储名，不是 1 层 MTP'],COLORS['CSA'])
    f.box(1095,160,445,192,'轻量顺序 Markov head',[f'W1 / W2 各 [{V},{R}]','前 token 查表 → logits bias → sample',f'按位置依次选 {K} 个候选'],COLORS['HCA'])
    f.arrow(505,250,565,250);f.arrow(1023,250,1082,250)
    f.box(60,410,720,185,'Confidence → prefix survival',[f'proj [1,{H+R}]，存储 BF16 / 参考计算 FP32','sigmoid 后 c_k = 条件接受概率估计','S_k = product(c_1…c_k)，E[length] = sum(S_k)'],COLORS['HCA'])
    f.box(830,410,710,185,'Target verify 与状态更新',['阈值 / engine profile 选择验证前缀','因果 target → accept/reject / correction/bonus','只提交接受/修正结果；未接受 draft 不当作历史'],COLORS['DSA'])
    f.arrow(780,500,817,500)
    f.box(60,650,1480,155,'三组数值分别解释',['HF nextn=1；inference + header 表明 3 个 DSpark stages。','checkpoint block=5；模型卡 vLLM example num_speculative_tokens=7，属于独立服务设置。','普通 MTP 使用独立 next-token predictor；DSpark 不套用 MTP 的 eh_proj/enorm/hnorm。'],'#54647b')
    f.text(60,857,['依据：固定 Flash-DSpark config/inference/header、DSpark paper v1、Ascend draft model/proposer；无设备收益推定。'],size=20,color='#54647b')
    figures.append({'id':3,'path':f.save('dspark-protocol.svg'),'caption':'DSpark 参数、context/draft 与验证协议','source':'按 Flash-DSpark 固定配置与公开参考绘制；服务配置、checkpoint 字段和实际 stage 不混用。'})
    (OUT / 'figure-inputs.json').write_text(json.dumps({'snapshot': family['updated'], 'models': {mid: {'revision': next(m['revision'] for m in family['models'] if m['id'] == mid), 'attentionTypeCounts': architectures[mid]['attentionTypeCounts'], 'compress_ratios': configs[mid].get('compress_ratios'), 'mainLogicalParameters': architectures[mid]['mainLogicalParameters']} for mid in ('v3', 'v4-flash', 'v4-pro','v4-flash-dspark')}, 'figures': figures}, ensure_ascii=False, indent=2) + '\n')
    return figures
