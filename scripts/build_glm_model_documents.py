"""Add the GLM entries to the independent model-document download centre.

Rows are a per-matrix operator inventory derived from the GLM-5.3-Flash structure
dataset plus the kernel each matrix is consumed by in the pinned recipe. They are NOT
a profiled call trace: the four profiling columns (duration / 计算利用率 / mte / scalar)
are written as 未实测 on every row, because the multi-card HiF8 recipe ships no
profiling data. GLM-5 and GLM-5.2 have no per-matrix evidence and are recorded as
documented gaps rather than padded out.
"""
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'dist/downloads/model-documents'
ARCH = ROOT / 'data/families/glm/glm-5.3-flash-architecture.json'
COMMIT = '3b36eeac4cde5e608f00cdd0560a715ff8925e15'
BLOB = f'https://gitcode.com/cann/cann-recipes-infer/blob/{COMMIT}/'
HF = 'https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json'
NM = '未实测'
NV = '未核验'

# ---------------------------------------------------------------- XLSX writer
def _sheet(rows):
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
           '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
           '<sheetData>']
    for ri, row in enumerate(rows, 1):
        cells = []
        for ci, val in enumerate(row):
            col = ''
            n = ci
            while True:
                col = chr(ord('A') + n % 26) + col
                n = n // 26 - 1
                if n < 0:
                    break
            cells.append(f'<c r="{col}{ri}" t="inlineStr"><is><t xml:space="preserve">'
                         f'{escape(str(val))}</t></is></c>')
        out.append(f'<row r="{ri}">' + ''.join(cells) + '</row>')
    out += ['</sheetData>', '</worksheet>']
    return ''.join(out)


def write_xlsx(path, sheets):
    """Minimal OOXML: one inline-string worksheet per (name, rows) pair."""
    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
          'officedocument.spreadsheetml.sheet.main+xml"/>',
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-'
          'officedocument.spreadsheetml.styles+xml"/>']
    for i in range(len(sheets)):
        ct.append(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" ContentType='
                  f'"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    ct.append('</Types>')

    wb_sheets = ''.join(
        f'<sheet name="{escape(n)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
        for i, (n, _) in enumerate(sheets))
    wb = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
          f'<sheets>{wb_sheets}</sheets></workbook>')
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
              '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
              '<borders count="1"><border/></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
              '</styleSheet>')
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/'
                 'officeDocument/2006/relationships/styles" Target="styles.xml"/>'
               + ''.join(
                   f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/'
                   f'officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>'
                   for i in range(len(sheets))) + '</Relationships>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
                 '2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ''.join(ct))
        z.writestr('_rels/.rels', root_rels)
        z.writestr('xl/workbook.xml', wb)
        z.writestr('xl/_rels/workbook.xml.rels', wb_rels)
        z.writestr('xl/styles.xml', styles)
        for i, (_, rows) in enumerate(sheets):
            z.writestr(f'xl/worksheets/sheet{i + 1}.xml', _sheet(rows))


# ------------------------------------------------------------ kernel mapping
def kernel_for(module_id, template):
    """Which NPU kernel consumes this weight, per the pinned recipe."""
    t = template
    if 'conv1d' in t:
        return ('prefill: torch.ops.cann_ops_transformer.causal_conv1d_fn(activation="silu")\n'
                'decode: causal_conv1d_update(activation="silu")\n融合：深度可分离因果卷积 + SiLU + 卷积状态缓存读写')
    if t.endswith('A_log') or t.endswith('dt_bias'):
        return ('prefill: flash_kda；decode: fused_recurrent_kda\n作为递推参数传入，不单独成 matmul\n'
                '融合：内核自带 l2norm、门控激活与 beta sigmoid')
    if 'o_norm' in t:
        return 'Glm53RMSNormGated（门控 RMSNorm，fp32 权重）'
    if module_id == 'attention' and 'self_attn' in t and 'q_a_proj' not in t and 'kv_' not in t and 'q_b_proj' not in t:
        return 'BF16 线性层（KDA 全部权重不量化）→ 普通 matmul'
    if any(k in t for k in ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa')) or (
            module_id == 'attention' and t.endswith('o_proj.weight')):
        return ('torch_npu.npu_quant_matmul(x1_dtype=hifloat8, x2_dtype=hifloat8)\n'
                '激活经 npu_dtype_cast 直接转 HiF8，不做缩放')
    if 'kv_b_proj' in t:
        return ('不参与 matmul：加载后由 prepare_absorbed_weights 拆成 w_k [64,256,512] 与 w_v [64,512,256]，'
                '原权重置空\n因此必须保留 BF16')
    if module_id == 'indexer':
        if 'kpool_compress' in t:
            return '_commit_pools：softmax(门控 + 位置嵌入) 在 kpool=4 组内加权合并'
        return ('decode: torch_npu.npu_lightning_indexer(layout_key="PA_BSND", sparse_mode=3)\n'
                'prefill: 无对应算子，torch matmul + topk，按 QUERY_BLOCK=1024 分块')
    if module_id == 'router':
        return 'torch_npu.npu_moe_gating_top_k(norm_type=1 sigmoid, group_select_mode=1)'
    if module_id == 'routed-experts':
        return ('torch_npu.npu_grouped_matmul(weight_dtype=float4_e2m1fn_x2, '
                'per_token_scale_dtype=float8_e8m0fnu)\n两段 GMM 之间 npu_swiglu_mx_quant 融合 SwiGLU 与再量化')
    if module_id == 'shared-expert':
        return ('torch_npu.npu_quant_matmul(scale_dtype=float8_e8m0fnu, group_sizes=[1,1,32])\n'
                'torch.ops.custom.npu_swiglu_group_quant 融合 clamp + SwiGLU + MX 量化')
    if module_id == 'ffn':
        return 'torch_npu.npu_quant_matmul(HiF8) + torch_npu.npu_swiglu（clamp 在外单独 in-place 执行）'
    if module_id.startswith('mhc'):
        side = '注意力侧' if module_id == 'mhc' else 'FFN 侧'
        return (f'torch.ops.custom.npu_hc_pre / npu_hc_post（{side}）\n'
                '融合：展平 → RMS 归一 → 与 hc_fn 矩阵乘 → 拆 pre/post/comb → Sinkhorn 20 轮 → 折叠/展开\n'
                'T/bs ≤ 128 且能被 16 整除时走融合 kernel，否则回退两个拆分 kernel')
    if module_id == 'norms' or t.endswith('layernorm.weight') or t.endswith('norm.weight'):
        return 'torch_npu.npu_rms_norm（不使用 npu_add_rms_norm：残差通路由 hc_pre/hc_post 持有）'
    if module_id == 'embed':
        return 'VocabParallelEmbedding'
    if module_id == 'lm_head':
        return 'ColumnParallelLinear（不量化）'
    return NV


def optype(node, module_id):
    if module_id in ('mhc', 'mhc-ffn'):
        return 'mhc'
    if module_id == 'indexer':
        return 'indexer'
    if module_id in ('router', 'routed-experts', 'shared-expert'):
        return 'moe'
    if module_id == 'norms':
        return 'norm'
    if module_id == 'ffn':
        return 'dense_mlp'
    if module_id == 'attention':
        return 'kda' if node['type'] == 'KDA' else 'dsa'
    return module_id


HEADERS_EXTRA = ['证据状态', '配置来源', '权重证据', 'PyTorch参考或调用', 'NVIDIA条件路径',
                 'AMD条件路径', '实现来源', '运行条件与缺口', '数学关系或推导']


def build_rows(template_headers):
    arch = json.loads(ARCH.read_text())
    rows = []
    for node in arch['nodes']:
        if node['group'] == 'vision':
            continue  # config level only: no matrices to enumerate
        for mod in node['modules']:
            for m in mod['matrices']:
                t = m['tensor_template']
                if node['group'] == 'decoder':
                    mid = f"decoder/layer_{node['number'] - 1}/{mod['id']}"
                else:
                    mid = f"global/{node['id']}"
                shape = '[' + ', '.join(str(x) for x in m['logical_shape']) + ']'
                cnt = m['count']
                rows.append([
                    mid,                                        # 模块
                    t + (f'  × {cnt}' if cnt > 1 else ''),      # 算子列表
                    kernel_for(mod['id'], t),                   # npu kernel
                    optype(node, mod['id']),                    # 算子类型
                    shape,                                      # inputshape
                    NV,                                         # output shape
                    'B,S,T；专家维 E=288，EP=8；其他轴见模型参数',  # dim
                    'B[1,128]；S_verify=1（不支持 MTP）；L_KV 独立；TP=1，EP=world_size',  # shape范围
                    m['stored_dtype'],                          # input dtype
                    NV,                                         # output dtype
                    '逻辑轴顺序；物理 format 未核验（enable_weight_nz=False）',  # input format
                    NV,                                         # output format
                    NV, NV, NV,                                 # dtype next1/2/3
                    'attn/dense DP，tp_size=1；MoE 按 Expert 维 EP=8',  # 并行资源占用情况
                    NM, NM, NM, NM,                             # duration/利用率/mte/scalar
                    kernel_for(mod['id'], t).split('融合：')[-1]
                    if '融合：' in kernel_for(mod['id'], t) else '未在本配方中标注融合',  # 融合算子分析
                    '配置与源码推导；无权重审计、无 profiling',   # 证据状态
                    HF,                                         # 配置来源
                    '未审计权重文件头；发布权重不量化集合见 config 的 modules_to_not_convert',  # 权重证据
                    BLOB + 'models/glm_5_3/models/modeling_glm53.py',  # PyTorch参考或调用
                    '未核验（本配方仅昇腾）',                      # NVIDIA条件路径
                    '未核验（本配方仅昇腾）',                      # AMD条件路径
                    BLOB + 'models/glm_5_3/',                    # 实现来源
                    'duration/利用率/mte/scalar 均未实测：多卡 HiF8 配方未提交任何 profiling 数据',  # 运行条件与缺口
                    f'逻辑参数 = {m["logical_parameters_each"]:,}（每份）× {cnt} 份',  # 数学关系或推导
                ])
    assert all(len(r) == len(template_headers) + len(HEADERS_EXTRA) for r in rows)
    return rows


GAP_ROW_NOTE = ('仅有配方配置类默认值，无逐矩阵结构，也未与厂商 config.json 核对；'
                '不能由 GLM-5.3-Flash 的结构推定本版本')


def gap_rows(name, cfg_url, n_headers):
    r = [f'{name}/资料缺口', '未展开', NV, 'gap', NV, NV, NV, NV, NV, NV, NV, NV, NV, NV, NV,
         NV, NM, NM, NM, NM, '未展开',
         '配置级记录，未逐层分析', cfg_url, '未审计', cfg_url,
         '未核验（本配方仅昇腾）', '未核验（本配方仅昇腾）',
         BLOB + 'models/', GAP_ROW_NOTE, '不做推导']
    assert len(r) == n_headers
    return [r]


PARAM_HEADERS = ['符号', '含义', '本模型值 / 场景输入', '来源', '条件与缺口']

PARAMS_53 = [
    ('H', 'config.hidden_size', '4096', HF, '官方配置'),
    ('L', 'config.num_hidden_layers', '45', HF, '另有 1 个 MTP 层，本配方不构建'),
    ('N', 'config.num_attention_heads', '64', HF, 'DSA 层'),
    ('Dq', 'qk_nope_head_dim', '256', HF, 'qk_rope_head_dim=0，全程 NoPE'),
    ('Dv', 'v_head_dim', '256', HF, '官方配置'),
    ('C', 'kv_lora_rank', '512', HF, 'DSA 每 token 缓存宽度'),
    ('Rq', 'q_lora_rank', '1536', HF, '官方配置'),
    ('Hl', 'linear_attn_config.num_heads', '64', HF, 'KDA'),
    ('Dl', 'linear_attn_config.head_dim', '128', HF, 'KDA 递推状态 [64,128,128]'),
    ('Kc', 'short_conv_kernel_size', '4', HF, 'KDA 卷积状态 [3, 8192]'),
    ('E', 'n_routed_experts', '288', HF, 'EP=8 时每卡 36 个'),
    ('K', 'num_experts_per_tok', '8', HF, '官方配置'),
    ('Es', 'n_shared_experts', '1', HF, '官方配置'),
    ('I', 'intermediate_size', '12288', HF, '前 3 层 dense MLP'),
    ('Ie', 'moe_intermediate_size', '2048', HF, '专家中间宽度'),
    ('V', 'vocab_size', '154880', HF, '官方配置'),
    ('hc', 'hc_mult', '4', HF, 'mHC 并行流数；hc_mix=(2+4)*4=24'),
    ('Tk', 'index_topk', '2048', HF, 'select_k = 2048/4 = 512'),
    ('Kp', 'index_kpool', '4', HF, 'pool 缓存按 4 压缩'),
    ('B', 'request batch', '1', BLOB + 'models/glm_5_3/config/glm_5_3_flash_ep8.yaml',
     '场景输入，README 覆盖 1/16/64/128；不代表设备上限'),
    ('S', '每 request 的 verify 长度', '1', BLOB + 'models/glm_5_3/models/modeling_glm53.py',
     '本配方拒绝 next_n>0，不支持 MTP，故固定 1'),
    ('TP', 'tensor parallel', '1', BLOB + 'models/glm_5_3/models/modeling_glm53.py',
     'cp/attn_tp/dense_tp/moe_tp/o_proj_tp/shared_tp 必须为 1'),
    ('EP', 'moe expert parallel', '8', BLOB + 'executor/core/config/inference_config.py',
     '由 world_size / moe_tp_size 派生'),
    ('duration', '算子耗时', NM, '—', '多卡 HiF8 配方未提交任何 profiling 数据'),
    ('计算利用率 / mte / scalar', 'profiling 指标', NM, '—',
     '同上；另一条单卡 CPU-MoE offload 路径有实测，但量化与框架不同，不可挪用'),
]

PARAMS_GAP = [
    ('H', 'config.hidden_size', '6144', None, '配方配置类默认值，未与厂商 config.json 核对'),
    ('L', 'config.num_hidden_layers', '78', None, '同上'),
    ('E', 'n_routed_experts', '256', None, '同上'),
    ('K', 'num_experts_per_tok', '8', None, '同上'),
    ('C', 'kv_lora_rank', '512', None, '同上'),
    ('Rq', 'q_lora_rank', '2048', None, '同上'),
    ('duration', '算子耗时', NM, '—', '未逐层分析'),
]


def write_csv(path, headers, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_ALL)
        w.writerow(headers)
        w.writerows(rows)


def main():
    manifest = json.loads((D / 'manifest.json').read_text())
    th = manifest['template']['headers'][:21]
    headers = th + HEADERS_EXTRA
    n = len(headers)

    rows53 = build_rows(th)
    entries = []

    specs = [
        ('glm-glm-5.3-flash', 'GLM-5.3-Flash', rows53,
         [(s, m, v, u, c) for s, m, v, u, c in PARAMS_53],
         dict(hf='https://huggingface.co/zai-org/GLM-5.3-Flash', config_status='配置核验',
              gap='视觉塔与 MTP 层仅配置级；无权重审计；无 profiling 数据',
              layers=45, status='逐矩阵展开；profiling 列全部未实测',
              config_source=HF)),
        ('glm-glm-5', 'GLM-5', gap_rows('GLM-5', BLOB + 'models/glm_5/models/configuration_glm.py', n),
         PARAMS_GAP,
         dict(hf='https://huggingface.co/zai-org/GLM-5', config_status='配置缺失',
              gap=GAP_ROW_NOTE, layers=0, status='资料缺口；未逐层分析',
              config_source=BLOB + 'models/glm_5/models/configuration_glm.py')),
        ('glm-glm-5.2', 'GLM-5.2', gap_rows('GLM-5.2', BLOB + 'models/glm_5_2/models/configuration_glm.py', n),
         PARAMS_GAP,
         dict(hf='https://huggingface.co/zai-org/GLM-5.2', config_status='配置缺失',
              gap=GAP_ROW_NOTE + '；仓库内也没有任何性能或精度数字', layers=0,
              status='资料缺口；未逐层分析',
              config_source=BLOB + 'models/glm_5_2/models/configuration_glm.py')),
    ]

    for mid, name, rows, params, meta in specs:
        prows = [[s, mean, v, u or meta['config_source'], c] for s, mean, v, u, c in params]
        (D / f'{mid}.json').write_text(json.dumps(dict(
            model=name, headers=headers, rows=rows,
            params={'H': 4096 if mid.endswith('5.3-flash') else 6144},
            config=meta['config_source'], configSource=meta['config_source'],
            parameterSources=[HF, BLOB + 'models/glm_5_3/'],
        ), ensure_ascii=False), encoding='utf-8')
        write_csv(D / f'{mid}.csv', headers, rows)
        write_csv(D / f'{mid}-parameters.csv', PARAM_HEADERS, prows)
        write_xlsx(D / f'{mid}.xlsx', [('算子表', [headers] + rows),
                                       ('模型参数', [PARAM_HEADERS] + prows)])
        manifest['models'].append(dict(
            id=mid, name=name, family='GLM', hf=meta['hf'], repo=None, revision=COMMIT,
            source=BLOB + 'models/', config_status=meta['config_status'], gap=meta['gap'],
            rows=len(rows), layers=meta['layers'], status=meta['status'],
            config_source=meta['config_source']))
        entries += [f'{mid}.xlsx', f'{mid}.csv', f'{mid}-parameters.csv']

    manifest['counts']['documents'] = len(manifest['models'])
    manifest['counts']['withLayerExpansion'] += 1
    manifest['counts']['withoutLayerExpansion'] += 2
    manifest['counts']['operatorRows'] += sum(len(r) for _, _, r, _, _ in specs)
    manifest['date'] = '2026-10-19'
    (D / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                     encoding='utf-8')

    checks = json.loads((D / 'file-checksums.json').read_text())
    known = {c['file'] for c in checks}
    for fn in entries:
        b = (D / fn).read_bytes()
        rec = dict(file=fn, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
        if fn in known:
            checks[[c['file'] for c in checks].index(fn)] = rec
        else:
            checks.append(rec)
    (D / 'file-checksums.json').write_text(json.dumps(checks, ensure_ascii=False, indent=1),
                                           encoding='utf-8')

    # The bundle must contain every model's workbook.
    with zipfile.ZipFile(D / 'model-documents-all.zip', 'a', zipfile.ZIP_DEFLATED) as z:
        have = set(z.namelist())
        for mid, *_ in specs:
            if f'{mid}.xlsx' not in have:
                z.write(D / f'{mid}.xlsx', f'{mid}.xlsx')

    print(f'GLM model documents: {len(specs)} entries '
          f'({len(rows53)} operator rows for GLM-5.3-Flash, 2 gap entries); '
          f'manifest now {manifest["counts"]["documents"]} documents.')


if __name__ == '__main__':
    main()
