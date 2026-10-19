"""Rebuild the GLM-5.3-Flash structure interface from the pinned CANN recipe shapes.

Every matrix below is transcribed from the implementation in cann-recipes-infer at commit
96e5813f3a62686c9f1f9abba7d07dcd94687971 (models/glm_5_3/). Shapes are the load-time
`[out, in]` form of each parameter; logical parameter counts are multiplied out from those
shapes. This is a code-derived structure, NOT a weight-header audit: `payloadBytes` stays
null on every node and no stored byte layout is claimed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/families/glm/glm-5.3-flash-architecture.json'

C = dict(
    hidden=4096, inter=12288, moe_inter=2048, heads=64, kv_lora=512, q_lora=1536,
    qk_nope=256, qk_rope=0, v=256, experts=288, shared=1, topk=8,
    idx_heads=32, idx_dim=128, idx_topk=2048, kpool=4,
    lin_heads=64, lin_dim=128, conv=4, hc=4, vocab=154880, layers=45,
    first_k_dense=3,
    vision_depth=24, vision_hidden=1024, vision_heads=16, vision_inter=4096,
)
H, I, MI, E = C['hidden'], C['inter'], C['moe_inter'], C['experts']
QKV = C['lin_heads'] * C['lin_dim']          # 8192
MIX = (2 + C['hc']) * C['hc']                # 24
HCD = C['hc'] * H                            # 16384


# Deployment storage formats produced by models/glm_5_3/utils/convert_model.py.
# These describe the converted deployment weights, not the released FP8 checkpoint.
HIF8 = 'uint8 · HiF8 W8A8（逐输出通道 float32 scale）'
MXFP4 = 'uint8 · MXFP4 W4A8（K 向 32 元素块，E8M0 uint8 scale，2 值/字节）'
MXFP8 = 'float8_e4m3fn · MXFP8 W8A8（K 向 32 元素块，E8M0 uint8 scale）'
BF16 = 'bfloat16（转换时跳过，不量化）'
FP32 = 'float32'


def mat(template, logical, dtype, count=1):
    """One matrix template. `logical` is the load-time [out, in] logical shape."""
    n = 1
    for d in logical:
        n *= d
    return dict(
        tensor_template=template,
        stored_dtype=dtype,
        stored_shape=list(logical),
        logical_shape=list(logical),
        logical_parameters_each=n,
        count=count,
    )


def module(mid, title, matrices, *, count=1, representative=False, selected=None):
    params = sum(m['logical_parameters_each'] * m['count'] for m in matrices)
    return dict(
        id=mid, title=title, representative=representative, count=count,
        selectedCount=count if selected is None else selected,
        parameters=params * count, matrices=matrices,
    )


def hc_module(kind):
    """mHC per-sublayer mixer parameters (npu_hc_pre / npu_hc_post operands)."""
    return [
        mat(f'model.layers.{{i}}.hc_{kind}_fn', [MIX, HCD], FP32),
        mat(f'model.layers.{{i}}.hc_{kind}_base', [MIX], FP32),
        mat(f'model.layers.{{i}}.hc_{kind}_scale', [3], FP32),
    ]


def kda_attention():
    p = 'model.layers.{i}.self_attn.'
    return [
        mat(p + 'q_proj.weight', [QKV, H], BF16),
        mat(p + 'k_proj.weight', [QKV, H], BF16),
        mat(p + 'v_proj.weight', [QKV, H], BF16),
        mat(p + 'q_conv1d', [QKV, C['conv']], BF16),
        mat(p + 'k_conv1d', [QKV, C['conv']], BF16),
        mat(p + 'v_conv1d', [QKV, C['conv']], BF16),
        mat(p + 'f_a_proj.weight', [C['lin_dim'], H], BF16),
        mat(p + 'f_b_proj.weight', [QKV, C['lin_dim']], BF16),
        mat(p + 'g_a_proj.weight', [C['lin_dim'], H], BF16),
        mat(p + 'g_b_proj.weight', [QKV, C['lin_dim']], BF16),
        mat(p + 'b_proj.weight', [C['lin_heads'], H], BF16),
        mat(p + 'A_log', [C['lin_heads']], FP32),
        mat(p + 'dt_bias', [QKV], FP32),
        mat(p + 'o_norm.weight', [C['lin_dim']], FP32),
        mat(p + 'o_proj.weight', [H, QKV], BF16),
    ]


def dsa_attention():
    p = 'model.layers.{i}.self_attn.'
    return [
        mat(p + 'q_a_proj.weight', [C['q_lora'], H], HIF8),
        mat(p + 'q_a_layernorm.weight', [C['q_lora']], BF16),
        mat(p + 'q_b_proj.weight', [C['heads'] * C['qk_nope'], C['q_lora']], HIF8),
        mat(p + 'kv_a_proj_with_mqa.weight', [C['kv_lora'], H], HIF8),
        mat(p + 'kv_a_layernorm.weight', [C['kv_lora']], BF16),
        mat(p + 'kv_b_proj.weight', [C['heads'] * (C['qk_nope'] + C['v']), C['kv_lora']], 'bfloat16（吸收式 MLA 必须保留，转换时显式排除）'),
        mat(p + 'o_proj.weight', [H, C['heads'] * C['v']], HIF8),
    ]


def indexer():
    p = 'model.layers.{i}.self_attn.indexer.'
    return [
        mat(p + 'wq_b.weight', [C['idx_heads'] * C['idx_dim'], C['q_lora']], BF16),
        mat(p + 'wk.weight', [C['idx_dim'], H], BF16),
        mat(p + 'k_norm.weight', [C['idx_dim']], BF16),
        mat(p + 'k_norm.bias', [C['idx_dim']], BF16),
        mat(p + 'weights_proj.weight', [C['idx_heads'], H], BF16),
        mat(p + 'index_kpool_compress_ape', [C['kpool'], C['idx_dim']], BF16),
        mat(p + 'index_kpool_compress_gate', [C['idx_dim'], H], BF16),
    ]


def dense_ffn():
    p = 'model.layers.{i}.mlp.'
    return [
        mat(p + 'gate_proj.weight', [I, H], HIF8),
        mat(p + 'up_proj.weight', [I, H], HIF8),
        mat(p + 'down_proj.weight', [H, I], HIF8),
    ]


def norms():
    return [
        mat('model.layers.{i}.input_layernorm.weight', [H], BF16),
        mat('model.layers.{i}.post_attention_layernorm.weight', [H], BF16),
    ]


def node(idx):
    """idx is 0-based; node number is 1-based."""
    is_dsa = idx % 4 == 3
    is_moe = idx >= C['first_k_dense']
    mods = []

    if is_dsa:
        mods.append(module('attention', '注意力 DSA：吸收式 MLA（全程 NoPE）', dsa_attention()))
        mods.append(module('indexer', 'Lightning Indexer（k-pool 压缩，top-k 稀疏选择）', indexer()))
    else:
        mods.append(module('attention', '注意力 KDA：短卷积 + 门控 delta 递推（线性注意力）', kda_attention()))

    if is_moe:
        mods.append(module(
            'router', 'MoE 路由 gate（fp32，sigmoid + noaux_tc）',
            [mat('model.layers.{i}.mlp.gate.weight', [E, H], 'float32（运行时 fp32；转换时不量化）'),
             mat('model.layers.{i}.mlp.gate.e_score_correction_bias', [E], FP32)]))
        mods.append(module(
            'routed-experts', f'路由专家（{E} 选 {C["topk"]}，代表模板 × {E}）',
            [mat('model.layers.{i}.mlp.experts.{e}.gate_proj.weight', [MI, H], MXFP4),
             mat('model.layers.{i}.mlp.experts.{e}.up_proj.weight', [MI, H], MXFP4),
             mat('model.layers.{i}.mlp.experts.{e}.down_proj.weight', [H, MI], MXFP4)],
            count=E, representative=True, selected=C['topk']))
        mods.append(module(
            'shared-expert', '共享专家（1 个，decode 可走侧流与路由专家重叠）',
            [mat('model.layers.{i}.mlp.shared_experts.gate_proj.weight',
                 [MI * C['shared'], H], MXFP8),
             mat('model.layers.{i}.mlp.shared_experts.up_proj.weight',
                 [MI * C['shared'], H], MXFP8),
             mat('model.layers.{i}.mlp.shared_experts.down_proj.weight',
                 [H, MI * C['shared']], MXFP8)]))
    else:
        mods.append(module('ffn', 'Dense MLP（clamped SwiGLU，limit=10.0）', dense_ffn()))

    mods.append(module('mhc', 'mHC 超连接：注意力侧混合矩阵（Sinkhorn 归一化）', hc_module('attn')))
    mods.append(module('mhc-ffn', 'mHC 超连接：FFN 侧混合矩阵（Sinkhorn 归一化）', hc_module('ffn')))
    mods.append(module('norms', '层归一化 RMSNorm', norms()))

    total = sum(m['parameters'] for m in mods)

    # Active linear parameters: same accounting, but only top-k of the routed experts.
    active = 0
    for m in mods:
        if m['id'] == 'routed-experts':
            per = m['parameters'] // m['count']
            active += per * m['selectedCount']
        else:
            active += m['parameters']

    return dict(
        id=f'decoder-{idx + 1}', group='decoder', number=idx + 1,
        type='MLA' if is_dsa else 'KDA',
        ffn='MoE' if is_moe else 'Dense',
        parameters=total,
        activeLinearParameters=active,
        payloadBytes=None,
        modules=mods,
    )


def vision_node(idx):
    """Config-level only. The recipe drops the vision tower, so no matrix is claimed.

    Depth and width come from the released config.json `vision_config`; the exclusion
    list in its `quantization_config` confirms blocks 0..23 with attn.qkv / attn.proj /
    attn.q_norm / attn.k_norm / norm1 / norm2 / mlp.{gate,up,down}_proj. Per-matrix
    shapes are NOT recorded because no implementation in the pinned recipe builds them.
    """
    return dict(
        id=f'vision-{idx + 1}', group='vision', number=idx + 1, type='Vision',
        ffn='Dense', parameters=None, activeLinearParameters=None, payloadBytes=None,
        modules=[
            dict(id='attention', title='视觉注意力（qkv / proj，含 q_norm、k_norm）',
                 representative=False, count=1, selectedCount=1, parameters=None, matrices=[]),
            dict(id='ffn', title='视觉 MLP（gate / up / down，clamped SwiGLU）',
                 representative=False, count=1, selectedCount=1, parameters=None, matrices=[]),
            dict(id='norms', title='norm1 / norm2', representative=False, count=1,
                 selectedCount=1, parameters=None, matrices=[]),
        ],
    )


def global_nodes():
    """Global components: embedding, final norm, lm_head, and the MTP layer.

    The MTP layer (index 45) is present in the released checkpoint — its per-layer
    entries appear once each in the config's exclusion list (eh_proj, enorm, hnorm,
    shared_head.norm, plus a 46th input_layernorm / post_attention_layernorm and a
    43rd mlp.gate) — and it is a DSA + MoE layer. The pinned recipe does not build it
    and rejects next_n > 0, so no matrix is claimed here.
    """
    return [
        dict(id='embed', group='global', number=0, type='Embedding', ffn='—',
             parameters=C['vocab'] * H, activeLinearParameters=C['vocab'] * H,
             payloadBytes=None, modules=[module(
                 'embed', '词嵌入 model.embed_tokens',
                 [mat('model.embed_tokens.weight', [C['vocab'], H], BF16)])]),
        dict(id='final-norm', group='global', number=0, type='Norm', ffn='—',
             parameters=H, activeLinearParameters=H, payloadBytes=None,
             modules=[module('norm', '末层归一 model.norm',
                             [mat('model.norm.weight', [H], BF16)])]),
        dict(id='lm-head', group='global', number=0, type='Output', ffn='—',
             parameters=C['vocab'] * H, activeLinearParameters=C['vocab'] * H,
             payloadBytes=None, modules=[module(
                 'lm_head', '输出投影 lm_head',
                 [mat('lm_head.weight', [C['vocab'], H], BF16)])]),
        dict(id='mtp', group='global', number=0, type='MTP', ffn='MoE',
             parameters=None, activeLinearParameters=None, payloadBytes=None,
             modules=[dict(id='mtp', title='MTP 层（第 45 层，DSA + MoE；本配方不构建）',
                           representative=False, count=1, selectedCount=1,
                           parameters=None, matrices=[])]),
    ]


def build():
    nodes = [node(i) for i in range(C['layers'])]
    nodes += [vision_node(i) for i in range(C['vision_depth'])]
    nodes += global_nodes()
    doc = dict(
        schemaVersion=1,
        modelId='glm-5.3-flash',
        label='GLM-5.3-Flash',
        evidence='derived',
        scope=(
            '结构与逐矩阵形状来自 cann-recipes-infer 固定快照 3b36eea 的 models/glm_5_3 实现，'
            '按张量声明形状逐项相乘得到逻辑参数；层排布、各维度与专家数已与智谱发布的 config.json 逐项核对一致。'
            '不是权重文件头审计，因此 payloadBytes 全部为空，也不代表发布权重的实际字节布局。'
            '发布 config.json 的 layer_types 与 full_attn_layers=[3,7,11,15,19,23,27,31,35,39,43] 确认 '
            '11 层 DSA / 34 层 KDA，mlp_layer_types 确认前 3 层 Dense、其余 42 层 MoE，'
            'first_k_dense_replace=3 与实现默认值一致，没有覆盖。'
            '据此推导的 45 层解码主干加 embed/lm_head/末层归一为 313,326,811,966；'
            '配方 README 自述“约 306B”与该口径不符，且仓库未给出推导过程，本站保留差异不做消解。'
            '视觉塔为配置级：深度 24 与宽度 1024 取自 vision_config，逐矩阵形状未记录，因为本配方丢弃视觉通路。'
            'MTP 层在检查点中存在（配置排除列表中各项出现一次）且为 DSA + MoE 层，但配方不构建、运行时拒绝 next_n>0。'
            'stored_dtype 中的 HiF8 / MXFP8 / MXFP4 是本配方转换脚本产出的部署格式；'
            '发布权重为 FP8 e4m3 动态激活，其不量化张量集合与本配方的判定一致。'
        ),
        source='https://gitcode.com/cann/cann-recipes-infer/blob/3b36eeac4cde5e608f00cdd0560a715ff8925e15/models/glm_5_3/models/modeling_glm53.py',
        templatePath=None,
        nodes=nodes,
        config=dict(
            hidden=H, experts=E, topK=C['topk'], shared=C['shared'],
            context=1048576, precision='发布 FP8；本配方部署为 Hybrid HiF8-MXFP8-MXFP4',
            heads=C['heads'],
        ),
        cache=dict(
            mla=('DSA 层每 token 只缓存 kv_lora_rank=512 个元素的压缩潜在向量（bf16），'
                 '因为 mla_use_nope=true，没有 RoPE 分量进入缓存；'
                 'SFA 内核所需的 rope 操作数由常零张量合成，不占缓存。'),
            kda=('KDA 层保留短卷积状态 [conv_kernel-1, 8192] = [3, 8192]（q/k/v 各一份）'
                 '与每头递推矩阵状态 [64, 128, 128]；融合递推路径下递推状态为 float32。'),
            indexer=('Indexer 另有 key / gate / pool 三份缓存，每份宽 128。'
                     'pool 缓存按 index_kpool=4 压缩，即每 4 个 token 落一条 128 维记录。'),
        ),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')

    kda = sum(n['type'] == 'KDA' for n in nodes if n['group'] == 'decoder')
    mla = sum(n['type'] == 'MLA' for n in nodes if n['group'] == 'decoder')
    dense = sum(n['ffn'] == 'Dense' for n in nodes if n['group'] == 'decoder')
    dec = [n for n in nodes if n['group'] == 'decoder']
    vis = [n for n in nodes if n['group'] == 'vision']
    total = sum(n['parameters'] for n in dec) + C['vocab'] * H * 2 + H
    print(f'GLM-5.3 structure: {len(dec)} decoder layers ({kda} KDA / {mla} DSA), '
          f'{dense} dense + {len(dec) - dense} MoE; {len(vis)} vision layers (config level); '
          f'derived decoder+embed+lm_head+norm total = {total:,}')


if __name__ == '__main__':
    build()
