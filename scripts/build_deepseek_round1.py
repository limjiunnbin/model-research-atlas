"""Build P0/P1 research JSON and Markdown from fixed official inputs (no upstream execution)."""
import argparse
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
OUT = ROOT / 'research/deepseek/round1'
CONFIG = 'hf-v3-config-json'
MODEL = 'gh-deepseek-v3-inference-model-py'
HF_MODEL = 'hf-v3-modeling_deepseek-py'
WEIGHTS = 'gh-deepseek-v3-readme_weights-md'
INDEX = 'hf-v3-model-safetensors-index-json'
DATE = '2026-10-01'


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def evaluate(expression, env):
    """Evaluate dimension arithmetic only; never eval source code or unrestricted expressions."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.FloorDiv): return a // b
        raise ValueError(f'Unsupported dimension expression: {expression}')
    return visit(ast.parse(str(expression), mode='eval').body)


def build(cache):
    manifest = read(DATA / 'sources.json')
    sources = {s['id']: s for s in manifest['sources']}
    raw = {}
    for sid, source in sources.items():
        content = (cache / source['cacheFile']).read_bytes()
        if hashlib.sha256(content).hexdigest() != source['sha256']:
            raise ValueError(f'Source hash mismatch: {sid}')
        raw[sid] = content.decode('utf-8')
    config = read(DATA / 'configs/DeepSeek-V3-config.json')
    demo = read(DATA / 'configs/DeepSeek-V3-demo-config.json')
    env = dict(H=config['hidden_size'], Vocab=config['vocab_size'], N_head=config['num_attention_heads'],
               R_q=config['q_lora_rank'], R_kv=config['kv_lora_rank'], D_nope=config['qk_nope_head_dim'],
               D_rope=config['qk_rope_head_dim'], D_v=config['v_head_dim'], I_dense=config['intermediate_size'],
               I_expert=config['moe_intermediate_size'], E=config['n_routed_experts'], K=config['num_experts_per_tok'],
               N_shared=config['n_shared_experts'], G=config['n_group'], G_keep=config['topk_group'])
    meanings = {'H':'残差隐藏宽度', 'Vocab':'词表大小', 'N_head':'全局 attention heads；本轮 MP=1',
                'R_q':'Q 低秩维度', 'R_kv':'KV 低秩维度', 'D_nope':'非位置 Q/K 子维',
                'D_rope':'位置 Q/K 子维', 'D_v':'V 子维', 'I_dense':'Dense FFN 中间宽度',
                'I_expert':'单专家中间宽度', 'E':'路由专家总数', 'K':'每 token 选择的专家数',
                'N_shared':'共享专家数', 'G':'专家组数', 'G_keep':'保留专家组数',
                'B':'padded 参考路径的 batch', 'L_q':'本次每请求新增 token 数',
                'L_kv':'历史与本次可见 KV 长度', 'N_tok':'本轮 padded 路径 B*L_q',
                'N_e':'代表专家收到的 token 数；动态值，不假设均匀路由'}
    proofs = {}

    def proof(sid, symbol):
        key = sid + ':' + symbol
        if key not in proofs:
            tree = ast.parse(raw[sid])
            found = None
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and symbol == node.name:
                    found = node
                if isinstance(node, ast.ClassDef):
                    for fn in node.body:
                        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and symbol == node.name + '.' + fn.name:
                            found = fn
            if found is None:
                raise ValueError(f'Missing source symbol: {symbol}')
            text = '\n'.join(raw[sid].splitlines()[found.lineno - 1:found.end_lineno])
            proofs[key] = {'source_id': sid, 'symbol': symbol, 'line': found.lineno, 'end': found.end_lineno,
                           'bodySha256': hashlib.sha256(text.encode()).hexdigest(),
                           'url': sources[sid]['url'] + f'#L{found.lineno}-L{found.end_lineno}',
                           'level': 'reference_source'}
        return key

    # Checkpoint fields mapped to the demo's names; absent fields are not silently inherited.
    field_map = {
        'vocab_size':'vocab_size', 'hidden_size':'dim', 'intermediate_size':'inter_dim',
        'moe_intermediate_size':'moe_inter_dim', 'num_hidden_layers':'n_layers',
        'first_k_dense_replace':'n_dense_layers', 'num_attention_heads':'n_heads',
        'n_routed_experts':'n_routed_experts', 'n_shared_experts':'n_shared_experts',
        'num_experts_per_tok':'n_activated_experts', 'n_group':'n_expert_groups',
        'topk_group':'n_limited_groups', 'routed_scaling_factor':'route_scale',
        'scoring_func':'score_func', 'q_lora_rank':'q_lora_rank', 'kv_lora_rank':'kv_lora_rank',
        'qk_nope_head_dim':'qk_nope_head_dim', 'qk_rope_head_dim':'qk_rope_head_dim', 'v_head_dim':'v_head_dim',
    }
    config_rows = []
    for key, value in config.items():
        if key in ('auto_map', 'architectures', 'initializer_range', 'transformers_version', 'attention_dropout'):
            continue
        target = field_map.get(key)
        config_rows.append({'field': key, 'value': value, 'evidence':'official', 'source_id': CONFIG,
                            'demoField': target, 'demoValue': demo.get(target) if target else None,
                            'comparison':'equal' if target and value == demo.get(target) else 'not_mapped'})
        if target and value != demo[target]:
            raise ValueError(f'Checkpoint/demo field mismatch: {key}')
    layer_ids = {'Dense':[], 'MoE':[]}
    for i in range(config['num_hidden_layers']):
        is_moe = i >= config['first_k_dense_replace'] and i % config['moe_layer_freq'] == 0
        layer_ids['MoE' if is_moe else 'Dense'].append(i)

    matrices = []
    def matrix(suffix, dims, owner, symbol, count='1', index_scale=False, source=HF_MODEL):
        shape = [evaluate(d, env) for d in dims]
        copies = evaluate(count, env)
        matrices.append({'id':owner + ':' + suffix, 'tensorTemplate':'model.layers.{i}.' + suffix if owner not in ('global', 'mtp') else suffix,
                         'owner':owner, 'shapeExpressions':dims, 'logicalShape':shape,
                         'logicalParametersEach':math.prod(shape), 'countExpression':count, 'count':copies,
                         'logicalParameters':math.prod(shape)*copies, 'evidence':'derived',
                         'source_ids':[CONFIG, source, INDEX], 'proof_id':proof(source, symbol),
                         'declaredScaleInIndex':index_scale,
                         'expectedScaleShape':[(x+127)//128 for x in shape] if index_scale else None,
                         'storedShape':None, 'storedDtype':None, 'payloadBytes':None,
                         'storageEvidence':'unknown; index provides names/shards, not tensor header shape/dtype'})
    for suffix, dims, scale in [
        ('self_attn.q_a_proj.weight',['R_q','H'],True),
        ('self_attn.q_a_layernorm.weight',['R_q'],False),
        ('self_attn.q_b_proj.weight',['N_head*(D_nope+D_rope)','R_q'],True),
        ('self_attn.kv_a_proj_with_mqa.weight',['R_kv+D_rope','H'],True),
        ('self_attn.kv_a_layernorm.weight',['R_kv'],False),
        ('self_attn.kv_b_proj.weight',['N_head*(D_nope+D_v)','R_kv'],True),
        ('self_attn.o_proj.weight',['H','N_head*D_v'],True),
    ]:
        matrix(suffix, dims, 'attention', 'DeepseekV3Attention.__init__', index_scale=scale)
    for suffix in ['input_layernorm.weight','post_attention_layernorm.weight']:
        matrix(suffix, ['H'], 'block_norm', 'DeepseekV3DecoderLayer.__init__')
    for suffix, dims in [('gate_proj.weight',['I_dense','H']),('up_proj.weight',['I_dense','H']),('down_proj.weight',['H','I_dense'])]:
        matrix('mlp.'+suffix,dims,'dense','DeepseekV3MLP.__init__',index_scale=True)
    matrix('mlp.gate.weight',['E','H'],'router','MoEGate.__init__')
    matrix('mlp.gate.e_score_correction_bias',['E'],'router','MoEGate.__init__')
    for suffix, dims in [('gate_proj.weight',['I_expert','H']),('up_proj.weight',['I_expert','H']),('down_proj.weight',['H','I_expert'])]:
        matrix('mlp.experts.{e}.'+suffix,dims,'routed','DeepseekV3MLP.__init__',count='E',index_scale=True)
    for suffix, dims in [('gate_proj.weight',['N_shared*I_expert','H']),('up_proj.weight',['N_shared*I_expert','H']),('down_proj.weight',['H','N_shared*I_expert'])]:
        matrix('mlp.shared_experts.'+suffix,dims,'shared','DeepseekV3MLP.__init__',index_scale=True)
    matrix('model.embed_tokens.weight',['Vocab','H'],'global','DeepseekV3Model.__init__')
    matrix('model.norm.weight',['H'],'global','DeepseekV3Model.__init__')
    matrix('lm_head.weight',['Vocab','H'],'global','DeepseekV3ForCausalLM.__init__')
    sums = {owner:sum(m['logicalParameters'] for m in matrices if m['owner']==owner)
            for owner in ('attention','block_norm','dense','router','routed','shared','global')}
    dense_params = sums['attention'] + sums['block_norm'] + sums['dense']
    moe_params = sums['attention'] + sums['block_norm'] + sums['router'] + sums['routed'] + sums['shared']
    totals = {'attention':sums['attention'], 'blockNorm':sums['block_norm'], 'denseFFN':sums['dense'],
              'singleRoutedExpert':sums['routed']//env['E'], 'allRoutedExperts':sums['routed'],
              'selectedRoutedExpertProxy':env['K']*(sums['routed']//env['E']),
              'sharedExperts':sums['shared'], 'routerIncludingCorrectionBias':sums['router'],
              'denseLayer':dense_params, 'moeLayer':moe_params, 'global':sums['global'],
              'mainUniqueLogical':sums['global']+len(layer_ids['Dense'])*dense_params+len(layer_ids['MoE'])*moe_params,
              'scope':'主干 61 层、独立 embedding/head、norm 与路由校正参数；排除 MTP 和量化 scale；配置+源码推导，非权重审计'}
    weight_index = json.loads(raw[INDEX])
    names = weight_index['weight_map']
    groups = collections.Counter()
    for name in names:
        match = re.match(r'model.layers\.(\d+)\.', name)
        groups[match.group(1) if match else 'global'] += 1
    selected_names = {name:shard for name, shard in names.items()
                      if not name.startswith('model.layers.') or
                      any(name.startswith(f'model.layers.{i}.') and ('.experts.' not in name or '.experts.0.' in name) for i in (0,3,61))}
    index_summary = {'source_id':INDEX, 'sourceSha256':sources[INDEX]['sha256'],
                     'tensorNames':len(names), 'shards':len(set(names.values())),
                     'indexDeclaredMetadata':weight_index['metadata'], 'namesByLayer':dict(groups),
                     'representativeNameToShard':selected_names,
                     'limit':'只读索引；metadata.total_size 未作为实测载荷，未读取 safetensors 文件头或权重'}
    # Verify presence of every declared main-model tensor template and companion scale name.
    for mat in matrices:
        ids = [None] if mat['owner']=='global' else layer_ids['Dense'] if mat['owner']=='dense' else layer_ids['MoE'] if mat['owner'] in ('router','routed','shared') else list(range(config['num_hidden_layers']))
        for i in ids:
            for e in range(env['E']) if mat['owner']=='routed' else [None]:
                name = mat['tensorTemplate'].replace('{i}',str(i)).replace('{e}',str(e))
                if name not in names:
                    raise ValueError(f'Declared tensor absent from index: {name}')
                if mat['declaredScaleInIndex'] and name.replace('.weight','.weight_scale_inv') not in names:
                    raise ValueError(f'Companion scale absent: {name}')
    write(DATA / 'research/v3-weight-index-summary.json',index_summary)

    steps = []
    def step(sid, title, op, inputs, outputs, relation, symbol='MLA.forward', branch='both', layer='both',
             dtype='激活/缓存由调用者 dtype 决定；本轮 shape 场景假设 BF16', extra=None, note=''):
        record = {'id':sid,'title':title,'op':op,'inputs':inputs,'outputs':outputs,'relation':relation,
                  'branch':branch,'layerKind':layer,'phases':['prefill','decode'], 'dtype':dtype,
                  'proof_id':proof(MODEL,symbol),'evidence':'derived','source_ids':[CONFIG,MODEL],
                  'backend':None,'backendEvidence':'unknown','duration':None,'note':note}
        if extra: record.update(extra)
        steps.append(record)
    X=['B','L_q','H']; Q=['B','L_q','N_head','D_nope+D_rope']; Qn=['B','L_q','N_head','D_nope']; Qr=['B','L_q','N_head','D_rope']
    C=['B','L_kv','R_kv']; PE=['B','L_kv','D_rope']; P=['B','L_q','N_head','L_kv']; V=['B','L_q','N_head','D_v']
    step('attn_norm','Attention 前 RMSNorm','norm',{'x':X},{'x_norm':X},'RMSNorm(x; gamma[H], eps)',symbol='Block.forward')
    step('q_a','Q 低秩下投影','linear',{'x_norm':X,'W_qa':['R_q','H']},{'q_low':['B','L_q','R_q']},'q_low=x_norm @ W_qa^T')
    step('q_norm','Q 低秩 RMSNorm','norm',{'q_low':['B','L_q','R_q']},{'q_low_norm':['B','L_q','R_q']},'RMSNorm(q_low; gamma[R_q])')
    step('q_b','Q 上投影','linear',{'q_low_norm':['B','L_q','R_q'],'W_qb':['N_head*(D_nope+D_rope)','R_q']},{'q_flat':['B','L_q','N_head*(D_nope+D_rope)']},'q_flat=q_low_norm @ W_qb^T')
    step('q_view','Q heads reshape','reshape',{'q_flat':['B','L_q','N_head*(D_nope+D_rope)']},{'q':Q},'reshape q_flat to heads')
    step('q_split','拆分 Q NoPE 与 RoPE','split',{'q':Q},{'q_nope':Qn,'q_pe':Qr},'split last axis [D_nope,D_rope]')
    step('q_rope','Q 位置旋转','rope',{'q_pe':Qr},{'q_rot':Qr},'apply_rotary_emb(q_pe, freqs_cis)',dtype='旋转先转 FP32/complex，返回输入 dtype',note='位置频率切片不是可训练矩阵')
    step('kv_a','KV 低秩及共享位置投影','linear',{'x_norm':X,'W_kva':['R_kv+D_rope','H']},{'kv_joint':['B','L_q','R_kv+D_rope']},'kv_joint=x_norm @ W_kva^T')
    step('kv_split','拆分潜变量与共享位置 Key','split',{'kv_joint':['B','L_q','R_kv+D_rope']},{'kv_low':['B','L_q','R_kv'],'k_pe':['B','L_q','D_rope']},'split last axis [R_kv,D_rope]')
    step('k_unsqueeze','为共享位置 Key 增加 head 轴','reshape',{'k_pe':['B','L_q','D_rope']},{'k_pe_single':['B','L_q','1','D_rope']},'unsqueeze head axis=2')
    step('k_rope','共享 Key 位置旋转','rope',{'k_pe_single':['B','L_q','1','D_rope']},{'k_rot_single':['B','L_q','1','D_rope']},'apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)',dtype='旋转先转 FP32/complex，返回输入 dtype')
    step('k_squeeze','共享位置 Key 去掉单 head 轴','reshape',{'k_rot_single':['B','L_q','1','D_rope']},{'k_rot':['B','L_q','D_rope']},'squeeze head axis for compressed positional cache')
    step('kv_norm','KV 低秩 RMSNorm','norm',{'kv_low':['B','L_q','R_kv']},{'c_new':['B','L_q','R_kv']},'RMSNorm(kv_low; gamma[R_kv])')
    step('kv_b','展开 KV 上投影','linear',{'c_new':['B','L_q','R_kv'],'W_kvb':['N_head*(D_nope+D_v)','R_kv']},{'kv_flat':['B','L_q','N_head*(D_nope+D_v)']},'kv_flat=c_new @ W_kvb^T',branch='naive')
    step('kv_view','展开 KV heads reshape','reshape',{'kv_flat':['B','L_q','N_head*(D_nope+D_v)']},{'kv_heads':['B','L_q','N_head','D_nope+D_v']},'reshape kv_flat to heads',branch='naive')
    step('kv_heads_split','拆分展开 Key 与 Value','split',{'kv_heads':['B','L_q','N_head','D_nope+D_v']},{'k_nope':Qn,'v_new':V},'split last axis [D_nope,D_v]',branch='naive')
    step('k_broadcast','共享位置 Key 广播','broadcast',{'k_rot_single':['B','L_q','1','D_rope']},{'k_rot_heads':Qr},'expand head axis without adding parameters',branch='naive')
    step('k_concat','构造完整 Key','concat',{'k_nope':Qn,'k_rot_heads':Qr},{'k_new':Q},'concat NoPE and rotated Key',branch='naive')
    step('q_concat','构造完整 Query','concat',{'q_nope':Qn,'q_rot':Qr},{'q_full':Q},'concat NoPE and rotated Query',branch='naive')
    step('k_cache','写展开 Key cache','cache_write',{'k_new':Q},{'k_cache':['B','L_kv','N_head','D_nope+D_rope']},'write [start_pos:start_pos+L_q]; read active [:L_kv]',branch='naive')
    step('v_cache','写展开 Value cache','cache_write',{'v_new':V},{'v_cache':['B','L_kv','N_head','D_v']},'write active new values',branch='naive')
    step('naive_scores','展开 QK 打分','einsum',{'q_full':Q,'k_cache':['B','L_kv','N_head','D_nope+D_rope']},{'scores':P},'scores=alpha*einsum(bqhd,bshd->bqhs)',branch='naive',extra={'einsum':'bqhd,bshd->bqhs'})
    step('absorb_weight','按需解量化 KV 上投影并分头','reshape',{'W_kvb':['N_head*(D_nope+D_v)','R_kv']},{'W_kvb_heads':['N_head','D_nope+D_v','R_kv']},'weight_dequant iff scale exists; view heads',branch='absorb',dtype='FP8 条件分支解量化到 torch.get_default_dtype()',note='不假设预先缓存了反量化权重；源码每次进入该分支进行条件处理')
    step('absorb_weight_split','区分潜变量 Key/Value 权重','split',{'W_kvb_heads':['N_head','D_nope+D_v','R_kv']},{'W_k':['N_head','D_nope','R_kv'],'W_v':['N_head','D_v','R_kv']},'slice key prefix and value suffix along axis 1',branch='absorb',extra={'axis':1})
    step('q_absorb','将 Key 上投影吸收到 Query','einsum',{'q_nope':Qn,'W_k':['N_head','D_nope','R_kv']},{'q_abs':['B','L_q','N_head','R_kv']},'einsum(bqhd,hdc->bqhc)',branch='absorb',extra={'einsum':'bqhd,hdc->bqhc'})
    step('latent_cache','写归一化 KV 潜变量 cache','cache_write',{'c_new':['B','L_q','R_kv']},{'c_cache':C},'cache normalized low-rank KV, not expanded K/V',branch='absorb')
    step('pe_cache','写共享位置 cache','cache_write',{'k_rot':['B','L_q','D_rope']},{'pe_cache':PE},'cache rotated shared positional key',branch='absorb')
    step('latent_score','潜变量打分','einsum',{'q_abs':['B','L_q','N_head','R_kv'],'c_cache':C},{'score_nope':P},'einsum(bqhc,bsc->bqhs)',branch='absorb',extra={'einsum':'bqhc,bsc->bqhs'})
    step('rope_score','共享位置打分','einsum',{'q_rot':Qr,'pe_cache':PE},{'score_rope':P},'einsum(bqhr,bsr->bqhs)',branch='absorb',extra={'einsum':'bqhr,bsr->bqhs'})
    step('score_combine','合并两路 attention 分数','add',{'score_nope':P,'score_rope':P},{'scores':P},'alpha*(score_nope+score_rope)',branch='absorb')
    step('attention_prob','Mask 与 attention softmax','masked_softmax',{'scores':P},{'prob':P},'initial prefill adds causal [L_q,L_q] mask; decode L_q=1 mask=None; softmax over L_kv',dtype='softmax 显式 FP32，然后 type_as(x)',note='本轮 prefill 为 start_pos=0；不声称支持带历史 chunked prefill 的方形 mask')
    step('naive_context','展开 Value 加权聚合','einsum',{'prob':P,'v_cache':['B','L_kv','N_head','D_v']},{'attn_heads':V},'einsum(bqhs,bshd->bqhd)',branch='naive',extra={'einsum':'bqhs,bshd->bqhd'})
    step('latent_context','潜变量加权聚合','einsum',{'prob':P,'c_cache':C},{'latent_context':['B','L_q','N_head','R_kv']},'einsum(bqhs,bsc->bqhc)',branch='absorb',extra={'einsum':'bqhs,bsc->bqhc'})
    step('value_expand','聚合后 Value 上投影','einsum',{'latent_context':['B','L_q','N_head','R_kv'],'W_v':['N_head','D_v','R_kv']},{'attn_heads':V},'einsum(bqhc,hdc->bqhd)',branch='absorb',extra={'einsum':'bqhc,hdc->bqhd'})
    step('attn_flatten','合并 attention heads','reshape',{'attn_heads':V},{'attn_flat':['B','L_q','N_head*D_v']},'flatten head and value axes')
    step('o_proj','Attention 输出投影','linear',{'attn_flat':['B','L_q','N_head*D_v'],'W_o':['H','N_head*D_v']},{'attn_out':X},'attn_out=attn_flat @ W_o^T')
    step('attn_residual','Attention 残差相加','add',{'x':X,'attn_out':X},{'after_attn':X},'after_attn=x+attn_out',symbol='Block.forward')
    step('ffn_norm','FFN 前 RMSNorm','norm',{'after_attn':X},{'ffn_input':X},'RMSNorm(after_attn; gamma[H], eps)',symbol='Block.forward')
    for sid,suffix,out in [('dense_gate','W_gate','g'),('dense_up','W_up','u')]:
        step(sid,'Dense '+('gate' if out=='g' else 'up')+' 投影','linear',{'ffn_input':X,suffix:['I_dense','H']},{out:['B','L_q','I_dense']},f'{out}=ffn_input @ {suffix}^T',symbol='MLP.forward',layer='Dense')
    step('dense_activation','Dense SwiGLU','multiply',{'g':['B','L_q','I_dense'],'u':['B','L_q','I_dense']},{'gu':['B','L_q','I_dense']},'SiLU(g)*u',symbol='MLP.forward',layer='Dense')
    step('dense_down','Dense down 投影','linear',{'gu':['B','L_q','I_dense'],'W_down':['H','I_dense']},{'ffn_out':X},'gu @ W_down^T',symbol='MLP.forward',layer='Dense')
    N=['N_tok','H']; NK=['N_tok','K']; NE=['N_e','H']; EI=['N_e','I_expert']; SI=['N_tok','N_shared*I_expert']
    step('moe_flatten','MoE token 展平','reshape',{'ffn_input':X},{'tokens':N},'N_tok=B*L_q for padded demo',symbol='MoE.forward',layer='MoE')
    step('router_linear','路由打分投影','linear',{'tokens':N,'W_router':['E','H']},{'logits':['N_tok','E']},'linear(tokens,W_router)',symbol='Gate.forward',layer='MoE',dtype='demo 未显式转 FP32；HF MoEGate.forward 显式将输入与权重转 FP32')
    step('router_sigmoid','路由 sigmoid','activation',{'logits':['N_tok','E']},{'original_scores':['N_tok','E']},'sigmoid(logits)',symbol='Gate.forward',layer='MoE')
    step('router_bias','选择分数加校正 bias','add',{'original_scores':['N_tok','E'],'correction_bias':['E']},{'choice_scores':['N_tok','E']},'selection only: original_scores+correction_bias',symbol='Gate.forward',layer='MoE',dtype='demo correction bias 构造为 FP32；保留 original_scores 用于最终权重')
    step('group_view','路由分组','reshape',{'choice_scores':['N_tok','E']},{'grouped':['N_tok','G','E//G']},'reshape to G groups',symbol='Gate.forward',layer='MoE')
    step('group_score','每组 top-2 分数求和','group_top2',{'grouped':['N_tok','G','E//G']},{'group_scores':['N_tok','G']},'sum(topk(grouped,2),last_axis)',symbol='Gate.forward',layer='MoE')
    step('group_topk','选择专家组','topk',{'group_scores':['N_tok','G']},{'selected_groups':['N_tok','G_keep']},'topk groups with G_keep=4',symbol='Gate.forward',layer='MoE',extra={'k':'G_keep'},dtype='输出为整型 group indices；score 精度依输入路径')
    step('group_mask','屏蔽未选择专家组','group_mask',{'choice_scores':['N_tok','E'],'selected_groups':['N_tok','G_keep']},{'masked_choice':['N_tok','E']},'unselected groups masked to -inf',symbol='Gate.forward',layer='MoE')
    step('expert_topk','选择路由专家','topk',{'masked_choice':['N_tok','E']},{'indices':NK},'topk masked_choice with K=8',symbol='Gate.forward',layer='MoE',extra={'k':'K'},dtype='输出为整型 expert indices；score 精度依输入路径')
    step('route_weights','取原分数、归一化与缩放','gather',{'original_scores':['N_tok','E'],'indices':NK},{'route_weights':NK},'gather ORIGINAL scores; normalize sum; multiply 2.5; type_as(tokens)',symbol='Gate.forward',layer='MoE',note='correction bias 不进入最终归一化权重；HF 分母额外加 1e-20')
    step('routed_init','初始化路由专家累加输出','zeros_like',{'tokens':N},{'routed_zero':N},'zeros_like(tokens)',symbol='MoE.forward',layer='MoE')
    step('dispatch','按专家 gather token','dispatch',{'tokens':N,'indices':NK},{'expert_tokens':NE,'expert_token_indices':['N_e'],'expert_slots':['N_e']},'bincount IDs; idx,top=torch.where(indices==e); x[idx]',symbol='MoE.forward',layer='MoE',dtype='expert_tokens 保持激活 dtype；token/slot indices 为整型',note='demo 用逐专家 gather 与加权 scatter；HF moe_infer 则排序，EP>1 时另含 all-to-all，未审阅生产框架')
    for sid,w,out in [('expert_gate','W_eg','eg'),('expert_up','W_eu','eu')]:
        step(sid,'代表路由专家 '+('gate' if out=='eg' else 'up'),'linear',{'expert_tokens':NE,w:['I_expert','H']},{out:EI},f'{out}=expert_tokens @ {w}^T',symbol='Expert.forward',layer='MoE',note='模板覆盖 E 个专家；每专家 N_e 动态，N_e=0 时跳过')
    step('expert_activation','路由专家 SwiGLU','multiply',{'eg':EI,'eu':EI},{'egu':EI},'SiLU(eg)*eu',symbol='Expert.forward',layer='MoE')
    step('expert_down','路由专家 down','linear',{'egu':EI,'W_ed':['H','I_expert']},{'expert_out':NE},'egu @ W_ed^T',symbol='Expert.forward',layer='MoE')
    step('expert_route_weight','取代表专家路由权重','expert_weight_gather',{'route_weights':NK,'expert_token_indices':['N_e'],'expert_slots':['N_e']},{'expert_weights':['N_e','1']},'weights[idx,top,None]',symbol='MoE.forward',layer='MoE')
    step('expert_weighted','代表专家输出乘路由权重','multiply',{'expert_out':NE,'expert_weights':['N_e','1']},{'weighted_expert':NE},'expert_out*expert_weights',symbol='MoE.forward',layer='MoE')
    step('weighted_scatter','路由专家加权 scatter 累加','scatter',{'weighted_expert':NE,'expert_token_indices':['N_e'],'routed_zero':N},{'routed_out':N},'y[idx] += weighted_expert; repeat for all nonempty local experts',symbol='MoE.forward',layer='MoE')
    for sid,w,out in [('shared_gate','W_sg','sg'),('shared_up','W_su','su')]:
        step(sid,'共享专家 '+('gate' if out=='sg' else 'up'),'linear',{'tokens':N,w:['N_shared*I_expert','H']},{out:SI},f'{out}=tokens @ {w}^T',symbol='MLP.forward',layer='MoE')
    step('shared_activation','共享专家 SwiGLU','multiply',{'sg':SI,'su':SI},{'sgu':SI},'SiLU(sg)*su',symbol='MLP.forward',layer='MoE')
    step('shared_down','共享专家 down','linear',{'sgu':SI,'W_sd':['H','N_shared*I_expert']},{'shared_out':N},'sgu @ W_sd^T',symbol='MLP.forward',layer='MoE')
    step('moe_combine','合并路由与共享专家','add',{'routed_out':N,'shared_out':N},{'moe_flat':N},'routed_out+shared_out; world_size>1 routed_out all_reduce before addition',symbol='MoE.forward',layer='MoE')
    step('moe_reshape','恢复 MoE 层 shape','reshape',{'moe_flat':N},{'ffn_out':X},'reshape to original [B,L_q,H]',symbol='MoE.forward',layer='MoE')
    step('ffn_residual','FFN 残差相加','add',{'after_attn':X,'ffn_out':X},{'layer_out':X},'after_attn+ffn_out',symbol='Block.forward')
    # Additional precision/parallel evidence is recorded separately from the logical step order.
    for sid,symbol in [(MODEL,'linear'),(MODEL,'Linear.__init__'),(MODEL,'RMSNorm.forward'),
                       (MODEL,'ColumnParallelLinear.__init__'),(MODEL,'RowParallelLinear.forward'),
                       (MODEL,'ParallelEmbedding.forward'),(MODEL,'Transformer.__init__'),(MODEL,'Transformer.forward'),
                       (MODEL,'Gate.__init__'),(MODEL,'MoE.__init__'),(MODEL,'apply_rotary_emb'),
                       (HF_MODEL,'MoEGate.forward'),(HF_MODEL,'DeepseekV3MoE.moe_infer'),
                       (HF_MODEL,'DeepseekV3Attention.forward'),(HF_MODEL,'DeepseekV3MLP.forward'),
                       (HF_MODEL,'DeepseekV3RMSNorm.forward'),
                       ('gh-deepseek-v3-inference-convert-py','main'),
                       ('gh-deepseek-v3-inference-kernel-py','act_quant'),
                       ('gh-deepseek-v3-inference-kernel-py','weight_dequant'),
                       ('gh-deepseek-v3-inference-kernel-py','fp8_gemm')]:
        proof(sid,symbol)
    checkpoints = read(DATA / 'research/selected-checkpoints.json')
    distill_bases = {'Qwen-1.5B':'Qwen/Qwen2.5-Math-1.5B','Qwen-7B':'Qwen/Qwen2.5-Math-7B',
                    'Qwen-14B':'Qwen/Qwen2.5-14B','Qwen-32B':'Qwen/Qwen2.5-32B',
                    'Llama-8B':'meta-llama/Llama-3.1-8B','Llama-70B':'meta-llama/Llama-3.3-70B-Instruct'}
    versions = []
    for item in checkpoints['selected']:
        item = dict(item)
        name = item['name']; base = None
        if '-Distill-' in name:
            kind='distill'; base=distill_bases[name.split('-Distill-',1)[1]]
            summary='R1 生成数据微调的其他底座；官方提示 config/tokenizer 有改动；不复用 V3 MLA/MoE 结构'
            sid='gh-deepseek-r1-readme-md'
        elif name in ('DeepSeek-R1','DeepSeek-R1-Zero'):
            kind='post_training'; base='deepseek-ai/DeepSeek-V3-Base'
            summary='官方明确以 V3-Base 训练；首轮尚未逐字段核对本 checkpoint 配置'
            sid='gh-deepseek-r1-readme-md'
        elif name == 'DeepSeek-V2':
            kind='architecture_origin'; summary='MLA 与 DeepSeekMoE 的演进起点；首轮只读概述'
            sid='gh-deepseek-v2-readme-md'
        elif name.startswith('DeepSeek-V3.2'):
            kind='architecture_increment'; summary='DSA 增量；V3.2 模型卡声明结构与 Exp 相同，逐配置/矩阵核验留到 P3'
            sid='hf-'+name.lower()+'-readme'
            base='deepseek-ai/DeepSeek-V3.2-Exp-Base'
        elif name.startswith('DeepSeek-V4'):
            kind='base' if name.endswith('-Base') else 'post_training'
            summary='CSA/HCA、mHC；Base 与后训练权重精度分别披露，首轮未读配置与实现'
            sid='hf-deepseek-v4-pro-readme'
            base=None if kind=='base' else 'deepseek-ai/'+name+'-Base'
        else:
            kind='base' if name.endswith('-Base') else 'post_training'
            summary='V3 的 MLA/MoE 主干；本轮只深读原始 V3 checkpoint'
            sid='gh-deepseek-v3-readme-md'
            base=None if kind=='base' else 'deepseek-ai/DeepSeek-V3-Base'
        item.update(kind=kind, baseModel=base, summary=summary, source_id=sid, evidence='official',
                    verifiedTo='config+reference_source+weight_index_names' if name=='DeepSeek-V3' else 'official_directory+model_card_or_README',
                    configVerified=name=='DeepSeek-V3')
        versions.append(item)
    version_data = {'schemaVersion':1,'familyId':'deepseek','snapshot':DATE,'models':versions,
                    'modes':[{'name':'DeepSeek-V4-Pro-Max','checkpoint':'deepseek-ai/DeepSeek-V4-Pro','kind':'inference_mode','source_id':'hf-deepseek-v4-pro-readme'},
                             {'name':'DeepSeek-V4-Flash-Max','checkpoint':'deepseek-ai/DeepSeek-V4-Flash','kind':'inference_mode','source_id':'hf-deepseek-v4-pro-readme'}],
                    'deferred':checkpoints['deferred'], 'limit':'17 个选定代表 checkpoint；目录有 V4.1、V3.1、R1-0528 等条目，后续再评估，非完整版本清单'}
    write(DATA / 'versions.json',version_data)
    conflicts = [
        {'id':'context','observation':'HF max_position_embeddings=163840；官方 V3 README 披露 128K；demo 配置未覆盖 ModelArgs.max_seq_len=16384',
         'sources':[CONFIG,'gh-deepseek-v3-readme-md',MODEL], 'resolution':'分别保留配置、披露和 demo 默认值；不能用其中任一数值替代已验证的服务上下文'},
        {'id':'mtp-count','observation':'README 披露主干 671B、含 MTP 的发布规模 685B/额外 14B；README_WEIGHTS 单列 MTP unique 11.5B（排除共享 embedding/head）',
         'sources':['gh-deepseek-v3-readme-md',WEIGHTS,INDEX], 'resolution':'主干推导总数单列；MTP exact unique 参数与共享 norm 的真实别名仍未知，不把披露的近似值强行配平'},
        {'id':'router-dtype','observation':'HF MoEGate.forward 显式 FP32 linear；demo Gate.forward 无相同显式转换，correction bias 构造为 FP32',
         'sources':[HF_MODEL,MODEL], 'resolution':'按实现分别记录 dtype，不把 HF 的精度语义直接转写成 demo 的执行路径'},
        {'id':'index-size','observation':f'索引 metadata.total_size={weight_index["metadata"]["total_size"]}，索引只给张量名与分片',
         'sources':[INDEX], 'resolution':'保留原值；实际 storage dtype、shape、payloadBytes 均未知，不能据索引 total_size 推断 FP8 实际载荷'},
    ]
    gaps = [
        {'id':'weight-headers','stage':'可选权重审计','question':'实际逐张量 dtype、shape、scale 与 payload，以及 MTP 共享参数存储别名','status':'unknown'},
        {'id':'mtp-runtime','stage':'P2','question':'目标框架的 MTP/speculative decoding 加载与执行路径；当前 demo 转换跳过 layer 61','status':'unknown'},
        {'id':'ascend-stack','stage':'P2','question':'目标 Ascend 型号、卡数/拓扑、CANN、torch_npu、框架及权重转换格式','status':'unknown'},
        {'id':'production-kernels','stage':'P2','question':'生产 attention、MoE、通信、量化 kernel 的条件映射与融合边界','status':'unknown'},
        {'id':'chunk-prefill','stage':'P2','question':'带历史的 chunked prefill/verify mask 路径；demo 当前方形 mask 不能直接证明这些场景','status':'not_verified'},
        {'id':'version-configs','stage':'P3/P4','question':'其他 checkpoint 的实际配置、精度与结构一致性；首轮只核对名称、revision、官方关系','status':'not_verified'},
        {'id':'schema-integration','stage':'P5a','question':'DeepSeek 网站数据适配、独立验证分派；新 cache 结构扩展在 P3/P5b 处理','status':'not_started'},
        {'id':'device-results','stage':'P6','question':'权重加载、设备数值正确性、profiling、性能与质量实验','status':'not_run'},
    ]
    result = {'schemaVersion':1,'kind':'deepseek_v3_round1_research','familyId':'deepseek','authored':DATE,
              'modelId':'deepseek-ai/DeepSeek-V3','revision':sources[CONFIG]['revision'],
              'referenceRevision':sources[MODEL]['revision'],'scope':'P0/P1；两个代表主干层，MP=1，初始 prefill 与常规 decode；非网站 compute 契约，非权重头审计/设备测试',
              'configPath':'data/families/deepseek/configs/DeepSeek-V3-config.json','configFields':config_rows,
              'symbols':{k:{'meaning':v,'value':env.get(k)} for k,v in meanings.items()},
              'layerMapping':layer_ids,'representatives':[{'kind':'Dense','sourceLayer':0,'displayLayer':1},{'kind':'MoE','sourceLayer':3,'displayLayer':4}],
              'matrices':matrices,'parameterTotals':totals,'steps':steps,'proofs':proofs,
              'weightIndexSummaryPath':'data/families/deepseek/research/v3-weight-index-summary.json',
              'scenarios':[{'phase':'prefill','B':2,'L_q':4,'L_kv':4,'start_pos':0,'N_e':1},
                           {'phase':'decode','B':2,'L_q':1,'L_kv':8,'start_pos':7,'N_e':1}],
              'cache':{'naivePerTokenPerLayerElements':'N_head*(D_nope+D_rope+D_v)',
                       'absorbPerTokenPerLayerElements':'R_kv+D_rope',
                       'naiveElements':env['N_head']*(env['D_nope']+env['D_rope']+env['D_v']),
                       'absorbElements':env['R_kv']+env['D_rope'],
                       'usageElementsFormula':'B*L_kv*layers*elements_per_token_per_layer',
                       'allocationFormula':'max_batch_size*max_seq_len*layers*elements_per_token_per_layer',
                       'byteFormula':'elements*itemsize (conditional on actual cache dtype)',
                       'limit':'MP=1；只含 K/V 或潜变量/位置 cache，排除 scores、RoPE 表、工作区、权重和框架管理开销；demo 按最大容量预分配'},
              'linearPrecision':{'weight_non_fp8':'F.linear', 'fp8_and_gemm_impl_bf16':'weight_dequant → F.linear',
                                 'fp8_and_gemm_impl_fp8':'act_quant → fp8_gemm',
                                 'limit':'linear 的条件分支；参考 Triton wrapper 已读，未运行或映射到 Ascend kernel'},
              'mtp':{'mainLayers':[0,60],'mtpLayer':61,'declaredModules':config['num_nextn_predict_layers'],
                     'shared':['embedding','output head'],'projectionLogicalShape':[env['H'],2*env['H']],
                     'projectionLogicalParameters':2*env['H']*env['H'],
                     'projectionEvidence':'derived from paper Eq.21 + checkpoint H, not tensor header',
                     'source_ids':[CONFIG,WEIGHTS,'v3-paper-v2','gh-deepseek-v3-inference-convert-py'],
                     'officialUniqueApprox':11_500_000_000,'exactUniqueParameters':None,
                     'demoExecuted':False,'limit':'转换显式跳过 model.layers.61；Transformer 只创建 n_layers 个 Block；MTP 单列，训练目标与 speculative decoding 支持分开'},
              'conflicts':conflicts,'gaps':gaps,'verification':{'weightHeaderAudit':False,'checkpointLoaded':False,'deviceTest':False,'performanceExperiment':False}}
    write(DATA / 'research/v3-round1.json',result)
    for source in manifest['sources']:
        sid=source['id']
        if source['kind']=='reference_code':
            source['readScope']='已读符号：'+', '.join(p['symbol'] for p in proofs.values() if p['source_id']==sid) if any(p['source_id']==sid for p in proofs.values()) else '只采集；未深读'
        elif sid==INDEX: source['readScope']='索引全部名称、分片和 metadata；核对主干模板名称与 scale 伴随名称；未读取文件头'
        elif sid==CONFIG: source['readScope']='全部配置字段；与 demo 对应字段逐项核对'
        elif sid=='v3-paper-v2': source['readScope']='第 2.2 节 MTP 和公式 21；未复验论文实验'
        elif source['kind']=='checkpoint_metadata': source['readScope']='固定 revision 的文件目录、论文标签与许可入口；未据此读取或验证其他 checkpoint 配置值'
        elif 'readme' in sid: source['readScope']='版本/底座、架构概述、参数与权重说明或许可入口；未复验评测'
        elif 'license' in sid: source['readScope']='确认 V3 代码 MIT 与独立模型许可入口；不扩展为法律结论'
        else: source['readScope']='配置或目录元数据核对'
    write(DATA / 'sources.json',manifest)
    export_report(result,version_data,sources)
    print(json.dumps({'mainUniqueLogical':totals['mainUniqueLogical'],'operatorDefinitions':len(steps),
                      'proofs':len(proofs),'selectedCheckpoints':len(versions)},ensure_ascii=False))


def export_report(d,versions,sources):
    def link(sid): return f'[{sid}]({sources[sid]["url"]})'
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(str(x).replace('|','\\|').replace('\n','<br>') for x in row)+' |' for row in rows])
    totals=d['parameterTotals']; cache=d['cache']; index=read(ROOT/d['weightIndexSummaryPath'])
    chunks=[
        '# DeepSeek 首轮研究：版本关系与 V3 代表层\n',
        f'研究日期：{DATE}。研究对象为 `{d["modelId"]}@{d["revision"]}`，官方推理源码固定到 `{d["referenceRevision"]}`。本轮完成配置、参考源码与索引名称的静态研究；主干逻辑参数推导为 **{totals["mainUniqueLogical"]:,}**，不含 MTP 和量化 scale。两类代表层分别为源码第 0 层 Dense（展示第 1 层）与源码第 3 层 MoE（展示第 4 层）。',
        '事实源为 [v3-round1.json](../../../data/families/deepseek/research/v3-round1.json)，版本关系为 [versions.json](../../../data/families/deepseek/versions.json)，输入哈希与访问范围见 [sources.json](../../../data/families/deepseek/sources.json)。[参数表 CSV](v3-parameters.csv) 和 [代表层算子 CSV](v3-representative-operators.csv) 从相同 JSON 导出；[验证记录](validation.json) 说明实际检查范围。',
        '## 1 版本关系与范围\n',
        table(['代表条目','关系与核验边界','来源'],[
            ['V2','MLA/DeepSeekMoE 的设计演进入口；不是 V3 权重底座关系',link('gh-deepseek-v2-readme-md')],
            ['V3-Base / V3','基础模型与后训练模型分列；本轮深读原始 V3',link('gh-deepseek-v3-readme-md')],
            ['R1-Zero / R1','官方说明基于 V3-Base 训练；配置相同性仍待逐字段核验',link('gh-deepseek-r1-readme-md')],
            ['R1-Distill 六版','Qwen2.5-Math 1.5B/7B、Qwen2.5 14B/32B、Llama-3.1 8B、Llama-3.3 70B Instruct；官方提示 config/tokenizer 调整',link('gh-deepseek-r1-readme-md')],
            ['V3.2-Exp / V3.2','Exp 从 V3.1-Terminus 引入 DSA；V3.2 模型卡声明与 Exp 同结构，具体配置/实现留到 P3',link('hf-deepseek-v3.2-exp-readme')+'；'+link('hf-deepseek-v3.2-readme')],
            ['V4-Flash / Pro 及 Base','模型卡披露 CSA/HCA、mHC；Base 与后训练版精度分列；Max 为推理模式',link('hf-deepseek-v4-pro-readme')],
        ]),
        f'本轮固定 {len(versions["models"])} 个代表 checkpoint 的 revision。作者目录还出现 V4.1、其他 V3 更新与 R1-0528 等条目，已列在 deferred；本轮未深读这些条目，也不把此表作为完整版本目录。目录采集范围为前 100 个条目。来源：{link("hf-directory")}。',
        '## 2 配置与参数口径\n',
        f'HF 配置对应字段与 demo 配置逐项相等；未覆盖字段单列。主干层型为 **{len(d["layerMapping"]["Dense"])} 层 Dense + {len(d["layerMapping"]["MoE"])} 层 MoE**。MTP 的源码/权重层号 61 不计入 61 个主干层。配置来源：{link(CONFIG)}；demo 配置来源：{link("gh-deepseek-v3-inference-configs-config_671b-json")}。',
        table(['配置字段','值','demo 对应字段'],[(x['field'],json.dumps(x['value'],ensure_ascii=False),x['demoField'] or '无直接映射；另记差异') for x in d['configFields']]),
        table(['参数项','逻辑参数','范围'],[
            ['Attention（含 Q/KV 低秩 norm）',f'{totals["attention"]:,}','每个主干层'],
            ['两次 block RMSNorm',f'{totals["blockNorm"]:,}','每个主干层'],
            ['Dense FFN',f'{totals["denseFFN"]:,}','每个 Dense 层'],
            ['一个路由专家',f'{totals["singleRoutedExpert"]:,}','gate/up/down 三矩阵'],
            ['全部路由专家',f'{totals["allRoutedExperts"]:,}','256 份；不等于每 token 激活'],
            ['选中路由专家权重代理',f'{totals["selectedRoutedExpertProxy"]:,}','8 份专家权重；不是完整 FLOPs 或性能'],
            ['共享专家',f'{totals["sharedExperts"]:,}','1 份'],
            ['Router',f'{totals["routerIncludingCorrectionBias"]:,}','256×7168 权重 + 256 校正参数'],
            ['Dense 主干层',f'{totals["denseLayer"]:,}','Attention + norm + Dense FFN'],
            ['MoE 主干层',f'{totals["moeLayer"]:,}','Attention + norm + router + 全部 routed/shared'],
            ['全局组件',f'{totals["global"]:,}','独立 embedding、output head 与 final norm'],
            ['主干合计',f'{totals["mainUniqueLogical"]:,}','3×Dense 层 + 58×MoE 层 + 全局'],
        ]),
        '矩阵均按 `[输出,输入]` 给出逻辑 shape，向量 norm 单列。主干 input embedding 与 output head 的 `tie_word_embeddings=false`，两份独立计数；MTP 的共享副本不加入主干合计。scale 的预期 block 表形状由 `ceil(out/128)×ceil(in/128)` 推导，实际 shape/dtype/payload 保持未知。参数声明依据固定 HF 参考类的 `__init__`，索引只用于核对名称。',
        '## 3 整体数据流与代表层\n',
        '```mermaid\nflowchart TD\n  T[Token IDs] --> E[Embedding]\n  E --> D[主干 0–2：MLA + Dense]\n  D --> M[主干 3–60：MLA + MoE]\n  M --> N[Final RMSNorm]\n  N --> H[Output head]\n  M -. 主干表示 .-> P[MTP：与后续 token embedding 分别归一化]\n  E -. 共享 embedding .-> P\n  P --> C[Concat 2H → projection H]\n  C --> B[MTP Transformer block，权重层号 61]\n  B -. 共享 output head .-> H\n```',
        '主干 block 为 `x + MLA(RMSNorm(x))`，随后 `x + FFN(RMSNorm(x))`。MoE 内部按 sigmoid、校正分数、组内 top-2 求和、选 4 组、选 8 专家、取原始分数归一化并乘 2.5、按专家 gather、专家计算、加权 scatter、共享专家相加组织。示意步骤对应数学与参考源码，生产框架可以采用其他重排、通信与融合方式。',
        'MTP 的 `[H,2H]` 投影由论文公式 21 与 checkpoint 的 H 推导，逻辑参数为 '+f'{d["mtp"]["projectionLogicalParameters"]:,}'+f'。图中的 MTP 是论文/权重说明的数据流；当前 demo 的转换脚本显式跳过 layer 61，Transformer 只创建主干，未验证 MTP 加载或 speculative decoding。来源：{link("v3-paper-v2")}（§2.2、Eq.21），{link(WEIGHTS)}，{link("gh-deepseek-v3-inference-convert-py")}。',
        '## 4 Prefill/decode 与两种 MLA 路径\n',
        '本轮以 padded、MP=1 的参考路径说明 shape：prefill 为 B=2、L_q=L_kv=4、start_pos=0；decode 为 B=2、L_q=1、L_kv=8、start_pos=7。N_tok=B×L_q；常规 decode 之外的 verify/ragged batching 未核验。naive 与 absorb 都可用于这两个阶段，它们是 attention 实现分支，不是阶段名称。',
        table(['步骤','输入/中间 shape','输出或 cache'],[
            ['Q 低秩分解','[B,L_q,H] → [B,L_q,R_q]','[B,L_q,128,128+64]'],
            ['KV 下投影','[B,L_q,H]','潜变量 [B,L_q,512] + 共享位置 Key [B,L_q,64]'],
            ['naive KV 展开','潜变量 → 每头 K_nope/V','K cache [B,L_kv,128,192]；V cache [B,L_kv,128,128]'],
            ['absorb Query 变换','Q_nope [B,L_q,128,128] × W_K [128,128,512]','Q_abs [B,L_q,128,512]'],
            ['absorb cache','归一化潜变量与旋转后共享位置 Key','C [B,L_kv,512]；PE [B,L_kv,64]'],
            ['两路 attention','naive 展开 QK；absorb 潜变量分数 + 位置分数','scores [B,L_q,128,L_kv]'],
            ['Value 聚合','naive 聚合 V；absorb 先聚合 C 再乘 W_V','[B,L_q,128,128] → [B,L_q,7168]'],
            ['MoE 专家输入','按实际路由 gather','[N_e,7168] → [N_e,2048] → [N_e,7168]'],
        ]),
        f'MP=1 时，naive 每 token/层为 **{cache["naiveElements"]:,}** 个 cache 元素，absorb 为 **{cache["absorbElements"]:,}** 个（512+64）。BF16 假设下分别为 81,920 和 1,152 字节/token/层；这是数学口径，不是实测显存。demo 按最大 batch/sequence 预分配；使用窗口 B×L_kv 的元素数与实际预分配容量分别记录。排除 attention scores、RoPE 表、权重、工作区与框架管理。来源：{link(MODEL)} 的 MLA.__init__/forward。',
        '## 5 参考实现、精度与证据边界\n',
        'FP8 权重在 `linear` 中有条件分支：非 FP8 用 F.linear；gemm_impl="bf16" 时先解量化再 F.linear；另一条分支为 act_quant→fp8_gemm。absorb 直接读取并按需解量化 W_kv_b，再做 einsum，不能把所有步骤强行对应为相同 GEMM wrapper。RMSNorm 的 demo 入口为 F.rms_norm；HF 参考 norm 显式转 FP32。算子表将精度说明与实际存储字段分开。',
        'demo 的 world_size 同时约束 attention heads、Dense/shared FFN 切分、词表切分与专家归属，不能将其直接改写为任意独立 TP/EP 布局。本轮使用 MP=1；列出的 all_reduce/all-to-all 仅是固定参考源码观察，未映射到生产框架或 Ascend kernel。',
        f'完整权重索引包含 {index["tensorNames"]:,} 个张量名称与 {index["shards"]} 个分片。研究只核对名称与 scale 伴随项，未读取文件头。实际 storedShape、storedDtype、payloadBytes、kernel 与 duration 为未知；没有 checkpoint 加载、GPU/NPU 数值测试或性能实验。来源：{link(INDEX)}。',
        '## 6 冲突与缺口\n',
        table(['项','观察','处理'],[(x['id'],x['observation'],x['resolution']) for x in d['conflicts']]),
        table(['下一阶段','待核验问题','状态'],[(x['stage'],x['question'],x['status']) for x in d['gaps']]),
        '## 7 复算与继续工作\n',
        '执行 `python3 scripts/validate_deepseek.py --research-root <源码快照目录>` 检查输入/函数体哈希、配置映射、层型、参数汇总、线性收缩、einsum、reshape/split、路由与 cache 公式；CSV 生成后检查与 JSON 的行列和值一致。验证器另做固定随机种子的 FP64 合成小矩阵 MLA 代数复算，不使用模型权重，也不导入执行上游代码。这项检查验证公式恒等关系，不能代替 checkpoint 数值验证。',
        '下一步为 P5a：将已审核的 V3 事实适配到 family/architecture/compute 契约，增加 DeepSeek 构建验证分派，接入来源与下载。P2 的设备实现研究在目标 Ascend 栈明确后开始；P3/P4 继续各自配置与结构核验。',
        '## 8 固定来源索引\n',
        table(['source_id','revision','已读取范围','来源'],[(sid,s.get('revision') or '访问日快照',s['readScope'],link(sid)) for sid,s in sources.items()]),
    ]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / 'report.md').write_text('\n\n'.join(chunks)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root',type=Path,required=True)
    build(parser.parse_args().research_root)
