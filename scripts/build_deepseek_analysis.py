"""Build static analytical tools from canonical model data and fixed public sources."""
import collections
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
OUT = ROOT / 'dist/assets/deepseek'
CACHE = Path.home() / '.cache/model-research-atlas'
DATE = '2026-10-02'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def plain(text):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', text))).strip()


def reference(source, lines, start, end):
    start = max(1, start); end = min(len(lines), end)
    body = '\n'.join(lines[start - 1:end])
    return {'sourceId': source['id'], 'revision': source['revision'], 'path': source['path'], 'line': start, 'end': end,
        'sourceSha256': source['sha256'], 'rangeSha256': hashlib.sha256(body.encode()).hexdigest(),
        'url': source['url'] + f'#L{start}-L{end}'}


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows = []; self.row = None; self.cell = None; self.table = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'table': self.table += 1
        if tag == 'tr': self.row = {'line': self.getpos()[0], 'table': self.table, 'cells': [], 'header': False, 'spanned': False}
        if tag in ('td', 'th') and self.row is not None:
            self.cell = []; self.row['header'] |= tag == 'th'
            self.row['spanned'] |= any(k in ('rowspan', 'colspan') and v != '1' for k, v in attrs)
        if tag in ('br', 'li') and self.cell is not None: self.cell.append(' ')

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row['cells'].append(plain(''.join(self.cell))); self.cell = None
        if tag == 'tr' and self.row is not None:
            self.row['end'] = self.getpos()[0]
            self.rows.append(self.row); self.row = None


def categories(text):
    keys = [('shape', r'shape|维度|轴|[BMNK]\s*[=：]|长度|大小|广播'), ('dtype', r'dtype|数据类型|FLOAT|INT\d|BOOL|量化类型'),
        ('layout', r'格式|layout|ND\b|NZ\b|BSH|TND|非连续|连续'), ('quantization', r'量化|scale|Scale|offset|Offset|groupSize'),
        ('platform', r'Atlas|Ascend|A2|A3|950|产品|芯片'), ('condition', r'必须|需要|一致|不能|不支持|仅支持|约束|范围')]
    result = [key for key, pattern in keys if re.search(pattern, text, re.I)]
    return result or ['parameter']


def contracts():
    sources = {s['id']: s for s in read(DATA / 'cann-sources.json')['sources']}
    operators = read(DATA / 'research/cann-operator-audit.json')['operators']
    result = []
    for op in operators:
        item = {'id': op['id'], 'directory': op['directory'], 'revision': op['revision'], 'tag': op['tag'],
            'topics': op['topics'], 'frameworkContext': op['frameworkContext'], 'apis': op['apiSymbols'],
            'documents': [], 'claims': [], 'productSupport': [], 'rules': [],
            'limits': ['范围为所选 API 与 tag；参数表中的类型/格式不自动推广到其他版本、平台或参数组合。', '源码契约与二进制安装、实际分派、ABI、运行成功分别记录。']}
        seen = set()
        for link in op['sources']:
            if link['role'] not in ('documentation', 'api'): continue
            source = sources[link['sourceId']]
            raw = (CACHE / 'deepseek-cann' / source['cacheFile']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == source['sha256']
            text = raw.decode(); lines = text.splitlines()
            if link['role'] == 'documentation':
                parser = Tables(); parser.feed(text); headers = {}; header_counts = collections.Counter()
                def context(line):
                    headings = [plain(x.lstrip('# ')) for x in lines[:line] if x.startswith('#')]
                    return ' / '.join(headings[-3:])
                item['documents'].append({'name': Path(source['path']).stem, 'sourceId': source['id'], 'url': source['url']})
                for row in parser.rows:
                    if row['header']:
                        headers[row['table']] = row; header_counts[row['table']] += 1; continue
                    hrow = headers.get(row['table'], {}); fields = hrow.get('cells', [])
                    if not fields or not row['cells']: continue
                    aligned = len(fields) == len(row['cells']) and not row['spanned'] and not hrow['spanned'] and header_counts[row['table']] == 1
                    joined = '；'.join(f'{fields[i] if aligned else "原始列 "+str(i+1)}: {value}' for i, value in enumerate(row['cells']))
                    key = (source['id'], row['line'], joined)
                    if key in seen: continue
                    seen.add(key)
                    item['claims'].append({'id': f'{op["id"]}-{len(item["claims"])+1}', 'apiDocument': Path(source['path']).stem,
                        'parameter': row['cells'][0], 'context': context(row['line']), 'categories': categories(joined),
                        'fields': dict(zip(fields, row['cells'])) if aligned else {}, 'rawCells': row['cells'], 'observedHeaders': fields,
                        'tableMapping': 'aligned_single_header' if aligned else 'raw_row_with_context',
                        'statement': joined, 'reference': reference(source, lines, row['line'], row['end'])})
                md_header = None
                for i, line in enumerate(lines):
                    match = re.search(r'<term>(.*?)</term>\s*[：:]\s*(不支持|支持)', line)
                    if match:
                        item['productSupport'].append({'apiDocument': Path(source['path']).stem, 'product': plain(match[1]), 'status': match[2], 'reference': reference(source, lines, i + 1, i + 1)})
                    if re.match(r'^\s*\|.*\|\s*$', line):
                        cells = [plain(s) for s in line.strip().strip('|').split('|')]
                        if all(re.fullmatch(r'[:\-\s]+', s) for s in cells): continue
                        next_line = lines[i + 1] if i + 1 < len(lines) else ''
                        if re.match(r'^\s*\|[ :\-|]+\|\s*$', next_line): md_header = cells; continue
                        if md_header and len(cells) == len(md_header):
                            joined = '；'.join(f'{k}: {v}' for k, v in zip(md_header, cells))
                            item['claims'].append({'id': f'{op["id"]}-{len(item["claims"])+1}', 'apiDocument': Path(source['path']).stem,
                                'parameter': cells[0], 'context': context(i+1), 'categories': categories(joined), 'fields': dict(zip(md_header, cells)),
                                'statement': joined, 'reference': reference(source, lines, i + 1, i + 1)})
                    else: md_header = None
                    if re.match(r'^\s*[-*]\s+', line) and re.search(r'必须|需要|仅支持|不支持|取值范围|不能|约束|要求|满足|一致|限制', line) and not match:
                        statement = plain(line.lstrip(' -*'))
                        item['claims'].append({'id': f'{op["id"]}-{len(item["claims"])+1}', 'apiDocument': Path(source['path']).stem,
                            'parameter': '补充条件', 'context': context(i+1), 'categories': categories(statement), 'fields': {}, 'statement': statement,
                            'reference': reference(source, lines, i + 1, i + 1)})
            elif not any(s['role'] == 'documentation' for s in op['sources']):
                # Preserve actual guards when this selection has no API document.
                for i, line in enumerate(lines):
                    if re.search(r'CHECK_|CheckParam|CheckDtype|CheckShape|CheckFormat|CheckSoc|SOC_VERSION|OP_LOGE', line):
                        item['claims'].append({'id': f'{op["id"]}-{len(item["claims"])+1}', 'apiDocument': Path(source['path']).name,
                            'parameter': 'API guard 上下文', 'categories': categories(line), 'fields': {},
                            'statement': plain('\n'.join(lines[max(0, i-1):min(len(lines), i+5)])),
                            'reference': reference(source, lines, max(1, i), min(len(lines), i+5))})
                if not item['claims']:
                    for i, line in enumerate(lines):
                        if re.search(r'aclnnStatus\s+aclnn\w+\s*\(', line):
                            end = next((j for j in range(i, min(len(lines), i+60)) if '{' in lines[j] or ';' in lines[j]), min(len(lines)-1, i+20))
                            item['claims'].append({'id': f'{op["id"]}-prototype', 'apiDocument': Path(source['path']).name,
                                'parameter': '所选公开 API 原型/转发入口', 'categories': ['parameter'], 'fields': {},
                                'statement': plain('\n'.join(lines[i:end+1])), 'reference': reference(source, lines, i+1, end+1)})
                            item['limits'].append('所选入口仅有原型/转发证据；未据此推定完整 shape、dtype 或内部实现条件。')
                            break
        def claim_ref(needle):
            claim = next((x for x in item['claims'] if needle in x['statement']), None)
            assert claim is not None, (op['id'], needle)
            return claim['reference']
        if op['id'] in ('rms-norm', 'swiglu'):
            dtype_ref = claim_ref('FLOAT32')
            item['rules'] += [{'id': op['id']+'-dtype', 'kind': 'enum', 'field': 'dtype', 'values': ['FLOAT16', 'BFLOAT16', 'FLOAT32'], 'label': '所选接口的输入 x 类型：FLOAT16/BFLOAT16/FLOAT32', 'reference': dtype_ref},
                {'id': op['id']+'-layout', 'kind': 'enum', 'field': 'layout', 'values': ['ND'], 'label': '所选接口输入采用 ND 格式', 'reference': claim_ref('ND')}]
        if op['id'] == 'rms-norm':
            item['rules'] += [{'id': 'rms-gamma-suffix', 'kind': 'suffix', 'field': 'gamma', 'label': 'gamma.shape 必须匹配 x 的尾部维度；不要求只能是一维', 'reference': claim_ref('gamma_shape')},
                {'id': 'rms-gamma-nonempty', 'kind': 'nonempty', 'field': 'gamma', 'label': 'gamma 不支持空 Tensor', 'reference': claim_ref('不支持空Tensor')},
                {'id': 'rms-rank', 'kind': 'rank', 'field': 'x', 'min': 1, 'max': 8, 'label': 'x 的秩为 1–8', 'reference': claim_ref('1-8')}]
        if op['id'] == 'swiglu':
            item['rules'].append({'id': 'swiglu-axis-even', 'kind': 'swiglu-axis', 'field': 'x', 'label': 'dim 在 [-rank, rank-1] 内，分割轴长度为偶数', 'reference': claim_ref('偶数')})
        if op['id'] == 'matmul':
            item['rules'].append({'id': 'matmul-2d-k', 'kind': 'matmul-k', 'field': 'x', 'label': '仅核对无转置二维子集：x 的 K 必须等于 weight 的 K', 'reference': claim_ref('Reduce维度')})
        if not item['claims']:
            raise ValueError('No contract evidence for '+op['id'])
        result.append(item)
    return result


def cost_models(family):
    result = []
    for model in family['models']:
        if not model.get('computePath'):
            assert model.get('computeScope'), model['id']
            continue
        arch = read(ROOT / model['architecturePath']); c = read(ROOT / model['configPath'])
        kind = 'v4' if c['model_type'] == 'deepseek_v4' else 'dsa' if c['model_type'] == 'deepseek_v32' else 'gqa' if c['model_type'] in ('qwen2', 'llama') else 'mla'
        groups = {}; routed = 0; non_expert_linears = 0; head_control = 0; excluded = []
        for node in arch['nodes']:
            if node['group'] in ('mtp', 'dspark'): continue
            for mod in node['modules']:
                if mod['id'] == 'routed': routed += mod['parameters']
                for m in mod['matrices']:
                    if m['role'] != 'weight' or len(m['logical_shape']) != 2: continue
                    if '.ape' in m['tensor_template']: continue
                    if mod['id'] != 'routed': non_expert_linears += m['logical_parameters_each'] * m['count']
                    if node['id'] == 'output-head' and m['logical_shape'] != [c['vocab_size'], c['hidden_size']]:
                        head_control += m['logical_parameters_each'] * m['count']
            if node['group'] != 'decoder': continue
            terms = []
            for mod in node['modules']:
                for m in mod['matrices']:
                    if m['role'] != 'weight' or len(m['logical_shape']) != 2: continue
                    if '.ape' in m['tensor_template']:
                        excluded.append({'tensor': m['tensor_template'], 'reason': '位置偏置表；不是每 token 的矩阵乘法'}); continue
                    terms.append({'tensor': m['tensor_template'], 'shape': m['logical_shape'], 'module': mod['id'],
                        'callsPerToken': mod['selectedCount'] if mod['representative'] else m['count'],
                        'inputGroups': c.get('o_groups', 1) if '.wo_a.weight' in m['tensor_template'] else 1,
                        'proofId': m['proof']})
            selected = next((mod['selectedCount'] for mod in node['modules'] if mod['id'] in ('routed', 'ffn')), 1)
            width = c.get('moe_intermediate_size', c.get('intermediate_size', 0)) if node['ffn'] == 'MoE' else c.get('intermediate_size', 0)
            group = {'type': node['type'], 'ratio': node.get('compressRatio', 0), 'ffn': node['ffn'], 'routing': node.get('routing'),
                'linears': terms, 'selectedExperts': selected, 'intermediate': width}
            key = json.dumps(group, sort_keys=True)
            if key not in groups: groups[key] = dict(group, count=0, nodes=[])
            groups[key]['count'] += 1; groups[key]['nodes'].append(node['id'])
        assert 0 <= non_expert_linears + routed <= arch['mainLogicalParameters'], model['id']
        result.append({'id': model['id'], 'name': model['name'], 'revision': model['revision'], 'kind': kind, 'context': model['facts']['context']['value'],
            'config': c, 'groups': list(groups.values()), 'mainLogicalParameters': arch['mainLogicalParameters'],
            'publishedWeightBytes': arch['fullPublishedPayloadBytes'], 'headControlElements': head_control,
            'defaultLogits': 'last' if kind in ('v4', 'dsa') else 'all',
            'parallelElements': {'replicated': arch['mainLogicalParameters'] - routed - non_expert_linears,
                'nonExpertLinears': non_expert_linears, 'routedExperts': routed},
            'excludedLinearCandidates': list({x['tensor']: x for x in excluded}.values()),
            'architecturePath': model['architecturePath'], 'configPath': model['configPath'], 'storageAuditPath': arch['storageAuditPath'],
            'parameterScope': arch['parameterScope'], 'auxiliaryScope': '主干计算器不自动执行 MTP/DSpark；共享副本、draft 参数及其协议见独立研究与演示。'})
    return result


def parallel_research():
    sources = read(DATA / 'analysis-sources.json')['sources']; by_path = {(s['repo'], s['path']): s for s in sources}
    references = []
    def ref(repo, path, needle, before=2, after=12):
        source = by_path[(repo, path)]; raw = (CACHE / 'deepseek-analysis' / source['cacheFile']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256']
        lines = raw.decode().splitlines(); i = next(i for i, line in enumerate(lines) if needle in line)
        record = reference(source, lines, i + 1 - before, i + 1 + after); references.append(record); return record
    v = 'vllm-project/vllm'; a = 'vllm-project/vllm-ascend'
    topics = [
        {'id': 'groups', 'title': 'TP、DP 与 EP 的进程组', 'summary': '所选 vLLM 开启 EP 后按 TP×DP 构造专家范围；attention 仍在各 DP replica 内按 TP 分片。PP 是另外的层分片维度，本计算器不建模 PP。',
            'refs': [ref(v, 'docs/serving/expert_parallel_deployment.md', 'EP_SIZE ='), ref(v, 'vllm/distributed/parallel_state.py', 'def initialize_model_parallel'), ref(v, 'vllm/config/parallel.py', 'enable_expert_parallel')]},
        {'id': 'weights', 'title': '每 rank 权重与冗余专家', 'summary': '容量公式采用两种明确假设：非专家权重复制，或非专家二维线性权重/embedding/head 理想均分到 TP；向量和位置表复制。具体模型存在复制的低秩投影和独立分派，不将理想均分写成实际 loader 布局。专家及冗余副本按 EP 均分；量化 scale、对齐、workspace、辅助模型另计。',
            'refs': [ref(v, 'docs/serving/expert_parallel_deployment.md', 'Expert Distribution Formula'), ref(v, 'vllm/config/parallel.py', 'num_redundant_experts'), ref(v, 'vllm/distributed/eplb/eplb_state.py', 'class EplbState')]},
        {'id': 'dispatch', 'title': 'MoE dispatch 与 combine', 'summary': '逻辑 token→expert 分配数为 N_global×top-k。发送量按远端路由比例和 hidden 向量字节计算；combine 使用相同宽度假设。每条逻辑 payload 只计发送一次，不再把接收重复相加。全局平均不代表最忙 rank，真实通信还受量化、padding、容量、token 去重及 SP 影响。',
            'refs': [ref(v, 'docs/serving/expert_parallel_deployment.md', 'Backend Selection Guide'), ref(v, 'vllm/distributed/device_communicators/all2all.py', 'class AgRsAll2AllManager'), ref(v, 'vllm/distributed/device_communicators/all2all.py', 'def dispatch(', after=18), ref(v, 'vllm/distributed/device_communicators/all2all.py', 'def combine(', after=10)]},
        {'id': 'collectives', 'title': 'TP collective 与算法边界', 'summary': '展示 ring AllReduce 的每 rank 发送字节公式。每层次数由用户指定，默认只是分析假设；不把 ring 公式推广成实际 HCCL/NCCL 算法或链路时间。AllGather/ReduceScatter 与 all-to-all 是独立算法，backend 名称不证明物理链路选择。',
            'refs': [ref(v, 'vllm/distributed/parallel_state.py', 'def get_tp_group'), ref(a, 'vllm_ascend/distributed/device_communicators/pyhccl.py', 'def all_reduce')]},
        {'id': 'eplb', 'title': 'EPLB 的映射、量化与 runner 条件', 'summary': 'EPLB 基于专家负载调整逻辑专家与物理副本映射，冗余副本消耗额外容量。所选 Ascend 文档明确区分 MRv1 与 MRv2 配置，且列出各量化格式的 Enabled/Rejected 条件；A2 不支持冗余专家的文档声明单独保存，不由其他平台代填。作者验证声明不等于本项目验证。',
            'refs': [ref(v, 'vllm/distributed/eplb/eplb_state.py', 'rebalance_experts'), ref(a, 'docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md', 'two EPLB integration paths', after=20), ref(a, 'docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md', 'Model Runner V2 Weight Formats', after=18)]},
        {'id': 'pd', 'title': 'Prefill/decode 分离与 cache 迁移', 'summary': 'P/D 分离把预填充和生成调度到不同实例，connector 协调 cache 保存/加载及完成信号。逻辑迁移量按有效主 KV 记录计，index cache 可独立列出；压缩状态、格式转换、分片重排、传输副本、网络协议和隐藏特征另计。需要两端布局与位置语义相容，不能只比较总字节。',
            'refs': [ref(v, 'docs/features/disagg_prefill.md', 'Disaggregated'), ref(v, 'vllm/distributed/kv_transfer/kv_connector/v1/base.py', 'def start_load_kv'), ref(v, 'vllm/distributed/kv_transfer/kv_connector/v1/base.py', 'def wait_for_save'), ref(a, 'docs/source/developer_guide/Design_Documents/disaggregated_prefill.md', 'Mooncake')]},
    ]
    return {'topics': topics, 'sources': sources, 'references': references, 'formulas': [
        {'name': 'EP', 'formula': 'EP = TP × DP（所选 vLLM EP 配置）'},
        {'name': '权重容量', 'formula': 'bytes/rank = w × [P_replicated + P_nonexpert / TP + P_expert × (1 + R/E) / EP]；复制假设将第二项除数改为 1'},
        {'name': '远端路由', 'formula': '均匀放置/路由假设：remote_fraction = (EP−1)/EP；也可手填 [0,1]'},
        {'name': 'MoE 逻辑发送', 'formula': '每 MoE 层 dispatch = B×Q×DP×K×H×activation_bytes×remote_fraction；combine 同宽；所有层再乘 MoE 层数'},
        {'name': 'Ring AllReduce', 'formula': '每 rank 每次发送 = 2×(TP−1)/TP × message_bytes；全组发送量再乘 TP'},
        {'name': 'P/D', 'formula': '按一个 DP replica 的有效主 KV 记录计；不含预留窗口空槽、FP32 压缩状态或其他协议数据'},
    ], 'limits': ['公开框架快照没有被组成兼容运行栈。', '容量与通信公式是条件模型，不生成耗时、带宽利用率或真实峰值显存。', '默认路由均匀；真实热专家、重复 token、padding 和拥塞需要额外分布证据。']}


def cross_family():
    catalog = read(ROOT / 'data/catalog.json'); records = []
    for ref in catalog['families']:
        family = read(ROOT / ref['path'])
        for model in family['models']:
            arch = read(ROOT / model['architecturePath']) if model.get('architecturePath') else None
            records.append({'key': family['id']+':'+model['id'], 'familyId': family['id'], 'family': family['name'], 'id': model['id'], 'name': model['name'],
                'revision': model.get('revision'), 'branch': model['branch'], 'summary': model['summary'], 'facts': model['facts'],
                'source': model.get('source'), 'detailPath': '#/family/'+family['id']+'/model/'+model['id'],
                'architecturePath': model.get('architecturePath'), 'configPath': model.get('configPath'), 'auditPath': model.get('auditPath'),
                'parameterScope': arch.get('parameterScope', arch.get('scope', '沿用该版事实来源的统计范围')) if arch else '该条目未提供独立架构表；未知事实保留未知',
                'cache': arch.get('cache') if arch else None,
                'evidenceDepth': '结构表可读；各事实仍按各自证据类别核对' if arch else '资料/历史条目；未补推结构',
                'inputFamilyPath': ref['path'], 'inputModelSha256': hashlib.sha256(json.dumps(model, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()})
    return {'schemaVersion': 1, 'snapshot': max([DATE]+[f['updated'] for f in catalog['families']]), 'families': [{'id': f['id'], 'name': f['name']} for f in catalog['families']],
        'models': records, 'dimensions': ['layers', 'hidden', 'attention', 'heads', 'experts', 'topK', 'shared', 'expertWidth', 'denseWidth', 'context', 'vocab', 'parameters', 'payload', 'quantization', 'mtpParameters', 'dsparkStages', 'engramParameters', 'activePrefillParameters', 'activeDecodeParameters', 'encoderLayers', 'decoderLayers', 'globalCacheBytesPerToken'],
        'limits': ['家族统计口径和读取深度不同；参数、发布载荷和辅助模块范围保留各自说明。', 'null 表示未知或未收录，不代表 0 或无该机制。', '配置相同、参数相近和模型名称相近不证明权重、能力或速度相同。', '不提供缺少统一实测条件的速度、价格或能力排名。']}


def build():
    family = read(DATA/'family.json'); models = cost_models(family); cann = contracts(); parallel = parallel_research()
    proofs = read(DATA/'research/site-proofs.json')['proofs']
    mechanisms = []
    for mid, title, symbols, source_filter, limit in [
        ('mla', 'MLA：展开与吸收', ['DeepseekV3Attention.forward'], lambda p: 'v3-' in p['source_id'], '教学合成矩阵，以参考展开投影为起点做代数吸收；不包含真实权重、低精度误差、位置外推或 kernel 融合。'),
        ('dsa', 'DSA：Indexer 与选择', ['Indexer.forward', 'MLA.forward'], lambda p: 'v3.2' in p['source_id'], 'Indexer 分数与 attention 分数分开；教学选择不证明模型质量。'),
        ('compression', 'V4：分块、overlap 与状态', ['Compressor.forward', 'Attention.forward'], lambda p: 'v4-flash' in p['source_id'] and 'dspark' not in p['source_id'], '固定教学 gating，仅展示分块/overlap/尾部状态；不模拟真实投影、RoPE、量化或全部 cache 管理。'),
        ('mhc', 'mHC：四流与 Sinkhorn', ['Block.hc_pre', 'Block.hc_post'], lambda p: 'v4-flash' in p['source_id'] and 'dspark' not in p['source_id'], 'pre/post 使用指定教学数值，子层为 tanh 示例；comb 用 Sinkhorn 展示行列和，不代表真实控制投影。'),
        ('dspark', 'DSpark：并行草稿与顺序验证', ['DSparkBlock.forward_embed', 'DSparkBlock.forward_head'], lambda p: 'dspark' in p['source_id'], '教学词表、logits、采样与 confidence；不体现真实训练校准或生产 scheduler。checkpoint block、native stage 和服务 token 数仍分别保存。'),
    ]:
        refs = []
        for symbol in symbols:
            candidates = [p for p in proofs.values() if p['symbol'] == symbol and source_filter(p)]
            assert candidates, (mid, symbol)
            refs.append(candidates[0])
        mechanisms.append({'id': mid, 'title': title, 'references': refs, 'limit': limit, 'evidence': 'interpretation'})
    data = {'schemaVersion': 1, 'familyId': 'deepseek', 'snapshot': DATE, 'scope': 'A1–A5 static analysis and teaching tools; no runtime validation',
        'costModels': models, 'excludedModels': [{'id':m['id'],'reason':m['computeScope'],'researchUrl':m.get('researchUrl')} for m in family['models'] if not m.get('computePath')],
        'deploymentModels': [{'id':m['id'],'name':m['name'],'context':m['facts']['context']['value'],
                              'storage':read(ROOT/m['architecturePath'])['deploymentStorage'],'researchUrl':m['researchUrl']} for m in family['models'] if m.get('deploymentAnalysisPath')],
        'cannContracts': cann, 'parallel': parallel, 'mechanisms': mechanisms,
        'costFormulas': [
            {'name': '投影收缩', 'formula': '2×B×Q×Σ(in×out×调用份数)；routed 专家按 top-k，APE 位置表与 embedding lookup 不算矩阵乘法'},
            {'name': 'Dense/GQA attention', 'formula': '2×B×N_head×Q×L_kv×(D_key+D_value)；有效因果位置为 Q×history + Q×(Q+1)/2'},
            {'name': 'DSA', 'formula': 'Indexer = 2×B×Q×L_kv×N_index_head×D_index；参考 dense score/mask 路径与 top-k 有效位置分列'},
            {'name': 'V4', 'formula': 'C=floor(L_kv/r)；每 query 有效位置=min(window,t)+min(index_topk,floor(t/r))（HCA 不设 index_topk）'},
            {'name': 'MLA cache', 'formula': 'latent=B×L_kv×(R_kv+D_rope)；expanded=B×L_kv×N_head×(D_nope+D_rope+D_value)'},
            {'name': 'GQA cache', 'formula': 'B×L_kv×2×N_KV×D_head 个元素'},
            {'name': 'V4 cache/state', 'formula': '主 KV 按 window+C；CSA index cache 独立；每 compressor 的两个 FP32 状态为 2×B×(coff×r)×(coff×D)，coff=2 当 r=4'},
            {'name': '输出', 'formula': 'head=2×B×logits_token_count×V×H；verify 必须选择全部 query logits'},
        ],
        'limits': ['成本使用完整模型、MP=1 的逻辑形状；buffer 按所选 batch 与上下文容量假设，实际 max_batch/max_seq、分页、分片与预留另计。',
            '仅统计声明的矩阵收缩；norm、softmax、RoPE、routing/top-k、Sinkhorn、非线性、量化转换与通信算术另列，不称完整 FLOPs。',
            'V4 槽位乘法是上界，masked 槽位是否跳过依实际 kernel；有效数学位置另列。',
            '参考 logits/score 张量的逻辑大小不证明设备分配；cache 精度是显式假设，FP8/FP4 模拟不当作实际缓存量化。',
            '投影输入+输出字节是逻辑消费代理，重复消费重复计，不作为物理 HBM 流量。',
            '主干与 MTP/DSpark 分开；未测延迟、吞吐、峰值显存和通信耗时始终为 null。'],
        'counts': {'costModels': len(models), 'costLayerGroups': sum(len(m['groups']) for m in models), 'costLayers': sum(g['count'] for m in models for g in m['groups']),
            'cannFamilies': len(cann), 'cannClaims': sum(len(c['claims']) for c in cann), 'codedNecessaryRules': sum(len(c['rules']) for c in cann),
            'parallelSources': len(parallel['sources']), 'parallelTopics': len(parallel['topics']), 'mechanisms': len(mechanisms)}}
    write(DATA/'analysis.json', data)
    family['analysisPath'] = 'data/families/deepseek/analysis.json'; write(DATA/'family.json', family)
    compare = cross_family(); write(ROOT/'data/analysis/cross-family.json', compare)
    catalog = read(ROOT/'data/catalog.json'); catalog['comparisonPath'] = 'data/analysis/cross-family.json'; write(ROOT/'data/catalog.json', catalog)
    write(DATA/'research/analysis-proofs.json', {'snapshot': DATE, 'parallel': parallel['references'], 'contracts': [claim['reference'] for item in cann for claim in item['claims']]})
    write(OUT/'DeepSeek-静态分析.json', data); write(OUT/'跨家族比较.json', compare)
    write_report(data, compare)
    print('Static analysis:', json.dumps(data['counts'], ensure_ascii=False), '; cross-family models:', len(compare['models']))
    return data


def write_report(data, compare):
    def table(headers, rows):
        clean = lambda x: str(x).replace('|', '\\|').replace('\n', ' ')
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] + ['| '+' | '.join(clean(x) for x in r)+' |' for r in rows])
    report = ['# DeepSeek 静态成本、算子契约与系统分析', '资料更新：'+max(data['snapshot'],compare['snapshot'])+'。21 个 checkpoint 的计算与并行模型、V4.1 部署缓存模型分别说明；设备实验不属于当前范围。',
        '## 1 成本模型', '覆盖 21 个 checkpoint 的全部 1,122 个主干层。参数来自 canonical 架构和各自完整文件头；计算器按声明矩阵收缩计数，主干、共享副本和 MTP/DSpark 独立。',
        table(['项目', '公式'], [(x['name'], x['formula']) for x in data['costFormulas']]),
        '\n'.join('- '+s for s in data['limits']),
        '成本表在 [静态分析工作簿](DeepSeek-静态分析.xlsx) 与 [示例 CSV](DeepSeek-cost-scenarios.csv) 中。网页可修改 batch、query/history、logits 和精度假设；每个结果保留输入，不写入未测时间或吞吐。']
    for model in data.get('deploymentModels',[]):
        s=model['storage']
        window=s['languageLayers']*s['window']*s['swaRecordBytes']
        draft=s['draftStages']*s['window']*s['swaRecordBytes']
        report += ['## 2 '+model['name']+' 部署缓存',
            '按真实 FP4/FP8 数据与块 scale 的打包容量统计。主 KV 为 NVFP4-like 变体，E2M1 数据加每 16 元素 E4M3 scale，省略第二级全局 scale；窗口为 MXFP8，E4M3 数据加每 32 元素 E8M0 scale；index 为 MXFP4，E2M1 数据加每 32 元素 E8M0 scale。',
            table(['项目','每请求容量 / 字节'],[
                ('全局 KV 与 index','(3 × floor(N/2) + N) × '+str(s['mainRecordBytes']+s['indexRecordBytes'])+'；偶数 N 为 890N'),
                ('Main KV 记录',s['mainRecordBytes']),('Index K 记录',s['indexRecordBytes']),('窗口 KV 记录',s['swaRecordBytes']),
                ('40 层 × 128 token 窗口',window),('3-stage DSpark context（可选）',draft),('FP32 压缩状态',s['compressionStateBytesPerRequest'])]),
            'N 为已缓存序列长度，batch B 按请求数相乘。B=1、N=1,048,576 时，全局缓存为 890 MiB；加语言窗口、DSpark context 和压缩状态约为 892.795 MiB。权重、分页对齐、临时工作区、候选池/分数张量和通信缓冲另计。',
            '参考代码的量化/反量化模拟与实际打包容量分开说明。V4.1 的 FLOPs 和并行通信未沿用旧 21 版公式；完整结构与边界见 [V4.1 Flash 研究](DeepSeek-V4.1-Flash-研究.md)，网页入口为 `#/family/deepseek/cost/v4.1-flash`。']
    report += ['## 3 CANN 算子契约', f'覆盖 {data["counts"]["cannFamilies"]} 个已固定源码家族、{data["counts"]["cannClaims"]} 条 API 参数表/补充条件/API guard 记录。参数表逐项保存所在文档和源码范围，不把同文档中不同平台/模式的条件合成无条件支持。',
        '网页有按 API 文档、参数和类别筛选的手册，以及已编码必要条件的检查器。检查器发现矛盾时指出条件；未发现矛盾只表示该子集没有冲突，其他条件保持未核对。',
        '下表用于定位算子家族。全部条件、章节上下文与固定行号见 [完整条件 CSV](DeepSeek-CANN-contracts.csv)、[静态分析 JSON](DeepSeek-静态分析.json) 和工作簿“算子条件”页。',
        table(['算子家族','API 文档数','条件数','框架上下文','固定来源'],[
            (c['id'],len(c['documents']),len(c['claims']),c['frameworkContext'],
             '；'.join('['+d['name']+']('+d['url']+')' for d in c['documents']) or '['+c['claims'][0]['apiDocument']+']('+c['claims'][0]['reference']['url']+')')
            for c in data['cannContracts']])]
    report += ['## 4 并行、通信与 P/D', table(['项目', '公式与条件'], [(f['name'], f['formula']) for f in data['parallel']['formulas']])]
    for topic in data['parallel']['topics']:
        report += ['### '+topic['title'], topic['summary'], '来源：'+ '；'.join('['+r['path']+']('+r['url']+')' for r in topic['refs'])]
    report += ['## 5 跨家族统一比较', f'沿用现有三个家族的 {len(compare["models"])} 个条目，包含历史/未核对条目；网页可按家族筛选、添加版本，并保留每个事实的证据类别和来源。',
        '\n'.join('- '+s for s in compare['limits']),
        '## 6 机制演示']
    for m in data['mechanisms']:
        report += ['### '+m['title'], m['limit'], '来源：'+ '；'.join('['+r['symbol']+']('+r['url']+')' for r in m['references'])]
    report += ['演示包含前后步进和可调输入。MLA 比较两条路径的分数和输出；DSA 分开 indexer 分数与 attention 分数；压缩保留未满窗口状态；mHC 展示 comb 的行列和；DSpark 提交接受前缀及校正/bonus，丢弃未验证草稿。',
        '## 7 复现与验证', '运行 collect_deepseek_analysis.py 获取固定公开输入，--verify-only 仅核对缓存；运行 build_deepseek_analysis.py 从既有 canonical 数据生成分析 JSON、手册和跨家族表。V4.1 输入由 build_deepseek_v41.py 固定并恢复。validate_deepseek_analysis.py 与 check_deepseek_analysis.mjs 分别验证来源/字段和独立数学基准，validate_deepseek_v41.py 核对本版结构和打包公式，check_deepseek_site.mjs 验证网页交互与下载。',
        '普通校验仅输出终端结果，浏览器截图使用临时目录；只有明确传入 --write-record 才保存校验记录。重建分析工作簿前用 check_deepseek_analysis.mjs --export-scenarios 生成当前导出输入。完整命令和当前附件覆盖范围见 [复现说明](README.md)。设备实验不属于本次验收。']
    (OUT/'DeepSeek-静态分析.md').write_text('\n\n'.join(report)+'\n')


if __name__ == '__main__':
    build()
