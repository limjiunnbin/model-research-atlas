"""Add the fixed V4.1 Flash config/reference study; keep upstream inputs outside Git."""
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
OUT = ROOT / 'dist/assets/deepseek'
CACHE = Path.home() / '.cache/model-research-atlas/deepseek-v41'
REVISION = '2cba9e42aa026125f3ed06c6d98c1db82f7ca027'
MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
ID = 'v4.1-flash'
DATE = '2026-10-08'
HASHES = {
    'README.md': '347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251',
    'config.json': '8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879',
    'inference/config.json': '2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809',
    'inference/model.py': '4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65',
    'inference/vision.py': '5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c',
    'inference/engram.py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897',
    'inference/kernel.py': '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455',
    'inference/README.md': '2834402823199ee24e9a42bdf36a0fc6daf94448f444cb062c042a057a798f1c',
    'DeepSeek_V41_Tech_Report.pdf': 'ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d',
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def url(path):
    return f'https://huggingface.co/{MODEL}/blob/{REVISION}/{path}'


def inputs(fetch=False):
    for name, expected in HASHES.items():
        path = CACHE / name
        if not path.is_file() and fetch:
            request = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true'
            raw = urllib.request.urlopen(request, timeout=60).read()
            assert hashlib.sha256(raw).hexdigest() == expected, name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        assert path.is_file(), f'Missing cached input {name}; run with --fetch.'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


def architecture():
    cfg = read(CACHE / 'config.json')
    c, v = cfg['text_config'], cfg['vision_config']
    native = read(CACHE / 'inference/config.json')
    for left, right in [('hidden_size', 'dim'), ('num_hidden_layers', 'n_layers'),
                        ('num_nextn_predict_layers', 'n_mtp_layers'),
                        ('kv_source_layer_ids', 'kv_source_layers'),
                        ('index_source_layer_ids', 'index_source_layers')]:
        assert c[left] == native[right], (left, right)
    assert c['compress_ratios'] == native['compress_ratios']
    proofs = {}

    def proof(path, symbol):
        key = path + ':' + symbol
        if key not in proofs:
            text = (CACHE / path).read_text()
            nodes = {}
            for cls in ast.parse(text).body:
                if isinstance(cls, ast.ClassDef):
                    for fn in cls.body:
                        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            nodes[cls.name + '.' + fn.name] = fn
            fn = nodes[symbol]
            body = '\n'.join(text.splitlines()[fn.lineno-1:fn.end_lineno])
            proofs[key] = {'path': path, 'symbol': symbol, 'revision': REVISION,
                           'line': fn.lineno, 'end': fn.end_lineno,
                           'sourceSha256': HASHES[path],
                           'functionSha256': hashlib.sha256(body.encode()).hexdigest(),
                           'url': url(path) + f'#L{fn.lineno}-L{fn.end_lineno}'}
        return key

    def mat(name, dims, symbol, count=1, path='inference/model.py', lookup=False):
        pid = proof(path, symbol)
        return {'tensor_template': name, 'logical_shape': dims,
                'logical_parameters_each': math.prod(dims), 'count': count,
                'role': 'weight', 'evidence': 'derived', 'proof': pid,
                'source': proofs[pid]['url'], 'stored_shape': None,
                'stored_dtype': None, 'payloadBytes': None, 'lookup': lookup}

    def mod(mid, title, tensors, topic, count=1, selected=1):
        return {'id': mid, 'title': title, 'matrices': tensors, 'count': count,
                'selectedCount': selected, 'representative': count > 1,
                'parameters': sum(t['logical_parameters_each']*t['count'] for t in tensors),
                'implementationTopic': topic}

    H, Q, N, D, J = (c[k] for k in ['hidden_size', 'q_lora_rank', 'num_attention_heads', 'head_dim', 'moe_intermediate_size'])
    HC, V = c['hc_mult'], c['vocab_size']
    KV = set(c['kv_source_layer_ids'])
    INDEX = set(c['index_source_layer_ids'])
    nodes = []
    for i in range(43):
        draft = i >= 40
        si = i-40 if draft else i
        prefix = ('mtp.' if draft else 'layers.') + str(si) + '.'
        E, K = (c['dspark_n_routed_experts'], c['dspark_num_experts_per_tok']) if draft else (c['n_routed_experts'], c['num_experts_per_tok'])
        mode = 'SWA' if c['compress_ratios'][i] == 0 else 'Full' if i in KV else 'Reindex' if i in INDEX else 'Reuse'
        topic = 'v41-dspark' if draft else 'v41-csa2'
        attn = [mat(prefix+'attn.'+name, dims, 'Attention.__init__') for name, dims in [
            ('attn_sink', [N]), ('wq_a.weight', [Q,H]), ('q_norm.weight', [Q]),
            ('wq_b.weight', [N*D,Q]), ('wkv.weight', [D,H]), ('kv_norm.weight', [D]),
            ('wo_a.weight', [c['o_groups']*c['o_lora_rank'],N*D//c['o_groups']]),
            ('wo_b.weight', [H,c['o_groups']*c['o_lora_rank']])]]
        modules = [mod('attention', '本层 Q、SWA KV 与分组输出投影', attn, topic)]
        if i in KV:
            tensors = [mat(prefix+'attn.compressor.wkv.weight', [D,H], 'Compressor.__init__'),
                       mat(prefix+'attn.compressor.norm.weight', [D], 'Compressor.__init__')]
            if c['compress_ratios'][i] > 1:
                tensors.append(mat(prefix+'attn.compressor.wgate.weight', [D,H], 'Compressor.__init__'))
            modules.append(mod('compressor', '本 Full 层拥有的全局 KV', tensors, 'v41-csa2'))
        if i in INDEX:
            tensors = [mat(prefix+'attn.indexer.'+name, dims, 'Indexer.__init__') for name, dims in [
                ('wq_b.weight', [c['index_n_heads']*c['index_head_dim'],Q]),
                ('weights_proj.weight', [c['index_n_heads'],H])]]
            if i in KV:
                tensors += [mat(prefix+'attn.indexer.'+name, dims, 'Indexer.__init__') for name, dims in [
                    ('wk.weight',[c['index_head_dim'],D]), ('k_norm.weight',[c['index_head_dim']])]]
            modules.append(mod('indexer', '本层重算 Top-K；K 只由 Full 层持有', tensors, 'v41-csa2'))
        router = [mat(prefix+'ffn.gate.'+name, dims, 'Gate.__init__') for name, dims in [
            ('weight',[E,H]), ('bias',[E]), ('bias_vl',[E])]]
        expert = [('w1.weight',[J,H]), ('w3.weight',[J,H]), ('w2.weight',[H,J])]
        modules += [mod('router', '文本/视觉分开的路由校正 bias', router, topic),
                    mod('routed', f'{E} 个路由专家 / 选 {K}', [mat(prefix+'ffn.experts.{e}.'+n,d,'Expert.__init__',E) for n,d in expert], topic,E,K),
                    mod('shared', '一个共享 SwiGLU 专家', [mat(prefix+'ffn.shared_experts.'+n,d,'Expert.__init__') for n,d in expert], topic)]
        mixing = (2+HC)*HC
        mhc = [mat(prefix+n,d,'Block.__init__') for n,d in [
            ('attn_norm.weight',[H]), ('ffn_norm.weight',[H]),
            ('hc_attn_fn',[mixing,HC*H]), ('hc_ffn_fn',[mixing,HC*H]),
            ('hc_attn_base',[mixing]), ('hc_ffn_base',[mixing]),
            ('hc_attn_scale',[3]), ('hc_ffn_scale',[3])]]
        modules.append(mod('mhc', 'Single-Pass mHC：使用上一子层的 pre_mix', mhc, 'v41-mhc'))
        if i in c['engram_layer_ids']:
            rows = c['engram_num_embeddings'][c['engram_layer_ids'].index(i)]
            tensors = [mat(prefix+'engram.embed.weight',[rows,c['engram_head_dim']],'ParallelEngramEmbedding.__init__',lookup=True),
                       mat(prefix+'engram.wkv.weight',[(HC+1)*H,(c['engram_max_ngram_size']-1)*c['engram_n_heads']*c['engram_head_dim']],'Engram.__init__'),
                       mat(prefix+'engram.q_weight',[HC,H],'Engram.__init__'),
                       mat(prefix+'engram.k_weight',[HC,H],'Engram.__init__')]
            modules.append(mod('engram', '条件内存：n-gram 哈希查表与上下文门控', tensors, 'v41-engram'))
        if draft and si == 0:
            modules.append(mod('context', '目标层 37/38/39 的 attention 输入均值拼接', [
                mat(prefix+'main_proj.weight',[H,3*H],'DSparkBlock.__init__'),
                mat(prefix+'main_norm.weight',[H],'DSparkBlock.__init__')], topic))
        if draft and si == 2:
            R = c['dspark_markov_rank']
            modules.append(mod('draft-heads', 'Markov 修正与 confidence（目标 LM head 共享）', [
                mat(prefix+'norm.weight',[H],'DSparkBlock.__init__'),
                mat(prefix+'markov_head.embed.weight',[V,R],'DSparkMarkovHead.__init__',lookup=True),
                mat(prefix+'markov_head.head.weight',[V,R],'DSparkMarkovHead.__init__'),
                mat(prefix+'confidence_head.proj.weight',[1,H+R],'DSparkConfidenceHead.__init__')], topic))
        nodes.append({'id': f'dspark-{si+1}' if draft else f'decoder-{i+1}',
                      'group': 'dspark' if draft else 'decoder', 'number': 0 if draft else i+1,
                      'sourceIndex': si, 'stage': 'draft' if draft else 'encoder' if i<20 else 'decoder',
                      'type': 'SWA' if mode=='SWA' else 'CSA2', 'attentionMode': mode,
                      'compressRatio': c['compress_ratios'][i], 'ffn': 'MoE',
                      'experts': E, 'topK': K, 'shared': 1, 'modules': modules,
                      'cacheKey': 'dspark' if draft else 'swa' if mode=='SWA' else 'csa2',
                      'implementationTopic': topic, 'payloadBytes': None})
    W, I = v['hidden_size'], v['intermediate_size']
    for i in range(v['num_hidden_layers']):
        prefix = f'vision.blocks.{i}.'
        fields = [('attn.wqkv.weight',[3*W,W],'Attention.__init__'),('attn.wqkv.bias',[3*W],'Attention.__init__'),
                  ('attn.wo.weight',[W,W],'Attention.__init__'),('attn.wo.bias',[W],'Attention.__init__'),
                  ('mlp.w1.weight',[2*I,W],'MLP.__init__'),('mlp.w2.weight',[W,I],'MLP.__init__'),
                  ('norm1.weight',[W],'Block.__init__'),('norm2.weight',[W],'Block.__init__')]
        nodes.append({'id':f'vision-{i+1}','group':'vision','number':i+1,'type':'Vision','ffn':'Dense',
                      'modules':[mod('vision','2D RoPE / 双向 attention / SwiGLU',[
                          mat(prefix+n,d,s,path='inference/vision.py') for n,d,s in fields],'v41-vision')],
                      'cacheKey':'vision','implementationTopic':'v41-vision','payloadBytes':None})
    globals_ = [
        ('embedding', [('embed.weight',[V,H],'Transformer.__init__')]),
        ('output-head', [('norm.weight',[H],'Transformer.__init__'),('head.weight',[V,H],'ParallelHead.__init__')]),
        ('vision-input', [('vision.patch_embed.proj.weight',[W,3*v['patch_size']**2],'PatchEmbed.__init__'),
                          ('vision.patch_embed.proj.bias',[W],'PatchEmbed.__init__'),('vision.norm.weight',[W],'ViT.__init__'),
                          ('aligner.w1.weight',[H,9*W],'Aligner.__init__'),('aligner.w1.bias',[H],'Aligner.__init__'),
                          ('aligner.w2.weight',[H,H],'Aligner.__init__'),('aligner.w2.bias',[H],'Aligner.__init__')]),
        ('image-delimiters', [(n,[H],'Transformer.__init__') for n in ['image_start','image_end','image_newline']]),
    ]
    for gid, fields in globals_:
        vision = gid == 'vision-input'
        tensors = [mat(n,d,s,path='inference/vision.py' if vision else 'inference/model.py',lookup=gid=='embedding') for n,d,s in fields]
        nodes.append({'id':gid,'group':'global','number':0,'type':gid,'ffn':None,
                      'modules':[mod(gid,gid,tensors,'v41-vision' if vision or gid=='image-delimiters' else 'v41-csa2')],
                      'payloadBytes':None})
    for n in nodes:
        n['parameters'] = sum(m['parameters'] for m in n['modules'])
        n['activeLinearParameters'] = sum(t['logical_parameters_each']*(m['selectedCount'] if m['representative'] else t['count'])
                                          for m in n['modules'] for t in m['matrices'] if len(t['logical_shape'])==2 and not t['lookup'])
        if n['id']=='embedding': n['activeLinearParameters'] = None
    eng = sum(m['parameters'] for n in nodes for m in n['modules'] if m['id']=='engram')
    vision = sum(n['parameters'] for n in nodes if n['group']=='vision' or n['id'] in ('vision-input','image-delimiters'))
    backbone = sum(n['parameters'] for n in nodes if n['group']!='dspark') - eng - vision
    arch = {'schemaVersion':1,'modelId':ID,'label':'DeepSeek-V4.1-Flash','evidence':'derived',
            'scope':'固定 HF/native 配置与可读参考构造；MP=1 逻辑矩阵；未审计本版分片 header，存储/载荷保持未知；论文生产路径与可读实现分开。',
            'source':url('config.json'),'referenceSource':url('inference/model.py'),'nodes':nodes,
            'config':{'hidden':H,'heads':N,'experts':384,'topK':6,'shared':1,'context':c['max_position_embeddings'],'streams':4,
                      'precision':'配置 FP8 + FP4 routed experts；逻辑形状来自参考构造，实际 stored shape/dtype/payload 未审计'},
            'cache':{
                'swa':'40 个主干层各有 [B,128,512] 局部 KV；真实部署按 MXFP8：E4M3 数据 + 每 32 通道一个 E8M0 scale，每记录 528 B；完整窗口合计 2703360×B B。活跃窗口不长期落 SSD，bounded replay 为近似恢复。',
                'csa2':'只统计 Full 层 0-based 2/8/14/20：前三个各 floor(N/2) 条、最后一个 N 条。每条 main KV=288 B，index K=68 B；打包全局缓存=(3×floor(N/2)+N)×356 B，N 为偶数时=890N B。不含窗口、FP32 compressor 状态、分页/对齐和 DSpark。Reindex/Reuse 不另存同一全局 cache。真实部署按上述数据与 scale 打包容量；参考模拟不用于容量计算。',
                'dspark':'三个 draft stage，各有独立 [B,128,512] 目标 context 窗口；prefill 写入 context，draft forward 五位置并行，Markov 顺序修正；不能并入目标主干缓存。',
                'vision':'单图双向 ViT；32 层，L_img patches 的 Q/K/V 与 attention 中间量；3×3 pixel-unshuffle 后约 ceil(h/3)×ceil(w/3) 个视觉 token；不分配自回归语言 KV。'},
            'mainLogicalParameters':backbone,'mtpLogicalParameters':None,
            'engramLogicalParameters':eng,'visionLogicalParameters':vision,
            'dsparkLogicalParameters':sum(n['parameters'] for n in nodes if n['group']=='dspark'),
            'parameterScope':'主干为参考构造的语言权重（不含 Engram、ViT/projector、DSpark、scale、非训练哈希/词表映射）；官方约 552B 主干与约 196B Engram 分列。DSpark embed/head 引用共享对象，未重复计数；逻辑构造不证明发布文件的别名或实际载荷。',
            'referenceProofs':proofs,'revision':REVISION,
            'cacheFormats':{'main':'NVFP4 风格变体：E2M1 / E4M3 scale 每16通道，省略第二级 global scale',
                            'swa':'MXFP8：E4M3 / E8M0 scale 每32通道',
                            'index':'MXFP4：E2M1 / E8M0 scale 每32通道',
                            'scope':'部署容量按真实打包数据与 scale；reference inplace 模拟只作为实现差异'},
            'deploymentStorage':{'mainRecordBytes':288,'indexRecordBytes':68,'swaRecordBytes':528,
                                 'fullOwners':[{'layer':i,'ratio':c['compress_ratios'][i]} for i in sorted(KV)],
                                 'languageLayers':40,'draftStages':3,'window':128,'compressionStateBytesPerRequest':24576},
            'sources':[{'path':p,'revision':REVISION,'sha256':h,'url':url(p)} for p,h in HASHES.items()]}
    return cfg, arch


def install(cfg, arch):
    write(DATA/'configs/DeepSeek-V4.1-Flash-config.json',cfg)
    write(DATA/(ID+'-architecture.json'),arch)
    c,v = cfg['text_config'],cfg['vision_config']
    family = read(DATA/'family.json')
    facts = {k:{'value':None,'evidence':'unknown','source':None} for k in family['models'][0]['facts']}
    def fact(key,value,source=None,evidence='official',note=None):
        facts[key]={'value':value,'evidence':evidence,'source':source or url('config.json')}
        if note: facts[key]['note']=note
    for key,value in {'layers':40,'hidden':5120,'heads':64,'experts':384,'topK':6,'shared':1,
                      'expertInput':5120,'expertWidth':2304,'context':1048576,'vocab':129280,
                      'visionLayers':32,'visionHidden':1024,'qRank':1280,'streams':4,
                      'headDim':512,'valueHeadDim':512,'indexTopK':512,'dsparkStages':3,
                      'draftBlock':5,'markovRank':256}.items(): fact(key,value)
    fact('parameters',552_000_000_000,url('DeepSeek_V41_Tech_Report.pdf'),note='官方约数：主干 552B，另有约 196B Engram；不等于全部发布张量/辅助模块之和。精确参考构造计数见结构 JSON。')
    fact('engramParameters',196_000_000_000,url('DeepSeek_V41_Tech_Report.pdf'),note='官方约数；两处条件内存，查表不是 dense GEMM。')
    fact('activePrefillParameters',8_000_000_000,url('DeepSeek_V41_Tech_Report.pdf'),note='论文 CED + bounded replay 部署口径；窗口回放另有固定成本。')
    fact('activeDecodeParameters',16_000_000_000,url('DeepSeek_V41_Tech_Report.pdf'))
    fact('encoderLayers',20,url('DeepSeek_V41_Tech_Report.pdf'))
    fact('decoderLayers',20,url('DeepSeek_V41_Tech_Report.pdf'))
    fact('globalCacheBytesPerToken',890,url('DeepSeek_V41_Tech_Report.pdf'),note='全局打包缓存，排除 SWA、临时状态、分页对齐与 DSpark；有限长度用 floor(N/2) 公式。')
    fact('attention','CED 20+20；SWA 2 / CSA2 Full 4 / Reindex 4 / Reuse 30',evidence='derived')
    fact('quantization','配置 FP8 / UE8M0 32×32 scales；routed experts FP4；global KV 为省略 global scale 的 NVFP4 变体（E2M1+E4M3/16）；index K MXFP4/32；SWA MXFP8（E4M3+E8M0/32）；实际存储未审计')
    fact('dsparkParameters',arch['dsparkLogicalParameters'],url('inference/model.py'),evidence='derived',note='参考构造；排除共享目标 embedding/head，未核对发布存储副本。')
    model = {'id':ID,'name':'DeepSeek-V4.1-Flash','branch':'主线','summary':'约 552B 主干 + 196B Engram；原生视觉；CED 20+20 层，CSA2 跨层共享、Single-Pass mHC 与 3-stage DSpark。',
             'tags':['长文档','通用对话','视觉理解','代码与代理'],'source':f'https://huggingface.co/{MODEL}/tree/{REVISION}',
             'revision':REVISION,'releaseDate':'2026-09-10','releaseDateSource':'https://api-docs.deepseek.com/zh-cn/updates/',
             'facts':facts,'sections':['v41'],'configPath':'data/families/deepseek/configs/DeepSeek-V4.1-Flash-config.json',
             'architecturePath':f'data/families/deepseek/{ID}-architecture.json','auditPath':None,'baseModel':None,
             'computeScope':'本版部署缓存按真实 NVFP4 变体 / MXFP4 / MXFP8 打包格式独立计算；FLOPs 与通信不套用旧 21-model 的 V4 路径。',
             'deploymentAnalysisPath':'#/family/deepseek/cost/'+ID,
             'researchUrl':'document.html?file=assets/deepseek/DeepSeek-V4.1-Flash-研究.md',
             'implementationLinks':[{'topic':t,'title':title,'path':f'#/family/deepseek/implementation/{t}/{ID}'} for t,title in [
                 ('v41-csa2','CED / CSA2 与 bounded replay'),('v41-engram','Engram 条件内存'),('v41-mhc','Single-Pass mHC'),
                 ('v41-dspark','V4.1 DSpark 与调度边界'),('v41-vision','原生视觉、ViT 与 projector')]]}
    family['models']=[m for m in family['models'] if m['id']!=ID]+[model]
    family.update(updated=DATE,description='MLA/DSA、V4 压缩注意力，以及 V4.1 Flash 的 CED、CSA2、Engram、原生视觉与 DSpark。',
                  scope='22 个代表 checkpoint：21 版完整文件头/计算研究，V4.1 Flash 为独立配置、固定源码与论文研究；设备实验不在本次范围。',
                  historyNote='架构与训练关系不证明相邻 checkpoint 直接继承权重。V4.1 Flash 已作为独立 CED/CSA2/Engram/视觉条目纳入；Max/1–100 reasoning effort 是推理设置。',
                  auditNote='21 个既有 checkpoint 完整审计 header/data_offsets；新增 V4.1 Flash 未做权重文件头审计，stored shape/dtype/payload 均为空。')
    family['overview']['conclusion']='V3.2 增加 DSA，V4 重建压缩与 mHC；V4.1 Flash 改为 CED、CSA2 缓存/索引共享和 Single-Pass mHC，加入 Engram 与原生视觉。主干、条件内存、草稿模块、论文部署与可读实现分列。'
    family['overview']['title']='从 MLA 到跨层缓存共享、条件内存与原生视觉'
    family['overview']['highlights']=[x for x in family['overview']['highlights'] if x['id']!=ID]+[{'id':ID,'label':'V4.1 Flash · CED / CSA2 / Engram / 视觉'}]
    family['technology']=[x for x in family['technology'] if x['id']!='v41']+[{'id':'v41','title':'V4.1 Flash','section':'v41','summary':'因果 Encoder–Decoder、跨层 KV/Top-K 共享、Engram、视觉与 Single-Pass mHC。'}]
    write(DATA/'family.json',family)
    catalog = read(ROOT/'data/catalog.json')
    ref = next(f for f in catalog['families'] if f['id']=='deepseek')
    ref.update(updated=DATE,description=family['description'],modelCount=len(family['models']),
               counts=['22 个代表 checkpoint','21 版文件头审计；V4.1 配置/源码','CED / CSA2 / Engram / 视觉 / DSpark'])
    write(ROOT/'data/catalog.json',catalog)
    versions = read(DATA/'versions.json')
    versions['models']=[m for m in versions['models'] if m['modelId']!=MODEL]+[{
        'name':model['name'],'modelId':MODEL,'revision':REVISION,'url':model['source'],
        'summary':model['summary'],'kind':'multimodal_causal_encoder_decoder','baseModel':None,
        'evidence':'official','verifiedTo':'fixed_config+native_reference+technical_report',
        'configVerified':True,'materials':{'modelCard':url('README.md'),'config':url('config.json'),
                                         'referenceCode':[url('inference/model.py'),url('inference/vision.py'),url('inference/engram.py')],
                                         'papers':[url('DeepSeek_V41_Tech_Report.pdf')]}}]
    versions['deferred']=[x for x in versions['deferred'] if x!=MODEL]
    versions['snapshot']=DATE
    versions['limit']='22 个选定代表 checkpoint；V4.1 为配置/源码/论文研究，另 21 版有完整文件头；非全量官方产品目录。'
    write(DATA/'versions.json',versions)
    report = read(DATA/'report.json')
    section = {'id':'v41','title':'12. V4.1 Flash：CED、CSA2、Engram 与原生视觉',
        'html':'<h2>V4.1 Flash 采用独立的多模态架构</h2><p>40 层语言主干分成 20 层因果 encoder 和 20 层 decoder，另有 32 层 ViT。官方披露约 552B 主干与 196B Engram；prefill/decode 激活约 8B/16B。配置为 384 选 6 MoE，一个共享专家，三层 DSpark 则为 128 选 3。</p><p>CSA2 在 0-based 2/8/14/20 四个 Full 层持有全局 KV；24/28/32/36 为 Reindex，其余 30 个 CSA2 层 Reuse；最前两层仅 SWA。前三份缓存 ratio=2，decoder 缓存 ratio=1。FP4 main KV 每条 288 B、index K 每条 68 B，打包全局缓存为 (3×floor(N/2)+N)×356 B，偶数长度时为 890N B；窗口、压缩状态与临时张量另计。</p><p>Single-Pass mHC 使用上一子层的 input-mixing 系数；Engram 位于 0-based 1/14，以 2/3/4-gram 哈希查表加入条件内存；ViT 用 2D RoPE 和 3×3 pixel-unshuffle。DSpark 五位置并行 logits、顺序 Markov 与 confidence 继续分开。</p><p>论文的 bounded replay 是近似状态恢复。可读参考仍对 prefill 遍历全部 40 层，先完整打分再掩 candidate，并用默认 dtype cache 模拟 FP4；不等于优化的生产 CED、稀疏候选打分或 Mega-mHC。本版提供按真实打包 FP4/FP8 格式计算的部署缓存页面；FLOPs 和通信不套用旧 21-model 的公式。本站未做本版权重头审计或设备运行。</p><p><a href="document.html?file=assets/deepseek/DeepSeek-V4.1-Flash-研究.md">阅读全文、训练与评测条件、版本比较和 Ascend 静态分析 →</a></p>'}
    write(DATA/'report.json',[s for s in report if s['id']!='v41']+[section])
    install_hardware()
    from build_deepseek_analysis import cross_family, write_report
    comparison = cross_family()
    comparison['snapshot']=DATE
    for key in ['engramParameters','activePrefillParameters','activeDecodeParameters','encoderLayers','decoderLayers','globalCacheBytesPerToken']:
        if key not in comparison['dimensions']:comparison['dimensions'].append(key)
    write(ROOT/'data/analysis/cross-family.json',comparison)
    write(OUT/'跨家族比较.json',comparison)
    analysis = read(DATA/'analysis.json')
    analysis['excludedModels']=[{'id':ID,'reason':model['computeScope'],'researchUrl':model['researchUrl']}]
    analysis['deploymentModels']=[{'id':ID,'name':model['name'],'context':c['max_position_embeddings'],
                                  'storage':arch['deploymentStorage'],'researchUrl':model['researchUrl']}]
    write(DATA/'analysis.json',analysis)
    write(OUT/'DeepSeek-静态分析.json',analysis)
    write_report(analysis,comparison)
    shutil.copyfile(ROOT/'research/deepseek/v4.1-flash.md',OUT/'DeepSeek-V4.1-Flash-研究.md')
    main_report=OUT/'DeepSeek-研究报告.md'
    marker='\n## 12. V4.1 Flash'
    original=main_report.read_text().split(marker)[0].rstrip()
    body=(ROOT/'research/deepseek/v4.1-flash.md').read_text().split('\n',1)[1]
    body='\n'.join('#'+line if line.startswith('## ') else line for line in body.splitlines())
    main_report.write_text(original+marker+'：CED、CSA2、Engram 与原生视觉\n'+body+'\n')
    roadmap=OUT/'DeepSeek-roadmap.md'
    heading=(ROOT/'roadmap_deepseek.md').read_text().split('\n## 1',1)[0].replace('research/deepseek/v4.1-flash.md','DeepSeek-V4.1-Flash-研究.md')
    roadmap.write_text(heading+'\n## 1'+roadmap.read_text().split('\n## 1',1)[1])
    csv_path = OUT/'DeepSeek-V4.1-Flash-逻辑矩阵.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(['组件','阶段/子系统','层型','CSA2 模式','模块','逻辑张量名','逻辑形状','单份参数','份数','来源','存储/载荷范围'])
        for n in arch['nodes']:
            for module in n['modules']:
                for t in module['matrices']:
                    w.writerow([n['id'],n.get('stage',n['group']),n['type'],n.get('attentionMode',''),module['title'],t['tensor_template'],
                                ' × '.join(map(str,t['logical_shape'])),t['logical_parameters_each'],t['count'],t['source'],
                                '仅参考逻辑构造；stored shape/dtype/payload 未审计'])
    update_downloads()


def install_hardware():
    h = read(DATA/'hardware.json')
    sources = [('v41-model','inference/model.py'),('v41-vision','inference/vision.py'),('v41-engram','inference/engram.py'),('v41-paper','DeepSeek_V41_Tech_Report.pdf')]
    for sid,path in sources:
        h['sources']=[s for s in h['sources'] if s['id']!=sid]+[{'id':sid,'title':'DeepSeek-V4.1-Flash / '+path,
            'url':url(path),'revision':REVISION,'sha256':HASHES[path],'accessed':DATE,'kind':'technical_report' if path.endswith('.pdf') else 'reference_code'}]
    entries = [
        ('v41-csa2','CED、CSA2 与 bounded replay',['Encoder 20 层','四份 Full cache','Reindex / Reuse','Decoder query + 局部 SWA'],
         'Full 4、Reindex 4、Reuse 30；全局缓存与本层 SWA 分开，候选池上限 16384、最终 top-k 512。',
         'Attention.__init__/_compress_kv 与 SharedAttentionRuntime 保留共享关系；可读 Indexer.forward 先全量 einsum 再 mask，Transformer.forward prefill 遍历全部 40 层。',
         '移植时需要真正按候选位置 gather/score、共享 cache 生命周期与近似 replay；旧 MLA/CSA 的接口或 kernel 名称不证明 CSA2 整体支持。'),
        ('v41-engram','Engram 条件内存',['Token 压缩与 2/3/4-gram','24 个哈希地址','FP8 查表与解量化','上下文门控加到四流 residual'],
         '0-based 1/14 两个模块；约 196B 条件内存，查表容量、访存与投影计算分列。',
         'ParallelEngramEmbedding.forward 本地查表、解量化并 all_reduce；视觉位置排除 n-gram；可读参考没有论文的 host RDMA 后台预取。',
         'NPU 研究涉及 Gather/Embedding、块缩放、投影、哈希状态和跨节点搬运；普通 Gather 可用不代表 Engram 的地址/预取/容量方案成立。'),
        ('v41-mhc','Single-Pass mHC',['四流 residual','使用上一子层 pre_mix','Attention/FFN','产生下一子层 mixing 系数'],
         'input-mixing 系数前移一子层，消除当前 reduction 的依赖；n=4 时论文融合路径流量从 20d 到 10d。',
         'Block.forward 返回 ffn_pre，下一层接收 pre_mix；可读实现仍拆成多次 PyTorch/Kernel 调用。',
         'Mega-mHC 是论文的融合部署路径；现有 CANN Sinkhorn/norm/matmul 的静态来源不证明该融合 kernel 或相同流量。'),
        ('v41-dspark','V4.1 DSpark 与服务边界',['目标层 37/38/39 attention 输入','三个 128选3 draft stage','五位置 logits','顺序 Markov / confidence','服务 target verification'],
         'draft rank=256、block=5、窗口128；主干不包含预训练 MTP，DSpark 独立训练再随策略对齐。',
         'DSparkBlock prefill 只写 context；forward_spec 提供草稿接口；generate.py 普通自回归，不包含 acceptance/rejection 或 confidence scheduler。',
         '静态前向存在不代表服务栈已支持；verification 长度策略需要真实引擎曲线，本站保持未知。'),
        ('v41-vision','原生视觉与 EPD 部署',['14×14 patch 线性投影','32 层双向 ViT / 2D RoPE','3×3 pixel-unshuffle','两层 MLP projector','语言 token 混合'],
         'ViT hidden=1024、heads=16；projector 9216→5120→5120；视觉 token 与文本联合预训练。',
         'vision.py 的 ViT/Aligner 与 Transformer.merge_image_embeddings；patch 个数和视觉 L_img 不用语言 L_KV 代替。',
         '论文 EPD 指视觉 Encoder / Prefill / Decode 三类服务，和 CED 的语言 Encoder 不同；NPU 端 SDPA/RoPE/重排/投影的候选映射不是整栈兼容证明。'),
    ]
    for sid,title,flow,meaning,code,hardware in entries:
        refs=[{'id':'v41-model'},{'id':'v41-paper'}]
        if sid in ('v41-vision','v41-engram'): refs.append({'id':sid})
        m={'id':sid,'title':title,'models':[ID],'flow':flow,'meaning':meaning,'code':code,'hardware':hardware,
           'limit':'固定源码与论文静态研究；未核验 V4.1 专用 Ascend 后端、实际存储、ABI 或设备数值/性能。设备实验不在本次范围。',
           'evidence':'配置/参考源码/论文；非设备实测','refs':refs}
        h['modules']=[x for x in h['modules'] if x['id']!=sid]+[m]
    h['updated']=DATE
    note={'models':[ID],'text':'V4.1 Flash 为独立 CED/CSA2/Engram/视觉/Single-Pass mHC 架构；已有 V4/MLA/Ascend 后端研究不自动覆盖本版。固定参考代码是可读实现，未实现论文全部生产优化。'}
    h['versionNotes']=[n for n in h['versionNotes'] if ID not in n.get('models',[])]+[note]
    write(DATA/'hardware.json',h)


def update_downloads():
    readme=(ROOT/'research/deepseek/README.md').read_text().replace('../../dist/assets/deepseek/','').replace('../../roadmap_deepseek.md','DeepSeek-roadmap.md').replace('(v4.1-flash.md)','(DeepSeek-V4.1-Flash-研究.md)')
    (OUT/'README.md').write_text(readme)
    family = read(DATA/'family.json')
    names = ['DeepSeek-V4.1-Flash-研究.md','DeepSeek-V4.1-Flash-逻辑矩阵.csv']
    for name in names:
        if not any(d['name']==name for d in family['downloads']):
            family['downloads'].append({'name':name,'path':'assets/deepseek/'+name})
    descriptions = {
        'DeepSeek-研究报告.md':('完整研究报告','当前研究','22 个版本；前 11 章为 21 版基础研究，第 12 章为 V4.1 专题。'),
        'DeepSeek-V4.1-Flash-研究.md':('V4.1 Flash 专题研究','当前研究','固定配置、参考源码、论文、真实 FP4/FP8 部署缓存；未做本版权重头审计。'),
        'README.md':('阅读与复现说明','当前研究','当前研究范围、附件口径与生成/校验命令。'),
        'DeepSeek-roadmap.md':('调研路线','当前研究','以开头的当前范围为准；旧阶段说明保留原始读取层级。'),
        'DeepSeek-研究报告包.zip':('完整研究包','当前研究','当前研究、表格、配置及固定来源；包含注明范围的既有基础数据，无权重或完整第三方源码。'),
        'DeepSeek-V4.1-Flash-逻辑矩阵.csv':('V4.1 Flash 逻辑矩阵','模型结构与存储','79 个组件、1,393 行参考逻辑矩阵；存储形状、dtype 和权重载荷未审计。'),
        'DeepSeek-all-layer-shapes.csv':('21 版全层计算表','模型结构与存储','21 个已审计 checkpoint，76,502 个声明步骤；不含 V4.1，重复阶段不可累加参数。'),
        'DeepSeek-模型与算子.xlsx':('21 版模型与算子工作簿','模型结构与存储','7 个 sheet，21 版结构/矩阵/接口与推测解码；V4.1 另见独立矩阵 CSV。'),
        'DeepSeek-compute.json':('21 版计算流 JSON','模型结构与存储','21 版主干、MTP/DSpark 的逐层声明；不含 V4.1 FLOPs。'),
        'DeepSeek-per-model-tables.zip':('21 版分模型表','模型结构与存储','21 份 CSV/XLSX 及字段说明；不含 V4.1。'),
        'DeepSeek-header-audits.zip':('21 版完整文件头摘要','模型结构与存储','各版独立分片文件头，不读取权重值；不含 V4.1。'),
        'v3-header-audit.json':('V3 文件头摘要','模型结构与存储','仅原始 V3 的 163 个分片；其他版本见完整文件头包。'),
        '字段与符号说明.md':('21 版计算字段说明','模型结构与存储','全层 CSV 与 7-sheet 工作簿的字段、别名、阶段及存储口径。'),
        'DeepSeek-静态分析.md':('静态分析说明','分析与比较','21 版成本/并行模型、V4.1 部署缓存、CANN 条件索引及机制说明。'),
        'DeepSeek-静态分析.json':('完整分析 JSON','分析与比较','21 版计算和并行、V4.1 部署缓存参数、30 个 CANN 算子家族与机制。'),
        'DeepSeek-静态分析.xlsx':('静态分析工作簿','分析与比较','5 个 sheet；21 版成本/并行示例、43 个跨家族条目、机制与算子条件。V4.1 缓存交互计算见网站。'),
        'DeepSeek-cost-scenarios.csv':('21 版成本示例','分析与比较','84 个固定成本场景；不含 V4.1，不含实测延迟或吞吐。'),
        'DeepSeek-parallel-scenarios.csv':('21 版并行示例','分析与比较','63 个固定 TP/DP 与路由假设场景；不含 V4.1。'),
        '跨家族比较.json':('跨家族比较 JSON','分析与比较','DeepSeek、Kimi、GLM 共 43 个条目；逐事实保留来源与核验深度。'),
        '跨家族比较.csv':('跨家族比较 CSV','分析与比较','43 个条目，包含 V4.1；未核验的历史项保留未知。'),
        'DeepSeek-CANN-contracts.csv':('完整 CANN 条件表','算子与源码证据','固定 v9.2.0-beta.2 的 30 个家族、4,207 条 API 条件；不等于设备运行支持证明。'),
        'DeepSeek-implementation-hardware-ascend.md':('实现与 Ascend 静态研究','算子与源码证据','16 个实现专题，含 5 个 V4.1 专题；实际 ABI、数值和性能未测。'),
        'DeepSeek-sources.csv':('实现专题来源索引','算子与源码证据','硬件/实现专题使用的固定来源，包含 V4.1；完整证明另见源码 JSON。'),
        'DeepSeek-source-proofs.json':('21 版参考函数证明','算子与源码证据','基础研究的固定 AST 函数与来源；V4.1 的 19 个函数证明在本版 architecture JSON 中。'),
        'DeepSeek-validation.json':('21 版基础校验摘要','算子与源码证据','既有 21 版校验快照；当前完整范围通过默认终端校验，不自动生成新记录。'),
        'DeepSeek-MTP-DSpark.json':('MTP / DSpark 基础协议','算子与源码证据','21 版参考构造、共享口径与服务边界；V4.1 的独立专家配置见本版专题。'),
        'DeepSeek-backend-trace.json':('基础后端追踪','算子与源码证据','固定版本的 dispatcher、wrapper、host/tiling 证据；未证明 V4.1 专用后端已接入。'),
        'DeepSeek-CANN-operators.json':('CANN 源码与本地包核查','算子与源码证据','30 个家族、固定 tag、桥接范围及本地 SDK 只读证据；不含设备实验。'),
        'R1-config-tokenizer-comparison.json':('R1 Distill 配置与 tokenizer 对比','算子与源码证据','六版原始 Distill 的配置/tokenizer 元数据比较；完整 BPE 见独立文件。'),
        'R1-full-tokenizer-audit.json':('R1 Distill 完整 tokenizer / BPE','算子与源码证据','六版完整公开 tokenizer 与四个 Qwen 底座；两个 Llama 底座 gated，差异未知。'),
    }
    for entry in family['downloads']:
        title,category,scope=descriptions[entry['name']]
        entry.update(title=title,category=category,scope=scope)
        raw=(ROOT/'dist'/entry['path']).read_bytes()
        entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    order={name:i for i,name in enumerate(descriptions)}
    family['downloads'].sort(key=lambda d:order[d['name']])
    write(DATA/'family.json',family)


def package_current():
    """Replace current files in the existing download, without new history/QA archives."""
    from build_deepseek_analysis import write_report
    write_report(read(DATA/'analysis.json'),read(ROOT/'data/analysis/cross-family.json'))
    update_downloads()
    family = read(DATA/'family.json')
    public = dict(family)
    public['downloads']=[]
    replacements = {'metadata/family.json':(json.dumps(public,ensure_ascii=False,indent=2)+'\n').encode()}
    for name in ['report.json','hardware.json','analysis.json',ID+'-architecture.json']:
        replacements['metadata/'+name]=(DATA/name).read_bytes()
    replacements['metadata/analysis-export-manifest.json']=(DATA/'research/analysis-export-manifest.json').read_bytes()
    replacements['configs/DeepSeek-V4.1-Flash-config.json']=(DATA/'configs/DeepSeek-V4.1-Flash-config.json').read_bytes()
    replacements['跨家族比较.json']=(ROOT/'data/analysis/cross-family.json').read_bytes()
    for name in ['README.md','DeepSeek-研究报告.md','DeepSeek-roadmap.md','DeepSeek-V4.1-Flash-研究.md','DeepSeek-V4.1-Flash-逻辑矩阵.csv','DeepSeek-静态分析.md','DeepSeek-静态分析.json',
                 'DeepSeek-静态分析.xlsx','跨家族比较.csv','DeepSeek-implementation-hardware-ascend.md','DeepSeek-sources.csv']:
        replacements[name]=(OUT/name).read_bytes()
    # Existing standalone metadata must match the same current files in the ZIP.
    # The new architecture is already served from data/ and included in the ZIP.
    for name,raw in replacements.items():
        if name.startswith('metadata/') and (OUT/name).is_file(): (OUT/name).write_bytes(raw)
    target=OUT/'DeepSeek-研究报告包.zip'
    CACHE.mkdir(parents=True,exist_ok=True)
    temporary=CACHE/'current-research-package.zip'
    with zipfile.ZipFile(target) as old, zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as new:
        for item in old.infolist():
            if item.filename not in replacements: new.writestr(item,old.read(item.filename))
        for name,raw in sorted(replacements.items()):
            item=zipfile.ZipInfo(name,(2026,10,8,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            item.external_attr=0o644<<16
            new.writestr(item,raw)
    with zipfile.ZipFile(temporary) as archive: assert archive.testzip() is None
    shutil.copyfile(temporary,target)
    update_downloads()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true',help='Fetch missing pinned public inputs into the external cache.')
    parser.add_argument('--package',action='store_true',help='Refresh the existing package after regenerating analytical exports.')
    args=parser.parse_args()
    if args.package: package_current()
    else:
        inputs(args.fetch)
        cfg,arch=architecture()
        install(cfg,arch)
        print(json.dumps({'model':ID,'components':len(arch['nodes']),'referenceBackboneParameters':arch['mainLogicalParameters'],
                          'referenceEngramParameters':arch['engramLogicalParameters'],'referenceDSparkParameters':arch['dsparkLogicalParameters']},ensure_ascii=False))
