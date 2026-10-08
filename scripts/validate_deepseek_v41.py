"""Validate the current V4.1 config/reference study without writing QA records."""
import argparse
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/families/deepseek'


def read(path):
    return json.loads(path.read_text())


def validate(sources=False):
    from build_deepseek_v41 import CACHE, HASHES, ID, REVISION
    family = read(DATA/'family.json')
    model = next(m for m in family['models'] if m['id']==ID)
    cfg = read(ROOT/model['configPath'])
    c,v = cfg['text_config'],cfg['vision_config']
    arch = read(ROOT/model['architecturePath'])
    assert cfg['model_type']=='deepseek_v41'
    assert model['revision']==arch['revision']==REVISION
    assert not model['auditPath'] and not model.get('computePath') and model['computeScope']
    assert model['facts']['payload']['value'] is None
    assert model['facts']['mtpParameters']['value'] is None
    assert model['facts']['parameters']['value']==552_000_000_000
    assert model['facts']['engramParameters']['value']==196_000_000_000
    assert model['facts']['activePrefillParameters']['value']==8_000_000_000
    assert model['facts']['activeDecodeParameters']['value']==16_000_000_000
    assert 'NVFP4' in arch['cacheFormats']['main'] and '省略' in arch['cacheFormats']['main']
    assert 'MXFP8' in arch['cacheFormats']['swa'] and '每32' in arch['cacheFormats']['swa']
    assert 'MXFP4' in arch['cacheFormats']['index']
    assert c['num_hidden_layers']==40 and v['num_hidden_layers']==32
    assert len(arch['nodes'])==79 and len({n['id'] for n in arch['nodes']})==79
    language = [n for n in arch['nodes'] if n['group']=='decoder']
    vision = [n for n in arch['nodes'] if n['group']=='vision']
    draft = [n for n in arch['nodes'] if n['group']=='dspark']
    assert [n['number'] for n in language]==list(range(1,41))
    assert [n['stage'] for n in language]==['encoder']*20+['decoder']*20
    assert [n['number'] for n in vision]==list(range(1,33))
    assert [n['sourceIndex'] for n in draft]==[0,1,2]
    assert collections.Counter(n['attentionMode'] for n in language)=={'SWA':2,'Full':4,'Reindex':4,'Reuse':30}
    assert [n['sourceIndex'] for n in language if n['attentionMode']=='Full']==[2,8,14,20]
    assert [n['sourceIndex'] for n in language if n['attentionMode']=='Reindex']==[24,28,32,36]
    assert c['candidate_source_layer_id']==20 and c['candidate_block_size']*c['candidate_topk_blocks']==16384
    assert c['index_topk']==512
    assert c['engram_layer_ids']==[1,14]
    assert [n['sourceIndex'] for n in language if any(m['id']=='engram' for m in n['modules'])]==[1,14]
    rows=[]
    for node in arch['nodes']:
        assert node['payloadBytes'] is None
        assert node['parameters']==sum(m['parameters'] for m in node['modules'])
        for module in node['modules']:
            assert module['parameters']==sum(t['logical_parameters_each']*t['count'] for t in module['matrices'])
            if module['representative']:
                assert (module['count'],module['selectedCount'])==((128,3) if node['group']=='dspark' else (384,6))
            for tensor in module['matrices']:
                assert tensor['stored_shape'] is None and tensor['stored_dtype'] is None and tensor['payloadBytes'] is None
                assert tensor['logical_parameters_each']==math.prod(tensor['logical_shape'])
                assert tensor['proof'] in arch['referenceProofs']
                assert tensor['source']==arch['referenceProofs'][tensor['proof']]['url']
                rows.append(tensor)
        if node['group']=='decoder':
            assert node['compressRatio']==c['compress_ratios'][node['sourceIndex']]
            assert any(m['id']=='compressor' for m in node['modules'])==(node['attentionMode']=='Full')
            assert any(m['id']=='indexer' for m in node['modules'])==(node['attentionMode'] in ('Full','Reindex'))
    # Independent count of conditional tables plus projection/gating, not a dense compute cost.
    table_parameters=sum(c['engram_num_embeddings'])*c['engram_head_dim']
    projection=2*(c['hc_mult']+1)*c['hidden_size']*((c['engram_max_ngram_size']-1)*c['engram_n_heads']*c['engram_head_dim'])
    gates=2*2*c['hc_mult']*c['hidden_size']
    assert table_parameters==196_613_849_600
    assert arch['engramLogicalParameters']==table_parameters+projection+gates==196_928_504_320
    assert abs(arch['mainLogicalParameters']/552_000_000_000-1)<0.01
    assert arch['dsparkLogicalParameters']==sum(n['parameters'] for n in draft)
    # Count each global KV owner once. Reindex/Reuse introduce no new global cache.
    main_record_bytes=512//2+512//16
    index_record_bytes=128//2+128//32
    assert main_record_bytes==288 and index_record_bytes==68
    storage=arch['deploymentStorage']
    assert storage['mainRecordBytes']==main_record_bytes and storage['indexRecordBytes']==index_record_bytes
    assert storage['swaRecordBytes']==512+512//32==528
    assert storage['languageLayers']*storage['window']*storage['swaRecordBytes']==2703360
    assert storage['draftStages']*storage['window']*storage['swaRecordBytes']==202752
    assert storage['compressionStateBytesPerRequest']==24576
    for length in [1,2,127,128,129,1023,1048576]:
        entries=sum(length//n['compressRatio'] for n in language if n['attentionMode']=='Full')
        expected=(3*(length//2)+length)*356
        assert entries*(main_record_bytes+index_record_bytes)==expected
        if length%2==0: assert expected==890*length
    assert 890*1048576==933232640
    assert 3*2*2*512*4==24576
    assert len([n for n in draft if any(m['id']=='context' for m in n['modules'])])==1
    assert len([n for n in draft if any(m['id']=='draft-heads' for m in n['modules'])])==1
    assert c['dspark_target_layer_ids']==[37,38,39] and c['dspark_block_size']==5
    assert all(not any(t['tensor_template'].endswith('head.weight') and 'markov_head' not in t['tensor_template'] for m in n['modules'] for t in m['matrices']) for n in draft)
    proofs=arch['referenceProofs']
    declared={s['path']:s for s in arch['sources']}
    for pid,proof in proofs.items():
        assert pid==proof['path']+':'+proof['symbol']
        assert proof['revision']==REVISION and proof['sourceSha256']==declared[proof['path']]['sha256']
        assert 0<proof['line']<=proof['end'] and len(proof['functionSha256'])==64
        assert proof['url'].endswith(f"#L{proof['line']}-L{proof['end']}")
    if sources:
        for path,sha in HASHES.items(): assert hashlib.sha256((CACHE/path).read_bytes()).hexdigest()==sha
        quant=(CACHE/'inference/kernel.py').read_text()
        model_text=(CACHE/'inference/model.py').read_text()
        assert 'fp8_block_size = 32' in model_text and 'scale_dtype = torch.float8_e8m0fnu' in model_text
        assert 'fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)' in model_text
        assert 'dtype=torch.float8_e4m3fn' in quant and 'dtype=torch.float4_e2m1fn_x2' in quant
        for proof in proofs.values():
            text=(CACHE/proof['path']).read_text()
            body='\n'.join(text.splitlines()[proof['line']-1:proof['end']])
            assert hashlib.sha256(body.encode()).hexdigest()==proof['functionSha256']
            cls_name,fn_name=proof['symbol'].split('.')
            cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==cls_name)
            fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==fn_name)
            assert (fn.lineno,fn.end_lineno)==(proof['line'],proof['end'])
    csv_path=ROOT/'dist/assets/deepseek/DeepSeek-V4.1-Flash-逻辑矩阵.csv'
    raw=csv_path.read_bytes()
    assert raw.startswith(b'\xef\xbb\xbf') and b'\n' not in raw.replace(b'\r\n',b'')
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        csv_rows=list(csv.reader(f))[1:]
    assert len(csv_rows)==len(rows)
    for row,tensor in zip(csv_rows,rows):
        assert row[5]==tensor['tensor_template'] and row[6]==' × '.join(map(str,tensor['logical_shape']))
        assert (int(row[7]),int(row[8]))==(tensor['logical_parameters_each'],tensor['count'])
        assert row[9]==tensor['source']
    comparison=read(ROOT/'data/analysis/cross-family.json')
    compared=next(m for m in comparison['models'] if m['key']=='deepseek:'+ID)
    assert compared['facts']==model['facts'] and compared['cache']==arch['cache']
    assert compared['auditPath'] is None and compared['parameterScope']==arch['parameterScope']
    analysis=read(DATA/'analysis.json')
    deployment=next(m for m in analysis['deploymentModels'] if m['id']==ID)
    assert deployment['storage']==storage and deployment['context']==c['max_position_embeddings']
    assert model['sections']==['v41']
    assert next(s for s in read(DATA/'report.json') if s['id']=='v41')['html']
    return {'status':'passed','model':ID,'languageLayers':40,'visionLayers':32,'draftStages':3,
            'logicalMatrixRows':len(rows),'sourceFunctions':len(proofs),'sourceBytesChecked':sources,
            'weightHeaderAudit':False,'hardwareExperiment':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',action='store_true')
    print(json.dumps(validate(parser.parse_args().sources),ensure_ascii=False))
