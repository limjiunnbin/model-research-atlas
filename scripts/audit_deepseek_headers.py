"""Audit fixed V3 safetensors headers using strictly validated HTTP byte ranges, never weight payload."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/families/deepseek'
REV='e815299b0bcbac849fa540c768ef21845365c9eb'
SIZES={'F8_E4M3':1,'F8_E5M2':1,'BF16':2,'F16':2,'F32':4,'F64':8,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1}


def byte_range(url,start,end):
    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'model-research-atlas-header-audit'})
    with urllib.request.urlopen(request,timeout=45) as response:
        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
        if response.status!=206 or not match or tuple(map(int,match.groups()[:2]))!=(start,end):
            raise ValueError('Server did not honor the exact byte range; refusing payload download')
        raw=response.read(end-start+2)
    if len(raw)!=end-start+1:raise ValueError('Byte range length mismatch')
    return raw,int(match.group(3))


def validate_record(result,shard):
    assert result['shard']==shard and result['revision']==REV
    assert result['url']==f'https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/{REV}/{shard}'
    assert re.fullmatch(r'[0-9a-f]{64}',result['headerSha256']) and 8<result['headerBytes']<=2_000_008
    ranges=sorted((t['data_offsets'][0],t['data_offsets'][1]) for t in result['tensors'].values())
    assert ranges[0][0]==0 and all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
    assert ranges[-1][1]+result['headerBytes']==result['fileBytes']
    for t in result['tensors'].values():
        assert t['data_offsets'][1]-t['data_offsets'][0]==math.prod(t['shape'])*SIZES[t['dtype']]
    return result


def one(shard,cache):
    destination=cache/shard.replace('.safetensors','.header.json')
    if destination.exists(): return validate_record(json.loads(destination.read_text()),shard)
    url=f'https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/{REV}/{shard}'
    for retry in range(3):
        try:
            prefix,size=byte_range(url,0,7); length=struct.unpack('<Q',prefix)[0]
            if not 0<length<=2_000_000:raise ValueError('Unexpected safetensors header size')
            raw,size2=byte_range(url,8,7+length)
            if size2!=size:raise ValueError('Shard size changed')
            header=json.loads(raw)
            result={'shard':shard,'revision':REV,'url':url,'fileBytes':size,'headerBytes':8+length,
                    'headerSha256':hashlib.sha256(prefix+raw).hexdigest(),'tensors':{k:v for k,v in header.items() if k!='__metadata__'}}
            validate_record(result,shard)
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result)+'\n')
            return result
        except Exception:
            if retry==2:raise
            time.sleep(0.2*(retry+1))


def audit(cache,index_path):
    index=json.loads(index_path.read_text()); weight_map=index['weight_map']; shards=sorted(set(weight_map.values()))
    cache.mkdir(parents=True,exist_ok=True)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(one,s,cache):s for s in shards}
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            if len(records)%16==0 or len(records)==len(shards): print(f'headers {len(records)}/{len(shards)}',flush=True)
    tensors={}; layers={}; dtype_counts=collections.Counter(); payload=0
    shard_records=[]
    for record in sorted(records,key=lambda r:r['shard']):
        shard_records.append({k:v for k,v in record.items() if k!='tensors'})
        for name,t in record['tensors'].items():
            assert name not in tensors and weight_map[name]==record['shard']
            t=dict(t);t['shard']=record['shard'];t['payloadBytes']=t['data_offsets'][1]-t['data_offsets'][0]
            tensors[name]=t;dtype_counts[t['dtype']]+=1;payload+=t['payloadBytes']
            match=re.match(r'model.layers\.(\d+)\.',name);layer=match.group(1) if match else 'global'
            row=layers.setdefault(layer,{'tensorCount':0,'parameterElements':0,'scaleElements':0,'payloadBytes':0})
            row['tensorCount']+=1;row['payloadBytes']+=t['payloadBytes']
            row['scaleElements' if name.endswith('weight_scale_inv') else 'parameterElements']+=math.prod(t['shape'])
    assert set(tensors)==set(weight_map)
    sample={name:t for name,t in tensors.items() if not name.startswith('model.layers.') or
            any(name.startswith(f'model.layers.{i}.') and ('.experts.' not in name or '.experts.0.' in name) for i in (0,3,61))}
    # All layers and expert copies must agree with each applicable normalized template.
    templates={}; copies=collections.Counter()
    for name,t in tensors.items():
        match=re.match(r'model.layers\.(\d+)\.',name)
        if not match:kind='global'
        else:kind='dense' if int(match[1])<3 else 'moe' if int(match[1])<61 else 'mtp'
        key=(kind,re.sub(r'\.experts\.\d+\.', '.experts.{e}.', re.sub(r'model.layers\.\d+\.', 'model.layers.{i}.',name)))
        signature=(t['dtype'],tuple(t['shape']),t['payloadBytes'])
        if key in templates: assert templates[key]==signature,key
        else:templates[key]=signature
        copies[key]+=1
    summary={'schemaVersion':1,'modelId':'deepseek-ai/DeepSeek-V3','revision':REV,'accessed':'2026-10-01',
             'scope':'全部 163 个 safetensors 分片的 header 与 data_offsets；不读取、加载或校验权重数值',
             'shards':shard_records,'tensorCount':len(tensors),'dtypeTensorCounts':dict(dtype_counts),
             'payloadBytes':payload,'fileBytes':sum(r['fileBytes'] for r in records),
             'metadataTransferredBytes':sum(r['headerBytes'] for r in records),'weightPayloadTransferredBytes':0,
             'layers':layers,'representativeTensors':sample,
             'templates':[{'kind':kind,'tensorTemplate':name,'storedDtype':value[0],'storedShape':list(value[1]),'payloadBytesEach':value[2],'copies':copies[(kind,name)]} for (kind,name),value in sorted(templates.items())],
             'indexDeclaredTotalSize':index['metadata']['total_size'],
             'mainParameterElements':sum(v['parameterElements'] for k,v in layers.items() if k!='61'),
             'mtpStoredParameterElements':layers['61']['parameterElements'],
             'limit':'MTP 存储副本与逻辑共享参数分开；文件头不能证明相同权重值或所有加载别名。原始 headers 留在研究缓存，仅分发摘要和固定来源。'}
    target=DATA/'research/v3-header-audit.json';target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['tensorCount','payloadBytes','metadataTransferredBytes','mainParameterElements','mtpStoredParameterElements','dtypeTensorCounts']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    a=p.parse_args();audit(a.cache,a.index)
