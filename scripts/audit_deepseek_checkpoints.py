"""Audit every fixed safetensors header and index; never read tensor payload or run a model."""
import collections
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.request

from collect_deepseek_round1 import write_json

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/families/deepseek';CACHE=Path.home()/'.cache/model-research-atlas/deepseek-header-followup'
SIZES={'F8_E4M3':1,'F8_E5M2':1,'F8_E8M0':1,'BF16':2,'F16':2,'F32':4,'F64':8,'I64':8,'I32':4,'I16':2,'I8':1,'U8':1,'BOOL':1,'F4':.5,'F4_E2M1':.5}


def byte_range(url,start,end):
 req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'model-research-atlas-header-followup'})
 with urllib.request.urlopen(req,timeout=35) as r:
  match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
  assert r.status==206 and match and tuple(map(int,match.groups()[:2]))==(start,end),'Exact Range required: '+url
  raw=r.read(end-start+2)
 assert len(raw)==end-start+1
 return raw,int(match.group(3))


def check(header,size,metadata_bytes):
 tensors={k:v for k,v in header.items() if k!='__metadata__'}
 ranges=sorted(t['data_offsets'] for t in tensors.values())
 assert ranges and ranges[0][0]==0 and all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
 assert ranges[-1][1]+metadata_bytes==size
 for t in tensors.values():assert t['data_offsets'][1]-t['data_offsets'][0]==math.prod(t['shape'])*SIZES[t['dtype']],t
 return tensors


def one(model,shard):
 name=model['name'];rev=model['revision'];url=f'https://huggingface.co/{model["modelId"]}/resolve/{rev}/{shard}'
 dest=CACHE/name/rev/(shard+'.header.json');raw_path=dest.with_suffix('.bin')
 if dest.exists():
  record=json.loads(dest.read_text());assert record['url']==url and record['revision']==rev
  raw=raw_path.read_bytes();assert hashlib.sha256(raw).hexdigest()==record['headerSha256']
  assert check(json.loads(raw[8:]),record['fileBytes'],len(raw))==record['tensors'];return record
 # Reuse the already checked original V3 metadata. Its raw hash was recorded during the original HTTP audit.
 old=Path.home()/'.cache/model-research-atlas/deepseek-v3-headers'/shard.replace('.safetensors','.header.json')
 if name=='DeepSeek-V3' and old.exists():
  record=json.loads(old.read_text());assert record['revision']==rev;check(record['tensors'],record['fileBytes'],record['headerBytes']);return record
 for retry in range(5):
  try:
   prefix,size=byte_range(url,0,7);length=struct.unpack('<Q',prefix)[0];assert 0<length<=4_000_000
   raw,size2=byte_range(url,8,7+length);assert size==size2
   combined=prefix+raw;header=json.loads(raw);tensors=check(header,size,len(combined))
   record={'shard':shard,'revision':rev,'url':url,'fileBytes':size,'headerBytes':len(combined),'headerSha256':hashlib.sha256(combined).hexdigest(),'tensors':tensors}
   dest.parent.mkdir(parents=True,exist_ok=True);raw_path.write_bytes(combined);dest.write_text(json.dumps(record)+'\n');return record
  except (urllib.error.URLError,TimeoutError,ConnectionError):
   if retry==4:raise
   time.sleep(min(1.5*2**retry,12))


def normalize(name,L):
 if name.startswith('model.layers.'):
  match=re.match(r'model.layers\.(\d+)\.',name);i=int(match.group(1));group='decoder' if i<L else 'mtp'
  template=re.sub(r'model.layers\.\d+\.', 'model.layers.{i}.',name)
 elif name.startswith('layers.'):
  i=int(name.split('.')[1]);group='decoder';template=re.sub(r'^layers\.\d+\.', 'layers.{i}.',name)
 elif name.startswith('mtp.'):
  i=int(name.split('.')[1]);group='auxiliary';template=re.sub(r'^mtp\.\d+\.', 'mtp.{i}.',name)
 else:i=None;group='global';template=name
 return group,i,re.sub(r'\.experts\.\d+\.', '.experts.{e}.',template)


def audit(model,sources):
 name=model['name'];rev=model['revision'];c=json.loads((DATA/'configs'/f'{name}-config.json').read_text());L=c['num_hidden_layers']
 output=DATA/'research/header-audits'/f'{name}.json'
 if output.exists():
  old=json.loads(output.read_text());assert old['revision']==rev;return old
 index_sources=[s for s in sources if s.get('modelId')==model['modelId'] and s.get('path')=='model.safetensors.index.json']
 index=json.loads((Path.home()/'.cache/model-research-atlas/deepseek-followup'/index_sources[0]['cacheFile']).read_text()) if index_sources else None
 metadata=json.loads((DATA/'research/checkpoint-metadata'/f'{name}.json').read_text())
 shards=sorted(set(index['weight_map'].values())) if index else sorted(s['rfilename'] for s in metadata['siblings'] if s['rfilename'].endswith('.safetensors'))
 records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  jobs=[pool.submit(one,model,shard) for shard in shards]
  for count,job in enumerate(concurrent.futures.as_completed(jobs),1):
   records.append(job.result())
   if count%32==0 or count==len(shards):print(name,f'{count}/{len(shards)} headers',flush=True)
 names=set();rows={};templates={};dtype=collections.Counter();group_stats=collections.defaultdict(collections.Counter)
 for record in sorted(records,key=lambda r:r['shard']):
  for tensor,t in record['tensors'].items():
   assert tensor not in names;names.add(tensor)
   if index:assert index['weight_map'][tensor]==record['shard']
   group,i,template=normalize(tensor,L);key=(group,i,template);size=t['data_offsets'][1]-t['data_offsets'][0]
   role='scale' if tensor.endswith(('.scale','.weight_scale_inv','.weight_scale')) else 'nontrainable_index' if tensor.endswith('tid2eid') else 'parameter'
   row={'group':group,'sourceIndex':i,'tensorTemplate':template,'storedShape':t['shape'],'storedDtype':t['dtype'],'payloadBytesEach':size,'role':role,'copies':1}
   if key in rows:
    assert all(rows[key][k]==row[k] for k in ('storedShape','storedDtype','payloadBytesEach','role')),tensor;rows[key]['copies']+=1
   else:rows[key]=row
   dtype[t['dtype']]+=1;group_stats[group]['tensors']+=1;group_stats[group]['payloadBytes']+=size;group_stats[group][role+'StoredElements']+=math.prod(t['shape'])
 if index:assert names==set(index['weight_map'])
 # Keep each actual layer/stage distinct: shared shape alone does not establish weight sharing.
 result={'schemaVersion':1,'modelId':model['modelId'],'name':name,'revision':rev,'accessed':'2026-10-01','scope':'全部分片 header、dtype/shape/连续 offsets 与索引；未读取权重数值','shards':[{k:v for k,v in r.items() if k!='tensors'} for r in sorted(records,key=lambda r:r['shard'])],'tensorCount':len(names),'dtypeTensorCounts':dict(dtype),'payloadBytes':sum(r['fileBytes']-r['headerBytes'] for r in records),'fileBytes':sum(r['fileBytes'] for r in records),'metadataCapturedBytes':sum(r['headerBytes'] for r in records),'weightPayloadTransferredBytes':0,'indexDeclaredTotalSize':index['metadata'].get('total_size') if index else None,'groups':{k:dict(v) for k,v in group_stats.items()},'templates':list(rows.values()),'limit':'参数是存储元素，不直接等于浮点逻辑参数；FP4 打包、量化 scale、共享别名与 DSpark/MTP 命名由固定构造/loader 独立对应。'}
 write_json(output,result);return result


def main():
 models=json.loads((DATA/'versions.json').read_text())['models']+json.loads((DATA/'research/followup-checkpoints.json').read_text())['models'];sources=json.loads((DATA/'followup-sources.json').read_text())['sources']
 results=[]
 for model in models:
  result=audit(model,sources);results.append({k:result[k] for k in ('name','modelId','revision','tensorCount','dtypeTensorCounts','payloadBytes','metadataCapturedBytes','weightPayloadTransferredBytes')})
  print('AUDITED',model['name'],result['tensorCount'],result['metadataCapturedBytes'],'metadata bytes',flush=True)
 write_json(DATA/'research/header-audits/summary.json',{'status':'passed','models':results,'count':len(results),'weightPayloadTransferredBytes':0,'scope':'全部选定 checkpoint 的全部文件头；非运行验证'})


if __name__=='__main__':main()
