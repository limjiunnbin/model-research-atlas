"""Resume static follow-up inputs at fixed revisions; never fetch weight payloads."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

from collect_deepseek_round1 import write_json

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/families/deepseek';CACHE=Path.home()/'.cache/model-research-atlas/deepseek-followup'
NEW=['DeepSeek-V4-Flash-DSpark','DeepSeek-V4-Pro-DSpark','DeepSeek-V4-Flash-0731','DeepSeek-V4-Pro-0813']
MANIFEST=DATA/'followup-sources.json'


def fetch(url,limit=32_000_000):
 if url.split('?')[0].endswith('.safetensors'):raise ValueError('No model-weight payload in this collector')
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'model-research-atlas-static-followup'}),timeout=40) as response:raw=response.read(limit+1)
 assert len(raw)<=limit,url
 return raw


def collect():
 CACHE.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'schemaVersion':1,'familyId':'deepseek','accessed':'2026-10-01','scope':'MTP、全 checkpoint 文件头、DSpark、完整 tokenizer 和后端静态补充；P6 暂缓','sources':[]}
 sources={s['id']:s for s in manifest['sources']}
 def save(s,raw):
  target=CACHE/s['cacheFile'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  if s.get('localPath'):
   target=ROOT/s['localPath'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  s.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),accessed='2026-10-01');sources[s['id']]=s
  write_json(MANIFEST,{**manifest,'sources':list(sources.values())})
 original=json.loads((DATA/'versions.json').read_text())['models']
 current_path=DATA/'research/followup-checkpoints.json'
 additions=json.loads(current_path.read_text())['models'] if current_path.exists() else []
 for name in NEW:
  if any(m['name']==name for m in additions):continue
  info=json.loads(fetch('https://huggingface.co/api/models/deepseek-ai/'+name))
  revision=info['sha'];info_raw=fetch(f'https://huggingface.co/api/models/deepseek-ai/{name}/revision/{revision}')
  info=json.loads(info_raw);assert info['sha']==revision
  sid='follow-'+name.lower()+'-metadata'
  save({'id':sid,'title':name+' 固定目录','modelId':'deepseek-ai/'+name,'revision':revision,'path':None,'url':f'https://huggingface.co/api/models/deepseek-ai/{name}/revision/{revision}','cacheFile':f'checkpoints/{name}/{revision}/metadata.json','localPath':f'data/families/deepseek/research/checkpoint-metadata/{name}.json','kind':'checkpoint_metadata','readScope':'固定模型目录；不凭 sibling 名称推断权重 shape/dtype','license':'见固定 checkpoint 的官方许可'},info_raw)
  additions.append({'name':name,'modelId':'deepseek-ai/'+name,'revision':revision,'url':f'https://huggingface.co/deepseek-ai/{name}/tree/{revision}','kind':'speculative' if name.endswith('DSpark') else 'release','baseModel':'deepseek-ai/'+('DeepSeek-V4-Flash' if 'Flash'in name else 'DeepSeek-V4-Pro') if name.endswith('DSpark') else None,'materialsSource':sid})
  write_json(current_path,{'accessed':'2026-10-01','scope':'保留首轮 17 模型快照；补充 2 个 DSpark 挂接 checkpoint 与 2 个带 DSpark 的正式发布','models':additions})
 models=original+additions
 tasks=[]
 for model in models:
  name=model['name'];revision=model['revision']
  info=json.loads((DATA/'research/checkpoint-metadata'/f'{name}.json').read_text())
  paths={s['rfilename'] for s in info['siblings']}
  chosen=['model.safetensors.index.json'] if 'model.safetensors.index.json'in paths else []
  if name in NEW:chosen+=['config.json','README.md']+[p for p in sorted(paths) if p.startswith('inference/') and p.endswith(('.py','.json')) or p.startswith('LICENSE')]
  for path in chosen:
   sid='follow-'+name.lower()+'-'+path.replace('/','-').replace('.','-')
   if sid in sources:
    assert (CACHE/sources[sid]['cacheFile']).is_file();continue
   url=f'https://huggingface.co/{model["modelId"]}/resolve/{revision}/{path}'
   local=f'data/families/deepseek/configs/{name}-config.json' if path=='config.json' else None
   tasks.append({'id':sid,'title':name+' / '+path,'modelId':model['modelId'],'revision':revision,'path':path,'url':url.replace('/resolve/','/blob/'),'rawUrl':url,'cacheFile':f'checkpoints/{name}/{revision}/{path}','localPath':local,'kind':'weight_index' if 'index.json'in path else 'checkpoint_config' if path=='config.json' else 'reference_code' if path.endswith('.py') else 'official_document','readScope':'固定版本采集；内容读取深度随当前证明/审计记录','license':'见固定 checkpoint 的官方许可'})
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  futures=[(s,pool.submit(fetch,s['rawUrl'])) for s in tasks]
  for s,job in futures:
   save(s,job.result());print(s['title'],sources[s['id']]['bytes'],flush=True)
 print('Follow-up checkpoint inputs:',len(models),'models;',len(sources),'sources',flush=True)


if __name__=='__main__':collect()
