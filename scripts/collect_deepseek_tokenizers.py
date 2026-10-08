"""Compare public Distill/baseline configs; preserve gated baseline gaps without bypassing access."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
from collect_deepseek_round1 import write_json

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/families/deepseek';CACHE=Path.home()/'.cache/model-research-atlas/deepseek-extended'
def fetch(u):
 with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'model-research-atlas'}),timeout=30) as r:return r.read(4000001)
def one(m):
 metadata=json.loads(fetch('https://huggingface.co/api/models/'+m['baseModel'])); base_revision=metadata['sha'];sources=[];files={};gaps=[]
 for mid,revision,name,paths in [(m['baseModel'],base_revision,'base-'+m['name'],['config.json','tokenizer_config.json']),(m['modelId'],m['revision'],m['name'],['tokenizer_config.json'])]:
  for path in paths:
   url=f'https://huggingface.co/{mid}/resolve/{revision}/{path}'
   try:raw=fetch(url)
   except urllib.error.HTTPError as exc:
    if exc.code not in (401,403):raise
    gaps.append({'modelId':mid,'revision':revision,'path':path,'status':'gated','httpStatus':exc.code});continue
   assert len(raw)<4000000
   file=f'baseline/{name}/{revision}/{path}';target=CACHE/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
   local=f'data/families/deepseek/research/r1-tokenizers/{name}-{path}';target=ROOT/local;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
   sid=f'tokenizer-{name.lower()}-{path.replace(".","-")}'
   sources.append({'id':sid,'title':mid+' / '+path,'modelId':mid,'revision':revision,'path':path,'url':f'https://huggingface.co/{mid}/blob/{revision}/{path}','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'cacheFile':file,'localPath':local,'accessed':'2026-10-01','kind':'checkpoint_config' if path=='config.json' else 'tokenizer_config','readScope':'配置字段与 chat_template 配置比较；未审计完整 tokenizer.json/BPE 词表','license':'见相应底座与蒸馏 checkpoint 许可'})
   files[('base-' if name.startswith('base-') else '')+path]=json.loads(raw)
 cfg=json.loads((DATA/'configs'/f'{m["name"]}-config.json').read_text()); bc=files.get('base-config.json');tc=files.get('tokenizer_config.json');btc=files.get('base-tokenizer_config.json')
 diff=lambda a,b:{k:{'base':a.get(k),'distill':b.get(k)} for k in set(a)|set(b) if a.get(k)!=b.get(k)}
 changes=diff(bc,cfg) if bc else None
 token_changes=diff(btc,tc) if btc is not None and tc is not None else None
 # Keep long templates out of prose; a hash and changed field name are sufficient for comparison.
 if token_changes:
  for k,v in token_changes.items():
   if any(len(str(value))>400 for value in v.values()):
    token_changes[k]={role:{'sha256':hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),'characters':len(str(value))} for role,value in v.items()}
 return sources,{'model':m['name'],'baseModel':m['baseModel'],'baseRevision':base_revision,'distillRevision':m['revision'],'configChanges':changes,'tokenizerConfigChanges':token_changes,'gaps':gaps,'limit':'配置级比较；不代表底座/蒸馏权重相同或 tokenizer.json/BPE 全量审计'}
def collect():
 models=[m for m in json.loads((DATA/'versions.json').read_text())['models'] if m['kind']=='distill'];manifest=json.loads((DATA/'extended-sources.json').read_text());records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for sources,record in pool.map(one,models):manifest['sources'].extend(sources);records.append(record);print(record['model'],'baseline', 'gated' if record['gaps'] else 'compared')
 write_json(DATA/'extended-sources.json',manifest);write_json(DATA/'research/r1-config-tokenizer-comparison.json',{'accessed':'2026-10-01','records':records})
if __name__=='__main__':collect()
