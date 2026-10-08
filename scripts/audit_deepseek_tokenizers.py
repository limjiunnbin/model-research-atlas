"""Compare all public tokenizer.json vocabulary/merge entries and special-token IDs at fixed revisions."""
import concurrent.futures
import hashlib
import json
import urllib.error
from pathlib import Path

from collect_deepseek_followup import DATA,CACHE,MANIFEST,fetch
from collect_deepseek_round1 import write_json


def audit():
 records=json.loads((DATA/'research/r1-config-tokenizer-comparison.json').read_text())['records']
 versions={m['name']:m for m in json.loads((DATA/'versions.json').read_text())['models']}
 manifest=json.loads(MANIFEST.read_text());sources={s['id']:s for s in manifest['sources']};tasks=[]
 for record in records:
  for role,model,revision in [('base',record['baseModel'],record['baseRevision']),('distill',versions[record['model']]['modelId'],record['distillRevision'])]:
   sid='full-tokenizer-'+role+'-'+record['model'].lower();file=f'tokenizers/{model}/{revision}/tokenizer.json'
   tasks.append((record,role,{'id':sid,'title':model+' / tokenizer.json','modelId':model,'revision':revision,'path':'tokenizer.json','url':f'https://huggingface.co/{model}/blob/{revision}/tokenizer.json','rawUrl':f'https://huggingface.co/{model}/resolve/{revision}/tokenizer.json','cacheFile':file,'kind':'full_tokenizer','readScope':'完整 model.vocab、BPE merges、added_tokens 与 config 特殊 ID；原始词表不随研究包分发','license':'见固定底座/checkpoint 许可'}))
 def one(args):
  record,role,s=args;dest=CACHE/s['cacheFile']
  try:raw=dest.read_bytes() if dest.exists() else fetch(s['rawUrl'])
  except urllib.error.HTTPError as exc:
   if exc.code not in (401,403):raise
   return record,role,s,None,{'modelId':s['modelId'],'revision':s['revision'],'path':'tokenizer.json','httpStatus':exc.code,'status':'gated'}
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);s.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),accessed='2026-10-01')
  return record,role,s,json.loads(raw),None
 models={};gaps={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for r,role,s,data,gap in pool.map(one,tasks):
   name=r['model'];models.setdefault(name,{})[role]=data
   if gap:gaps.setdefault(name,[]).append(gap)
   else:sources[s['id']]=s;write_json(MANIFEST,{**manifest,'sources':list(sources.values())})
   print(name,role,'gated' if gap else f'{s["bytes"]} bytes',flush=True)
 results=[]
 for r in records:
  name=r['model'];pair=models[name];d=pair['distill'];b=pair.get('base');cfg=json.loads((DATA/'configs'/f'{name}-config.json').read_text())
  dv=d['model']['vocab'];added={a['id']:a['content'] for a in d['added_tokens']};all_ids=set(dv.values())|set(added)
  special={}
  inverse={v:k for k,v in dv.items()}
  for k in ('bos_token_id','eos_token_id','pad_token_id'):
   value=cfg.get(k);ids=value if isinstance(value,list) else [value] if isinstance(value,int) else []
   special[k]={'id':value,'present':all(i in all_ids for i in ids) if ids else None,'tokens':[added.get(i,inverse.get(i)) for i in ids]}
  assert max(all_ids)<cfg['vocab_size'] and all(v['present'] is not False for v in special.values()),name
  item={'model':name,'distillRevision':r['distillRevision'],'baseModel':r['baseModel'],'baseRevision':r['baseRevision'],'distillVocabularyEntries':len(dv),'distillMerges':len(d['model'].get('merges',[])),'addedTokens':len(d['added_tokens']),'maxTokenId':max(all_ids),'embeddingRows':cfg['vocab_size'],'specialTokens':special,'gaps':gaps.get(name,[]),'comparison':None,'scope':'完整词表与每个 merge/added token 比较；非 tokenizer 训练或模型质量实验'}
  if b is not None:
   bv=b['model']['vocab'];common=set(bv)&set(dv);changed={t:{'base':bv[t],'distill':dv[t]} for t in common if bv[t]!=dv[t]};bm=b['model'].get('merges',[]);dm=d['model'].get('merges',[])
   item['comparison']={'baseVocabularyEntries':len(bv),'vocabularyAdded':len(set(dv)-set(bv)),'vocabularyRemoved':len(set(bv)-set(dv)),'commonTokenIdsChanged':len(changed),'changedIdExamples':dict(list(sorted(changed.items()))[:12]),'mergeListIdentical':bm==dm,'baseMerges':len(bm),'differentMergePositions':sum(x!=y for x,y in zip(bm,dm))+abs(len(bm)-len(dm)),'addedTokensIdentical':b['added_tokens']==d['added_tokens'],'normalizerIdentical':b.get('normalizer')==d.get('normalizer'),'preTokenizerIdentical':b.get('pre_tokenizer')==d.get('pre_tokenizer'),'postProcessorIdentical':b.get('post_processor')==d.get('post_processor'),'decoderIdentical':b.get('decoder')==d.get('decoder')}
  results.append(item)
 write_json(DATA/'research/r1-full-tokenizer-audit.json',{'status':'passed_with_gated_baseline_gaps','records':results,'publicDistillsAudited':6,'publicBaselinesCompared':4,'gatedBaselines':2,'scope':'逐条完整 vocab/merge 检查，特殊 token ID 与 embedding 上限一致；不以字节差异自动推定编码行为全部相同'})


if __name__=='__main__':audit()
