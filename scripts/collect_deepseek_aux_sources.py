"""Collect fixed DSpark/MTP, tokenizer and backend source inputs for static analysis."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.error

from collect_deepseek_followup import DATA,CACHE,MANIFEST,fetch
from collect_deepseek_round1 import write_json


def collect():
 manifest=json.loads(MANIFEST.read_text());sources={s['id']:s for s in manifest['sources']}
 backend=Path.home()/'.cache/model-research-atlas/deepseek-backends';tasks=[]
 def add(repo,rev,path):
  sid='aux-'+repo.replace('/','-')+'-'+path.replace('/','-').replace('.','-')
  if sid in sources:return
  tasks.append({'id':sid,'title':repo+' / '+path,'repo':repo,'revision':rev,'path':path,'url':f'https://github.com/{repo}/blob/{rev}/{path}','rawUrl':f'https://raw.githubusercontent.com/{repo}/{rev}/{path}','cacheFile':f'code/{repo}/{rev}/{path}','kind':'reference_code' if path.endswith(('.py','.cu','.cpp','.h')) else 'official_document','readScope':'固定来源采集；具体符号/条件由研究证明记录','license':'见该固定 repo 的 LICENSE；完整代码不分发'})
 ds=json.loads((CACHE/'deepspec-tree.json').read_text())
 for path in ds['paths']:
  if path in ['README.md','LICENSE','NOTICE','requirements.txt'] or path.endswith('.py') and (path.startswith('deepspec/modeling/dspark/') or path.startswith('deepspec/eval/dspark/') or path in ['deepspec/eval/base_evaluator.py','deepspec/trainer/dspark_trainer.py']):add(ds['repo'],ds['revision'],path)
 desired={
  'vllm-project/vllm':['vllm/models/deepseek_v4/nvidia/dspark.py','vllm/models/deepseek_v4/nvidia/mtp.py','vllm/models/deepseek_v4/amd/dspark.py','vllm/model_executor/model_loader/mtp_validation.py','vllm/config/speculative.py','vllm/v1/worker/gpu/spec_decode/dspark/speculator.py','vllm/v1/worker/gpu/spec_decode/dspark/utils.py','vllm/v1/worker/gpu/spec_decode/adaptive_verification.py','vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py','vllm/v1/sample/rejection_sampler.py'],
  'vllm-project/vllm-ascend':['vllm_ascend/models/deepseek_mtp.py','vllm_ascend/models/deepseek_v4/dspark.py','vllm_ascend/models/deepseek_v4/__init__.py','vllm_ascend/spec_decode/dspark_proposer.py','vllm_ascend/worker/v2/spec_decode/dspark/speculator.py','vllm_ascend/patch/worker/patch_v2/patch_dspark.py','vllm_ascend/ops/triton/v2/spec_decode/resample.py','vllm_ascend/ops/triton/v2/spec_decode/prepare_dflash_inputs.py'],
 }
 for repo,paths in desired.items():
  tree=json.loads((backend/(repo.replace('/','-')+'-tree.json')).read_text())
  for path in paths:
   if path in tree['paths']:add(repo,tree['revision'],path)
  # Follow the actual filenames at this revision rather than inventing a standard proposer path.
  for path in tree['paths']:
   if path.startswith('vllm/v1/spec_decode/') and any(w in path for w in ('dspark','adaptive_verification','acceptance_estimator')) and path.endswith('.py'):add(repo,tree['revision'],path)
 tasks.append({'id':'dspark-paper-v1','title':'DSpark paper v1','revision':'2607.05147v1','path':None,'url':'https://arxiv.org/html/2607.05147v1','rawUrl':'https://arxiv.org/html/2607.05147v1','cacheFile':'paper/2607.05147v1.html','kind':'official_paper','readScope':'半自回归、Markov/置信度头、prefix survival 和吞吐剖面调度；不复验作者性能','license':'arXiv distribution; only original analysis and links are shipped'})
 tasks=[s for s in tasks if s['id'] not in sources]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  jobs=[(s,pool.submit(fetch,s['rawUrl'])) for s in tasks]
  for s,job in jobs:
   raw=job.result();path=CACHE/s['cacheFile'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
   s.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),accessed='2026-10-01');sources[s['id']]=s;write_json(MANIFEST,{**manifest,'sources':list(sources.values())});print(s['title'],len(raw),flush=True)
 print('Auxiliary fixed inputs collected:',len(sources),flush=True)


if __name__=='__main__':collect()
