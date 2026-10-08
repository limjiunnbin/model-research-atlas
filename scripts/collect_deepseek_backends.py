"""Snapshot selected public backend files; fixed commits are metadata, not a tested compatible stack."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect_deepseek_round1 import fetch,write_json

ROOT=Path(__file__).resolve().parents[1]; CACHE=Path.home()/'.cache/model-research-atlas/deepseek-backends'
REVISIONS={'vllm-project/vllm-ascend':'a8fcedb03d93e60efceddbfc912406f7fa491d57','vllm-project/vllm':'bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f','deepseek-ai/TileKernels':'66258df6175d2f630ffecb04c5ab66bff8a2ae6a','huggingface/transformers':'984bc11b0882ff1e5b34ba717ea357e069ceced9'}
SELECTION={
 'vllm-project/vllm-ascend':[
  'README.md','LICENSE','vllm_ascend/attention/mla_v1.py','vllm_ascend/attention/dsa_v1.py',
  'vllm_ascend/ops/mla.py','vllm_ascend/ops/dsa.py','vllm_ascend/ops/linear.py','vllm_ascend/ops/layernorm.py',
  'vllm_ascend/ops/fused_moe/fused_moe.py','vllm_ascend/ops/fused_moe/moe_mlp.py',
  'vllm_ascend/ops/fused_moe/token_dispatcher.py','vllm_ascend/quantization/methods/w8a8/fp8_block.py',
  'vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py',
  'vllm_ascend/models/deepseek_v4/model.py','vllm_ascend/models/deepseek_v4/compressor.py',
  'vllm_ascend/models/deepseek_v4/indexer.py','vllm_ascend/models/deepseek_v4/mtp.py',
 ],
 'vllm-project/vllm':['LICENSE','vllm/model_executor/models/deepseek_v2.py','vllm/model_executor/models/deepseek_mtp.py',
                     'vllm/model_executor/layers/attention/mla_attention.py','vllm/model_executor/layers/attention/sparse_mla_attention.py'],
 'deepseek-ai/TileKernels':['README.md','LICENSE','tile_kernels/config.py',
  'tile_kernels/mhc/pre_big_fuse_kernel.py','tile_kernels/mhc/pre_big_fuse_asc.py',
  'tile_kernels/mhc/post_kernel.py','tile_kernels/mhc/post_asc.py','tile_kernels/mhc/sinkhorn_kernel.py','tile_kernels/mhc/sinkhorn_asc.py',
  'tile_kernels/moe/topk_gate_kernel.py','tile_kernels/moe/topk_gate_asc.py'],
 'huggingface/transformers':['src/transformers/models/qwen2/modeling_qwen2.py','src/transformers/models/llama/modeling_llama.py','src/transformers/modeling_utils.py'],
}

def collect():
 sources=[]; tasks=[]
 for repo,paths in SELECTION.items():
  revision=REVISIONS[repo]
  for path in paths:
   tasks.append((repo,revision,path))
 def one(args):
  repo,revision,path=args; url=f'https://raw.githubusercontent.com/{repo}/{revision}/{path}';raw=fetch(url)
  file=f'{repo}/{revision}/{path}'; target=CACHE/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  sid='hf-transformers-'+Path(path).name.replace('.','-') if repo=='huggingface/transformers' else repo.replace('/','-')+'-'+path.replace('/','-').replace('.','-')
  return {'id':sid,'title':repo+' / '+path,
          'repo':repo,'revision':revision,'path':path,'url':f'https://github.com/{repo}/blob/{revision}/{path}',
          'rawUrl':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'cacheFile':file,
          'accessed':'2026-10-01','kind':'backend_code' if path.endswith('.py') else 'backend_document',
          'license':'see fixed LICENSE','readScope':'采集；函数核验范围随后写入后端证据表'}
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for source in pool.map(one,tasks):sources.append(source);print(source['title'],source['bytes'])
 write_json(ROOT/'data/families/deepseek/backend-sources.json',{'schemaVersion':1,'familyId':'deepseek','accessed':'2026-10-01','scope':'独立固定源码快照；未据此声明兼容运行栈','sources':sources})

if __name__=='__main__':collect()
