"""Collect fixed CANN operator sources and audit a local package without running an operator."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/families/deepseek'
CACHE=Path.home()/'.cache/model-research-atlas/deepseek-cann'
GIT_CACHE=Path.home()/'.cache/model-research-atlas/cann-source'
REPOS={
 'ops-transformer':'5f33f1e23d41fe0a047d01b7c7ae278777cac1f5',
 'ops-nn':'30ef7dd563c8a4b74c3161835c8e47d1d96f87b6',
 'ops-math':'0a2ce5b57caec6068d9e5658b740c2d41482aa15',
}
TAG='v9.2.0-beta.2'
PLUGIN_REV='d83570a35dfe0d8e9869c3ecfca6647cfccdd9c8'
# These are operator source families, not a claim that every model uses every branch.
PROFILES=[
 ('fused-infer-attention','ops-transformer','attention/fused_infer_attention_score',('mla','gqa','dspark'),'npu_fused_infer_attention_score / v2 / workspace / out'),
 ('sparse-flash-attention','ops-transformer','attention/sparse_flash_attention',('dsa',),'sparse attention dispatcher'),
 ('sparse-flash-mla','ops-transformer','attention/sparse_flash_mla',('compression',),'V4 sparse MLA'),
 ('mixed-quant-sparse-mla','ops-transformer','attention/mixed_quant_sparse_flash_mla',('compression','precision'),'V4 mixed-quant sparse MLA'),
 ('sparse-mla-metadata','ops-transformer','attention/sparse_flash_mla_metadata',('compression',),'metadata preparation'),
 ('lightning-indexer','ops-transformer','attention/lightning_indexer',('dsa',),'indexer'),
 ('lightning-indexer-v2','ops-transformer','attention/lightning_indexer_v2',('compression',),'V4 indexer'),
 ('quant-lightning-indexer','ops-transformer','attention/quant_lightning_indexer',('dsa',),'quant indexer'),
 ('quant-lightning-indexer-v2','ops-transformer','attention/quant_lightning_indexer_v2',('compression',),'_C_ascend.npu_quant_lightning_indexer_v2'),
 ('compressor','ops-transformer','attention/compressor',('compression',),'_C_ascend.compressor / aclnnCompressor'),
 ('mhc-pre','ops-transformer','mhc/mhc_pre',('mhc',),'official MhcPre; framework custom HcPreV2 kept separate'),
 ('mhc-post','ops-transformer','mhc/mhc_post',('mhc',),'official MhcPost; framework custom HcPost kept separate'),
 ('grouped-matmul','ops-transformer','gmm/grouped_matmul',('moe','precision'),'npu_grouped_matmul'),
 ('grouped-matmul-swiglu','ops-transformer','gmm/grouped_matmul_swiglu_quant',('moe',),'grouped_matmul_swiglu_quant'),
 ('grouped-matmul-swiglu-v2','ops-transformer','gmm/grouped_matmul_swiglu_quant_v2',('moe',),'V2 API alternative'),
 ('moe-dispatch-v2','ops-transformer','mc2/moe_distribute_dispatch_v2',('moe',),'npu_moe_distribute_dispatch_v2'),
 ('moe-dispatch-v3','ops-transformer','mc2/moe_distribute_dispatch_v3',('moe',),'plugin V3/V4 alternatives'),
 ('moe-combine-v2','ops-transformer','mc2/moe_distribute_combine_v2',('moe',),'npu_moe_distribute_combine_v2'),
 ('moe-combine-v3','ops-transformer','mc2/moe_distribute_combine_v3',('moe',),'plugin V3/V4 alternatives'),
 ('rotary-position','ops-transformer','posembedding/rotary_position_embedding',('mla','dsa','gqa'),'official rotary operator; custom inplace partial rotary is distinct'),
 ('rms-norm','ops-nn','norm/rms_norm',('precision',),'npu_rms_norm'),
 ('add-rms-norm','ops-nn','norm/add_rms_norm',('precision',),'npu_add_rms_norm; custom bias path is distinct'),
 ('swiglu','ops-nn','activation/swi_glu',('ffn','moe'),'npu_swiglu generated op_api'),
 ('swiglu-group-quant','ops-nn','activation/swiglu_group_quant',('moe','precision'),'npu_swiglu_group_quant'),
 ('matmul','ops-nn','matmul/mat_mul_v3',('ffn','precision'),'Mm/Matmul API candidates; not an unconditional GEMM dispatch'),
 ('quant-matmul','ops-nn','matmul/quant_batch_matmul_v4',('precision',),'quantized GEMM candidate'),
 ('dynamic-mx-quant','ops-nn','quant/dynamic_mx_quant',('precision',),'MX quantization'),
 ('add','ops-math','math/add',('precision',),'mathematical building block; not a per-step kernel assignment'),
 ('mul','ops-math','math/mul',('precision',),'mathematical building block; not a per-step kernel assignment'),
 ('cast','ops-math','experimental/math/cast',('precision',),'experimental Cast; availability not production selection'),
]

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def git_repo(repo):
 for name in (repo+'-metadata',repo+'-9.2.0-beta.2'):
  path=GIT_CACHE/name
  if (path/'.git').exists():
   assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()==REPOS[repo]
   return path
 path=GIT_CACHE/(repo+'-metadata');path.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','clone','--quiet','--depth','1','--filter=blob:none','--no-checkout','--branch',TAG,f'https://atomgit.com/cann/{repo}.git',str(path)],check=True,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()==REPOS[repo]
 return path

def source_bytes(source):
 for attempt in range(4):
  try:
   with urllib.request.urlopen(source['rawUrl'],timeout=30) as response:raw=response.read(8_000_001)
   assert len(raw)<=8_000_000
   if source.get('gitBlobSha1'):assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==source['gitBlobSha1'],source['id']
   return raw
  except (OSError,TimeoutError):
   if attempt==3:raise
   time.sleep(min(2**attempt,4))

def selected_files(paths,prefix):
 files=[p for p in paths if p.startswith(prefix+'/') and '/tests/' not in p and '/test/' not in p]
 roles={}
 for role,items,limit in [
  ('documentation',[p for p in files if '/docs/' in p and p.endswith('.md') and '/docs/en/' not in p],10),
  ('api',[p for p in files if '/op_api/' in p and Path(p).name.startswith('aclnn_') and p.endswith(('.cpp','.h'))],40),
  ('definition',[p for p in files if '/op_host/' in p and p.endswith('_def.cpp')],1),
  ('tiling',[p for p in files if '/op_host/' in p and 'tiling' in Path(p).name and p.endswith('.cpp')],2),
  ('kernel_entry',[p for p in files if '/op_kernel/' in p and p.endswith('.cpp')],2),
  ('kernel_body',[p for p in files if '/op_kernel/' in p and p.endswith(('.h','.hpp')) and not any(t in p for t in ('tiling','struct','utils','common','interface','config','key'))],2),
 ]:
  # Cover an arch35 branch when present, then retain a separate generic/earlier-generation path.
  ordered=sorted(items,key=lambda p:('arch35' not in p,p.count('/'),p))
  for p in ordered[:limit]:roles[p]=role
 return roles

def collect():
 source_map={};profiles=[]
 def add(repo,rev,path,role):
  sid='cann-'+repo.replace('/','-')+'-'+path.replace('/','-').replace('.','-')
  source_map[sid]={'id':sid,'repo':repo,'revision':rev,'path':path,'title':repo+' / '+path,'url':f'https://atomgit.com/{repo}/blob/{rev}/{path}' if repo.startswith('cann/') else f'https://github.com/{repo}/blob/{rev}/{path}',
   'rawUrl':f'https://raw.gitcode.com/{repo}/raw/{rev}/{path}' if repo.startswith('cann/') else f'https://raw.githubusercontent.com/{repo}/{rev}/{path}',
   'cacheFile':f'code/{repo}/{rev}/{path}','kind':'official_document' if role in ('documentation','license') else 'reference_code','readScope':'fixed '+role+'; static CANN API/host/tiling/kernel availability; no build or device execution','license':'upstream LICENSE at the fixed revision; complete source is not redistributed','accessed':'2026-10-01','role':role}
  return sid
 for repo in REPOS:
  cwd=git_repo(repo);paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],cwd=cwd,text=True).splitlines()
  for p in ('README.md','LICENSE'):
   if p in paths:add('cann/'+repo,REPOS[repo],p,'license' if p=='LICENSE' else 'documentation')
  for oid,project,prefix,topics,caller in PROFILES:
   if project!=repo:continue
   files=selected_files(paths,prefix);assert files,(repo,prefix)
   profiles.append({'id':oid,'repo':'cann/'+repo,'revision':REPOS[repo],'tag':TAG,'directory':prefix,'topics':list(topics),'frameworkContext':caller,'sources':[{ 'sourceId':add('cann/'+repo,REPOS[repo],p,role),'role':role} for p,role in files.items()]})
  rows=subprocess.check_output(['git','ls-tree','-r','HEAD'],cwd=cwd,text=True).splitlines()
  hashes={line.split('\t',1)[1]:line.split()[2] for line in rows}
  for source in source_map.values():
   if source['repo']=='cann/'+repo:source['gitBlobSha1']=hashes[source['path']]
 plugin=GIT_CACHE/'op-plugin'
 keep=('FusedInferAttentionScore','GroupedMatmul','MoeDistributeDispatchV2','MoeDistributeCombineKernelV2','RmsNormKernelOpApi','AddRmsNormV2','QuantLightningIndexer','SparseFlashAttentionKernel','SwigluGroupQuantKernel','MatmulKernelNpuOpApi','MmKernelNpuOpApi','QuantMatmulKernelNpuOpApi')
 for p in plugin.rglob('*'):
  if p.is_file() and (p.name=='LICENSE' or p.name=='op_plugin_functions.yaml' or p.suffix=='.cpp' and any(x in p.name for x in keep)):
   relative=p.relative_to(plugin).as_posix();sid=add('Ascend/op-plugin',PLUGIN_REV,relative,'binding');dest=CACHE/source_map[sid]['cacheFile'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 def download(source):
  dest=CACHE/source['cacheFile'];raw=dest.read_bytes() if dest.exists() else source_bytes(source)
  if source.get('gitBlobSha1'):assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==source['gitBlobSha1']
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);source.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for index,_ in enumerate(pool.map(download,source_map.values()),1):
   if index%35==0:print('CANN inputs',index,'/',len(source_map),flush=True)
 dump(DATA/'cann-sources.json',{'snapshot':'2026-10-01','sources':list(source_map.values()),'scope':'CANN v9.2.0-beta.2 operator source families plus a separately fixed op-plugin bridge; source availability is distinct from the installed operator binaries'})
 dump(DATA/'research/cann-source-selection.json',{'tag':TAG,'repos':REPOS,'pluginRevision':PLUGIN_REV,'operators':profiles})
 print('Fixed CANN source files:',len(source_map),'; operator families:',len(profiles),flush=True)

def audit_package(instance):
 # Output only vendor-relative paths and selected metadata; exclude instance/host/user/network information.
 script='''import pathlib,json,hashlib,platform
root=pathlib.Path.home()/"Ascend/cann"
assert root.exists(),"CANN installation not found"
versions=[]
for p in sorted((root/"share/info").glob("*/version.info")):
 raw=p.read_bytes(); fields=dict(line.split("=",1) for line in raw.decode().splitlines() if "=" in line)
 versions.append({"path":str(p.relative_to(root)),"sha256":hashlib.sha256(raw).hexdigest(),"version":fields.get("Version")})
sources=[]
for relative in ("compiler/version.info","opp/version.info","aarch64-linux/asc/include/adv_api/activation/swiglu.h","aarch64-linux/asc/impl/adv_api/detail/activation/swiglu/swiglu_3510_impl.h","aarch64-linux/asc/include/adv_api/normalization/rmsnorm.h","aarch64-linux/asc/include/adv_api/matmul/matmul.h"):
 p=root/relative
 if p.is_file():
  raw=p.read_bytes();sources.append({"path":relative,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"lines":len(raw.decode(errors="replace").splitlines())})
headers=sorted(str(p.relative_to(root)) for p in (root/"aarch64-linux/include/aclnnop").rglob("*.h"))
libraries=sorted(str(p.relative_to(root)) for p in (root/"aarch64-linux/lib64").glob("*opapi*.so"))
print(json.dumps({"capture":"local Multipass installation; static file inspection","version":(root/"opp/version.info").read_text().splitlines()[0].split("=",1)[1],"architecture":platform.machine(),"componentVersions":versions,"files":sources,"aclnnopHeaders":headers,"opapiLibraries":libraries,"operatorLibraryComponents":[v for v in versions if pathlib.Path(v["path"]).parts[2].startswith("ops-")],"operatorExecution":False}))
'''
 raw=subprocess.check_output(['multipass','exec',instance,'--','python3','-'],input=script.encode())
 value=json.loads(raw);assert value['version']=='9.2.0-beta.2'
 value['scope']='本地安装提供编译器/AscendC SDK 与高级 API 源码；所检查安装路径未发现 ops-nn/ops-math/ops-transformer 的 libopapi 和独立算子头文件。不能将 SDK 高级 API 等同已安装完整 aclnn 算子库。对应算子源码继续由匹配 tag 核对；未安装、编译、加载或运行任何算子。'
 dump(DATA/'research/cann-package-audit.json',value)
 print('CANN package audited:',value['version'],len(value['componentVersions']),'components; no operator execution')

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--multipass-instance');parser.add_argument('--sources-only',action='store_true');args=parser.parse_args()
 if args.multipass_instance:audit_package(args.multipass_instance)
 if not args.multipass_instance or args.sources_only:collect()
