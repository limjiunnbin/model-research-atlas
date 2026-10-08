"""Package the validated research assets and update download hashes without changing facts."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/families/deepseek';OUT=ROOT/'dist/assets/deepseek'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def packaged_markdown(text,mapping,prefixes=()):
 def link(match):
  label,target=match.groups();path,separator,anchor=target.partition('#')
  if path.startswith(('https://','http://')) or not path:return match.group(0)
  destination=mapping.get(path)
  for source,replacement in prefixes:
   if destination is None and path.startswith(source):destination=replacement+path[len(source):]
  if destination is None:return label+'（仓库文件 `'+target+'`）'
  return '['+label+']('+destination+(separator+anchor if separator else '')+')'
 return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
def archive(target,files):
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,path in sorted(files):
   info=zipfile.ZipInfo(name,(2026,10,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
   z.writestr(info,path.read_bytes(),compresslevel=9)
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
def package():
 family=read(DATA/'family.json');compute=read(DATA/'compute.json');manifest=read(DATA/'research/export-manifest.json')
 from research_exports import export
 export(ROOT,family)
 shutil.copyfile(DATA/'compute.json',OUT/'DeepSeek-compute.json')
 shutil.copyfile(DATA/'research/atlas-validation.json',OUT/'DeepSeek-validation.json')
 shutil.copyfile(DATA/'research/site-proofs.json',OUT/'DeepSeek-source-proofs.json')
 shutil.copyfile(DATA/'research/r1-config-tokenizer-comparison.json',OUT/'R1-config-tokenizer-comparison.json')
 for source,target in [('speculation.json','DeepSeek-MTP-DSpark.json'),('r1-full-tokenizer-audit.json','R1-full-tokenizer-audit.json'),('backend-trace.json','DeepSeek-backend-trace.json')]:shutil.copyfile(DATA/'research'/source,OUT/target)
 shutil.copyfile(DATA/'research/cann-operator-audit.json',OUT/'DeepSeek-CANN-operators.json')
 archive(OUT/'DeepSeek-header-audits.zip',[(p.name,p) for p in (DATA/'research/header-audits').glob('*.json')])
 readme=(ROOT/'research/deepseek/README.md').read_text().replace('../../dist/assets/deepseek/','').replace('../../roadmap_deepseek.md','DeepSeek-roadmap.md').replace('(v4.1-flash.md)','(DeepSeek-V4.1-Flash-研究.md)')
 (OUT/'README.md').write_text(readme)
 roadmap=packaged_markdown((ROOT/'roadmap_deepseek.md').read_text(),{'research/deepseek/README.md':'README.md','research/deepseek/v4.1-flash.md':'DeepSeek-V4.1-Flash-研究.md','research/deepseek/validation/browser-validation.json':'metadata/browser-validation.json','data/families/deepseek/research/v3-round1.json':'round1/v3-round1.json'},[('dist/assets/deepseek/',''),('research/deepseek/round1/','round1/')])
 (OUT/'DeepSeek-roadmap.md').write_text(roadmap)
 metadata=OUT/'metadata';metadata.mkdir(exist_ok=True)
 shutil.copytree(DATA/'research/r1-tokenizers',metadata/'r1-tokenizers',dirs_exist_ok=True)
 public_family=dict(family);public_family['downloads']=[]
 write(metadata/'family.json',public_family)
 for name in ['report.json','hardware.json','sources.json','extended-sources.json','backend-sources.json','followup-sources.json','cann-sources.json','analysis.json','analysis-sources.json']:
  shutil.copyfile(DATA/name,metadata/name)
 for path in DATA.glob('*-architecture.json'):
  if path.name!='v4.1-flash-architecture.json':shutil.copyfile(path,metadata/path.name)
 for name in ['export-manifest.json','source-validation.json','schema-validation.json','privacy-validation.json','cann-package-audit.json','cann-source-selection.json','analysis-proofs.json','analysis-validation.json','analysis-math-validation.json','analysis-export-manifest.json','analysis-export-validation.json','analysis-scenarios.json']:
  source=DATA/'research'/name
  if source.is_file():shutil.copyfile(source,metadata/name)
 browser=ROOT/'research/deepseek/validation/browser-validation.json'
 if browser.is_file():shutil.copyfile(browser,metadata/'browser-validation.json')
 # The early report stays an explicitly historical round-one package, not a current completion declaration.
 early=OUT/'round1';early.mkdir(exist_ok=True)
 for p in (ROOT/'research/deepseek/round1').iterdir():
  if p.suffix in ('.md','.csv','.json') and p.name!='tables.json':
   if p.suffix=='.md':
    early_text=packaged_markdown(p.read_text(),{'../README.md':'../README.md','report.md':'report.md','v3-parameters.csv':'v3-parameters.csv','v3-representative-operators.csv':'v3-representative-operators.csv','validation.json':'validation.json','../../../roadmap_deepseek.md':'../DeepSeek-roadmap.md','../../../data/families/deepseek/research/v3-round1.json':'v3-round1.json','../../../data/families/deepseek/versions.json':'versions.json','../../../data/families/deepseek/sources.json':'sources.json'})
    (early/p.name).write_text(early_text)
   else:shutil.copyfile(p,early/p.name)
 for source,name in [(DATA/'research/v3-round1.json','v3-round1.json'),(DATA/'versions.json','versions.json'),(DATA/'sources.json','sources.json')]:shutil.copyfile(source,early/name)
 fields='''# DeepSeek 计算与导出字段

全层 CSV 覆盖选定 21 个 checkpoint 的主干、MTP 与 DSpark，用同一 compute.json 的模板和层映射展开。prefill/decode 分开，metadata、共享别名与逻辑权重记录分开；重复阶段引用同一份权重，不能把逐行参数相加为模型总数。DSpark prefilling 只初始化 context cache，矩阵声明行不等于每阶段必然执行。

B 是 padded 参考 batch，L_q 是本次新增长度，常规 decode=1；L_kv 是可见历史+本次长度。N_tok=B×L_q 仅适用于本参考口径；N_e 动态，专家模板份数与 per-token top-k 分开。V4 的 C=floor(L_kv/r)，窗口、压缩记录、indexer cache 与 FP32 状态另计，未完成压缩窗口不生成新 token。

source_index_0based 是实际模板索引：主干为 display_layer−1，普通 V4 MTP namespace 为 mtp.0，DSpark 三个 stage 为 mtp.0/1/2，V3 权重 MTP 层号为 61。参考函数/源码调用集合是上下文证据，不是实际 kernel 顺序。框架模块、dispatcher 与 kernel 层级不可互换。

全部 21 版的 stored shape/dtype/payload 来自各自全分片 header；原始 V3 为 163 分片。FP4/I8 容器、F8_E8M0/F32 scale 和 I64 非训练表分别统计；逻辑权重形状与存储形状独立，cache 精度模拟不当作实测存储。DSpark 的 mtp.* namespace、3 个 stage、HF nextn=1、block=5 和服务示例 speculative_tokens=7 分别保留。共享 embedding/head 存储副本排除逻辑重复计数，训练别名与权重值相同仍未由 header 证明。

duration 为空表示未测，不能补 0 或预测值。设备实验已按用户要求移出本次范围，没有 profiler、吞吐或质量实测。工作簿是独立研究格式，与仓库未提供的原始私有 Excel 模板不作兼容声明。

CSV 为 UTF-8 BOM、CRLF；XLSX 中主干参数等数字保持数值类型。原始配置、当前来源/函数哈希、完整 JSON 与验证记录在报告包中。源码与权重不随包分发。
'''
 (OUT/'字段与符号说明.md').write_text(fields)
 files=[(Path(p['path']).name,ROOT/'dist'/p['path']) for p in manifest['perModel']]
 archive(OUT/'DeepSeek-per-model-tables.zip',files+[('字段与符号说明.md',OUT/'字段与符号说明.md')])
 paths=['DeepSeek-研究报告.md','README.md','DeepSeek-roadmap.md','DeepSeek-all-layer-shapes.csv','DeepSeek-模型与算子.xlsx','DeepSeek-compute.json','DeepSeek-source-proofs.json','DeepSeek-validation.json','R1-config-tokenizer-comparison.json','R1-full-tokenizer-audit.json','DeepSeek-MTP-DSpark.json','DeepSeek-backend-trace.json','DeepSeek-CANN-operators.json','DeepSeek-header-audits.zip','v3-header-audit.json','字段与符号说明.md',
        'DeepSeek-implementation-hardware-ascend.md','DeepSeek-sources.csv','DeepSeek-per-model-tables.zip']
 paths += ['DeepSeek-静态分析.md','DeepSeek-静态分析.json','DeepSeek-静态分析.xlsx','DeepSeek-cost-scenarios.csv','DeepSeek-parallel-scenarios.csv','DeepSeek-CANN-contracts.csv','跨家族比较.json','跨家族比较.csv']
 extras=[('configs/'+p.name,p) for p in (DATA/'configs').glob('*-config.json')]+[(p.relative_to(OUT).as_posix(),p) for sub in ('figures','metadata','round1') for p in (OUT/sub).rglob('*') if p.is_file()]
 archive(OUT/'DeepSeek-研究报告包.zip',[(p,OUT/p) for p in paths]+extras)
 paths.append('DeepSeek-研究报告包.zip')
 family['downloads']=[{'name':p,'path':'assets/deepseek/'+p,'bytes':(OUT/p).stat().st_size,'sha256':hashlib.sha256((OUT/p).read_bytes()).hexdigest()} for p in paths]
 compute['downloads']=[d for d in family['downloads'] if d['name'] in ['DeepSeek-all-layer-shapes.csv','DeepSeek-per-model-tables.zip','DeepSeek-compute.json','字段与符号说明.md','DeepSeek-研究报告.md','DeepSeek-模型与算子.xlsx']]
 # Avoid recursive hashes: the downloadable compute JSON intentionally omits its own downloads manifest.
 downloadable=dict(compute);downloadable['downloads']=[]
 (OUT/'DeepSeek-compute.json').write_text(json.dumps(downloadable,ensure_ascii=False,indent=2)+'\n')
 entry=next(d for d in family['downloads'] if d['name']=='DeepSeek-compute.json');raw=(OUT/entry['name']).read_bytes();entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 compute['downloads']=[dict(d) for d in family['downloads'] if d['name'] in ['DeepSeek-all-layer-shapes.csv','DeepSeek-per-model-tables.zip','DeepSeek-compute.json','字段与符号说明.md','DeepSeek-研究报告.md','DeepSeek-模型与算子.xlsx']]
 write(DATA/'compute.json',compute);write(DATA/'family.json',family)
 # Rebuild the deterministic report archive after updating the downloadable compute file.
 archive(OUT/'DeepSeek-研究报告包.zip',[(p,OUT/p) for p in paths if p!='DeepSeek-研究报告包.zip']+extras)
 raw=(OUT/'DeepSeek-研究报告包.zip').read_bytes();entry=next(d for d in family['downloads'] if d['name']=='DeepSeek-研究报告包.zip');entry.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());write(DATA/'family.json',family)
 if (DATA/'v4.1-flash-architecture.json').is_file():
  from build_deepseek_v41 import package_current
  package_current()
  family=read(DATA/'family.json')
 print('DeepSeek package:',len(family['downloads']),'downloads;',manifest['expandedRows'],'expanded CSV rows')
if __name__=='__main__':package()
