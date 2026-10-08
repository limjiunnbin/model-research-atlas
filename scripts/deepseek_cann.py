"""Build CANN source availability and bridge evidence, separately from runtime validation."""
import hashlib
import re


def build_cann(e):
 from build_deepseek_atlas import DATA,read,write
 selection=read(DATA/'research/cann-source-selection.json');package=read(DATA/'research/cann-package-audit.json')
 ranges=[]
 def reference(sid,role,needle=None):
  source=e.sources[sid];lines=e.text(sid).splitlines()
  patterns={'api':r'aclnnStatus\s+aclnn\w+','definition':r'(class\s+\w+|OP_ADD|REGISTER_OP)','tiling':r'(REGISTER_OP_TILING|IMPL_OP_OPTILING|graphStatus|namespace)','kernel_entry':r'(extern "C"|__global__|__aicore__|namespace)','kernel_body':r'(class\s+\w+|void\s+(Process|Compute|Init)|namespace)','documentation':r'^#+\s+','binding':r'(EXEC_.*NPU.*CMD|DO_COMPATIBILITY|func: npu_swiglu)'}
  index=next((i for i,line in enumerate(lines) if needle in line),None) if needle else next((i for i,line in enumerate(lines) if re.search(patterns.get(role,r'\S'),line)),0)
  if index is None:return None
  start=max(0,index-2);end=min(len(lines),index+15);body='\n'.join(lines[start:end])
  record={'sourceId':sid,'repo':source['repo'],'revision':source['revision'],'path':source['path'],'role':role,'line':start+1,'end':end,'sourceSha256':source['sha256'],'rangeSha256':hashlib.sha256(body.encode()).hexdigest(),'url':source['url']+f'#L{start+1}-L{end}','level':'固定源码范围；未执行或编译；不是 Python AST 函数体证明'}
  ranges.append(record);return record
 operators=[]
 for item in selection['operators']:
  entry=dict(item);entry['references']=[];apis=set();observed=set();capabilities=set()
  for link in item['sources']:
   source=e.sources[link['sourceId']];text=e.text(source['id'])
   observed.update(re.findall(r'\baclnn[A-Z][A-Za-z0-9]+',text))
   apis.update(re.findall(r'aclnnStatus\s+(aclnn[A-Z][A-Za-z0-9]+)\s*\(',text))
   if link['role']=='documentation':apis.update(re.findall(r'^#+\s+(aclnn[A-Z][A-Za-z0-9]+)\b',text,re.M))
   capabilities.update(re.findall(r'\b(?:DT_FLOAT16|DT_BF16|DT_FLOAT|DT_INT8|DT_HIFLOAT8|DT_FLOAT8_E4M3FN|DT_FLOAT8_E5M2|DT_FLOAT4_E2M1|DT_FLOAT8_E8M0)\b',text))
   entry['references'].append(reference(source['id'],link['role']))
  entry['apiSymbols']=sorted(api for api in apis if not api.startswith('aclnnInner') and 'Test' not in api and not any(t in api for t in ('Common','Base','Process')))
  entry['apiSymbolsObservedInContext']=sorted(observed);entry['dtypeTokensObserved']=sorted(capabilities)
  entry['evidence']='对应 CANN v9.2.0-beta.2 的公开源码已读取；API/host/tiling/kernel 文件与范围逐一哈希。dtype token 集合是源码上下文，不是无条件 dtype 支持矩阵。'
  entry['runtimeStatus']='未编译、加载或执行；实际库/分支、ABI、数值与性能未实测（本次范围外）'
  if item['id'] in ('mhc-pre','mhc-post','rotary-position','add-rms-norm'):
   entry['relationship']='公开官方实现与框架 custom-op 独立列出；相似算子名/数学定义不证明框架直接调用该 CANN 实现'
  else:entry['relationship']='公开来源定位；具体 Python/框架调用须与下面 bridge 及已固定 vLLM-Ascend 上下文对应'
  operators.append(entry)
 plugin_sources=[s for s in e.sources.values() if s.get('repo')=='Ascend/op-plugin']
 bridges=[]
 for source in plugin_sources:
  if source['path'].endswith('.cpp'):
   text=e.text(source['id'])
   calls=sorted(set(re.findall(r'(?:EXEC_[A-Z_]*NPU[A-Z_]*CMD|DO_COMPATIBILITY)\s*\(\s*(aclnn[A-Za-z0-9_]+)',text)))
   if calls:
    refs=[reference(source['id'],'binding',call) for call in calls]
    bridges.append({'path':source['path'],'revision':source['revision'],'apiCandidates':calls,'references':refs,'scope':'同一函数中的 API 是条件候选集合，不是串行执行链；Python v2 后缀不等于 aclnn V2'})
  elif source['path'].endswith('op_plugin_functions.yaml'):
   for symbol in ('npu_swiglu(Tensor','exec: aclnnSwiGlu'):
    record=reference(source['id'],'binding',symbol)
    if record:bridges.append({'path':source['path'],'revision':source['revision'],'apiCandidates':['aclnnSwiGlu'],'references':[record],'scope':'npu_swiglu 的 gen_opapi schema 显式指定 aclnnSwiGlu；代码生成声明与实际安装构建分开'})
 fia=next(b for b in bridges if b['path'].endswith('FusedInferAttentionScoreV2KernelNpuOpApi.cpp'))
 assert {'aclnnFusedInferAttentionScoreV4','aclnnFusedInferAttentionScoreV5'}<=set(fia['apiCandidates'])
 known={api for o in operators for api in o['apiSymbols']}
 result={'snapshot':'2026-10-01','scope':'纠正此前将 CANN 笼统列为闭源的判断；30 个相关算子源码家族、独立 op-plugin bridge 与本地 SDK 安装证据。源码可读、安装库存在、编译正确性和设备实测分列。',
  'package':package,'tag':selection['tag'],'repositories':selection['repos'],'pluginRevision':selection['pluginRevision'],'operators':operators,'bridges':bridges,'codeRanges':ranges,
  'counts':{'operatorFamilies':len(operators),'bridgeFiles':len({b['path'] for b in bridges}),'bridgeEntries':len(bridges),'sourceFiles':len([s for s in e.sources.values() if s['id'].startswith('cann-')]),'codeRanges':len(ranges),'apiSymbols':len(known)},
  'findings':['CANN ops-transformer/ops-nn/ops-math 存在与本地 9.2.0-beta.2 对应的固定开源 tag，相关源文件的原始字节还核对 Git blob SHA-1。','npu_fused_infer_attention_score_v2 的所选 op-plugin 源码按设备/格式条件调用 aclnn V4 或 V5；不能只从 Python 名称推断 CANN API 版本。','MoE dispatch/combine 与 grouped_matmul 同样包含多版本/格式的条件选择，不按调用集合串成时序。','MhcPre/Post 官方算子与 vLLM-Ascend 自定义 HcPre/Post 记录为独立实现，未据名字认定直接接线。','本地 SDK 的 AscendC SwiGLU/RMSNorm/Matmul 头文件与实现可读取；所检查安装目录未发现三个独立算子库的 libopapi 与头文件，匹配 tag 的源码补充追踪。'],
  'limits':['本地 SDK 安装与公开 tag 的版本对应不证明二进制逐字节同源或与框架 ABI 兼容','所选 op-plugin 是独立源码快照，不宣称已经安装于虚拟机或与选定 vLLM-Ascend 组成兼容运行栈','未安装/更新虚拟机软件，未编译、加载模型、执行算子或进行设备/仿真实验；设备实验不在本次范围']}
 write(DATA/'research/cann-operator-audit.json',result);return result
