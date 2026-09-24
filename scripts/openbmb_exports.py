"""Export configuration-derived logical shapes, explicitly separating runtime unknowns."""
from pathlib import Path
import json,csv,io,zipfile,hashlib,collections
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data/openbmb';OUT=ROOT/'dist/reports/openbmb/compute';OUT.mkdir(parents=True,exist_ok=True)
source=json.loads((SRC/'sources.json').read_text());layers=[];steps=[];coverage=[]
UNKNOWN='未核验';linear_api='torch.nn.functional.linear（数学参考，非模型调用链确认）'
def emit(base,op,ins,outs,weight='',note='',api=linear_api,evidence='配置与公式推导'):
 steps.append(dict(**base,step=len([s for s in steps if s['model']==base['model'] and s['component']==base['component'] and s['layer_1based']==base['layer_1based']])+1,operator=op,input_shape=ins,output_shape=outs,logical_weight_shape=weight or UNKNOWN,torch_reference=api,evidence=evidence,note=note or '逻辑数学分解；不证明物理张量命名、融合方式或权重格式'))
def component(meta,c,path,kind='language',count=None,opaque=False):
 n=count or c.get('num_hidden_layers') or c.get('encoder_layers') or c.get('num_layers');h=c.get('hidden_size',c.get('d_model',c.get('hidden_dim')));heads=c.get('num_attention_heads',c.get('encoder_attention_heads',c.get('num_heads')));kv=c.get('num_key_value_heads',heads);d=c.get('head_dim',c.get('kv_channels'));d=d or (h//heads if h and heads and h%heads==0 else None);ff=c.get('intermediate_size',c.get('encoder_ffn_dim',c.get('ffn_dim')))
 if not n:return
 for i in range(n):
  typ=(c.get('layer_types') or c.get('mixer_types') or ['full_attention']*n)[i]
  if 'q_lora_rank' in c:typ='MLA'
  if kind=='vision':typ='vision_attention'
  if opaque:typ='specialized_unverified'
  base=dict(model=meta['model'],component=path,layer_1based=i+1,config_revision=meta['revision'],config_source=meta['url'],config_pointer='/'+path.replace('.','/') if path!='root' else '/',attention_type=typ)
  layers.append(dict(**base,hidden_size=h or UNKNOWN,heads=heads or UNKNOWN,kv_heads=kv or UNKNOWN,head_dim=d or UNKNOWN,intermediate_size=ff or UNKNOWN,head_dim_evidence='config' if ('head_dim'in c or 'kv_channels'in c) else 'hidden_size/heads 推导' if d else UNKNOWN,layer_scope='配置层；drop/truncation/运行路径另行核验'))
  x=f'[B,T,{h}]' if h else '[B,T,H_unknown]'
  emit(base,'layer_boundary',x,x,api=UNKNOWN,evidence='配置接口',note='T取决于文本、视觉压缩/切片或音频块；并非真实运行张量记录')
  if typ in ['linear_attention','lightning-attn','specialized_unverified']:
   emit(base,'recurrent_or_specialized_core',x,x,note='状态布局、门控/卷积、累加精度和专用核调用未确认；不能套用完整attention',api=UNKNOWN,evidence='结构类型确认；内部shape未知')
  elif typ=='MLA':
   rq=c['q_lora_rank'];rk=c['kv_lora_rank'];dr=c['qk_rope_head_dim'];dn=c['qk_nope_head_dim']
   emit(base,'query_low_rank_down',x,f'[B,T,{rq}]',f'[{rq},{h}]')
   emit(base,'query_low_rank_up',f'[B,T,{rq}]',f'[B,T,{heads*(dn+dr)}]',f'[{heads*(dn+dr)},{rq}]')
   emit(base,'kv_compression_plus_rope',x,f'[B,T,{rk+dr}]',f'[{rk+dr},{h}]')
   emit(base,'MLA_attention_and_output',f'Q:[B,{heads},T,{dn+dr}]; KV-compressed:[B,S,{rk}]',x,note='value head尺寸/展开与吸收实现未按checkpoint代码确认，保留未知',api=UNKNOWN,evidence='部分shape推导')
  elif all([h,heads,kv,d]):
   for name,k in [('q',heads),('k',kv),('v',kv)]:
    gate=c.get('attn_output_gate',False) and name=='q'
    emit(base,name+'_logical_projection',x,f'[B,T,{k*d}]',UNKNOWN if gate else f'[{k*d},{h}]',note='仅Q逻辑分支；额外输出门及实际融合权重布局未确认' if gate else '')
   emit(base,'attention_core',f'Q:[B,{heads},T,{d}]; K,V:[B,{kv},S,{d}]',f'[B,{heads},T,{d}]',api='torch.nn.functional.scaled_dot_product_attention（数学参考；mask/GQA/位置编码须适配）',note='这是选定KV上的数学参考，非稠密执行声明；MiniCPM4/4.1及SALA的minicpm4分支可能涉及评分、TopK选块与稀疏分派，这些未展开。prefill T为输入长度，decode通常T=1；S为参与读取KV长度。不得物化[B,H,T,S]来估算融合核内存。')
   emit(base,'output_logical_projection',f'[B,T,{heads*d}]',x,f'[{h},{heads*d}]')
  if h and ff and not opaque:
   experts=c.get('num_experts',1)
   emit(base,'FFN_expand_per_expert' if experts>1 else 'FFN_expand',x,f'[B,T,{ff}]',f'[{ff},{h}]',note=f'每个专家的数学分支；专家总数{experts}，top-k={c.get("num_experts_per_tok",UNKNOWN)}；门控分支/激活/调度另行确认' if experts>1 else '扩展分支的逻辑shape；实际门控分支与融合布局未在本表确认为权重张量')
   emit(base,'FFN_contract_per_expert' if experts>1 else 'FFN_contract',f'[B,T,{ff}]',x,f'[{h},{ff}]')
  emit(base,'normalization_position_residual',x,x,api=UNKNOWN,evidence='待实现核验',note='归一化类型、RoPE/位置编码、残差缩放及先后顺序不可只靠形状确认；不计作已审计算子')
for meta in source['models']:
 raw=(SRC/'configs'/(meta['model']+'.json')).read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['sha256'];c=json.loads(raw);before=len(layers)
 def walk(x,path='root'):
  if not isinstance(x,dict):return
  if any(k in x for k in ['num_hidden_layers','encoder_layers','num_layers']):component(meta,x,path,'vision' if 'vision'in path else 'language',opaque=('encoder_config' in path or 'dit_config' in path))
  for key,val in x.items():
   if isinstance(val,dict) and key not in ['rope_scaling']:walk(val,key if path=='root' else path+'.'+key)
 walk(c)
 if c.get('residual_lm_num_layers'):
  rc=dict(c['lm_config']);component(meta,rc,'residual_lm',count=c['residual_lm_num_layers'],opaque=meta['model']!='VoxCPM2')
 base=dict(model=meta['model'],component='global_and_unresolved',layer_1based=0,config_revision=meta['revision'],config_source=meta['url'],config_pointer='/',attention_type='global')
 emit(base,'embedding_head_connectors_and_optional_modules',UNKNOWN,UNKNOWN,api=UNKNOWN,evidence='未完成权重/全模型调用链审计',note='此行显式保留embedding、lm_head、模态连接/压缩、VAE/波形解码或动作生成中未逐算子展开的部分；不是全模型完整计算图')
 if meta['model']=='MiniCPM-RobotTrack':
  widths=[c['backbone_config']['hidden_size'],4096,1024,512,256,128,c['num_waypoints']*c['action_dim']]
  for i,(a,b) in enumerate(zip(widths,widths[1:]),1):emit({**base,'component':'FunnelTrajectoryHead','layer_1based':i},'trajectory_linear',f'[B,{a}]',f'[B,{b}]',f'[{b},{a}]',note='源码确认6个Linear；最后reshape为[B,8,3]，GELU/LayerNorm/Dropout/tanh见源码',api='torch.nn.Linear',evidence='固定源码与配置确认')
 coverage.append({'model':meta['model'],'expanded_config_layers':len(layers)-before,'coverage':'配置层与逻辑计算形状；专用组件部分未知','weight_header_audit':'未执行','runtime_backend_validation':'未执行','source':meta['url']})
def csv_text(rows):
 out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);return '\ufeff'+out.getvalue()
backends=[]
for s in steps:
 backends.append({k:s[k] for k in ['model','component','layer_1based','step','operator','torch_reference','config_source']}|{'NVIDIA_CUDA':'未核验具体模型分派/内核；普通PyTorch语义不等于已验证CUDA路径','AMD_ROCm':'未核验具体模型分派/内核；ROCm可复用torch接口，但不证明扩展兼容','Ascend':'未核验逐算子API与完整运行栈；不可从CUDA或通用架构推定支持','backend_evidence':'未运行模型/硬件；不提供虚构厂商接口名'})
for name,rows in [('OpenBMB-layer-config.csv',layers),('OpenBMB-layer-compute-shapes.csv',steps),('OpenBMB-backend-api-map.csv',backends),('OpenBMB-model-coverage.csv',coverage)]: (OUT/name).write_text(csv_text(rows),encoding='utf-8')
# Verified repository-level interfaces, deliberately not claimed as every checkpoint's call path.
impl=source['implementation_sources'];sparse=next(x for x in impl if x['repo']=='infllmv2_cuda_impl');robot=next(x for x in impl if x['repo']=='MiniCPM-Robot');vox=next(x for x in impl if x['repo']=='VoxCPM')
apis=[dict(scope='InfLLM-V2 CUDA仓库',api=a,source=sparse['url'],evidence='固定README声明；未执行、未确认每个checkpoint启用') for a in ['infllmv2_attn_stage1','infllmv2_sparse_attn_fwd','infllmv2_sparse_attn_bwd']]+[dict(scope='RobotTrack动作头',api='FunnelTrajectoryHead.forward / torch.nn.Linear',source=robot['url'],evidence='固定源码确认'),dict(scope='VoxCPM2',api='MiniCPMModel / VoxCPMLocEnc / VoxCPMLocDiTV2 / UnifiedCFM',source=vox['url'],evidence='固定源码构造路径确认；不等于专用后端API')]
(OUT/'OpenBMB-verified-implementation-interfaces.csv').write_text(csv_text(apis),encoding='utf-8')
notes=f'''# OpenBMB 逐层表：字段、来源与覆盖边界

研究日期：2026-09-24。覆盖原报告的 **17 个具体模型**，展开 {len(layers)} 个配置层、{len(steps)} 行逻辑步骤。不是 Kimi 表，也不是权重头审计或已验证的全模型部署图。

## 下载与阅读

- [逐层配置 CSV](OpenBMB-layer-config.csv)
- [逐层逻辑 shape / 计算步骤 CSV](OpenBMB-layer-compute-shapes.csv)
- [PyTorch / NVIDIA / AMD / Ascend 映射 CSV](OpenBMB-backend-api-map.csv)
- [已核验实现接口 CSV](OpenBMB-verified-implementation-interfaces.csv)
- [17 模型覆盖说明 CSV](OpenBMB-model-coverage.csv)
- [按模型拆分 ZIP](OpenBMB-per-model-tables.zip)

## 如何理解这些表

配置字段是固定 checkpoint 的公开信息；矩阵形状是根据配置推导的数学表示，不声称与权重文件中真实张量名称、融合打包或量化载荷一致。未下载权重，未运行模型。`layer_1based` 从 1 开始；0 表示全局/尚未展开的组件。`component` 对应 config 路径，不一定等于state_dict路径。

`B` 为batch，`T` 为当前输入长度，`S` 为参与attention的KV长度；视觉T取决于切片、压缩与padding，音频T取决于分块。线性权重采用 `[out,in]` 数学约定。显式 `head_dim` 优先；仅缺失时用hidden/head推导并标注，不能用hidden/head覆盖显式维度。归一化、位置编码、门控、残差等未知细节单独列行，不能把表中行顺序当作已审计执行次序。

线性/递归注意力层的输入输出宽度与层类型可以由配置核对，但专用状态、门控和kernel内布局未确认时保留“未核验”。MiniCPM3的低秩Q/KV单列，未将其伪装成普通GQA。视觉层数量按配置记录，运行时drop/truncation应另查。VoxCPM2 residual_lm按已核验构造代码复制lm_config并替换层数；旧VoxCPM residual细节仍标未知。AudioVAE、连接器、压缩器和RobotManip动作内部没有伪造逐层尺寸；覆盖表与全局行明确这些缺口。

## 算子与设备接口

`torch_reference` 是数学参考，不是每个模型实际调用栈。NVIDIA/AMD/Ascend列目前没有逐checkpoint运行验证，保持未知；不能只因PyTorch提供一个API就断言各设备都有等价内核。已确认的InfLLM-V2 CUDA入口放在独立实现接口CSV，不代表全部MiniCPM默认调用它。TopK在该仓库attention内核外执行。

PyTorch参考：[linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear.html)、[SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)、[ROCm语义](https://docs.pytorch.org/docs/2.14/notes/hip.html)。这些滚动/版本文档只支持通用API语义，不替代模型实现证据。此前报告中的Ascend/FlagOS支持声明仍按各自版本和功能范围理解。

## 复核

每行保留模型、组件、层号、固定配置URL与revision；`sources.json`记录配置SHA256及补查源码来源。生成器从仓库内17份配置重建CSV，并检查哈希、层数、唯一行键及ZIP内容。CSV为UTF-8 BOM，便于Excel读取；没有地点或时区字段。

|范围|证据|
|---|---|
|配置与逻辑shape|config确认或明确公式推导|
|RobotTrack六层动作Linear|固定源码+配置|
|专用设备API/完整状态形状|未核验，不补猜测|
'''
(OUT/'字段与来源说明.md').write_text(notes);(OUT/'sources.json').write_text(json.dumps(source,ensure_ascii=False,indent=2))
class StableZip(zipfile.ZipFile):
 def writestr(self,name,data):
  info=zipfile.ZipInfo(name,date_time=(2026,9,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
  return super().writestr(info,data)
with StableZip(OUT/'OpenBMB-per-model-tables.zip','w') as z:
 for m in source['models']:
  name=m['model']
  for suffix,rows in [('layers',layers),('shapes',steps),('backends',backends)]:z.writestr(name+'/'+suffix+'.csv',csv_text([r for r in rows if r['model']==name]))
 z.writestr('字段与来源说明.md',notes);z.writestr('sources.json',json.dumps(source,ensure_ascii=False,indent=2));z.writestr('OpenBMB-model-coverage.csv',csv_text(coverage));z.writestr('OpenBMB-verified-implementation-interfaces.csv',csv_text(apis))
assert len(coverage)==17 and len({(r['model'],r['component'],r['layer_1based']) for r in layers})==len(layers)
assert len({(r['model'],r['component'],r['layer_1based'],r['step']) for r in steps})==len(steps)
with zipfile.ZipFile(OUT/'OpenBMB-per-model-tables.zip') as z:assert z.testzip() is None
(OUT/'validation.json').write_text(json.dumps({'status':'passed','models':17,'config_layers':len(layers),'logical_steps':len(steps),'backend_rows':len(backends),'weight_or_hardware_test':False},ensure_ascii=False,indent=2))
print('OpenBMB export:',len(layers),'config layers;',len(steps),'logical steps;',len(coverage),'models')

# Keep the complete offline report bundle in sync with the separate exports.
report_dir=OUT.parent
with StableZip(report_dir/'OpenBMB-研究报告包.zip','w') as z:
 for p in sorted(report_dir.rglob('*')):
  if not p.is_file() or p.name in ['index.html','OpenBMB-研究报告包.zip']:continue
  content=p.read_bytes()
  if p.suffix=='.html':
   content=content.decode().replace('../../index.html#/','https://limjiunnbin.github.io/model-research-atlas/index.html#/').replace('href="index.html"','href="https://limjiunnbin.github.io/model-research-atlas/reports/openbmb/"').replace('<script type="module" src="../../document-links.js"></script>','').encode()
  z.writestr(p.relative_to(report_dir).as_posix(),content)
