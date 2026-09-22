import json,csv
from pathlib import Path
r=Path(__file__).resolve().parents[1]; d=r/'data/families/kimi'; f=json.loads((d/'family.json').read_text())
def classify(t):
 n=t['tensor_template']
 if 'self_attn.' in n:return 'attention'
 if '.experts.' in n:return 'routed'
 if 'shared_experts' in n:return 'shared'
 if 'routed_expert_' in n:return 'latent'
 if '.gate.' in n:return 'router'
 if '.mlp.' in n:return 'ffn'
 if 'res_' in n:return 'residual'
 return 'norm'
titles={'attention':'注意力 Attention','routed':'路由专家 MoE','shared':'共享专家','latent':'潜空间投影 LatentMoE','router':'路由器 Router','ffn':'稠密前馈 Dense FFN','residual':'跨层读取 AttnRes','norm':'归一化 Norm','vision':'视觉编码器','projector':'视觉连接器'}
for m in f['models']:
 if not m['configPath']:continue
 cfg=json.loads((r/m['configPath']).read_text()); t=cfg.get('text_config',cfg); v=cfg.get('vision_config',{})
 source=m['name']; base=m['id']=='k2-base'; audited=bool(m['auditPath']); a=json.loads((r/m['auditPath']).read_text()) if audited else None
 if base:a=json.loads((d/'k2-instruct-layers.json').read_text());source='Kimi-K2-Instruct'
 file=r/'dist/assets/kimi'/f'{source}-tensor-templates.csv'
 templates=list(csv.DictReader(file.open())) if file.exists() else []
 for x in templates:
  x['logical_parameters_each']=int(x['logical_parameters_each']);x['count']=int(x['count'])
 nodes=[]; n=m['facts']['layers']['value']; full=t.get('linear_attn_config',{}).get('full_attn_layers',[])
 for i in range(1,n+1):
  typ='KDA' if full and i not in full else 'MLA' if m['facts']['attention']['value']!='GQA' else 'GQA'
  group=next((x for x in a['groups'] if x['group']=='decoder' and x['layer_1based']==i),None) if a else None
  mats=[]
  for x in templates:
   if x['group']!='decoder':continue
   cat=classify(x)
   if i==1 and cat in ['routed','shared','latent','router']:continue
   if i>1 and cat=='ffn':continue
   if m['id']=='k3' and cat=='attention' and ((x['count']==69 and typ!='KDA') or (x['count']==24 and typ!='MLA')):continue
   mats.append(dict(x,module=cat))
  modules=[]
  for cat in dict.fromkeys(x['module'] for x in mats):
   rows=[x for x in mats if x['module']==cat]; each=sum(x['logical_parameters_each'] for x in rows); count=m['facts']['experts']['value'] if cat=='routed' else 1
   modules.append({'id':cat,'title':titles[cat],'representative':cat=='routed','count':count,'selectedCount':m['facts']['topK']['value'] if cat=='routed' else 1,'parameters':each*count,'matrices':rows})
  if not modules:
   # Config-only views deliberately expose no fabricated weight tensors.
   modules=[{'id':'attention','title':titles['attention'],'representative':False,'count':1,'selectedCount':1,'parameters':None,'matrices':[]},{'id':'ffn','title':'Dense FFN' if i==1 or not m['facts']['experts']['value'] else 'MoE / 配置级','representative':False,'count':1,'selectedCount':1,'parameters':None,'matrices':[]}]
  nodes.append({'id':'decoder-'+str(i),'group':'decoder','number':i,'type':typ,'ffn':'Dense' if i==1 or not m['facts']['experts']['value'] else 'MoE','parameters':group['logical_parameters'] if group else None,'activeLinearParameters':group.get('active_linear_parameters') if group else None,'payloadBytes':group['payload_bytes'] if group and audited else None,'modules':modules})
 for i in range(1,(m['facts']['visionLayers']['value'] or 0)+1):
  g=next((x for x in a['groups'] if x['group']=='vision' and x['layer_1based']==i),None) if a else None
  rows=[x for x in templates if x['group']=='vision']
  modules=[]
  for cat in ['attention','ffn','norm']:
   rr=[x for x in rows if ('mlp' in x['tensor_template'] if cat=='ffn' else 'norm' in x['tensor_template'] if cat=='norm' else 'mlp' not in x['tensor_template'] and 'norm' not in x['tensor_template'])]
   modules.append({'id':cat,'title':'视觉前馈 MLP' if cat=='ffn' else titles[cat],'representative':False,'count':1,'selectedCount':1,'parameters':sum(x['logical_parameters_each'] for x in rr) if rr else None,'matrices':rr})
  nodes.append({'id':'vision-'+str(i),'group':'vision','number':i,'type':'Vision','ffn':'MLP','parameters':g['logical_parameters'] if g else None,'activeLinearParameters':g.get('active_linear_parameters') if g else None,'payloadBytes':g['payload_bytes'] if g and audited else None,'modules':modules})
 if a:
  for g in a['groups']:
   if g['group'] in ['decoder','vision']:continue
   rows=[x for x in templates if x['group']==g['group']]
   nodes.append({'id':g['group'],'group':g['group'],'number':0,'type':g['group'],'ffn':None,'parameters':g['logical_parameters'],'activeLinearParameters':g.get('active_linear_parameters'),'payloadBytes':g['payload_bytes'] if audited else None,'modules':[{'id':'components','title':'全部权重与辅助张量','count':1,'selectedCount':1,'representative':False,'parameters':g['logical_parameters'],'matrices':rows}]})
 arch={'schemaVersion':1,'modelId':m['id'],'label':m['name'],'evidence':'derived' if a else 'official','scope':'配置推导；尺寸模板来自同尺寸 Instruct，不代表 Base 文件头已审计。' if base else '固定检查点文件头与官方配置；只展示形状和参数，不加载真实权重。' if audited else '仅配置级结构；未审计的矩阵与参数保持为空。','source':m['source'],'templatePath':'assets/kimi/'+file.name if file.exists() else None,'nodes':nodes,'config':{'hidden':m['facts']['hidden']['value'],'experts':m['facts']['experts']['value'],'topK':m['facts']['topK']['value'],'shared':m['facts']['shared']['value'],'context':m['facts']['context']['value'],'precision':m['facts']['quantization']['value'],'heads':m['facts']['heads']['value']},'cache':{'mla':'优化 MLA 通常缓存每层每 token 576 个元素；BF16 下为 1152 字节。参考实现可能缓存展开的 K/V，实际内存取决于引擎。','kda':'KDA 保留固定矩阵状态。K3 为每层 96 × 128 × 128 元素，另有短卷积状态；FP32 状态假设下计算，不能当作实测显存。','vision':'视觉编码阶段的激活和注意力工作区，不等同于语言生成的 KV 缓存。'}}
 p=d/(m['id']+'-architecture.json');p.write_text(json.dumps(arch,ensure_ascii=False,separators=(',',':')));m['architecturePath']=str(p.relative_to(r))
(d/'family.json').write_text(json.dumps(f,ensure_ascii=False,indent=2))
print('Architecture files',sum(bool(m.get('architecturePath')) for m in f['models']))
