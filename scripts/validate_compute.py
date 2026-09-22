"""Validate exported compute data; optionally recheck pinned local source bodies."""
import argparse, ast, collections, csv, gzip, hashlib, io, json, math, re, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())
def sha(b): return hashlib.sha256(b).hexdigest()
def dims(s): return tuple(map(int,s.split('x')))
def validate(research=None):
 d=read(ROOT/'data/families/kimi/compute.json');f=read(ROOT/'data/families/kimi/family.json')
 fm={m['id']:m for m in f['models']};expected=collections.Counter();stats=collections.Counter();totals={}
 for m in d['models']:
  cfg=read(ROOT/m['config']);c=cfg.get('text_config',cfg);v=cfg.get('vision_config',{});H=c['hidden_size'];h=c['num_attention_heads'];qr=c['q_lora_rank'];kr=c['kv_lora_rank'];dn=c['qk_nope_head_dim'];dr=c['qk_rope_head_dim'];dv=c['v_head_dim'];E=c.get('num_experts',c.get('n_routed_experts'));latent=c.get('routed_expert_hidden_size',H)
  arch=read(ROOT/fm[m['id']]['architecturePath']);an={n['id']:n for n in arch['nodes']};assert set(an)=={n['id'] for n in m['nodes']}
  language=[n for n in m['nodes'] if n['group']=='decoder'];assert [n['layer'] for n in language]==list(range(1,c['num_hidden_layers']+1))
  assert sum(n['group']=='vision' for n in m['nodes'])==v.get('vt_num_hidden_layers',0)
  if m['id']=='k3':
   assert [n['layer'] for n in language if n['type']=='MLA']==c['linear_attn_config']['full_attn_layers']
   assert [n['layer'] for n in language if n['type']=='KDA']==c['linear_attn_config']['kda_layers']
  for n in m['nodes']:
   steps=d['templates'][n['template']]['steps'];ts=[t for s in steps for t in s['tensors']];assert len({t['tensor_template'] for t in ts})==len(ts)
   original={t['tensor_template'].replace('.experts.{i}.','.experts.{e}.'):(t['logical_shape'],t['stored_shape'],t['stored_dtype'],mod['count'] if mod['representative'] else 1) for mod in an[n['id']]['modules'] for t in mod['matrices']}
   assert {t['tensor_template']:(t['logical_shape'],t['stored_shape'],t['stored_dtype'],t['multiplicity']) for t in ts}==original
   assert sum(s['parameters'] for s in steps)==n['parameters']==an[n['id']]['parameters']
   stats['components']+=1;stats['steps']+=len(steps)
   for s in steps:
    assert all(s[k] for k in ['input','output','phase','dtype','relation']);assert set(s['apis'])=={'torch','nvidia','amd','ascend'}
    for a in [s['reference']]+list(s['apis'].values()):
     assert all(p in d['proofs'] for p in a['proofs'])
    for backend,a in s['apis'].items():
     stats[backend+'_unknown_steps']+=a['level']=='未知'
   lookup={t['tensor_template']:t for t in ts}
   for t in ts:
    name=t['tensor_template'];shape=t['logical_shape'];stored=dims(t['stored_shape']);mult=t['multiplicity'];logical=None if shape=='metadata' else dims(shape)
    if logical: assert math.prod(logical)==t['logical_parameters_each']
    target=None
    if n['group']=='decoder':
     if '.self_attn.' in name:
      key=name.split('.self_attn.')[1]
      target={'q_a_proj.weight':(qr,H),'q_a_layernorm.weight':(qr,),'q_b_proj.weight':(h*(dn+dr),qr),'kv_a_proj_with_mqa.weight':(kr+dr,H),'kv_a_layernorm.weight':(kr,),'kv_b_proj.weight':(h*(dn+dv),kr),'o_proj.weight':(H,h*dv)}.get(key) if n['type']=='MLA' else None
      if n['type']=='KDA':
       lc=c['linear_attn_config'];kh=lc['num_heads'];kd=lc['head_dim'];kw=kh*kd
       target={'q_proj.weight':(kw,H),'k_proj.weight':(kw,H),'v_proj.weight':(kw,H),'g_proj.weight':(kw,H),'o_proj.weight':(H,kw),'b_proj.weight':(kh,H),'f_a_proj.weight':(kd,H),'f_b_proj.weight':(kw,kd),'o_norm.weight':(kd,),'dt_bias':(kw,)}.get(key)
       if key in ['q_conv1d.weight','k_conv1d.weight','v_conv1d.weight']:target=(kw,1,lc['short_conv_kernel_size'])
     if t['owner'] in ['ffn','shared','routed'] and logical and len(logical)==2 and (name.endswith('.weight') or name.endswith('.weight_packed')):
      width=c['intermediate_size'] if t['owner']=='ffn' else c['moe_intermediate_size']*(c.get('num_shared_experts',c.get('n_shared_experts',1)) if t['owner']=='shared' else 1)
      hidden=latent if t['owner']=='routed' else H
      down='.down_proj.' in name or '.w2.' in name;target=(hidden,width) if down else (width,hidden)
     if '.experts.{e}.' in name:assert mult==E
    elif n['group']=='vision':
     C=v['vt_hidden_size'];Q=v.get('qkv_hidden_size',C);F=v['vt_intermediate_size'];key=name.split('.{i}.')[-1]
     target={'wqkv.weight':(3*Q,C),'wo.weight':(C,Q),'mlp.fc0.weight':(F,C),'mlp.fc1.weight':(C,F),'norm0.weight':(C,),'norm1.weight':(C,)}.get(key)
    if target is not None:assert logical==target,(m['id'],n['id'],name,logical,target);stats['config_shape_checks']+=1
    if name.endswith('.weight_packed'):
     factor=8 if t['stored_dtype']=='I32' else 2;assert stored==(logical[0],logical[1]//factor)
     scale=lookup[name.replace('weight_packed','weight_scale')];assert dims(scale['stored_shape'])==(logical[0],logical[1]//32)
     assert scale['stored_dtype']==('BF16' if factor==8 else 'U8');stats['packed_scale_checks']+=1
    if t['stored_dtype']=='F8_E4M3':
     scale=lookup[name.replace('.weight','.weight_scale_inv')];assert dims(scale['stored_shape'])==tuple(math.ceil(x/128) for x in logical);assert scale['stored_dtype']=='F32';stats['fp8_scale_checks']+=1
    if m['id']!='k2-base':
     concrete=name.replace('{i}',str(n['layer']-1));key=re.sub(r'\.experts\.\d+\.', '.experts.{e}.',concrete)
     expected[(m['name'],key,t['stored_dtype'],t['stored_shape'],shape,t['logical_parameters_each'])]+=mult
  totals[m['id']]=sum(n['parameters'] for n in m['nodes'])
 actual=collections.Counter()
 with gzip.open(ROOT/'dist/assets/kimi/全部张量清单.csv.gz','rt',encoding='utf-8-sig') as h:
  for x in csv.DictReader(h):
   if x['model'] not in {m['name'] for m in d['models']}:continue
   key=re.sub(r'\.experts\.\d+\.', '.experts.{e}.',x['tensor'])
   actual[(x['model'],key,x['stored_dtype'],x['stored_shape'],x['logical_shape'],int(x['logical_parameters']))]+=1;stats['raw_tensor_rows']+=1
 assert expected==actual,('raw inventory mismatch',list((expected-actual).items())[:3],list((actual-expected).items())[:3])
 if research:
  sources=read(research/'sources.json');cache={}
  for pid,p in d['proofs'].items():
   src=next(x for x in sources if x.get('repo')==p['repo'] and x.get('revision')==p['revision'] and x.get('path')==p['path'])
   if src['file'] not in cache:
    raw=(research/src['file']).read_bytes();assert sha(raw)==p['sourceSha256'];text=raw.decode();tree=ast.parse(text);funcs={}
    for scope in [tree]+[x for x in ast.walk(tree) if isinstance(x,ast.ClassDef)]:
     for fn in scope.body:
      if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):funcs[(scope.name+'.' if isinstance(scope,ast.ClassDef) else '')+fn.name]=fn
    cache[src['file']]=(text.splitlines(),funcs)
   lines,funcs=cache[src['file']];parent=d['proofs'].get(p.get('parentProof'),p);fn=funcs[parent['symbol']]
   assert (fn.lineno,fn.end_lineno)==(parent['line'],parent['end'])
   body='\n'.join(lines[fn.lineno-1:fn.end_lineno])
   if 'parentProof' in p:
    a=body.index(p['startNeedle']);b=body.index(p['endNeedle'],a+len(p['startNeedle']));body=body[a:b]
   else:assert set(p['verifiedCalls'])<={ast.unparse(x.func) for x in ast.walk(fn) if isinstance(x,ast.Call)}
   assert sha(body.encode())==p['functionSha256'],pid;stats['source_proofs_rechecked']+=1
 for x in d['downloads']:
  raw=(ROOT/'dist'/x['path']).read_bytes();assert len(raw)==x['bytes'] and sha(raw)==x['sha256']
  if x['name'].endswith('.csv'):
   assert raw.startswith(b'\xef\xbb\xbf');rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))));assert len(rows)==d['counts']['steps'];assert all(None not in r for r in rows)
   assert sum(int(r['逻辑参数计数_本步骤']) for r in rows)==sum(totals.values()) if 'shapes' in x['name'] else True
 with zipfile.ZipFile(ROOT/'dist/assets/kimi/compute/Kimi-per-model-tables.zip') as z:
  assert z.testzip() is None and len(z.namelist())==17
  for m in d['models']:
   count=sum(len(d['templates'][n['template']]['steps']) for n in m['nodes'])
   for suffix in ['compute-shapes','backend-api']:
    raw=z.read(m['id']+'-'+suffix+'.csv');assert raw.startswith(b'\xef\xbb\xbf');assert len(list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))))==count
 assert d['counts']['components']==stats['components'] and d['counts']['steps']==stats['steps']
 result={'status':'passed','counts':dict(stats),'logicalParametersByModel':totals,'limitations':d['notes']};return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path);p.add_argument('--output',type=Path);args=p.parse_args();r=validate(args.research_root)
 if args.output:args.output.write_text(json.dumps(r,ensure_ascii=False,indent=2))
 print(json.dumps(r,ensure_ascii=False))
