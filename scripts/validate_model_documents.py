"""Validate the generated independent model documents without regenerating the private template."""
from pathlib import Path
import json,hashlib,zipfile,csv
ROOT=Path(__file__).resolve().parents[1]
def validate():
 d=ROOT/'dist/downloads/model-documents'
 if not (d/'manifest.json').exists():return
 manifest=json.loads((d/'manifest.json').read_text());models=manifest['models']
 assert len({m['id'] for m in models})==len(models)
 for m in models:
  data=json.loads((d/(m['id']+'.json')).read_text());assert len(data['rows'])==m['rows']
  assert data['headers'][:21]==manifest['template']['headers'][:21]
  with (d/(m['id']+'.csv')).open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.reader(f)))==m['rows']+1
  with zipfile.ZipFile(d/(m['id']+'.xlsx')) as z:assert z.testzip() is None
 for r in json.loads((d/'file-checksums.json').read_text()):
  p=d/r['file'];assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['file']
 with zipfile.ZipFile(d/'model-documents-all.zip') as z:
  assert z.testzip() is None
  for m in models:assert m['id']+'.xlsx' in z.namelist()
 k=next(m for m in models if m['name']=='Kimi-K3');a=json.loads((d/(k['id']+'.json')).read_text());lang={r[0] for r in a['rows'] if r[0].startswith('decoder/layer_')};assert len(lang)==93
 assert set(json.loads((d/'K3-template-cell-audit.json').read_text())[0]) >= {'original','verified_or_formula','comparison_status','fixed_sources'}
 # Regressions from the independent semantic review: preserve platform conditions.
 compute=json.loads((ROOT/'data/families/kimi/compute.json').read_text())
 for cm in compute['models']:
  m=next(m for m in models if m['name']==cm['name'])
  rows=json.loads((d/(m['id']+'.json')).read_text())['rows'];i=0
  for n in cm['nodes']:
   for step in compute['templates'][n['template']]['steps']:
    r=rows[i];i+=1
    for platform,col in [('torch',24),('nvidia',25),('amd',26),('ascend',2)]:
     a=step['apis'][platform]
     assert all(a.get(k,'') in r[col] for k in ['chain','level','condition']),(m['id'],i,platform)
  assert i==len(rows)
 kr=json.loads((d/'kimi-kimi-k3.json').read_text())['rows']
 kd=[r for r in kr if r[3]=='kda'];assert len(kd)==69
 assert all('H_state=' in r[29] and 'L_KV=' not in r[29] for r in kd)
 for m in models:
  md=json.loads((d/(m['id']+'.json')).read_text())
  for r in md['rows']:
   if 'vision' in r[0] and r[1] in ['position_and_cache','position_and_layout']:
    assert 'L_img' in r[5] and 'L_KV' not in r[5]
   assert 'audio_config.optional_decoder/layer_' not in r[0]
  if md.get('isEncoder'):
   assert all('L_KV' not in r[4]+r[5] for r in md['rows'])
   assert all(md.get('encodingAxis','L_text') in r[28] for r in md['rows'])
 assert json.loads((d/'openbmb-minicpm3-4b.json').read_text())['params']['V']==64
 assert json.loads((d/'openbmb-minicpm3-rag-lora.json').read_text())['params']['V']=='未核验'
 print('Model documents OK:',len(models),'independent XLSX; files, CSV rows, ZIP CRC and K3 93 layers checked.')
if __name__=='__main__':validate()
