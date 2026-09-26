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
 print('Model documents OK:',len(models),'independent XLSX; files, CSV rows, ZIP CRC and K3 93 layers checked.')
if __name__=='__main__':validate()
