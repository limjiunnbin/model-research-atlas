"""Validate data contracts, then copy authored JSON into the static website."""
import json,shutil
from research_exports import export
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
import openbmb_exports
catalog=read(ROOT/'data/catalog.json');assert catalog['schemaVersion']==1
assert len({f['id'] for f in catalog['families']})==len(catalog['families'])
reports=catalog.get('reports',[])
assert len({r['id'] for r in reports})==len(reports)
assert not ({r['id'] for r in reports}&{f['id'] for f in catalog['families']})
for report in reports:
 assert report['name'] and report['description'] and report['repositoryCount']>0
 path=Path(report['path']);assert not path.is_absolute() and '..' not in path.parts
 assert (ROOT/'dist'/path/'index.html').is_file(),report['path']
checks=0
for ref in catalog['families']:
 f=read(ROOT/ref['path']);assert ref['id']==f['id']
 export(ROOT,f)
 (ROOT/ref['path']).write_text(json.dumps(f,ensure_ascii=False,indent=2))
 ids=[m['id'] for m in f['models']];assert len(ids)==len(set(ids))
 report=read(ROOT/f['reportPath']);section_ids={s['id'] for s in report}
 for m in f['models']:
  assert m['name'] and m['branch'] and m['summary']
  for k,v in m['facts'].items():
   assert v['evidence'] in ['official','derived','interpretation','unknown'],(m['id'],k)
   assert v['value'] is not None or v['evidence']=='unknown',(m['id'],k)
  assert all(s in section_ids for s in m['sections'])
  for key in ['configPath','auditPath','architecturePath']:
   if m.get(key):assert (ROOT/m[key]).is_file(),m[key]
  if m.get('architecturePath'):
   a=read(ROOT/m['architecturePath']);nodes=a['nodes'];assert len({n['id'] for n in nodes})==len(nodes)
   language=[n for n in nodes if n['group']=='decoder'];assert len(language)==m['facts']['layers']['value'];assert [n['number'] for n in language]==list(range(1,len(language)+1))
   assert len([n for n in nodes if n['group']=='vision'])==(m['facts']['visionLayers']['value'] or 0)
   if m['id']=='k3':assert sum(n['type']=='KDA' for n in language)==69 and sum(n['type']=='MLA' for n in language)==24 and language[-1]['type']=='MLA'
   if m.get('auditPath'):
    for n in language:
     assert sum(x['parameters'] for x in n['modules'])==n['parameters'],(m['id'],n['id'],'module sum',sum(x['parameters'] for x in n['modules']),n['parameters'])
   checks+=len(nodes)
 if f.get('hardwarePath'):
  h=read(ROOT/f['hardwarePath']);assert h['schemaVersion']==1 and h['familyId']==f['id']
  source_ids={x['id'] for x in h['sources']};assert len(source_ids)==len(h['sources'])
  module_ids={x['id'] for x in h['modules']};assert len(module_ids)==len(h['modules'])
  for group in ['modules','platforms','optimizations']:
   assert len({x['id'] for x in h[group]})==len(h[group])
   for item in h[group]:
    assert item['refs'] and all(r['id'] in source_ids for r in item['refs'])
    if group=='modules':assert all(m in ids for m in item['models']) and len(item['flow'])>1
    if group=='optimizations':assert item['priority'] in ['P0','P1','P2'] and item['metric'] and item['risk']
  for m in f['models']:
   for link in m.get('implementationLinks',[]):assert link['topic'] in module_ids and link['path'].endswith('/'+m['id'])
  for path in h['downloads'].values():assert (ROOT/'dist'/path).is_file()
 for fig in f['figures']:assert (ROOT/'dist'/fig['path']).is_file()
 if f.get('computePath'):
  from validate_compute import validate
  validate()
 for download in f['downloads']:assert (ROOT/'dist'/download['path']).stat().st_size==download['bytes']
shutil.copytree(ROOT/'data',ROOT/'dist/data',dirs_exist_ok=True)
for name in ['index.html','app.js','styles.css','structure.js','hardware.js','compute.js']:assert (ROOT/'dist'/name).stat().st_size>0
print(f'Build OK: {len(catalog["families"])} families, {checks} structure components, all asset references valid.')

# Every shipped Markdown file is an allowed reader source.
markdown_files=sorted(p.relative_to(ROOT/"dist").as_posix() for p in (ROOT/"dist").rglob("*.md"))
(ROOT/"dist/documents.json").write_text(json.dumps({"files":markdown_files},ensure_ascii=False,indent=2))
