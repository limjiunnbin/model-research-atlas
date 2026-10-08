"""Optional full Draft 2020-12 schema check (requires jsonschema in the selected QA environment)."""
import json
import argparse
from importlib.metadata import version
from pathlib import Path
from jsonschema import Draft202012Validator

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--write-record',action='store_true',help='Explicitly save a validation record; default is terminal output only.')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
schema=json.loads((ROOT/'schema.json').read_text())
Draft202012Validator.check_schema(schema)
files={'family':['data/families/deepseek/family.json'],'report':['data/families/deepseek/report.json'],
       'compute':['data/families/deepseek/compute.json'],'hardware':['data/families/deepseek/hardware.json'],
       'catalog':['data/catalog.json'],'analysis':['data/families/deepseek/analysis.json'],'comparison':['data/analysis/cross-family.json'],'architecture':[str(p.relative_to(ROOT)) for p in (ROOT/'data/families/deepseek').glob('*-architecture.json')]}
count=0
for definition,paths in files.items():
 validator=Draft202012Validator({'$schema':schema['$schema'],'$defs':schema['$defs'],'$ref':'#/$defs/'+definition})
 for path in paths:
  validator.validate(json.loads((ROOT/path).read_text()));count+=1
print(f'DeepSeek Draft 2020-12 schema validation: {count} documents passed')
if args.write_record:
 (ROOT/'data/families/deepseek/research/schema-validation.json').write_text(json.dumps({'status':'passed','draft':'2020-12','validator':'jsonschema '+version('jsonschema'),'documents':count,'files':files},ensure_ascii=False,indent=2)+'\n')
