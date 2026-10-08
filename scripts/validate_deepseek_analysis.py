"""Check analytical provenance, canonical joins and static-only boundaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
CACHE = Path.home() / '.cache/model-research-atlas'


def read(path):
    return json.loads(path.read_text())


def validate(sources=False, exports=False, write_record=False):
    d = read(DATA/'analysis.json'); family = read(DATA/'family.json'); core = read(DATA/'research/site-proofs.json')
    assert d['schemaVersion'] == 1 and d['familyId'] == family['id'] == 'deepseek'
    assert family['acceptanceScope']['deviceExperimentsRequired'] is False
    excluded = {m['id'] for m in d.get('excludedModels', [])}
    cost_ids = {m['id'] for m in d['costModels']}
    assert not (excluded & cost_ids)
    assert cost_ids | excluded == {m['id'] for m in family['models']}
    for item in d.get('excludedModels', []):
        model = next(m for m in family['models'] if m['id'] == item['id'])
        assert item['reason'] == model['computeScope'] and not model.get('computePath')
    layer_count = 0
    for model in d['costModels']:
        raw_model = next(m for m in family['models'] if m['id'] == model['id'])
        arch = read(ROOT/model['architecturePath']); cfg = read(ROOT/model['configPath'])
        assert model['config'] == cfg and model['revision'] == raw_model['revision']
        assert model['mainLogicalParameters'] == arch['mainLogicalParameters']
        assert model['publishedWeightBytes'] == arch['fullPublishedPayloadBytes']
        assert sum(model['parallelElements'].values()) == arch['mainLogicalParameters']
        nodes = {n['id']: n for n in arch['nodes'] if n['group'] == 'decoder'}; joined = []
        for group in model['groups']:
            assert len(group['nodes']) == group['count']
            joined += group['nodes']
            for nid in group['nodes']:
                node = nodes[nid]
                assert group['type'] == node['type'] and group['ratio'] == node.get('compressRatio', 0) and group['ffn'] == node['ffn']
                expected = []
                for mod in node['modules']:
                    for matrix in mod['matrices']:
                        if matrix['role'] == 'weight' and len(matrix['logical_shape']) == 2 and '.ape' not in matrix['tensor_template']:
                            expected.append((matrix['tensor_template'], matrix['logical_shape'], mod['selectedCount'] if mod['representative'] else matrix['count']))
                assert [(x['tensor'], x['shape'], x['callsPerToken']) for x in group['linears']] == expected, nid
            for term in group['linears']:
                assert term['proofId'] in core['proofs'] and len(term['shape']) == 2 and all(isinstance(x,int) and x>0 for x in term['shape'])
        assert len(joined) == len(set(joined)) and set(joined) == set(nodes)
        layer_count += len(nodes)
    assert layer_count == 1122 == d['counts']['costLayers']
    sources_by_id = dict(core['sources'])
    locations = {sid: CACHE/'deepseek-cann'/s['cacheFile'] for sid,s in sources_by_id.items() if sid.startswith('cann-')}
    added = read(DATA/'analysis-sources.json')['sources']
    assert len(added) == 21 and all(len(s['revision']) == 40 and len(s['sha256']) == 64 for s in added)
    for s in added:
        assert s['id'] not in sources_by_id
        sources_by_id[s['id']] = s; locations[s['id']] = CACHE/'deepseek-analysis'/s['cacheFile']
    if sources:
        from collect_deepseek_analysis import collect
        collect(verify_only=True)
    checked_ranges = 0
    def check_ref(r):
        nonlocal checked_ranges
        s = sources_by_id[r['sourceId']]
        assert r['revision'] == s['revision'] and r['path'] == s['path'] and r['sourceSha256'] == s['sha256']
        assert r['url'] == s['url']+f'#L{r["line"]}-L{r["end"]}' and 0<r['line']<=r['end']
        if sources:
            raw = locations[r['sourceId']].read_bytes()
            assert hashlib.sha256(raw).hexdigest() == s['sha256']
            lines = raw.decode().splitlines(); assert r['end'] <= len(lines)
            text = '\n'.join(lines[r['line']-1:r['end']])
            assert hashlib.sha256(text.encode()).hexdigest() == r['rangeSha256']
        checked_ranges += 1
    operators = read(DATA/'research/cann-operator-audit.json')['operators']
    assert {x['id'] for x in d['cannContracts']} == {x['id'] for x in operators}
    for c in d['cannContracts']:
        assert c['claims'] and c['tag'] == 'v9.2.0-beta.2'
        assert len({r['id'] for r in c['claims']}) == len(c['claims'])
        for claim in c['claims']:
            assert claim['statement'] and claim['categories']; check_ref(claim['reference'])
            if claim.get('tableMapping') == 'aligned_single_header':
                assert len(claim['fields']) == len(claim['rawCells']) == len(claim['observedHeaders'])
            elif claim.get('tableMapping') == 'raw_row_with_context': assert claim['fields'] == {}
        for record in c['rules'] + c['productSupport']: check_ref(record['reference'])
    for topic in d['parallel']['topics']:
        assert topic['summary'] and topic['refs']
        for r in topic['refs']: check_ref(r)
    assert len(d['parallel']['topics']) == 6 and len(d['mechanisms']) == 5
    for mechanism in d['mechanisms']:
        assert mechanism['limit'] and mechanism['references']
        for ref in mechanism['references']:
            assert ref in core['proofs'].values()
    compare = read(ROOT/'data/analysis/cross-family.json')
    catalog = read(ROOT/'data/catalog.json')
    expected_keys = {f['id']+':'+m['id'] for f in catalog['families'] for m in read(ROOT/f['path'])['models']}
    assert len(compare['models']) == len(expected_keys) and {m['key'] for m in compare['models']} == expected_keys
    for m in compare['models']:
        input_family = read(ROOT/m['inputFamilyPath']); raw_model = next(x for x in input_family['models'] if x['id'] == m['id'])
        assert raw_model['facts'] == m['facts']
        assert hashlib.sha256(json.dumps(raw_model, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == m['inputModelSha256']
        for fact in m['facts'].values():
            assert fact['evidence'] in ('official','derived','interpretation','unknown')
            assert fact['value'] is not None or fact['evidence'] == 'unknown'
    if exports:
        from validate_deepseek_analysis_exports import validate as validate_exports
        export_result = validate_exports()
    else: export_result = None
    result = {'status':'passed', 'date':d['snapshot'], 'costModels':len(d['costModels']), 'costLayers':layer_count,
        'cannFamilies':len(d['cannContracts']), 'cannTableRowsAndConditions':sum(len(c['claims']) for c in d['cannContracts']),
        'codedNecessaryRules':sum(len(c['rules']) for c in d['cannContracts']), 'parallelSources':len(added), 'sourceRanges':checked_ranges,
        'sourceBytesChecked':sources, 'crossFamilyModels':len(compare['models']), 'mechanisms':5, 'exports':export_result,
        'deviceExperiments':'excluded_from_current_scope_by_user', 'limits':['No runtime stack, model inference, timing or hardware validation; static formulas and synthetic teaching calculations only.']}
    if write_record:
        (DATA/'research/analysis-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',action='store_true');p.add_argument('--exports',action='store_true')
    args=p.parse_args();print(json.dumps(validate(args.sources,args.exports),ensure_ascii=False))
