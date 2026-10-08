"""Check every exported CSV/XLSX value and archive entry against the canonical atlas."""
import csv
import hashlib
import io
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
OUT = ROOT / 'dist/assets/deepseek'
SHAPE_HEADERS = ['model_id', 'model', 'component', 'group', 'source_index_0based', 'display_layer', 'type', 'phase', 'path', 'step_id', 'operation', 'input_shape', 'output_shape', 'dtype', 'logical_weight_reference_parameters', 'stored_weight_evidence', 'source_ids', 'reference_functions', 'source_urls', 'math_relation', 'duration', 'scope']
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main', 'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def read(path):
    return json.loads(path.read_text())


def csv_values(raw):
    assert raw.startswith(b'\xef\xbb\xbf'), 'CSV must include UTF-8 BOM'
    assert b'\r\n' in raw and b'\n' not in raw.replace(b'\r\n', b''), 'CSV must use CRLF'
    return list(csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline='')))


def stringify(value):
    return '' if value is None else str(value)


def shape_row(model, node, step, compute):
    proofs = [compute['proofs'][pid] for pid in step['reference']['proofs']]
    return [stringify(v) for v in [model['id'], model['name'], node['id'], node['group'], node['sourceIndex'], node['layer'], node['type'], step['phase'], step['path'], step['id'], step['title'], step['input'], step['output'], step['dtype'], step['parameters'], model['weightEvidence'], ';'.join(p['source_id'] for p in proofs), ';'.join(p['symbol'] for p in proofs), ';'.join(p['url'] for p in proofs), step['relation'], None, step['note']]]


def display(value):
    return 'unknown' if value is None else ' × '.join(map(str, value)) if isinstance(value, list) else str(value)


def matrix_key(node):
    return json.dumps([node.get('group'), node.get('type'), node.get('ffn'), node.get('routing'), node.get('compressRatio'), [[m['id'], [[x['tensor_template'], x['logical_shape'], x['count'], x['role']] for x in m['matrices']]] for m in node['modules']]], ensure_ascii=False)


def workbook_expected(family, compute, evidence, hardware):
    summary, matrices, steps = [], [], []
    for model in family['models']:
        arch = read(ROOT / model['architecturePath'])
        summary.append([model['name'], model['revision'], model['facts']['layers']['value'], model['facts']['attention']['value'], model['facts']['hidden']['value'], model['facts']['parameters']['value'], arch['mtpLogicalParameters'] if arch['mtpLogicalParameters'] is not None else 'unknown',arch.get('dsparkLogicalParameters',0),arch.get('dspark',{}).get('stages',0),arch.get('dspark',{}).get('blockSize','不适用'), model['facts']['quantization']['value'], '完整本 checkpoint 文件头' if model['auditPath'] else '配置+源码推导', model['baseModel'] or '未声明权重底座'])
        seen = set()
        for node in arch['nodes']:
            key = matrix_key(node)
            if key in seen:
                continue
            seen.add(key)
            layers = ','.join(str(n['number'] or n['id']) for n in arch['nodes'] if matrix_key(n) == key)
            for module in node['modules']:
                for matrix in module['matrices']:
                    proof = evidence['proofs'].get(matrix.get('proof'))
                    matrices.append([model['name'], layers, node['type'], module['title'], matrix['tensor_template'], display(matrix['logical_shape']), matrix['logical_parameters_each'], matrix['count'], matrix['logical_parameters_each'] * matrix['count'], display(matrix['stored_shape']), display(matrix['stored_dtype']), matrix['payloadBytes'], matrix['role'], matrix['evidence'], proof['symbol'] if proof else '文件头/共享口径', proof['url'] if proof else matrix['source']])
    for template in compute['templates'].values():
        name = next(m['name'] for m in family['models'] if m['id'] == template['modelId'])
        for step in template['steps']:
            proofs = [compute['proofs'][pid] for pid in step['reference']['proofs']]
            steps.append([name, template['id'], template['layerType'], template.get('ffn') or 'global', step['phase'], step['path'], step['title'], step['input'], step['output'], step['parameters'], step['relation'], ';'.join(p['symbol'] for p in proofs), ';'.join(p['url'] for p in proofs), step['note']])
    sources = [[s['id'], s.get('title') or s.get('modelId') or s.get('repo'), s.get('revision'), s.get('path') or 'metadata', s['kind'], s['sha256'], s['readScope'], s['url']] for s in evidence['sources'].values()]
    gaps = [[o['priority'], o['title'], o['status'], o['scope'], o['observation'], o['proposal'], o['metric'], o['risk']] for o in hardware['optimizations']]
    gaps.append(['说明','本次验收范围','静态研究与网站','结构/来源/shape/导出/网站','当前没有硬件，设备实验已移出本次范围','未来需要时单独发起设备研究','本次按静态与网站校验验收','未测 duration 等保持空值，不等于 0 或测试通过'])
    spec=read(DATA/'research/speculation.json');token=read(DATA/'research/r1-full-tokenizer-audit.json');spec_rows=[]
    compact=lambda v:json.dumps(v,ensure_ascii=False,separators=(',',':'))
    for model in spec['models']:
        urls=';'.join(evidence['proofs'][i]['url'] for i in model['referenceProofs'])
        for k,v in model['config'].items():spec_rows.append([model['name'],'checkpoint 配置',k,compact(v) if isinstance(v,(dict,list)) else str(v),urls,'MTP namespace 与 DSpark 实际 stage 分开；服务 token 数另记'])
        for c in model['conflicts']:spec_rows.append([model['name'],'配置/服务口径',c['field'],compact(c),'固定 config/inference/header/model card',c['resolution']])
    for p in spec['protocol']:spec_rows.append(['MTP / DSpark','服务协议',p['phase'],p['operation'],';'.join(evidence['proofs'][i]['url'] for i in p['proofs']),p.get('limit','静态协议；没有设备执行验证')])
    token_rows=[]
    for r in token['records']:
        c=r['comparison'];token_rows.append([r['model'],r['baseModel'],r['distillVocabularyEntries'],r['distillMerges'],r['addedTokens'],r['embeddingRows'],c['commonTokenIdsChanged'] if c else 'gated',c['differentMergePositions'] if c else 'gated',('是' if c['addedTokensIdentical'] else '否') if c else 'gated',compact(r['specialTokens']),'原底座 gated，公开 Distill 已核对' if r['gaps'] else '完整 vocab/merge/config ID 已核对'])
    return {'版本概览': summary, '矩阵模板': matrices, '算子模板': steps, '来源': sources, '实验与缺口': gaps,'推测解码':spec_rows,'Tokenizer核对':token_rows}


def sheet_cells(archive, path, shared):
    cells = {}
    for cell in ET.fromstring(archive.read(path)).findall('.//s:sheetData/s:row/s:c', NS):
        value = cell.find('s:v', NS)
        kind = cell.get('t')
        assert kind != 'e', f'Workbook error at {path}:{cell.get("r")}'
        assert cell.find('s:f', NS) is None, 'Research workbook should contain fixed snapshot values'
        if kind == 's':
            parsed = shared[int(value.text)]
        elif kind == 'inlineStr':
            parsed = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
        elif value is None:
            parsed = None
        elif kind in ('str', 'b'):
            parsed = value.text
        else:
            parsed = float(value.text)
        cells[cell.get('r')] = (parsed, kind)
    return cells


def cell_address(row, column):
    letters = ''
    while column:
        column, r = divmod(column - 1, 26)
        letters = chr(65 + r) + letters
    return letters + str(row)


def validate():
    family, compute = read(DATA / 'family.json'), read(DATA / 'compute.json')
    # This existing workbook is the 21-checkpoint full-header/compute export.
    # Config/reference additions have separate exports and no borrowed audit values.
    family = dict(family)
    compute_ids = {m['id'] for m in compute['models']}
    family['models'] = [m for m in family['models'] if m['id'] in compute_ids]
    evidence, hardware = read(DATA / 'research/site-proofs.json'), read(DATA / 'hardware.json')
    manifest = read(DATA / 'research/export-manifest.json')
    downloadable = dict(compute)
    downloadable['downloads'] = []
    normalized = (json.dumps(downloadable, ensure_ascii=False, indent=2) + '\n').encode()
    assert hashlib.sha256(normalized).hexdigest() == manifest['computeSha256'], 'Export facts differ from canonical compute snapshot'
    assert (OUT / 'DeepSeek-compute.json').read_bytes() == normalized
    expected = [SHAPE_HEADERS]
    for model in compute['models']:
        for node in model['nodes']:
            for step in compute['templates'][node['template']]['steps']:
                expected.append(shape_row(model, node, step, compute))
    rows = csv_values((OUT / 'DeepSeek-all-layer-shapes.csv').read_bytes())
    assert rows == expected, 'Expanded CSV does not match every canonical step'
    assert len(rows) - 1 == manifest['expandedRows'] == compute['counts']['steps']
    per_model = {}
    for entry in manifest['perModel']:
        raw = (ROOT / 'dist' / entry['path']).read_bytes()
        subset = [SHAPE_HEADERS] + [row for row in rows[1:] if row[0] == entry['modelId']]
        assert csv_values(raw) == subset and len(subset) - 1 == entry['rows'], entry['modelId']
        per_model[Path(entry['path']).name] = raw
    assert len(per_model) == len(family['models']) == 21
    workbook = workbook_expected(family, compute, evidence, hardware)
    checked_cells = 0
    with zipfile.ZipFile(ROOT / 'dist' / manifest['workbook']['path']) as archive:
        assert archive.testzip() is None
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in item.findall('.//s:t', NS)) for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        names = []
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            name = sheet.get('name');names.append(name)
            target = rels[sheet.get('{'+NS['r']+'}id')]
            path = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
            cells = sheet_cells(archive, path, shared)
            spec = next(s for s in manifest['workbook']['sheets'] if s['name'] == name)
            assert len(workbook[name]) == spec['rows']
            for ri, row in enumerate(workbook[name], start=6):
                assert len(row) == spec['columns']
                for ci, expected_value in enumerate(row, start=1):
                    addr = cell_address(ri, ci);value, kind = cells.get(addr, (None, None))
                    assert value == expected_value or value is None and expected_value == '', (name, addr, value, expected_value)
                    if isinstance(expected_value, (int, float)):
                        assert kind in (None, 'n'), (name, addr, 'number serialized as text')
                    checked_cells += 1
            last = 5 + spec['rows']
            assert not any(int(''.join(filter(str.isdigit, addr))) > last and value is not None for addr, (value, _) in cells.items()), name
        assert names == list(workbook)
    with zipfile.ZipFile(OUT / 'DeepSeek-per-model-tables.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(per_model) | {'字段与符号说明.md'}
        for name, raw in per_model.items():
            assert archive.read(name) == raw
    with zipfile.ZipFile(OUT / 'DeepSeek-研究报告包.zip') as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist()))
        for name in archive.namelist():
            path = DATA / name if name.startswith('configs/') else OUT / name
            if name.startswith('metadata/') and name.endswith('-architecture.json'):
                path = DATA / Path(name).name
            assert archive.read(name) == path.read_bytes(), name
        from html.parser import HTMLParser
        from urllib.parse import unquote, urlsplit
        class ResourceLinks(HTMLParser):
            def __init__(self): super().__init__();self.targets=[]
            def handle_starttag(self,tag,attrs):
                self.targets.extend(value for key,value in attrs if key in ('href','src') and value)
        names=set(archive.namelist())
        for name in names:
            if not name.endswith('.md'): continue
            body=archive.read(name).decode('utf-8');parser=ResourceLinks();parser.feed(body)
            import re
            targets=parser.targets+re.findall(r'\]\(([^)]+)\)',body)
            for target in targets:
                url=urlsplit(target)
                if url.scheme or not url.path: continue
                resolved=posixpath.normpath(posixpath.join(posixpath.dirname(name),unquote(url.path)))
                assert resolved in names or resolved=='DeepSeek-研究报告包.zip',(name,target,'missing packaged document/image')
    return {'csvRows': len(rows) - 1, 'csvColumns': len(SHAPE_HEADERS), 'perModelCSVs': len(per_model), 'xlsxSheets': len(workbook), 'xlsxCells': checked_cells, 'zipArchives': 2}


if __name__ == '__main__':
    print(json.dumps(validate(), ensure_ascii=False))
