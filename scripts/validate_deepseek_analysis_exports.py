"""Compare every analytical CSV/XLSX cell with canonical JSON; no authoring here."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from validate_deepseek_exports import NS, sheet_cells, cell_address, csv_values

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/families/deepseek';OUT=ROOT/'dist/assets/deepseek'


def read(path): return json.loads(path.read_text())


def expected_tables():
    d=read(DATA/'analysis.json');scenarios=read(DATA/'research/analysis-scenarios.json');compare=read(ROOT/'data/analysis/cross-family.json')
    costs=[]
    for s in scenarios['costScenarios']:
        i,r=s['input'],s['result'];m=next(m for m in d['costModels'] if m['id']==s['modelId'])
        costs.append([s['name'],s['modelId'],m['revision'],i['phase'],i['path'],i['batch'],i['query'],i['history'],i['logits'],i['cacheBytes'],i['activationBytes'],i['weightBytes'],r['projectionFlops'],r['coreFlops'],r['usefulCoreFlops'],r['indexFlops'],r['headFlops']+r['headControlFlops'],r['contractionFlops'],r['mainKvBytes'],r['indexCacheBytes'],r['compressionStateBytes'],r['sequenceCacheBytes'],r['mainLogicalWeightBytes'],r['publishedWeightBytes'],r['projectionIOBytes'],r['declaredTensors']['scoreBytes'],r['measuredLatency'],r['measuredThroughput'],'声明矩阵收缩；逻辑字节与精度假设；主干范围；设备未测'])
    parallel=[]
    for s in scenarios['parallelScenarios']:
        i,r=s['input'],s['result']
        parallel.append([s['name'],i['tp'],i['dp'],r['ep'],i['policy'],i['batch'],i['query'],i['history'],i['redundantExperts'],i['weightBytes'],i['activationBytes'],r['remoteFraction'],i['allReduces'],r['logicalParametersPerRank'],r['logicalWeightBytesPerRank'],r['dispatchSendBytesPerMoeLayer'],r['combineSendBytesPerMoeLayer'],r['moeSendBytes'],r['uniformMoeSendBytesPerRank'],r['tpRingSendBytesPerRank'],r['tpRingSendBytes'],r['pdMainKvBytesPerReplica'],r['indexCacheBytesPerReplica'],r['measuredCommunicationTime'],r['measuredRankPeakMemory'],'均分/路由/ring/精度为显式假设；量化 metadata、padding、ABI 与运行布局另核对'])
    comparison=[]
    for m in compare['models']:
        f=m['facts'];comparison.append([m['family'],m['name'],m['branch'],m['revision'],f.get('layers',{}).get('value'),f.get('attention',{}).get('value'),f.get('parameters',{}).get('value'),f.get('payload',{}).get('value'),f.get('quantization',{}).get('value'),m['evidenceDepth'],m['parameterScope'] if isinstance(m['parameterScope'],str) else json.dumps(m['parameterScope'],ensure_ascii=False,separators=(',',':')),json.dumps(m['cache'],ensure_ascii=False,separators=(',',':')) if m['cache'] is not None else None,m['source']])
    mechanisms=[[m['title'],m['id'],m['limit'],';'.join(r['symbol'] for r in m['references']),';'.join(r['url'] for r in m['references'])] for m in d['mechanisms']]
    contracts=[]
    for c in d['cannContracts']:
        for r in c['claims']:
            ref=r['reference'];contracts.append([c['id'],r['apiDocument'],r.get('context','API 入口'),r['parameter'],' / '.join(r['categories']),r['statement'],ref['revision'],ref['line'],ref['end'],ref['url'],ref['rangeSha256'],'合并单元格保留原始列/上下文；未自动映射契约' if r.get('tableMapping')=='raw_row_with_context' else '固定表行/条件；不证明实际支持'])
    return {'成本场景':costs,'并行场景':parallel,'跨家族比较':comparison,'机制要点':mechanisms,'算子条件':contracts}


def validate(write_record=False):
    manifest=read(DATA/'research/analysis-export-manifest.json');expected=expected_tables();cells_checked=0;csv_rows=0
    assert hashlib.sha256((DATA/'analysis.json').read_bytes()).hexdigest()==manifest['analysisSha256']
    with zipfile.ZipFile(ROOT/'dist'/manifest['path']) as z:
        assert z.testzip() is None
        shared=[''.join(t.text or '' for t in item.findall('.//s:t',NS)) for item in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
        rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))};names=[]
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS):
            name=sheet.get('name');names.append(name);target=rels[sheet.get('{'+NS['r']+'}id')]
            source=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target))
            cells=sheet_cells(z,source,shared);spec=next(s for s in manifest['sheets'] if s['name']==name)
            assert len(expected[name])==spec['rows'] and all(len(r)==spec['columns'] for r in expected[name])
            for ri,row in enumerate(expected[name],start=6):
                for ci,v in enumerate(row,start=1):
                    addr=cell_address(ri,ci);actual,kind=cells.get(addr,(None,None))
                    if isinstance(v,(int,float)):
                        assert kind in (None,'n') and isinstance(actual,(int,float)) and math.isclose(actual,v,rel_tol=1e-13,abs_tol=1e-12),(name,addr,actual,v)
                    else: assert actual==v or actual is None and v=='',(name,addr,actual,v)
                    cells_checked+=1
            if spec['csv']:
                rows=csv_values((OUT/spec['csv']).read_bytes());assert len(rows)-1==len(expected[name])
                assert len(rows[0])==spec['columns'];csv_rows+=len(rows)-1
                for actual,row in zip(rows[1:],expected[name]):
                    for a,v in zip(actual,row):
                        if isinstance(v,(int,float)): assert math.isclose(float(a),v,rel_tol=1e-13,abs_tol=1e-12)
                        else: assert a==('' if v is None else v)
    expected_cells = sum(len(expected[s['name']])*s['columns'] for s in manifest['sheets'])
    assert names==list(expected) and cells_checked==manifest['dataCells']==expected_cells
    result={'status':'passed','date':read(DATA/'analysis.json')['snapshot'],'xlsxSheets':len(names),'xlsxDataCells':cells_checked,'csvDataRows':csv_rows,'csvFiles':4,'unknownMeasurements':'blank','scope':'every analytical export value compared with canonical JSON'}
    if write_record:
        (DATA/'research/analysis-export-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__': print(json.dumps(validate(),ensure_ascii=False))
