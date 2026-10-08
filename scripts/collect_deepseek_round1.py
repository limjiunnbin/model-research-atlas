"""Collect small, fixed official inputs; never download model weights or execute upstream code."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
DATE = '2026-10-01'
V3_REV = 'e815299b0bcbac849fa540c768ef21845365c9eb'
COMMITS = {
    'DeepSeek-V2': 'ec98ee3cbffc32104cd55dba8af884b3d772602a',
    'DeepSeek-V3': '9b4e9788e4a3a731f7567338ed15d3ec549ce03b',
    'DeepSeek-R1': '0cf78561f1d51c84a21b2190626b21116d5c68bb',
    'DeepSeek-V3.2-Exp': '87e509a2e5a100d221c97df52c6e8be7835f0057',
}


def fetch(url):
    if url.endswith('.safetensors'):
        raise ValueError('This collector never downloads model-weight shards')
    request = urllib.request.Request(url, headers={'User-Agent': 'model-research-atlas-deepseek-round1'})
    limit = 16_000_000 if url.endswith('model.safetensors.index.json') else 4_000_000
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'Research input exceeds {limit} byte limit: {url}')
    return raw


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def collect_materials(cache, entries):
    """Verify official file pointers for each representative without reading other configs/weights."""
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        tasks=[(entry,pool.submit(fetch,f'https://huggingface.co/api/models/{entry["modelId"]}/revision/{entry["revision"]}')) for entry in entries]
        for entry,future in tasks:
            raw=future.result(); info=json.loads(raw)
            if info['sha']!=entry['revision']: raise ValueError(f'Revision mismatch: {entry["name"]}')
            files={f['rfilename'] for f in info.get('siblings',[])}
            blob=f'https://huggingface.co/{entry["modelId"]}/blob/{entry["revision"]}/'
            paper_ids=[t.split(':',1)[1] for t in info.get('tags',[]) if t.startswith('arxiv:')]
            code=[p for p in sorted(files) if p.endswith('.py') and (p.startswith('modeling_') or p.startswith('inference/'))]
            entry['materials']={
                'modelCard':blob+'README.md' if 'README.md' in files else None,
                'config':blob+'config.json' if 'config.json' in files else None,
                'weightIndex':blob+'model.safetensors.index.json' if 'model.safetensors.index.json' in files else None,
                'license':[blob+p for p in sorted(files) if p.upper().startswith('LICENSE')],
                'referenceCode':[blob+p for p in code],
                'papers':['https://arxiv.org/abs/'+p for p in paper_ids],
                'status':'fixed-revision file names / paper tags checked; content read only where separately listed in sources.json',
                'note':'没有权重索引可能是单文件权重；不凭文件目录推断张量 shape、dtype 或 builtin 框架版本',
            }
            sid='hf-'+entry['name'].lower()+'-metadata'
            file=f'hf/{entry["name"]}/metadata-{entry["revision"]}.json'
            target=cache/file; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
            local_path=f'data/families/deepseek/research/checkpoint-metadata/{entry["name"]}.json'
            local_target=ROOT/local_path; local_target.parent.mkdir(parents=True,exist_ok=True); local_target.write_bytes(raw)
            entry['materialsSource']=sid
            records.append({'id':sid,'modelId':entry['modelId'],'revision':entry['revision'],
                            'url':f'https://huggingface.co/api/models/{entry["modelId"]}/revision/{entry["revision"]}',
                            'path':None,'cacheFile':file,'localPath':local_path,'kind':'checkpoint_metadata',
                            'accessed':DATE,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
                            'license':'metadata pointers; use the referenced version-specific license',
                            'readScope':'固定 revision 的文件目录与论文标签；未读取其他 checkpoint 配置值或权重'})
    return records


def collect(cache):
    cache.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/api/models?author=deepseek-ai&limit=100&full=true'
    directory = fetch(url)
    models = json.loads(directory)
    if isinstance(models, dict):
        raise ValueError('Unexpected HF directory response')
    inventory = {m['id'].split('/', 1)[1]: m for m in models}
    if inventory['DeepSeek-V3']['sha'] != V3_REV:
        raise ValueError('V3 revision changed; review and explicitly update the selected fixed revision')
    # Selected representatives, not a claim of exhaustive model-directory coverage.
    selected = [
        'DeepSeek-V2', 'DeepSeek-V3-Base', 'DeepSeek-V3', 'DeepSeek-R1-Zero', 'DeepSeek-R1',
        'DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-7B',
        'DeepSeek-R1-Distill-Qwen-14B', 'DeepSeek-R1-Distill-Qwen-32B',
        'DeepSeek-R1-Distill-Llama-8B', 'DeepSeek-R1-Distill-Llama-70B',
        'DeepSeek-V3.2-Exp', 'DeepSeek-V3.2', 'DeepSeek-V4-Flash-Base',
        'DeepSeek-V4-Flash', 'DeepSeek-V4-Pro-Base', 'DeepSeek-V4-Pro',
    ]
    directory_path='data/families/deepseek/research/hf-directory-snapshot.json'
    (ROOT/directory_path).parent.mkdir(parents=True,exist_ok=True)
    (ROOT/directory_path).write_bytes(directory)
    sources = [{
        'id': 'hf-directory', 'url': url, 'revision': None, 'path': None,
        'accessed': DATE, 'kind': 'official_directory', 'sha256': hashlib.sha256(directory).hexdigest(),
        'bytes': len(directory), 'cacheFile': 'hf-directory.json', 'localPath': directory_path,
        'readScope': '选定作者目录的前 100 个条目；用于核对名称与 revision，不宣称完整产品目录',
        'license': 'metadata; underlying model licenses remain separate',
    }]
    (cache / 'hf-directory.json').write_bytes(directory)
    entries = [{'name': name, 'modelId': f'deepseek-ai/{name}', 'revision': inventory[name]['sha'],
                'url': f'https://huggingface.co/deepseek-ai/{name}/tree/{inventory[name]["sha"]}'}
               for name in selected]
    sources.extend(collect_materials(cache,entries))
    write_json(DATA / 'research/selected-checkpoints.json', {
        'accessed': DATE, 'directorySource': 'hf-directory', 'selected': entries,
        'deferred': [m['id'] for m in models if any(s in m['id'] for s in ('DeepSeek-V2', 'DeepSeek-V3', 'DeepSeek-V4', 'DeepSeek-R1')) and m['id'].split('/', 1)[1] not in selected],
    })
    tasks = []
    for repo, commit in COMMITS.items():
        verified = json.loads(fetch(f'https://api.github.com/repos/deepseek-ai/{repo}/commits/{commit}'))
        if verified['sha'] != commit:
            raise ValueError(f'Commit verification failed: {repo}')
        paths = ['README.md']
        if repo == 'DeepSeek-V3':
            paths += ['README_WEIGHTS.md', 'LICENSE-CODE', 'LICENSE-MODEL',
                      'inference/model.py', 'inference/kernel.py', 'inference/convert.py',
                      'inference/fp8_cast_bf16.py', 'inference/configs/config_671B.json']
        for path in paths:
            sid = f'gh-{repo.lower()}-{path.replace("/", "-").replace(".", "-").lower()}'
            raw_url = f'https://raw.githubusercontent.com/deepseek-ai/{repo}/{commit}/{path}'
            record = {'id': sid, 'repo': f'deepseek-ai/{repo}', 'revision': commit, 'path': path,
                      'url': f'https://github.com/deepseek-ai/{repo}/blob/{commit}/{path}',
                      'rawUrl': raw_url, 'cacheFile': f'github/{repo}/{path}', 'localPath': None,
                      'kind': 'reference_code' if path.endswith('.py') else 'official_document',
                      'license': 'MIT (V3 code); DeepSeek model license (V3 model)' if repo == 'DeepSeek-V3' else 'see fixed upstream repository license; not independently audited',
                      'readScope': '待阅读；采集不等于逐函数核验'}
            if path == 'inference/configs/config_671B.json':
                record['localPath'] = 'data/families/deepseek/configs/DeepSeek-V3-demo-config.json'
            tasks.append(record)
    for path in ['config.json', 'model.safetensors.index.json', 'modeling_deepseek.py', 'configuration_deepseek.py']:
        tasks.append({'id': 'hf-v3-' + path.replace('.', '-'), 'modelId': 'deepseek-ai/DeepSeek-V3',
                      'revision': V3_REV, 'path': path,
                      'url': f'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/{V3_REV}/{path}',
                      'rawUrl': f'https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/{V3_REV}/{path}',
                      'cacheFile': f'hf/DeepSeek-V3/{path}',
                      'localPath': 'data/families/deepseek/configs/DeepSeek-V3-config.json' if path == 'config.json' else None,
                      'kind': 'checkpoint_config' if path == 'config.json' else 'weight_index' if 'index' in path else 'reference_code',
                      'license': 'see gh-deepseek-v3-license-code / gh-deepseek-v3-license-model',
                      'readScope': '待阅读；未下载任何 safetensors 分片'})
    for name in ['DeepSeek-V3.2-Exp', 'DeepSeek-V3.2', 'DeepSeek-V4-Flash', 'DeepSeek-V4-Pro']:
        rev = inventory[name]['sha']
        path = 'README.md'
        tasks.append({'id': 'hf-' + name.lower() + '-readme', 'modelId': f'deepseek-ai/{name}',
                      'revision': rev, 'path': path,
                      'url': f'https://huggingface.co/deepseek-ai/{name}/blob/{rev}/{path}',
                      'rawUrl': f'https://huggingface.co/deepseek-ai/{name}/resolve/{rev}/{path}',
                      'cacheFile': f'hf/{name}/{path}', 'localPath': None, 'kind': 'official_model_card',
                      'license': 'MIT as disclosed by model card; exact license files not audited',
                      'readScope': '版本关系与架构概述；未读取配置、权重或实现'})
    tasks.append({'id': 'v3-paper-v2', 'url': 'https://arxiv.org/html/2412.19437v2',
                  'rawUrl': 'https://arxiv.org/html/2412.19437v2', 'revision': '2412.19437v2',
                  'path': None, 'cacheFile': 'paper/2412.19437v2.html', 'localPath': None,
                  'kind': 'official_paper', 'license': 'arXiv distribution terms; only research references are committed',
                  'readScope': '第 2.2 节 MTP 与公式 21；不是全文实验复验'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [(record, pool.submit(fetch, record['rawUrl'])) for record in tasks]
        for record, future in futures:
            raw = future.result()
            target = cache / record['cacheFile']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if record['localPath']:
                target = ROOT / record['localPath']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            record.update(accessed=DATE, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            sources.append(record)
            print(record['id'], len(raw), record['sha256'][:12])
    write_json(DATA / 'sources.json', {'schemaVersion': 1, 'familyId': 'deepseek', 'accessed': DATE,
                                     'scope': 'P0/P1 research manifest, not website hardware.sources', 'sources': sources})


def restore(cache):
    """Rehydrate the exact recorded snapshots without choosing new revisions."""
    manifest=json.loads((DATA/'sources.json').read_text())
    for source in manifest['sources']:
        target=cache/source['cacheFile']
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==source['sha256']:
            continue
        raw=(ROOT/source['localPath']).read_bytes() if source['localPath'] else fetch(source.get('rawUrl',source['url']))
        if len(raw)!=source['bytes'] or hashlib.sha256(raw).hexdigest()!=source['sha256']:
            raise ValueError(f'Fixed snapshot changed or unavailable: {source["id"]}')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
        print('restored',source['id'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--restore',action='store_true',help='Restore recorded sources; do not refresh the official directory')
    args = parser.parse_args()
    (restore if args.restore else collect)(args.cache)
