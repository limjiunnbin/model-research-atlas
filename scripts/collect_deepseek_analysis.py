"""Fetch fixed public parallelism sources; never install or execute their code."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
CACHE = Path.home() / '.cache/model-research-atlas/deepseek-analysis'
REPOS = {
    'vllm-project/vllm': ('bcee730b1a9d25f0fd283a0ef6c19133ebeebf4f', [
        'LICENSE', 'docs/serving/expert_parallel_deployment.md',
        'docs/serving/data_parallel_deployment.md', 'docs/serving/parallelism_scaling.md',
        'docs/features/disagg_prefill.md', 'vllm/config/parallel.py',
        'vllm/distributed/parallel_state.py', 'vllm/distributed/device_communicators/all2all.py',
        'vllm/distributed/eplb/eplb_state.py', 'vllm/distributed/eplb/eplb_utils.py',
        'vllm/distributed/kv_transfer/README.md', 'vllm/config/kv_transfer.py',
        'vllm/distributed/kv_transfer/kv_connector/v1/base.py',
    ]),
    'vllm-project/vllm-ascend': ('a8fcedb03d93e60efceddbfc912406f7fa491d57', [
        'LICENSE', 'docs/source/developer_guide/Design_Documents/disaggregated_prefill.md',
        'docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md',
        'docs/source/user_guide/feature_guide/expert_parallelism_load_balancer.md',
        'docs/source/user_guide/feature_guide/epd_disaggregation.md',
        'vllm_ascend/distributed/parallel_state.py',
        'vllm_ascend/distributed/device_communicators/pyhccl.py',
        'vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py',
    ]),
}


def collect(cache=CACHE, verify_only=False):
    manifest = DATA / 'analysis-sources.json'
    existing = {s['id']: s for s in json.loads(manifest.read_text())['sources']} if manifest.is_file() else {}
    if verify_only:
        sources = json.loads(manifest.read_text())['sources']
    else:
        sources = []
        for repo, (revision, paths) in REPOS.items():
            tree_file = CACHE / (repo.split('/')[-1] + '-tree.json')
            blobs = {x['path']: x['sha'] for x in json.loads(tree_file.read_text())['tree'] if x['type'] == 'blob'} if tree_file.exists() else {}
            for path in paths:
                sources.append({'id': 'analysis-' + repo.split('/')[-1] + '-' + path.replace('/', '-').replace('.', '-'), 'repo': repo,
                    'revision': revision, 'path': path, 'cacheFile': f'code/{repo}/{revision}/{path}',
                    'url': f'https://github.com/{repo}/blob/{revision}/{path}',
                    'rawUrl': f'https://raw.githubusercontent.com/{repo}/{revision}/{path}',
                    'kind': 'official_document' if path.endswith('.md') else 'reference_code',
                    'gitBlobSha1': blobs.get(path), 'readScope': 'static parallelism/communication contract; not a tested runtime stack',
                    'accessed': '2026-10-01', 'license': 'Apache-2.0 at the selected upstream revision; full upstream code is not redistributed'})

    def one(source):
        dest = cache / source['cacheFile']
        if dest.exists():
            raw = dest.read_bytes()
        else:
            assert not verify_only, source['id']
            request = urllib.request.Request(source['rawUrl'], headers={'User-Agent': 'model-research-atlas'})
            raw = urllib.request.urlopen(request, timeout=45).read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        previous = existing.get(source['id'])
        if previous and (previous['repo'], previous['revision'], previous['path']) == (source['repo'], source['revision'], source['path']):
            assert len(raw) == previous['bytes'] and sha == previous['sha256'], 'Fixed source drift: '+source['id']
        if source.get('gitBlobSha1'):
            assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == source['gitBlobSha1'], source['id']
        if verify_only:
            assert len(raw) == source['bytes'] and sha == source['sha256'], source['id']
        else:
            source.update(bytes=len(raw), sha256=sha)
        return len(raw)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        counts = list(pool.map(one, sources))
    if not verify_only:
        manifest.write_text(json.dumps({'snapshot': '2026-10-01', 'scope': 'A3 fixed public parallelism inputs, separate from the completed 579 core inputs', 'sources': sources}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'analysisSources': len(sources), 'bytes': sum(counts), 'mode': 'verified' if verify_only else 'restored_and_verified', 'weightsDownloaded': False}))
    return sources


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    collect(args.cache, args.verify_only)
