"""Restore exact public research inputs from manifests, or verify cached source/function hashes."""
import argparse
import ast
import concurrent.futures
import hashlib
import json
from pathlib import Path

from collect_deepseek_round1 import fetch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/families/deepseek'
GROUPS = {'sources.json': 'round1', 'extended-sources.json': 'extended', 'backend-sources.json': 'backends','followup-sources.json':'followup','cann-sources.json':'cann'}


def read(path):
    return json.loads(path.read_text())


def source_url(source):
    if source.get('rawUrl'):
        return source['rawUrl']
    if source.get('repo') and source.get('path'):
        return f'https://raw.githubusercontent.com/{source["repo"]}/{source["revision"]}/{source["path"]}'
    url = source['url']
    if '/blob/' in url and 'huggingface.co/' in url:
        return url.replace('/blob/', '/resolve/', 1)
    if '/blob/' in url and 'github.com/' in url:
        return url.replace('https://github.com/', 'https://raw.githubusercontent.com/', 1).replace('/blob/', '/', 1)
    return url


def validate(research_root=None, verify_only=True):
    inputs = []
    locations = {}
    for manifest, group in GROUPS.items():
        cache = research_root / group if research_root else Path.home() / '.cache/model-research-atlas' / ('deepseek-' + group)
        for source in read(DATA / manifest)['sources']:
            relative = Path(source['cacheFile'])
            assert not relative.is_absolute() and '..' not in relative.parts
            path = cache / relative
            inputs.append((source, path))
            assert source['id'] not in locations
            locations[source['id']] = (source, path)

    def one(item):
        source, path = item
        if not path.exists() and not verify_only:
            # Rolling API responses may change formatting. Prefer the exact shipped legal metadata snapshot.
            local = ROOT / source['localPath'] if source.get('localPath') else None
            if local and local.is_file():raw=local.read_bytes()
            else:
                from collect_deepseek_followup import fetch as fetch_large_public_input
                raw=fetch_large_public_input(source_url(source)) if source['bytes']>4_000_000 else fetch(source_url(source))
            assert len(raw) == source['bytes'] and hashlib.sha256(raw).hexdigest() == source['sha256'], f'Upstream snapshot drift: {source["id"]}'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        raw = path.read_bytes()
        assert len(raw) == source['bytes'] and hashlib.sha256(raw).hexdigest() == source['sha256'], source['id']
        if source.get('gitBlobSha1'):
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==source['gitBlobSha1'],source['id']
        if source.get('localPath'):
            assert (ROOT / source['localPath']).read_bytes() == raw, source['localPath']
        return len(raw)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        byte_counts = list(pool.map(one, inputs))
    evidence = read(DATA / 'research/site-proofs.json')
    assert evidence['sources'] == {s['id']: s for s, _ in inputs}
    texts, functions = {}, {}
    for proof_id, proof in evidence['proofs'].items():
        source, path = locations[proof['source_id']]
        assert proof['sourceSha256'] == source['sha256'] and proof['revision'] == source['revision'] and proof['path'] == source['path'], proof_id
        if source['id'] not in texts:
            text = path.read_text();texts[source['id']] = text
            symbols = {}
            for node in ast.parse(text).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols[node.name] = node
                elif isinstance(node, ast.ClassDef):
                    symbols.update({node.name + '.' + f.name: f for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))})
            functions[source['id']] = symbols
        node = functions[source['id']][proof['symbol']]
        assert (node.lineno, node.end_lineno) == (proof['line'], proof['end']), proof_id
        body = '\n'.join(texts[source['id']].splitlines()[node.lineno - 1:node.end_lineno])
        assert hashlib.sha256(body.encode()).hexdigest() == proof['functionSha256'], proof_id
        assert proof['url'] == source['url'] + f'#L{node.lineno}-L{node.end_lineno}', proof_id
    cpp_ranges = 0
    trace = read(DATA / 'research/backend-trace.json')
    for backend in trace['paths']:
        for record in backend['ranges']:
            source, path = locations[record['sourceId']]
            assert source['revision'] == trace['revision'] and source['repo'] == trace['repo']
            assert record['sourceSha256'] == source['sha256'] and record['path'] == source['path']
            lines = path.read_text().splitlines()
            assert 0 < record['line'] <= record['end'] <= len(lines)
            body = '\n'.join(lines[record['line'] - 1:record['end']])
            assert record['symbolOrStatement'] in body
            assert hashlib.sha256(body.encode()).hexdigest() == record['rangeSha256']
            assert record['url'] == source['url'] + f'#L{record["line"]}-L{record["end"]}'
            cpp_ranges += 1
    cann_ranges=read(DATA/'research/cann-operator-audit.json')['codeRanges']
    for record in cann_ranges:
        source,path=locations[record['sourceId']]
        assert (record['repo'],record['revision'],record['path'],record['sourceSha256'])==(source['repo'],source['revision'],source['path'],source['sha256'])
        lines=path.read_text().splitlines();assert 0<record['line']<=record['end']<=len(lines)
        body='\n'.join(lines[record['line']-1:record['end']])
        assert hashlib.sha256(body.encode()).hexdigest()==record['rangeSha256']
        assert record['url']==source['url']+f'#L{record["line"]}-L{record["end"]}'
    return {'sourceFiles': len(inputs), 'sourceBytes': sum(byte_counts), 'functions': len(evidence['proofs']), 'cppCodeRanges': cpp_ranges, 'cannSourceRanges':len(cann_ranges), 'trackedSnapshotReferences': sum(bool(s.get('localPath')) for s, _ in inputs), 'uniqueTrackedSnapshots': len({s['localPath'] for s, _ in inputs if s.get('localPath')}), 'mode': 'verified_cache' if verify_only else 'restored_and_verified', 'weightsDownloaded': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, help='Public input cache with round1/, extended/, backends/, followup/ and cann/ subdirectories')
    parser.add_argument('--verify-only', action='store_true', help='Do not fetch or write cache files')
    args = parser.parse_args()
    print(json.dumps(validate(args.research_root, args.verify_only), ensure_ascii=False))
