"""Collect fixed checkpoint configs and official implementation inputs for P2/P3/P4."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from collect_deepseek_round1 import fetch,write_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/families/deepseek'
DATE='2026-10-01'


def collect(cache):
    versions=json.loads((DATA/'versions.json').read_text())
    tasks=[]
    for model in versions['models']:
        name=model['name']; revision=model['revision']; model_id=model['modelId']
        tasks.append((name,revision,model_id,'config.json','checkpoint_config',f'data/families/deepseek/configs/{name}-config.json'))
        if name in ('DeepSeek-V2','DeepSeek-V3-Base','DeepSeek-R1','DeepSeek-R1-Zero'):
            tasks.append((name,revision,model_id,'modeling_deepseek.py','reference_code',None))
        if name in ('DeepSeek-V3.2-Exp','DeepSeek-V3.2','DeepSeek-V4-Flash','DeepSeek-V4-Pro'):
            metadata=json.loads((ROOT/'data/families/deepseek/research/checkpoint-metadata'/f'{name}.json').read_text())
            for f in metadata['siblings']:
                path=f['rfilename']
                if path.startswith('inference/') and path.endswith(('.py','.json')) or path.startswith('modeling_') and path.endswith('.py'):
                    tasks.append((name,revision,model_id,path,'reference_code' if path.endswith('.py') else 'reference_config',None))
    sources=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[]
        for name,revision,model_id,path,kind,local in tasks:
            url=f'https://huggingface.co/{model_id}/resolve/{revision}/{path}'
            jobs.append(((name,revision,model_id,path,kind,local,url),pool.submit(fetch,url)))
        for args,job in jobs:
            name,revision,model_id,path,kind,local,url=args; raw=job.result()
            file=f'extended/{name}/{revision}/{path}'; target=cache/file
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(raw)
            if local:
                target=ROOT/local;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            sid=f'ext-{name.lower()}-{path.replace("/","-").replace(".","-")}'
            sources.append({'id':sid,'title':name+' / '+path,'modelId':model_id,'revision':revision,'path':path,
                            'url':f'https://huggingface.co/{model_id}/blob/{revision}/{path}','rawUrl':url,
                            'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'cacheFile':file,'localPath':local,
                            'accessed':DATE,'kind':kind,'readScope':'采集完成，逐模块核验进度由研究数据记录',
                            'license':'见该 checkpoint 固定 revision 的官方 LICENSE；完整源码不随研究包分发'})
            print(name,path,len(raw))
    write_json(DATA/'extended-sources.json',{'schemaVersion':1,'familyId':'deepseek','accessed':DATE,'sources':sources})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    collect(p.parse_args().cache)
