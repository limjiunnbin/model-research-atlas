"""Render research Markdown and CSV exports from each family's canonical JSON."""
import csv,hashlib,json

def export(root,family):
 if not family.get('hardwarePath'):return
 d=json.loads((root/family['hardwarePath']).read_text());sources=d['sources']
 def refs(rows):
  parts=[]
  for r in rows:
   s=next(s for s in sources if s['id']==r['id']);url=s['url']+('#L'+str(r['line']) if r.get('line') else '')
   parts.append(f"[{s['title']}]({url})")
  return '来源：'+'；'.join(parts)
 md=['# '+family['name']+' 实现、硬件与 Ascend 优化研究','研究快照：'+d['updated'],d['scope']]
 for m in d['modules']:
  md+=['## '+m['title'],' → '.join(m['flow'])]+[m[k] for k in ['meaning','code','hardware','limit']]+[refs(m['refs'])]
 md+=['## 硬件与支持边界']
 for p in d['platforms']:md+=['### '+p['name']]+[p[k] for k in ['precision','support','stack','boundary']]+[refs(p['refs'])]
 md+=['## Ascend 优先验证与优化']
 for o in d['optimizations']:
  md+=['### '+o['priority']+' '+o['title'],o['status']+' · '+o['scope']]+[label+'：'+o[key] for label,key in [('观察','observation'),('方案','proposal'),('指标','metric'),('风险','risk')]]+[refs(o['refs'])]
 md+=['## 统一实验协议']+d['protocol']+['## 版本边界']+[n['text'] for n in d['versionNotes']]
 md+=['## 来源索引']+[f"- {s['id']} [{s['title']}]({s['url']})；{s['revision']}；访问 {s['accessed']}" for s in sources]
 (root/'dist'/d['downloads']['report']).write_text('\n\n'.join(md),encoding='utf-8')
 with (root/'dist'/d['downloads']['sources']).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['id','title','url','revision','sha256','accessed','kind']);w.writeheader();w.writerows(sources)
 for path in d['downloads'].values():
  b=(root/'dist'/path).read_bytes();entry=next(x for x in family['downloads'] if x['path']==path);entry.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
