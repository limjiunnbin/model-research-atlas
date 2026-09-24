import {marked} from './vendor/marked-marked.esm.js';
import DOMPurify from './vendor/dompurify-purify.es.mjs';
const root=new URL('./',import.meta.url),article=document.querySelector('#document'),status=document.querySelector('#status');
function safeURL(value,base){try{const u=new URL(value,base);return ['http:','https:'].includes(u.protocol)?u:null}catch{return null}}
export function renderMarkdown(text,source){
 const fragment=DOMPurify.sanitize(marked.parse(text.replace(/^\uFEFF/,''),{gfm:true}),{USE_PROFILES:{html:true},RETURN_DOM_FRAGMENT:true,FORBID_TAGS:['style','form','input','button','iframe','object','embed','base','link','meta'],FORBID_ATTR:['style','srcset','name']});
 const used=new Set([...fragment.querySelectorAll('[id]')].map(e=>e.id));
 for(const h of fragment.querySelectorAll('h1,h2,h3,h4,h5,h6')){
  if(h.id)continue;const base=h.textContent.toLowerCase().trim().replace(/[^\p{L}\p{N}\s_-]/gu,'').replace(/\s/g,'-')||'section';let id=base,n=0;while(used.has(id))id=base+'-'+(++n);h.id=id;used.add(id);
 }
 for(const a of fragment.querySelectorAll('a[href]')){
  const value=a.getAttribute('href');if(value.startsWith('#'))continue;const u=safeURL(value,source);if(!u){a.removeAttribute('href');continue}
  if(u.origin===root.origin&&u.pathname.startsWith(root.pathname)&&u.pathname.toLowerCase().endsWith('.md')){
   const reader=new URL('document.html',root);reader.searchParams.set('file',decodeURIComponent(u.pathname.slice(root.pathname.length)));reader.hash=u.hash;a.href=reader.href;
  }else{a.href=u.href;if(u.origin!==root.origin){a.target='_blank';a.rel='noopener noreferrer'}}
 }
 for(const img of fragment.querySelectorAll('img')){const u=safeURL(img.getAttribute('src'),source);if(u)img.src=u.href;else img.removeAttribute('src');img.loading='lazy'}
 for(const table of fragment.querySelectorAll('table')){const wrap=document.createElement('div');wrap.className='table-scroll';table.replaceWith(wrap);wrap.append(table)}
 return fragment;
}
try{
 const file=new URL(location.href).searchParams.get('file');const list=await fetch(new URL('documents.json',root)).then(r=>{if(!r.ok)throw Error('文档目录无法读取');return r.json()});
 if(!list.files.includes(file))throw Error('找不到该文档，请从研究入口选择文档。');
 const source=new URL(file,root);if(source.origin!==root.origin||!source.pathname.startsWith(root.pathname))throw Error('无效的文档地址');
 const response=await fetch(source);if(!response.ok)throw Error(`文档读取失败（${response.status}）`);
 article.replaceChildren(renderMarkdown(await response.text(),source));status.hidden=true;
 const raw=document.querySelector('#raw');raw.href=source.href;raw.hidden=false;
 const collection=document.querySelector('#collection');collection.href=new URL(file.startsWith('reports/openbmb/')?'reports/openbmb/':'index.html#/family/kimi/sources',root).href;collection.textContent=file.startsWith('reports/openbmb/')?'OpenBMB 报告与下载':'Kimi 来源与下载';
 document.title=(article.querySelector('h1')?.textContent||file.split('/').pop())+' · 文档阅读';
 const target=decodeURIComponent(location.hash.slice(1));if(target)requestAnimationFrame(()=>document.getElementById(target)?.scrollIntoView());
}catch(error){status.textContent=error.message}
