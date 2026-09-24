// Upgrade Markdown links throughout static pages and dynamically rendered views.
const root=new URL('./',import.meta.url);
function update(){
 for(const a of document.querySelectorAll('a[href]:not([data-document-link])')){
  let url;try{url=new URL(a.getAttribute('href'),document.baseURI)}catch{continue}
  if(url.origin!==root.origin||!url.pathname.startsWith(root.pathname)||!url.pathname.toLowerCase().endsWith('.md'))continue;
  a.dataset.documentLink='true';
  const reader=new URL('document.html',root);reader.searchParams.set('file',decodeURIComponent(url.pathname.slice(root.pathname.length)));reader.hash=url.hash;
  if(a.hasAttribute('download')){
   const read=document.createElement('a');read.href=reader.href;read.textContent='在线阅读';read.className=a.className;read.dataset.documentLink='true';a.before(read,document.createTextNode(' · '));
   if(!a.textContent.includes('下载'))a.append(document.createTextNode('（原始 MD 下载）'));
  }else{
   a.href=reader.href;
   const raw=document.createElement('a');raw.href=url.href;raw.download='';raw.textContent='原始 MD 下载';raw.dataset.documentLink='true';raw.className='raw-document';a.after(document.createTextNode(' · '),raw);
  }
 }
}
update();new MutationObserver(update).observe(document.body,{childList:true,subtree:true});
