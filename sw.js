const CACHE='ai-wardrobe-v5-4-1';
const ASSETS=['/','/index.html','/manifest.webmanifest','/local-ai.js'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS))));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
  const keys=await caches.keys();
  // Only remove obsolete app-shell caches. Never delete Transformers.js'
  // persistent model cache or AI Wardrobe model caches during a deployment.
  await Promise.all(keys.filter(k=>k.startsWith('ai-wardrobe-v')&&k!==CACHE).map(k=>caches.delete(k)));
  await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
  if(event.request.method!=='GET')return;
  event.respondWith(fetch(event.request).then(response=>{
    const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy));return response;
  }).catch(()=>caches.match(event.request).then(r=>r||caches.match('/index.html'))));
});
