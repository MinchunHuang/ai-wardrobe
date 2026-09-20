const PROFILE={
  auto:{maxDim:1792,maxBytes:500000},
  outfit:{maxDim:2048,maxBytes:650000},
  single:{maxDim:2048,maxBytes:650000},
  accessory:{maxDim:2560,maxBytes:750000}
};

const CAT_MAP = {
  outerwear:{name:'外套',cat:'外套'}, jacket:{name:'外套',cat:'外套'}, coat:{name:'外套',cat:'外套'}, blazer:{name:'西裝外套',cat:'外套'}, cardigan:{name:'針織外套',cat:'外套'},
  shirt:{name:'襯衫',cat:'上衣'}, 't-shirt':{name:'T-shirt',cat:'上衣'}, sweater:{name:'針織／毛衣',cat:'上衣'}, hoodie:{name:'帽T',cat:'上衣'}, top:{name:'上衣',cat:'上衣'},
  pants:{name:'長褲',cat:'下身'}, trousers:{name:'長褲',cat:'下身'}, jeans:{name:'牛仔褲',cat:'下身'}, shorts:{name:'短褲',cat:'下身'}, skirt:{name:'裙子',cat:'下身'}, dress:{name:'洋裝',cat:'洋裝'},
  shoes:{name:'鞋子',cat:'鞋包'}, bag:{name:'包包',cat:'鞋包'}, handbag:{name:'包包',cat:'鞋包'}, backpack:{name:'後背包',cat:'鞋包'}, belt:{name:'腰帶',cat:'配件'},
  watch:{name:'手錶',cat:'配件'}, glasses:{name:'眼鏡',cat:'配件'}, sunglasses:{name:'太陽眼鏡',cat:'配件'}, necklace:{name:'項鍊',cat:'配件'}, earring:{name:'耳環',cat:'配件'}, bracelet:{name:'手鍊',cat:'配件'}, ring:{name:'戒指',cat:'配件'}, scarf:{name:'圍巾',cat:'配件'}, hat:{name:'帽子',cat:'配件'}
};

function yieldUI(){return new Promise(r=>setTimeout(r,20));}
async function blobFromCanvas(canvas,q){return new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('encode failed')),'image/jpeg',q));}
async function preprocess(file,mode){
  const cfg=PROFILE[mode]||PROFILE.auto;
  const bm=await createImageBitmap(file,{imageOrientation:'from-image'});const sw=bm.width,sh=bm.height;let max=cfg.maxDim,q=.88,blob,canvas;
  for(let attempt=0;attempt<7;attempt++){
    const scale=Math.min(1,max/Math.max(sw,sh)),w=Math.max(1,Math.round(sw*scale)),h=Math.max(1,Math.round(sh*scale));
    canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;canvas.getContext('2d',{alpha:false}).drawImage(bm,0,0,w,h);blob=await blobFromCanvas(canvas,q);
    if(blob.size<=cfg.maxBytes)break;
    canvas.width=canvas.height=1;
    if(q>.68)q-=.07;else max=Math.max(960,Math.round(max*.88));
  }
  bm.close?.();
  const dataUrl=await new Promise((resolve,reject)=>{const fr=new FileReader();fr.onload=()=>resolve(fr.result);fr.onerror=reject;fr.readAsDataURL(blob)});
  const out={dataUrl,width:canvas.width,height:canvas.height,sourceWidth:sw,sourceHeight:sh,bytes:blob.size};canvas.width=canvas.height=1;return out;
}
async function cropOriginal(file,bbox){
  const bm=await createImageBitmap(file,{imageOrientation:'from-image'});const [x,y,w,h]=bbox||[0,0,1,1],pad=.035;const x0=Math.max(0,x-pad),y0=Math.max(0,y-pad),x1=Math.min(1,x+w+pad),y1=Math.min(1,y+h+pad);const sx=Math.round(x0*bm.width),sy=Math.round(y0*bm.height),sw=Math.max(1,Math.round((x1-x0)*bm.width)),sh=Math.max(1,Math.round((y1-y0)*bm.height));const scale=Math.min(1,1200/Math.max(sw,sh));const c=document.createElement('canvas');c.width=Math.max(1,Math.round(sw*scale));c.height=Math.max(1,Math.round(sh*scale));c.getContext('2d').drawImage(bm,sx,sy,sw,sh,0,0,c.width,c.height);bm.close?.();const blob=await new Promise((res,rej)=>c.toBlob(b=>b?res(b):rej(new Error('crop failed')),'image/webp',.91));c.width=c.height=1;return URL.createObjectURL(blob);
}
function colorDistance(a,b){return Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)}
function similarity(a,b){if(a.cat!==b.cat)return 0;const color=Math.max(0,1-colorDistance(a.avgColor||[128,128,128],b.avgColor||[128,128,128])/150),aspect=Math.max(0,1-Math.abs((a.aspect||1)-(b.aspect||1))/Math.max(.2,Math.max(a.aspect||1,b.aspect||1)));return Math.round((color*.62+aspect*.38)*100)}
function findDuplicates(items){const pairs=[];for(let i=0;i<items.length;i++)for(let j=i+1;j<items.length;j++){if(items[i].sourceIndex===items[j].sourceIndex)continue;const score=similarity(items[i],items[j]);if(score>=76)pairs.push({a:items[i].id,b:items[j].id,score})}return pairs.sort((a,b)=>b.score-a.score).slice(0,8)}
async function sampleColor(file,bbox){try{const bm=await createImageBitmap(file,{imageOrientation:'from-image'});const [x,y,w,h]=bbox||[0,0,1,1];const c=document.createElement('canvas');c.width=32;c.height=32;c.getContext('2d').drawImage(bm,x*bm.width,y*bm.height,w*bm.width,h*bm.height,0,0,32,32);bm.close?.();const d=c.getContext('2d').getImageData(0,0,32,32).data;let r=0,g=0,b=0,n=0;for(let i=0;i<d.length;i+=16){r+=d[i];g+=d[i+1];b+=d[i+2];n++}c.width=c.height=1;return [Math.round(r/n),Math.round(g/n),Math.round(b/n)]}catch{return[128,128,128]}}
async function analyzeOne(file,sourceIndex,mode,onProgress){
  const prep=await preprocess(file,mode);onProgress?.({type:'route',sourceIndex,mode,sourceWidth:prep.sourceWidth,sourceHeight:prep.sourceHeight,processedWidth:prep.width,processedHeight:prep.height});
  const res=await fetch('/api/analyze',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({image:prep.dataUrl,mode})});const body=await res.json().catch(()=>({}));if(!res.ok)throw new Error(body.error||`Server AI ${res.status}`);
  const resolvedMode=body.debug?.resolvedMode||mode;
  onProgress?.({type:'resolvedRoute',sourceIndex,requestedMode:mode,resolvedMode,router:body.debug?.router||null});
  const items=[];
  for(let i=0;i<(body.items||[]).length;i++){
    const it=body.items[i],meta=CAT_MAP[it.category]||{name:it.label||'單品',cat:'未分類'};let photo=it.cutout_url||it.mask_url||null,photoType=photo?'mask':'bbox-crop';
    if(!photo&&it.bbox)photo=await cropOriginal(file,it.bbox);if(!photo&&it.visualization_url){photo=it.visualization_url;photoType='visualization'};if(!photo)continue;
    const avg=await sampleColor(file,it.bbox||[0,0,1,1]);const bbox=it.bbox||[0,0,1,1],confidence=Number(it.confidence||0);
    items.push({id:`${sourceIndex}-srv-${i}-${crypto.randomUUID?.()||Math.random().toString(36).slice(2)}`,name:meta.name,cat:meta.cat,label:it.label||it.category,sourceIndex,sourceName:file.name,sourceWidth:prep.sourceWidth,sourceHeight:prep.sourceHeight,processedWidth:prep.width,processedHeight:prep.height,requestedMode:mode,routeMode:resolvedMode,classificationConfidence:Math.round(confidence*100),needsGenericCategoryReview:confidence<.42||meta.cat==='未分類',reviewReason:'Server AI 信心偏低，請確認類別',photo,photoType,bbox,areaRatio:Math.max(.001,bbox[2]*bbox[3]),aspect:bbox[2]/Math.max(.001,bbox[3]),avgColor:avg,serverDebug:body.debug||null});
  }
  return items;
}
window.AIWardrobeSegmentation={version:'4.4-routed-server',modelId:'vufinder/sam3',async health(){const r=await fetch('/api/analyze');return r.json()},async segmentFiles(files,onProgress,options={}){const all=[];const mode=options.mode||'auto';for(let i=0;i<files.length;i++){onProgress?.({type:'file',index:i,total:files.length,name:files[i].name});onProgress?.({type:'inference',sourceIndex:i,stage:'start'});const one=await analyzeOne(files[i],i,mode,onProgress);all.push(...one);onProgress?.({type:'inference',sourceIndex:i,stage:'done',count:one.length,mode:one[0]?.routeMode||mode});await yieldUI()}return{items:all,duplicates:findDuplicates(all)}}};
window.dispatchEvent(new CustomEvent('aiwardrobe-segmenter-ready'));
