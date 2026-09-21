import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

const ORT_VERSION = '1.30.0';
const CACHE_NAME = 'ai-wardrobe-models-v5-3';

// Runtime / memory profile must be defined before any model state is initialized.
// V5.3 accidentally referenced these names before defining them, which caused the
// ES module to abort during evaluation on every device and left the UI stuck at
// "準備本機 AI…".
const IS_IOS = /iPhone|iPad|iPod/i.test(navigator.userAgent) ||
  (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
const MOBILE_MEMORY_GUARD = IS_IOS || /Android/i.test(navigator.userAgent);
const WORKING_MAX_SIDE = IS_IOS ? 1600 : 1920;
const OUTPUT_MAX_SIDE = IS_IOS ? 900 : 1200;

// V5.1: route by scene. Outfit uses human-clothes parsing; single/flat uses fashion detection + foreground matting.
const OUTFIT_MODEL = 'Xenova/segformer_b2_clothes';
const DETECTOR_ID = 'louisJLN/yolo8-fashionpedia::yolov8n';
const DETECTOR_URL = 'https://huggingface.co/louisJLN/yolo8-fashionpedia/resolve/main/results/yolov8n-fashionpedia-1.onnx';
const CUTOUT_ID = 'edgetools/u2netp';
const CUTOUT_URL = 'https://huggingface.co/edgetools/u2netp/resolve/main/u2netp.onnx';

const FP_CLASSES = [
  'shirt, blouse','top, t-shirt, sweatshirt','sweater','cardigan','jacket','vest','pants','shorts','skirt','coat','dress','jumpsuit','cape',
  'glasses','hat','headband, head covering, hair accessory','tie','glove','watch','belt','leg warmer','tights, stockings','sock','shoe','bag, wallet','scarf','umbrella',
  'hood','collar','lapel','epaulette','sleeve','pocket','neckline','buckle','zipper','applique','bead','bow','flower','fringe','ribbon','rivet','ruffle','sequin','tassel'
];
const FP_META = {
  0:['襯衫／上衣','上衣'],1:['T-shirt／上衣','上衣'],2:['針織／毛衣','上衣'],3:['開襟衫','外套'],4:['外套','外套'],5:['背心','上衣'],
  6:['長褲','下身'],7:['短褲','下身'],8:['裙子','下身'],9:['大衣','外套'],10:['洋裝','洋裝'],11:['連身褲','洋裝'],12:['披肩','外套'],
  13:['眼鏡','配件'],14:['帽子','配件'],15:['髮飾／頭飾','配件'],16:['領帶','配件'],17:['手套','配件'],18:['手錶','配件'],19:['腰帶','配件'],
  20:['腿套','配件'],21:['絲襪／褲襪','配件'],22:['襪子','配件'],23:['鞋子','鞋包'],24:['包包','鞋包'],25:['圍巾','配件'],26:['雨傘','配件']
};
const MAIN_SINGLE_CLASSES = new Set([0,1,2,3,4,5,6,7,8,9,10,11,12]);
const ACCESSORY_CLASSES = new Set([13,14,15,16,17,18,19,20,21,22,23,24,25,26]);

const OUTFIT_MAP = {
  'Upper-clothes':{name:'上衣',cat:'上衣'},
  'Pants':{name:'長／短褲',cat:'下身'},
  'Skirt':{name:'裙子',cat:'下身'},
  'Dress':{name:'洋裝',cat:'洋裝'},
  'Bag':{name:'包包',cat:'鞋包'},
  'Belt':{name:'腰帶',cat:'配件'},
  'Hat':{name:'帽子',cat:'配件'},
  'Sunglasses':{name:'眼鏡',cat:'配件'},
  'Scarf':{name:'圍巾',cat:'配件'}
};
const HUMAN_LABELS = new Set(['Face','Hair','Left-arm','Right-arm','Left-leg','Right-leg','Torso-skin']);

let outfitPipelinePromise = null;
let detectorSessionPromise = null;
let cutoutSessionPromise = null;
let runtimeInfo = {
  version:'5.3.1-memory-safe',
  outfitModel:OUTFIT_MODEL,
  singleDetector:DETECTOR_ID,
  singleCutout:CUTOUT_ID,
  backend:IS_IOS?'iPhone 安全模式 · 尚未載入':'尚未載入',
  downloadedBytes:0
};

function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function yieldUI(){return new Promise(r=>setTimeout(r,16));}
function iou(a,b){
  const x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]),x2=Math.min(a[2],b[2]),y2=Math.min(a[3],b[3]);
  const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1),aa=Math.max(0,a[2]-a[0])*Math.max(0,a[3]-a[1]),bb=Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
  return inter/Math.max(1e-6,aa+bb-inter);
}
function nms(cands,iouThr=.45,maxDet=12){
  const q=[...cands].sort((a,b)=>b.score-a.score),out=[];
  while(q.length&&out.length<maxDet){const a=q.shift();out.push(a);for(let i=q.length-1;i>=0;i--)if(iou(a.box,q[i].box)>iouThr)q.splice(i,1);}return out;
}
function normalizedBox(src,w,h){return [src[0]/w,src[1]/h,src[2]/w,src[3]/h].map(v=>clamp(v,0,1));}
function colorDistance(a,b){return Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2);}
function similarity(a,b){if(a.cat!==b.cat)return 0;const color=Math.max(0,1-colorDistance(a.avgColor||[128,128,128],b.avgColor||[128,128,128])/150),aspect=Math.max(0,1-Math.abs((a.aspect||1)-(b.aspect||1))/Math.max(.2,Math.max(a.aspect||1,b.aspect||1)));return Math.round((color*.62+aspect*.38)*100);}
function findDuplicates(items){const pairs=[];for(let i=0;i<items.length;i++)for(let j=i+1;j<items.length;j++){if(items[i].sourceIndex===items[j].sourceIndex)continue;const score=similarity(items[i],items[j]);if(score>=76)pairs.push({a:items[i].id,b:items[j].id,score});}return pairs.sort((a,b)=>b.score-a.score).slice(0,8);}

function getOrt(){
  const ort=window.ort;if(!ort)throw new Error('ONNX Runtime Web 尚未載入');
  ort.env.wasm.wasmPaths=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  // iOS Safari/WKWebView gets a conservative memory profile. One WASM thread avoids
  // extra SharedArrayBuffer/thread stacks while we validate stability.
  ort.env.wasm.numThreads=IS_IOS?1:Math.max(1,Math.min(4,(navigator.hardwareConcurrency||2)-1));
  return ort;
}
async function fetchCached(url,onProgress,label){
  let cached=null;
  if('caches'in window){try{cached=await(await caches.open(CACHE_NAME)).match(url);}catch{}}
  if(cached){const b=await cached.arrayBuffer();onProgress?.({type:'modelPart',label,status:'ready',cached:true,bytes:b.byteLength});return b;}
  const r=await fetch(url,{mode:'cors'});if(!r.ok)throw new Error(`${label} 下載失敗 HTTP ${r.status}`);
  const total=Number(r.headers.get('content-length'))||0;let bytes;
  if(r.body&&total){const rd=r.body.getReader(),chunks=[];let loaded=0;while(true){const {done,value}=await rd.read();if(done)break;chunks.push(value);loaded+=value.byteLength;onProgress?.({type:'modelPart',label,status:'progress',progress:Math.round(loaded/total*100)});}bytes=new Uint8Array(loaded);let o=0;for(const c of chunks){bytes.set(c,o);o+=c.byteLength;}}
  else bytes=new Uint8Array(await r.arrayBuffer());
  if('caches'in window){try{await(await caches.open(CACHE_NAME)).put(url,new Response(bytes,{headers:{'content-type':'application/octet-stream'}}));}catch{}}
  runtimeInfo.downloadedBytes+=bytes.byteLength;onProgress?.({type:'modelPart',label,status:'ready',cached:false,bytes:bytes.byteLength});return bytes.buffer;
}
async function createOrtSession(url,onProgress,label){
  const ort=getOrt(),buf=await fetchCached(url,onProgress,label);
  // V5.3: do not use WebGPU on iPhone/iPad. Safari can terminate the tab under GPU
  // memory pressure; stability matters more than peak throughput for this benchmark.
  if(!IS_IOS&&navigator.gpu){try{const s=await ort.InferenceSession.create(buf,{executionProviders:['webgpu'],graphOptimizationLevel:'all'});runtimeInfo.backend='WebGPU';return s;}catch(e){console.warn(label,'WebGPU fallback',e);}}
  const s=await ort.InferenceSession.create(buf,{executionProviders:['wasm'],graphOptimizationLevel:'all'});runtimeInfo.backend=IS_IOS?'WASM · iPhone 安全模式':'WASM';return s;
}
async function getDetector(onProgress){if(!detectorSessionPromise)detectorSessionPromise=createOrtSession(DETECTOR_URL,onProgress,'Fashion detector').catch(e=>{detectorSessionPromise=null;throw e});return detectorSessionPromise;}
async function getCutout(onProgress){if(!cutoutSessionPromise)cutoutSessionPromise=createOrtSession(CUTOUT_URL,onProgress,'Foreground cutout').catch(e=>{cutoutSessionPromise=null;throw e});return cutoutSessionPromise;}
async function getOutfitPipeline(onProgress){
  if(!outfitPipelinePromise){
    onProgress?.({type:'modelPart',label:'Outfit parser',status:'loading'});
    outfitPipelinePromise=pipeline('image-segmentation',OUTFIT_MODEL,{dtype:'q8'}).then(p=>{onProgress?.({type:'modelPart',label:'Outfit parser',status:'ready'});return p;}).catch(e=>{outfitPipelinePromise=null;throw e});
  }
  return outfitPipelinePromise;
}

async function probeImageDimensions(file){
  try{
    const ab=await file.slice(0,Math.min(file.size,524288)).arrayBuffer(),v=new DataView(ab);
    if(v.byteLength>=24&&v.getUint32(0)===0x89504e47&&v.getUint32(4)===0x0d0a1a0a)return {w:v.getUint32(16),h:v.getUint32(20)};
    if(v.byteLength>=4&&v.getUint16(0)===0xffd8){let o=2;while(o+9<v.byteLength){if(v.getUint8(o)!==0xff){o++;continue}const m=v.getUint8(o+1);if(m===0xd8||m===0xd9){o+=2;continue}if(o+4>v.byteLength)break;const len=v.getUint16(o+2);if(len<2)break;if([0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf].includes(m)&&o+9<v.byteLength)return {h:v.getUint16(o+5),w:v.getUint16(o+7)};o+=2+len;}}
  }catch{}
  return null;
}
async function loadBitmap(file,maxSide=WORKING_MAX_SIDE){
  if('createImageBitmap'in window){
    const dim=await probeImageDimensions(file);
    if(dim?.w&&dim?.h){
      const sc=Math.min(1,maxSide/Math.max(dim.w,dim.h)),rw=Math.max(1,Math.round(dim.w*sc)),rh=Math.max(1,Math.round(dim.h*sc));
      try{return await createImageBitmap(file,{imageOrientation:'from-image',resizeWidth:rw,resizeHeight:rh,resizeQuality:'high'});}catch(e){console.warn('resized createImageBitmap fallback',e);}
    }
    const full=await createImageBitmap(file,{imageOrientation:'from-image'});
    if(Math.max(full.width,full.height)<=maxSide)return full;
    const sc=Math.min(1,maxSide/Math.max(full.width,full.height)),rw=Math.max(1,Math.round(full.width*sc)),rh=Math.max(1,Math.round(full.height*sc)),c=document.createElement('canvas');c.width=rw;c.height=rh;c.getContext('2d').drawImage(full,0,0,rw,rh);full.close?.();const small=await createImageBitmap(c);c.width=c.height=1;return small;
  }
  return new Promise((resolve,reject)=>{const u=URL.createObjectURL(file),img=new Image();img.onload=async()=>{try{const sc=Math.min(1,maxSide/Math.max(img.naturalWidth,img.naturalHeight)),rw=Math.max(1,Math.round(img.naturalWidth*sc)),rh=Math.max(1,Math.round(img.naturalHeight*sc)),c=document.createElement('canvas');c.width=rw;c.height=rh;c.getContext('2d').drawImage(img,0,0,rw,rh);resolve(await createImageBitmap(c));c.width=c.height=1;}catch(e){reject(e)}finally{URL.revokeObjectURL(u)}};img.onerror=reject;img.src=u;});
}
async function bitmapToBlobURL(bm,maxSide=1600){const scale=Math.min(1,maxSide/Math.max(bm.width,bm.height)),w=Math.max(1,Math.round(bm.width*scale)),h=Math.max(1,Math.round(bm.height*scale)),c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').drawImage(bm,0,0,w,h);const b=await new Promise((res,rej)=>c.toBlob(x=>x?res(x):rej(new Error('影像暫存失敗')),'image/jpeg',.92));c.width=c.height=1;return URL.createObjectURL(b);}
async function canvasBlobURL(c,type='image/webp',quality=.92){const b=await new Promise((res,rej)=>c.toBlob(x=>x?res(x):rej(new Error('影像輸出失敗')),type,quality));return URL.createObjectURL(b);}

// ---------- Outfit / human-worn clothes ----------
function maskPixelValue(mask,p){
  const data=mask?.data;if(!data)return 0;const ch=Math.max(1,mask.channels||Math.round(data.length/Math.max(1,(mask.width||1)*(mask.height||1)))||1),base=p*ch;
  let v=0;
  if(ch===1)v=Number(data[base]||0);
  else {const lim=Math.min(3,ch);for(let c=0;c<lim;c++)v=Math.max(v,Number(data[base+c]||0));}
  if(v<=1.5)v*=255;
  return clamp(v,0,255);
}
async function segmentationResultToMaskCanvas(seg){
  const mask=seg.mask;
  // Transformers.js RawImage masks may render through toCanvas() with an opaque alpha channel.
  // Build alpha from the mask pixel values first; otherwise the whole photo becomes foreground.
  if(mask?.data&&mask.width&&mask.height){
    const c=document.createElement('canvas');c.width=mask.width;c.height=mask.height;const ctx=c.getContext('2d'),img=ctx.createImageData(mask.width,mask.height),pixels=mask.width*mask.height;
    for(let p=0;p<pixels;p++){const a=maskPixelValue(mask,p)>8?255:0,j=p*4;img.data[j]=img.data[j+1]=img.data[j+2]=255;img.data[j+3]=a;}
    ctx.putImageData(img,0,0);return c;
  }
  if(mask instanceof HTMLCanvasElement)return mask;
  if(mask?.toCanvas){
    const raw=await mask.toCanvas(),ctx=raw.getContext('2d',{willReadFrequently:true}),im=ctx.getImageData(0,0,raw.width,raw.height),d=im.data;
    // Ignore source alpha; use luminance as the semantic foreground mask.
    for(let i=0;i<d.length;i+=4){const v=Math.max(d[i],d[i+1],d[i+2]);d[i]=d[i+1]=d[i+2]=255;d[i+3]=v>8?255:0;}ctx.putImageData(im,0,0);return raw;
  }
  throw new Error('Outfit parser mask 格式不支援');
}
async function makeSemanticCutout(bm,segments,labels){
  const masks=[];for(const s of segments){if(labels.includes(s.label))masks.push(await segmentationResultToMaskCanvas(s));}
  if(!masks.length)return null;
  // Keep the working alpha canvas at model/preview resolution instead of full 24MP source resolution.
  const mw=masks[0].width,mh=masks[0].height,alpha=document.createElement('canvas');alpha.width=mw;alpha.height=mh;const actx=alpha.getContext('2d');
  for(const m of masks){actx.globalCompositeOperation='source-over';actx.drawImage(m,0,0,mw,mh);}
  const ad=actx.getImageData(0,0,mw,mh).data;let minX=mw,minY=mh,maxX=-1,maxY=-1;
  for(let y=0;y<mh;y+=2)for(let x=0;x<mw;x+=2){if(ad[(y*mw+x)*4+3]>40){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);}}
  if(maxX<minX||maxY<minY){alpha.width=alpha.height=1;return null;}
  const pad=Math.max(3,Math.round(Math.max(maxX-minX,maxY-minY)*.035)),mx0=clamp(minX-pad,0,mw),my0=clamp(minY-pad,0,mh),mx1=clamp(maxX+pad,0,mw),my1=clamp(maxY+pad,0,mh),mcw=mx1-mx0,mch=my1-my0;
  const sx=bm.width/mw,sy=bm.height/mh,x0=mx0*sx,y0=my0*sy,cw=mcw*sx,ch=mch*sy,scale=Math.min(1,OUTPUT_MAX_SIDE/Math.max(cw,ch)),ow=Math.max(1,Math.round(cw*scale)),oh=Math.max(1,Math.round(ch*scale));
  const out=document.createElement('canvas');out.width=ow;out.height=oh;const ctx=out.getContext('2d');ctx.drawImage(bm,x0,y0,cw,ch,0,0,ow,oh);ctx.globalCompositeOperation='destination-in';ctx.drawImage(alpha,mx0,my0,mcw,mch,0,0,ow,oh);ctx.globalCompositeOperation='source-over';
  const pix=ctx.getImageData(0,0,ow,oh).data;let r=0,g=0,b=0,n=0;for(let i=0;i<pix.length;i+=16)if(pix[i+3]>50){r+=pix[i];g+=pix[i+1];b+=pix[i+2];n++;}
  const url=await canvasBlobURL(out);alpha.width=alpha.height=out.width=out.height=1;return {url,box:[x0,y0,cw,ch],avgColor:n?[Math.round(r/n),Math.round(g/n),Math.round(b/n)]:[128,128,128]};
}
function maskAreaRatio(seg){
  const m=seg.mask;if(!m?.data||!m.width||!m.height)return 0;const pixels=m.width*m.height;let n=0;for(let p=0;p<pixels;p++)if(maskPixelValue(m,p)>8)n++;return n/Math.max(1,pixels);
}
async function analyzeOutfit(file,sourceIndex,onProgress,preSegments=null){
  const bm=await loadBitmap(file,WORKING_MAX_SIDE),url=await bitmapToBlobURL(bm,Math.min(WORKING_MAX_SIDE,1400)),pipe=await getOutfitPipeline(onProgress),segments=preSegments||await pipe(url),items=[];URL.revokeObjectURL(url);
  const byLabel=new Map();for(const s of segments){if(!byLabel.has(s.label))byLabel.set(s.label,[]);byLabel.get(s.label).push(s);}
  const plans=[];for(const [label,meta] of Object.entries(OUTFIT_MAP))if(byLabel.has(label))plans.push({labels:[label],label,meta,confidence:92});
  const shoeLabels=['Left-shoe','Right-shoe'];if(shoeLabels.some(x=>byLabel.has(x)))plans.push({labels:shoeLabels,label:'Shoes',meta:{name:'鞋子',cat:'鞋包'},confidence:90});
  let idx=0;for(const p of plans){const segs=segments.filter(s=>p.labels.includes(s.label)),cut=await makeSemanticCutout(bm,segs,p.labels);if(!cut)continue;const bbox=normalizedBox(cut.box,bm.width,bm.height),area=bbox[2]*bbox[3];if(area<.002)continue;items.push({id:`${sourceIndex}-outfit-${idx++}-${Math.random().toString(36).slice(2)}`,name:p.meta.name,cat:p.meta.cat,label:p.label,sourceIndex,sourceName:file.name,sourceWidth:bm.width,sourceHeight:bm.height,requestedMode:'outfit',routeMode:'outfit',classificationConfidence:p.confidence,needsGenericCategoryReview:false,reviewReason:'',layerWarning:p.label==='Upper-clothes'?'人物多層穿搭仍可能把外套與內搭合成同一個上身區域。':'',photo:cut.url,photoType:'semantic-mask',bbox,areaRatio:area,aspect:bbox[2]/Math.max(.001,bbox[3]),avgColor:cut.avgColor,localDebug:{engine:'SegFormer clothes parser'}});}
  bm.close?.();onProgress?.({type:'inference',sourceIndex,stage:'done',count:items.length,mode:'outfit',backend:'local'});return {items,segments};
}
async function sceneHasHuman(file,onProgress){
  const bm=await loadBitmap(file,900),url=await bitmapToBlobURL(bm,900),pipe=await getOutfitPipeline(onProgress),segments=await pipe(url);URL.revokeObjectURL(url);bm.close?.();
  let humanHits=0;for(const s of segments)if(HUMAN_LABELS.has(s.label))humanHits+=maskAreaRatio(s)||.01;
  return {hasHuman:humanHits>.012,segments,humanHits};
}

// ---------- Single / flat garment ----------
function prepareYoloInput(bm,size,ort){const scale=Math.min(size/bm.width,size/bm.height),dw=Math.round(bm.width*scale),dh=Math.round(bm.height*scale),padX=(size-dw)/2,padY=(size-dh)/2,c=document.createElement('canvas');c.width=size;c.height=size;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,size,size);ctx.drawImage(bm,padX,padY,dw,dh);const rgba=ctx.getImageData(0,0,size,size).data,plane=size*size,data=new Float32Array(plane*3);for(let i=0,p=0;i<rgba.length;i+=4,p++){data[p]=rgba[i]/255;data[plane+p]=rgba[i+1]/255;data[plane*2+p]=rgba[i+2]/255;}c.width=c.height=1;return {tensor:new ort.Tensor('float32',data,[1,3,size,size]),scale,padX,padY,size,sourceWidth:bm.width,sourceHeight:bm.height};}
function decodeYoloDetection(tensor,conf=.18){
  const d=tensor.dims.map(Number),nc=FP_CLASSES.length;let count,channels,at;
  if(d.length!==3)throw new Error(`Fashion detector 輸出不符：${d.join('×')}`);
  if(d[1]===4+nc){channels=d[1];count=d[2];at=(n,c)=>tensor.data[c*count+n];}
  else if(d[2]===4+nc){count=d[1];channels=d[2];at=(n,c)=>tensor.data[n*channels+c];}
  else if(d[2]===6){return Array.from({length:d[1]},(_,n)=>({box:[tensor.data[n*6],tensor.data[n*6+1],tensor.data[n*6+2],tensor.data[n*6+3]],score:tensor.data[n*6+4],classId:Math.round(tensor.data[n*6+5])})).filter(x=>x.score>=conf);}
  else throw new Error(`Fashion detector channels 不符：${d.join('×')}`);
  const out=[];for(let n=0;n<count;n++){const cx=at(n,0),cy=at(n,1),w=at(n,2),h=at(n,3),scores=[];for(let c=0;c<nc;c++)scores.push({classId:c,score:Number(at(n,4+c))});scores.sort((a,b)=>b.score-a.score);const top=scores[0];if(!top||top.score<conf)continue;out.push({box:[cx-w/2,cy-h/2,cx+w/2,cy+h/2],score:top.score,classId:top.classId,classCandidates:scores.slice(0,3)});}return out;
}
function inverseDetBox(box,prep){const [x1,y1,x2,y2]=box;const sx1=clamp((x1-prep.padX)/prep.scale,0,prep.sourceWidth),sy1=clamp((y1-prep.padY)/prep.scale,0,prep.sourceHeight),sx2=clamp((x2-prep.padX)/prep.scale,0,prep.sourceWidth),sy2=clamp((y2-prep.padY)/prep.scale,0,prep.sourceHeight);return [sx1,sy1,Math.max(1,sx2-sx1),Math.max(1,sy2-sy1)];}
async function detectFashion(bm,onProgress,allowedSet){const ort=getOrt(),s=await getDetector(onProgress),name=s.inputNames?.[0]||'images',meta=s.inputMetadata?.[0]||{},dims=meta.shape||meta.dimensions||[],h=Number(dims[dims.length-2]),size=Number.isFinite(h)&&h>0?h:640,prep=prepareYoloInput(bm,size,ort),start=performance.now(),out=await s.run({[name]:prep.tensor}),pred=Object.values(out).find(t=>t?.dims?.length===3);if(!pred)throw new Error('Fashion detector 沒有 detection tensor');let dets=decodeYoloDetection(pred,.15).filter(x=>allowedSet.has(x.classId));for(const d of dets){if(Math.max(...d.box.map(Math.abs))<=2.2)d.box=d.box.map(v=>v*size);}dets=nms(dets,.45,8);for(const d of dets)d.sourceBox=inverseDetBox(d.box,prep);return {dets,ms:Math.round(performance.now()-start),size};}
function choosePrimary(dets,bm){if(!dets.length)return null;return [...dets].sort((a,b)=>{const aa=(a.sourceBox[2]*a.sourceBox[3])/(bm.width*bm.height),bb=(b.sourceBox[2]*b.sourceBox[3])/(bm.width*bm.height);return (b.score*Math.pow(bb,.28))-(a.score*Math.pow(aa,.28));})[0];}
function cropCanvas(bm,box,pad=.06,maxSide=OUTPUT_MAX_SIDE){const [x,y,w,h]=box,px=w*pad,py=h*pad,x0=clamp(x-px,0,bm.width),y0=clamp(y-py,0,bm.height),x1=clamp(x+w+px,0,bm.width),y1=clamp(y+h+py,0,bm.height),cw=x1-x0,ch=y1-y0,sc=Math.min(1,maxSide/Math.max(cw,ch)),ow=Math.max(1,Math.round(cw*sc)),oh=Math.max(1,Math.round(ch*sc)),c=document.createElement('canvas');c.width=ow;c.height=oh;c.getContext('2d').drawImage(bm,x0,y0,cw,ch,0,0,ow,oh);return {canvas:c,sourceBox:[x0,y0,cw,ch]};}
function u2TensorFromCanvas(c,ort){const t=document.createElement('canvas');t.width=t.height=320;const ctx=t.getContext('2d',{willReadFrequently:true});ctx.drawImage(c,0,0,320,320);const rgba=ctx.getImageData(0,0,320,320).data,plane=320*320,data=new Float32Array(plane*3),mean=[.485,.456,.406],std=[.229,.224,.225];let max=1;for(let i=0;i<rgba.length;i+=4)max=Math.max(max,rgba[i],rgba[i+1],rgba[i+2]);for(let i=0,p=0;i<rgba.length;i+=4,p++){data[p]=(rgba[i]/max-mean[0])/std[0];data[plane+p]=(rgba[i+1]/max-mean[1])/std[1];data[plane*2+p]=(rgba[i+2]/max-mean[2])/std[2];}t.width=t.height=1;return new ort.Tensor('float32',data,[1,3,320,320]);}
async function applyU2Net(c,onProgress){
  const ort=getOrt(),s=await getCutout(onProgress),input=s.inputNames?.[0]||'input.1',tensor=u2TensorFromCanvas(c,ort),out=await s.run({[input]:tensor}),sal=Object.values(out)[0],arr=sal.data;
  let lo=Infinity,hi=-Infinity;for(let i=0;i<arr.length;i++){lo=Math.min(lo,arr[i]);hi=Math.max(hi,arr[i]);}
  const md=document.createElement('canvas');md.width=md.height=320;const ctx=md.getContext('2d'),im=ctx.createImageData(320,320),den=Math.max(1e-6,hi-lo);let strong=0;
  // U2Net's documented postprocess is min-max -> 0..255 alpha. Do not hard-threshold dark garments.
  for(let i=0;i<arr.length;i++){const v=clamp((arr[i]-lo)/den,0,1),a=Math.round(v*255),j=i*4;im.data[j]=im.data[j+1]=im.data[j+2]=255;im.data[j+3]=a;if(a>64)strong++;}
  ctx.putImageData(im,0,0);const coverage=strong/(320*320);
  const mask=document.createElement('canvas');mask.width=c.width;mask.height=c.height;const mctx=mask.getContext('2d');mctx.imageSmoothingEnabled=true;mctx.drawImage(md,0,0,c.width,c.height);
  const outc=document.createElement('canvas');outc.width=c.width;outc.height=c.height;const o=outc.getContext('2d');o.drawImage(c,0,0);o.globalCompositeOperation='destination-in';o.drawImage(mask,0,0);o.globalCompositeOperation='source-over';
  const pix=o.getImageData(0,0,outc.width,outc.height).data;let r=0,g=0,b=0,n=0;for(let i=0;i<pix.length;i+=16)if(pix[i+3]>55){r+=pix[i];g+=pix[i+1];b+=pix[i+2];n++;}
  let url,maskAccepted=coverage>=.12&&coverage<=.96;
  if(maskAccepted)url=await canvasBlobURL(outc);
  else url=await canvasBlobURL(c,'image/webp',.92); // Better to show the whole detected garment than a tiny wrong saliency island.
  md.width=md.height=mask.width=mask.height=outc.width=outc.height=1;
  return {url,coverage,maskAccepted,avgColor:n?[Math.round(r/n),Math.round(g/n),Math.round(b/n)]:[128,128,128]};
}
async function analyzeSingle(file,sourceIndex,onProgress,accessory=false){
  const bm=await loadBitmap(file,WORKING_MAX_SIDE),allowed=accessory?ACCESSORY_CLASSES:MAIN_SINGLE_CLASSES,{dets,ms,size}=await detectFashion(bm,onProgress,allowed),best=choosePrimary(dets,bm);
  // Do not keep detector and cutout models resident at the same time on iOS.
  if(IS_IOS){await releaseDetectorOnly();await memoryYield();}
  let box=best?.sourceBox||[0,0,bm.width,bm.height],meta=best?FP_META[best.classId]:['單品（請確認）',accessory?'配件':'上衣'],label=best?FP_CLASSES[best.classId]:'unknown',conf=best?Math.round(best.score*100):0;
  const rawCandidates=(best?.classCandidates||[]).filter(x=>allowed.has(x.classId)).slice(0,3).map(x=>({name:FP_META[x.classId]?.[0]||FP_CLASSES[x.classId],label:FP_CLASSES[x.classId],score:Math.round(x.score*100)}));
  const margin=rawCandidates.length>1?(rawCandidates[0].score-rawCandidates[1].score):100;
  const crop=cropCanvas(bm,box,best?.score>.22?.08:.02),cut=await applyU2Net(crop.canvas,onProgress),bbox=normalizedBox(crop.sourceBox,bm.width,bm.height);
  if(IS_IOS){await releaseCutoutOnly();await memoryYield();}
  const uncertain=!best||conf<78||margin<14;
  const finalName=uncertain?'單品（請確認）':meta[0],finalCat=uncertain?(accessory?'配件':meta[1]):meta[1];
  const reasons=[];if(uncertain)reasons.push(`分類器不夠確定${rawCandidates.length?`；候選：${rawCandidates.map(x=>`${x.name} ${x.score}%`).join('、')}`:''}`);if(!cut.maskAccepted)reasons.push('去背遮罩信心不足，先保留完整偵測區域，避免只剩圖案或碎片');
  const item={id:`${sourceIndex}-single-${Math.random().toString(36).slice(2)}`,name:finalName,cat:finalCat,label,sourceIndex,sourceName:file.name,sourceWidth:bm.width,sourceHeight:bm.height,requestedMode:accessory?'accessory':'single',routeMode:accessory?'accessory':'single',classificationConfidence:conf,classificationCandidates:rawCandidates,needsGenericCategoryReview:uncertain,reviewReason:reasons.join('；'),layerWarning:!cut.maskAccepted?'⚠ 去背模型未通過完整度 Gate，本卡先顯示主要偵測區域。':'',photo:cut.url,photoType:cut.maskAccepted?'detector+foreground-mask':'detector-crop-fallback',bbox,areaRatio:bbox[2]*bbox[3],aspect:bbox[2]/Math.max(.001,bbox[3]),avgColor:cut.avgColor,localDebug:{engine:'FashionPedia detector + U2Netp',detectorMs:ms,inputSize:size,detections:dets.length,maskCoverage:cut.coverage,maskAccepted:cut.maskAccepted}};
  crop.canvas.width=crop.canvas.height=1;bm.close?.();onProgress?.({type:'inference',sourceIndex,stage:'done',count:1,mode:item.routeMode,ms,backend:runtimeInfo.backend});return [item];
}

async function analyzeOne(file,sourceIndex,mode,onProgress){
  if(mode==='outfit'){const r=(await analyzeOutfit(file,sourceIndex,onProgress)).items;if(IS_IOS){await releaseOutfitOnly();await memoryYield();}return r;}
  if(mode==='single')return analyzeSingle(file,sourceIndex,onProgress,false);
  if(mode==='accessory')return analyzeSingle(file,sourceIndex,onProgress,true);
  const scene=await sceneHasHuman(file,onProgress),resolved=scene.hasHuman?'outfit':'single';onProgress?.({type:'resolvedRoute',sourceIndex,requestedMode:'auto',resolvedMode:resolved,router:{kind:'SegFormer-human-cues',score:scene.humanHits}});
  if(resolved==='outfit'){const r=(await analyzeOutfit(file,sourceIndex,onProgress,scene.segments)).items;if(IS_IOS){await releaseOutfitOnly();await memoryYield();}return r;}
  if(IS_IOS){await releaseOutfitOnly();await memoryYield();}
  return analyzeSingle(file,sourceIndex,onProgress,false);
}

async function releaseDetectorOnly(){if(detectorSessionPromise){try{const s=await detectorSessionPromise;await s?.release?.();}catch{}detectorSessionPromise=null;}}
async function releaseCutoutOnly(){if(cutoutSessionPromise){try{const s=await cutoutSessionPromise;await s?.release?.();}catch{}cutoutSessionPromise=null;}}
async function releaseOutfitOnly(){if(outfitPipelinePromise){try{const p=await outfitPipelinePromise;await p?.dispose?.();}catch{}outfitPipelinePromise=null;}}
async function memoryYield(ms=80){await new Promise(r=>setTimeout(r,ms));}

async function releaseLocalModels(){
  await Promise.allSettled([releaseDetectorOnly(),releaseCutoutOnly(),releaseOutfitOnly()]);
  runtimeInfo.backend=IS_IOS?'模型已釋放 · iPhone 安全模式':'模型已釋放（快取保留）';
  if(IS_IOS)await memoryYield(120);
}

window.AIWardrobeSegmentation={
  version:'5.3.1-memory-safe',
  modelId:'V5.3.1 Memory-Safe · SegFormer + FashionPedia + U2Netp',
  getRuntimeInfo(){return {...runtimeInfo};},
  async health(){return {ok:true,local:true,...runtimeInfo};},
  async segmentFiles(files,onProgress,options={}){const all=[],mode=options.mode||'auto';try{for(let i=0;i<files.length;i++){onProgress?.({type:'file',index:i,total:files.length,name:files[i].name});onProgress?.({type:'inference',sourceIndex:i,stage:'start'});const items=await analyzeOne(files[i],i,mode,onProgress);all.push(...items);if(IS_IOS){await releaseLocalModels();onProgress?.({type:'memory',status:'released-one',index:i});}await memoryYield(IS_IOS?120:16);}return {items:all,duplicates:findDuplicates(all),runtime:{...runtimeInfo}};}finally{if(MOBILE_MEMORY_GUARD||options.releaseAfterBatch){await releaseLocalModels();onProgress?.({type:'memory',status:'released'});}}}
};
window.dispatchEvent(new CustomEvent('aiwardrobe-segmenter-ready'));
