const ORT_VERSION = '1.30.0';
const MODEL_ID = 'HasanKabir23/Yolov26s-DeepFashion2';
const MODEL_URL = 'https://huggingface.co/HasanKabir23/Yolov26s-DeepFashion2/resolve/main/best.onnx';
const MODEL_CACHE = 'ai-wardrobe-models-v5';
const CLASS_NAMES = [
  'short_sleeve_top','long_sleeve_top','short_sleeve_outwear','long_sleeve_outwear','vest','sling',
  'shorts','trousers','skirt','short_sleeve_dress','long_sleeve_dress','vest_dress','sling_dress'
];
const CLASS_META = {
  short_sleeve_top:{name:'短袖上衣',cat:'上衣'},
  long_sleeve_top:{name:'長袖上衣',cat:'上衣'},
  short_sleeve_outwear:{name:'短袖外套',cat:'外套'},
  long_sleeve_outwear:{name:'長袖外套',cat:'外套'},
  vest:{name:'背心',cat:'上衣'},
  sling:{name:'細肩帶上衣',cat:'上衣'},
  shorts:{name:'短褲',cat:'下身'},
  trousers:{name:'長褲',cat:'下身'},
  skirt:{name:'裙子',cat:'下身'},
  short_sleeve_dress:{name:'短袖洋裝',cat:'洋裝'},
  long_sleeve_dress:{name:'長袖洋裝',cat:'洋裝'},
  vest_dress:{name:'背心洋裝',cat:'洋裝'},
  sling_dress:{name:'細肩帶洋裝',cat:'洋裝'}
};
const MODE_CFG = {
  auto:{conf:.24,iou:.48,classAware:true,maxDet:12},
  outfit:{conf:.20,iou:.46,classAware:true,maxDet:12},
  single:{conf:.20,iou:.52,classAware:false,maxDet:5}
};

let sessionPromise = null;
let runtimeInfo = {backend:'尚未載入',inputSize:null,model:MODEL_ID,modelBytes:null};

function yieldUI(){return new Promise(r=>setTimeout(r,16));}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function sigmoid(x){return 1/(1+Math.exp(-x));}
function iou(a,b){
  const x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]),x2=Math.min(a[2],b[2]),y2=Math.min(a[3],b[3]);
  const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
  const aa=Math.max(0,a[2]-a[0])*Math.max(0,a[3]-a[1]);
  const bb=Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
  return inter/Math.max(1e-6,aa+bb-inter);
}
function nms(cands,cfg){
  const sorted=[...cands].sort((a,b)=>b.score-a.score), keep=[];
  while(sorted.length && keep.length<cfg.maxDet){
    const a=sorted.shift(); keep.push(a);
    for(let i=sorted.length-1;i>=0;i--){
      const b=sorted[i];
      if(cfg.classAware && a.classId!==b.classId)continue;
      if(iou(a.box,b.box)>cfg.iou)sorted.splice(i,1);
    }
  }
  return keep;
}
function colorDistance(a,b){return Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2);}
function similarity(a,b){
  if(a.cat!==b.cat)return 0;
  const color=Math.max(0,1-colorDistance(a.avgColor||[128,128,128],b.avgColor||[128,128,128])/150);
  const aspect=Math.max(0,1-Math.abs((a.aspect||1)-(b.aspect||1))/Math.max(.2,Math.max(a.aspect||1,b.aspect||1)));
  return Math.round((color*.62+aspect*.38)*100);
}
function findDuplicates(items){
  const pairs=[];
  for(let i=0;i<items.length;i++)for(let j=i+1;j<items.length;j++){
    if(items[i].sourceIndex===items[j].sourceIndex)continue;
    const score=similarity(items[i],items[j]);
    if(score>=76)pairs.push({a:items[i].id,b:items[j].id,score});
  }
  return pairs.sort((a,b)=>b.score-a.score).slice(0,8);
}

async function fetchModel(onProgress){
  let cached=null;
  if('caches' in window){
    try{const c=await caches.open(MODEL_CACHE);cached=await c.match(MODEL_URL);}catch{}
  }
  if(cached){
    const buf=await cached.arrayBuffer();
    runtimeInfo.modelBytes=buf.byteLength;
    onProgress?.({type:'model',info:{status:'progress',progress:100,cached:true}});
    return buf;
  }
  const res=await fetch(MODEL_URL,{mode:'cors'});
  if(!res.ok)throw new Error(`模型下載失敗 HTTP ${res.status}`);
  const total=Number(res.headers.get('content-length'))||0;
  let bytes;
  if(res.body && total){
    const reader=res.body.getReader(),chunks=[];let loaded=0;
    while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);loaded+=value.byteLength;onProgress?.({type:'model',info:{status:'progress',progress:Math.round(loaded/total*100)}});}
    bytes=new Uint8Array(loaded);let off=0;for(const c of chunks){bytes.set(c,off);off+=c.byteLength;}
  }else{
    bytes=new Uint8Array(await res.arrayBuffer());
    onProgress?.({type:'model',info:{status:'progress',progress:100}});
  }
  runtimeInfo.modelBytes=bytes.byteLength;
  if('caches' in window){try{const c=await caches.open(MODEL_CACHE);await c.put(MODEL_URL,new Response(bytes,{headers:{'content-type':'application/octet-stream'}}));}catch{}}
  return bytes.buffer;
}

function getOrt(){
  const ort=window.ort;
  if(!ort)throw new Error('ONNX Runtime Web 尚未載入');
  ort.env.wasm.wasmPaths=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.wasm.numThreads=Math.max(1,Math.min(4,(navigator.hardwareConcurrency||2)-1));
  return ort;
}
async function createSession(onProgress){
  const ort=getOrt();
  const model=await fetchModel(onProgress);
  if(navigator.gpu){
    try{
      const s=await ort.InferenceSession.create(model,{executionProviders:['webgpu'],graphOptimizationLevel:'all'});
      runtimeInfo.backend='WebGPU';return s;
    }catch(err){console.warn('WebGPU session failed, falling back to WASM',err);}
  }
  const s=await ort.InferenceSession.create(model,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  runtimeInfo.backend='WASM';return s;
}
async function getSession(onProgress){
  if(!sessionPromise)sessionPromise=createSession(onProgress).catch(err=>{sessionPromise=null;throw err;});
  const s=await sessionPromise;
  const inputName=s.inputNames?.[0]||'images';
  const meta=s.inputMetadata?.[0]||{};
  const dims=meta.shape||meta.dimensions||meta.dims||[];
  const h=Number(dims[dims.length-2]),w=Number(dims[dims.length-1]);
  runtimeInfo.inputSize=(Number.isFinite(h)&&h>0&&h===w)?h:640;
  onProgress?.({type:'model',info:{status:'ready',backend:runtimeInfo.backend,inputSize:runtimeInfo.inputSize,bytes:runtimeInfo.modelBytes}});
  return s;
}

async function loadBitmap(file){
  if('createImageBitmap' in window)return createImageBitmap(file,{imageOrientation:'from-image'});
  return new Promise((resolve,reject)=>{const u=URL.createObjectURL(file),img=new Image();img.onload=async()=>{try{resolve(await createImageBitmap(img));}catch(e){reject(e);}finally{URL.revokeObjectURL(u);}};img.onerror=reject;img.src=u;});
}
function letterboxTensor(bm,size,ort){
  const scale=Math.min(size/bm.width,size/bm.height),dw=Math.round(bm.width*scale),dh=Math.round(bm.height*scale),padX=(size-dw)/2,padY=(size-dh)/2;
  const c=document.createElement('canvas');c.width=size;c.height=size;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,size,size);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(bm,padX,padY,dw,dh);
  const rgba=ctx.getImageData(0,0,size,size).data,plane=size*size,data=new Float32Array(plane*3);
  for(let i=0,p=0;i<rgba.length;i+=4,p++){data[p]=rgba[i]/255;data[plane+p]=rgba[i+1]/255;data[2*plane+p]=rgba[i+2]/255;}
  c.width=c.height=1;
  return {tensor:new ort.Tensor('float32',data,[1,3,size,size]),scale,padX,padY,size,sourceWidth:bm.width,sourceHeight:bm.height};
}
function outputTensors(outputs){
  const arr=Object.values(outputs||{});
  const pred=arr.find(t=>t?.dims?.length===3);
  const proto=arr.find(t=>t?.dims?.length===4);
  if(!pred)throw new Error('模型輸出格式不符：找不到 detection tensor');
  if(!proto)throw new Error('模型輸出格式不符：找不到 segmentation prototype');
  return {pred,proto};
}
function decodeCandidates(pred,proto,cfg,inputSize){
  const pd=pred.dims.map(Number),md=proto.dims.map(Number),maskDim=md[1]||32;
  const rawChannels=4+CLASS_NAMES.length+maskDim;
  let layout='raw-cn',count,channels,at;
  if(pd[1]===rawChannels){channels=pd[1];count=pd[2];at=(n,c)=>pred.data[c*count+n];}
  else if(pd[2]===rawChannels){count=pd[1];channels=pd[2];at=(n,c)=>pred.data[n*channels+c];layout='raw-nc';}
  else if(pd[2]===6+maskDim){count=pd[1];channels=pd[2];at=(n,c)=>pred.data[n*channels+c];layout='processed';}
  else if(pd[1]===6+maskDim){channels=pd[1];count=pd[2];at=(n,c)=>pred.data[c*count+n];layout='processed-cn';}
  else throw new Error(`未支援的 YOLO segmentation 輸出：${pd.join('×')} / proto ${md.join('×')}`);
  const out=[];
  for(let n=0;n<count;n++){
    let x1,y1,x2,y2,score,classId,coeffStart;
    if(layout.startsWith('processed')){
      x1=at(n,0);y1=at(n,1);x2=at(n,2);y2=at(n,3);score=at(n,4);classId=Math.round(at(n,5));coeffStart=6;
    }else{
      const cx=at(n,0),cy=at(n,1),w=at(n,2),h=at(n,3);let best=-Infinity,bestId=-1;
      for(let c=0;c<CLASS_NAMES.length;c++){const s=at(n,4+c);if(s>best){best=s;bestId=c;}}
      score=best;classId=bestId;coeffStart=4+CLASS_NAMES.length;x1=cx-w/2;y1=cy-h/2;x2=cx+w/2;y2=cy+h/2;
    }
    if(score<cfg.conf||classId<0||classId>=CLASS_NAMES.length)continue;
    if(Math.max(Math.abs(x1),Math.abs(y1),Math.abs(x2),Math.abs(y2))<=2.2){x1*=inputSize;y1*=inputSize;x2*=inputSize;y2*=inputSize;}
    x1=clamp(x1,0,inputSize);y1=clamp(y1,0,inputSize);x2=clamp(x2,0,inputSize);y2=clamp(y2,0,inputSize);
    if(x2-x1<4||y2-y1<4)continue;
    const coeff=new Float32Array(maskDim);for(let m=0;m<maskDim;m++)coeff[m]=at(n,coeffStart+m)||0;
    out.push({box:[x1,y1,x2,y2],score,classId,coeff});
  }
  return {candidates:out,maskDim,layout};
}
function inverseBox(box,prep){
  const [x1,y1,x2,y2]=box;
  const sx1=clamp((x1-prep.padX)/prep.scale,0,prep.sourceWidth),sy1=clamp((y1-prep.padY)/prep.scale,0,prep.sourceHeight);
  const sx2=clamp((x2-prep.padX)/prep.scale,0,prep.sourceWidth),sy2=clamp((y2-prep.padY)/prep.scale,0,prep.sourceHeight);
  return [sx1,sy1,Math.max(1,sx2-sx1),Math.max(1,sy2-sy1)];
}
function buildProtoMask(det,proto,inputSize){
  const md=proto.dims.map(Number),c=md[1],mh=md[2],mw=md[3],plane=mh*mw,out=new Uint8ClampedArray(plane*4);
  const bx1=det.box[0]*mw/inputSize,by1=det.box[1]*mh/inputSize,bx2=det.box[2]*mw/inputSize,by2=det.box[3]*mh/inputSize;
  let active=0;
  for(let p=0;p<plane;p++){
    const x=p%mw,y=(p/mw)|0;let a=0;
    if(x>=bx1&&x<=bx2&&y>=by1&&y<=by2){let z=0;for(let k=0;k<c;k++)z+=det.coeff[k]*proto.data[k*plane+p];a=sigmoid(z)>0.5?255:0;}
    if(a)active++;
    const q=p*4;out[q]=out[q+1]=out[q+2]=255;out[q+3]=a;
  }
  const cv=document.createElement('canvas');cv.width=mw;cv.height=mh;cv.getContext('2d').putImageData(new ImageData(out,mw,mh),0,0);
  return {canvas:cv,mw,mh,active};
}
async function canvasBlob(canvas){return new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('透明單品輸出失敗')),'image/webp',.92));}
async function makeCutout(bm,det,proto,prep){
  const src=inverseBox(det.box,prep),[sx,sy,sw,sh]=src,pad=.035;
  const px=sw*pad,py=sh*pad,x0=clamp(sx-px,0,bm.width),y0=clamp(sy-py,0,bm.height),x1=clamp(sx+sw+px,0,bm.width),y1=clamp(sy+sh+py,0,bm.height);
  const cw=x1-x0,ch=y1-y0,scale=Math.min(1,1200/Math.max(cw,ch)),ow=Math.max(1,Math.round(cw*scale)),oh=Math.max(1,Math.round(ch*scale));
  const out=document.createElement('canvas');out.width=ow;out.height=oh;const octx=out.getContext('2d');octx.drawImage(bm,x0,y0,cw,ch,0,0,ow,oh);
  const pm=buildProtoMask(det,proto,prep.size),full=document.createElement('canvas');full.width=prep.size;full.height=prep.size;const fctx=full.getContext('2d');fctx.imageSmoothingEnabled=true;fctx.drawImage(pm.canvas,0,0,prep.size,prep.size);
  const mask=document.createElement('canvas');mask.width=ow;mask.height=oh;const mctx=mask.getContext('2d');
  const mx0=x0*prep.scale+prep.padX,my0=y0*prep.scale+prep.padY,mw=cw*prep.scale,mh=ch*prep.scale;mctx.drawImage(full,mx0,my0,mw,mh,0,0,ow,oh);
  octx.globalCompositeOperation='destination-in';octx.drawImage(mask,0,0);octx.globalCompositeOperation='source-over';
  const pix=octx.getImageData(0,0,ow,oh).data;let r=0,g=0,b=0,n=0,alpha=0;for(let i=0;i<pix.length;i+=16){if(pix[i+3]>40){r+=pix[i];g+=pix[i+1];b+=pix[i+2];n++;alpha++;}}
  const blob=await canvasBlob(out),url=URL.createObjectURL(blob);pm.canvas.width=pm.canvas.height=full.width=full.height=mask.width=mask.height=out.width=out.height=1;
  return {url,avgColor:n?[Math.round(r/n),Math.round(g/n),Math.round(b/n)]:[128,128,128],sourceBox:src,maskFill:pm.active/(pm.mw*pm.mh)};
}
function normalizedBox(src,w,h){return [src[0]/w,src[1]/h,src[2]/w,src[3]/h].map(v=>clamp(v,0,1));}
function resolveAutoMode(dets){
  if(dets.length<=1)return 'single';
  let strongOverlap=0;for(let i=0;i<dets.length;i++)for(let j=i+1;j<dets.length;j++)if(iou(dets[i].box,dets[j].box)>.35)strongOverlap++;
  return strongOverlap?'outfit':'auto';
}

async function analyzeOne(file,sourceIndex,mode,onProgress){
  if(mode==='accessory')throw new Error('V5.0 先驗證核心衣物模型；配件近拍模型會在核心 Gate 通過後加入。');
  const ort=getOrt(),session=await getSession(onProgress),size=runtimeInfo.inputSize||640,bm=await loadBitmap(file),prep=letterboxTensor(bm,size,ort),start=performance.now();
  onProgress?.({type:'route',sourceIndex,mode,sourceWidth:bm.width,sourceHeight:bm.height,processedWidth:size,processedHeight:size,backend:runtimeInfo.backend});
  const inputName=session.inputNames?.[0]||'images',outputs=await session.run({[inputName]:prep.tensor}),ms=Math.round(performance.now()-start),{pred,proto}=outputTensors(outputs),cfg=MODE_CFG[mode]||MODE_CFG.auto,decoded=decodeCandidates(pred,proto,cfg,size);
  let dets=nms(decoded.candidates,cfg),resolvedMode=mode;
  if(mode==='auto')resolvedMode=resolveAutoMode(dets);
  if(mode==='single'&&dets.length>1){
    // 單件模式：跨類別 NMS 已經比較積極，另外移除非常小的重疊殘片。
    const biggest=Math.max(...dets.map(d=>(d.box[2]-d.box[0])*(d.box[3]-d.box[1])));
    dets=dets.filter(d=>((d.box[2]-d.box[0])*(d.box[3]-d.box[1]))>biggest*.06);
  }
  onProgress?.({type:'resolvedRoute',sourceIndex,requestedMode:mode,resolvedMode,router:{kind:'local-core-heuristic'}});
  const items=[];
  for(let i=0;i<dets.length;i++){
    const d=dets[i],label=CLASS_NAMES[d.classId],meta=CLASS_META[label],cut=await makeCutout(bm,d,proto,prep),bbox=normalizedBox(cut.sourceBox,bm.width,bm.height),confidence=Math.round(d.score*100);
    items.push({
      id:`${sourceIndex}-local-${i}-${crypto.randomUUID?.()||Math.random().toString(36).slice(2)}`,
      name:meta.name,cat:meta.cat,label,sourceIndex,sourceName:file.name,
      sourceWidth:bm.width,sourceHeight:bm.height,processedWidth:size,processedHeight:size,
      requestedMode:mode,routeMode:resolvedMode,classificationConfidence:confidence,
      needsGenericCategoryReview:confidence<42,
      reviewReason:confidence<42?'Local AI 分類信心偏低，建議人工確認':'',
      layerWarning:mode==='outfit'&&['short_sleeve_top','long_sleeve_top','short_sleeve_outwear','long_sleeve_outwear'].includes(label)?'人物多層穿搭仍是本模型的主要壓力測試項目。':'',
      photo:cut.url,photoType:'instance-mask',bbox,
      areaRatio:Math.max(.001,bbox[2]*bbox[3]),aspect:bbox[2]/Math.max(.001,bbox[3]),avgColor:cut.avgColor,
      localDebug:{backend:runtimeInfo.backend,inferenceMs:ms,inputSize:size,predShape:pred.dims,protoShape:proto.dims,layout:decoded.layout,maskFill:cut.maskFill}
    });
    await yieldUI();
  }
  bm.close?.();
  onProgress?.({type:'inference',sourceIndex,stage:'done',count:items.length,mode:resolvedMode,ms,backend:runtimeInfo.backend});
  return items;
}

window.AIWardrobeSegmentation={
  version:'5.0-local-core',
  modelId:MODEL_ID,
  getRuntimeInfo(){return {...runtimeInfo};},
  async health(){return {ok:true,local:true,...runtimeInfo};},
  async segmentFiles(files,onProgress,options={}){
    const all=[],mode=options.mode||'auto';
    for(let i=0;i<files.length;i++){
      onProgress?.({type:'file',index:i,total:files.length,name:files[i].name});
      onProgress?.({type:'inference',sourceIndex:i,stage:'start'});
      const one=await analyzeOne(files[i],i,mode,onProgress);all.push(...one);await yieldUI();
    }
    return {items:all,duplicates:findDuplicates(all),runtime:{...runtimeInfo}};
  }
};
window.dispatchEvent(new CustomEvent('aiwardrobe-segmenter-ready'));
