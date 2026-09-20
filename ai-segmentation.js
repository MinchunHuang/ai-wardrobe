import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

const MODEL_ID = 'Xenova/segformer_b2_clothes';
let segmenterPromise = null;

const CLASS_MAP = {
  'Upper-clothes': { name: '上衣', cat: '上衣' },
  'Pants': { name: '長褲', cat: '下身', lowerBody: true },
  'Skirt': { name: '裙子', cat: '下身', lowerBody: true },
  'Dress': { name: '洋裝', cat: '洋裝' },
  'Bag': { name: '包包', cat: '鞋包' },
  'Hat': { name: '帽子', cat: '配件' },
  'Sunglasses': { name: '眼鏡', cat: '配件' },
  'Belt': { name: '腰帶', cat: '配件' },
  'Scarf': { name: '圍巾', cat: '配件' },
  'Left-shoe': { name: '鞋子', cat: '鞋包', group: 'Shoes' },
  'Right-shoe': { name: '鞋子', cat: '鞋包', group: 'Shoes' },
};

// V4.1: class-specific sanity gates. They intentionally err on the side of asking
// the user to add a missed item rather than presenting obvious segmentation noise.
const AREA_GATES = {
  'Upper-clothes': [0.012, 0.70],
  'Pants': [0.010, 0.65],
  'Skirt': [0.010, 0.55],
  'Dress': [0.020, 0.80],
  'Bag': [0.0025, 0.35],
  'Belt': [0.0018, 0.09],
  'Hat': [0.0015, 0.18],
  'Sunglasses': [0.00035, 0.06],
  'Scarf': [0.0012, 0.22],
  'Shoes': [0.0015, 0.18],
};

async function getSegmenter(onProgress) {
  if (!segmenterPromise) {
    segmenterPromise = pipeline('image-segmentation', MODEL_ID, {
      dtype: 'q8',
      progress_callback: (info) => onProgress?.({ type: 'model', info }),
    });
  }
  return segmenterPromise;
}

async function prepareImage(file, maxDim = 1280) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();
  return { canvas, url: canvas.toDataURL('image/jpeg', 0.92), width, height };
}

function rawMaskValue(mask, mx, my) {
  const channels = mask.channels || Math.max(1, Math.round(mask.data.length / (mask.width * mask.height)));
  return mask.data[(my * mask.width + mx) * channels] || 0;
}

// Keep coherent objects instead of every isolated pixel classified as the same label.
// For each raw semantic mask, keep the dominant component plus any substantial secondary
// component (useful for shoes/hats straps), while rejecting tiny detached islands.
function cleanMask(mask, threshold = 24) {
  const w = mask.width, h = mask.height;
  const on = new Uint8Array(w*h);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    if (rawMaskValue(mask,x,y) > threshold) on[y*w+x]=1;
  }
  const seen = new Uint8Array(w*h);
  const comps=[];
  const qx=new Int32Array(w*h), qy=new Int32Array(w*h);
  const dirs=[[1,0],[-1,0],[0,1],[0,-1]];
  for (let sy=0;sy<h;sy++) for (let sx=0;sx<w;sx++) {
    const start=sy*w+sx;
    if(!on[start]||seen[start]) continue;
    let head=0,tail=0; qx[tail]=sx; qy[tail]=sy; tail++; seen[start]=1;
    const pixels=[];
    while(head<tail){
      const x=qx[head],y=qy[head]; head++; pixels.push(y*w+x);
      for(const [dx,dy] of dirs){const nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=w||ny>=h)continue;const i=ny*w+nx;if(on[i]&&!seen[i]){seen[i]=1;qx[tail]=nx;qy[tail]=ny;tail++;}}
    }
    comps.push(pixels);
  }
  comps.sort((a,b)=>b.length-a.length);
  const keep = new Uint8Array(w*h);
  const largest = comps[0]?.length || 0;
  for (const comp of comps) {
    if (comp.length < Math.max(10, largest*0.12)) break;
    for (const i of comp) keep[i]=1;
  }
  return { width:w, height:h, keep, source:mask };
}

function cleanedMaskValue(cm, x, y, targetW, targetH) {
  const mx = Math.min(cm.width - 1, Math.floor(x * cm.width / targetW));
  const my = Math.min(cm.height - 1, Math.floor(y * cm.height / targetH));
  if (!cm.keep[my*cm.width+mx]) return 0;
  return rawMaskValue(cm.source,mx,my);
}

function buildCutout(sourceCanvas, cleanMasks) {
  const w = sourceCanvas.width, h = sourceCanvas.height;
  const src = sourceCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h);
  const out = new ImageData(new Uint8ClampedArray(src.data), w, h);
  let minX=w, minY=h, maxX=-1, maxY=-1, count=0;
  let sumR=0,sumG=0,sumB=0;
  for (let y=0; y<h; y++) {
    for (let x=0; x<w; x++) {
      const p = y*w+x;
      let a=0;
      for (const m of cleanMasks) a=Math.max(a, cleanedMaskValue(m,x,y,w,h));
      // Slightly harder alpha floor than V4 removes translucent model halos.
      if (a > 32) {
        const alpha=Math.max(0,Math.min(255,Math.round((a-20)*1.16)));
        out.data[p*4+3]=alpha;
        minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y); count++;
        sumR+=src.data[p*4]; sumG+=src.data[p*4+1]; sumB+=src.data[p*4+2];
      } else out.data[p*4+3]=0;
    }
  }
  if (!count || count/(w*h) < 0.00025) return null;
  const pad=Math.round(Math.max(w,h)*0.012);
  minX=Math.max(0,minX-pad); minY=Math.max(0,minY-pad); maxX=Math.min(w-1,maxX+pad); maxY=Math.min(h-1,maxY+pad);
  const full=document.createElement('canvas'); full.width=w; full.height=h; full.getContext('2d').putImageData(out,0,0);
  const cw=maxX-minX+1, ch=maxY-minY+1;
  const crop=document.createElement('canvas'); crop.width=cw; crop.height=ch;
  crop.getContext('2d').drawImage(full,minX,minY,cw,ch,0,0,cw,ch);
  return {
    photo: crop.toDataURL('image/png'),
    areaRatio: count/(w*h),
    aspect: cw/ch,
    avgColor:[Math.round(sumR/count),Math.round(sumG/count),Math.round(sumB/count)],
    bbox:[minX,minY,cw,ch],
    touchesBottom:(maxY >= h-3),
    touchesLeft:(minX <= 2),
    touchesRight:(maxX >= w-3),
  };
}

function areaPass(key, ratio){
  const [min,max]=AREA_GATES[key] || [0.001,0.9];
  return ratio>=min && ratio<=max;
}

function resolvePerImageConflicts(items){
  // Semantic human parsing sometimes emits both Pants and Skirt for the same lower body.
  // Remove only obvious tiny residuals. If both are substantial, keep both but mark them
  // ambiguous so the UI asks the user instead of confidently lying.
  const pants=items.find(x=>x.label==='Pants');
  const skirt=items.find(x=>x.label==='Skirt');
  if(pants&&skirt){
    const hi=Math.max(pants.areaRatio,skirt.areaRatio), lo=Math.min(pants.areaRatio,skirt.areaRatio);
    if(lo < hi*0.32){
      const loser=pants.areaRatio<skirt.areaRatio?pants:skirt;
      items=items.filter(x=>x!==loser);
      const winner=items.find(x=>x===pants||x===skirt);
      if(winner && winner.label==='Skirt' && winner.touchesBottom){
        winner.needsCategoryReview=true;
        winner.reviewReason='下身被模型判為裙子，但照片裁切/寬鬆版型可能讓長褲被誤判';
        winner.name='下身（請確認）';
      }
    } else {
      pants.needsCategoryReview=true; skirt.needsCategoryReview=true;
      pants.reviewReason=skirt.reviewReason='模型同時偵測到 Pants 與 Skirt，請確認實際類別';
      pants.name='下身（褲？）'; skirt.name='下身（裙？）';
    }
  } else if(skirt && skirt.touchesBottom){
    // Cropped product photos like the user's third example are a common failure mode.
    skirt.needsCategoryReview=true;
    skirt.reviewReason='下身延伸到照片底部，寬鬆長褲可能被 human-parsing 模型誤判為裙子';
    skirt.name='下身（請確認）';
  }
  return items;
}

async function segmentOne(file, sourceIndex, onProgress) {
  const prepared = await prepareImage(file);
  const segmenter = await getSegmenter(onProgress);
  onProgress?.({type:'inference', sourceIndex, stage:'start'});
  const output = await segmenter(prepared.url);
  const grouped = new Map();
  for (const result of output) {
    const meta = CLASS_MAP[result.label];
    if (!meta) continue;
    const key = meta.group || result.label;
    if (!grouped.has(key)) grouped.set(key,{...meta,key,labels:[],masks:[]});
    const g=grouped.get(key); g.labels.push(result.label); g.masks.push(cleanMask(result.mask));
  }
  let results=[];
  for (const [key,g] of grouped) {
    const cut=buildCutout(prepared.canvas,g.masks);
    if (!cut || !areaPass(key,cut.areaRatio)) continue;
    results.push({
      id:`${sourceIndex}-${key}-${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}`,
      name:g.name, cat:g.cat, label:g.labels.join('+'), sourceIndex,
      sourceName:file.name, sourceWidth:prepared.width, sourceHeight:prepared.height,
      layerWarning:g.labels.includes('Upper-clothes') ? '外套與內搭可能被合併為同一個上身區域' : '',
      ...cut,
    });
  }
  results=resolvePerImageConflicts(results);
  onProgress?.({type:'inference', sourceIndex, stage:'done', count:results.length});
  return results;
}

function colorDistance(a,b){return Math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)}
function similarity(a,b){
  if(a.cat!==b.cat) return 0;
  const color=Math.max(0,1-colorDistance(a.avgColor,b.avgColor)/150);
  const aspect=Math.max(0,1-Math.abs(a.aspect-b.aspect)/Math.max(.2,Math.max(a.aspect,b.aspect)));
  const area=Math.max(0,1-Math.abs(a.areaRatio-b.areaRatio)/Math.max(.04,Math.max(a.areaRatio,b.areaRatio)));
  return Math.round((color*.58+aspect*.27+area*.15)*100);
}
function findDuplicates(items){
  const pairs=[];
  for(let i=0;i<items.length;i++) for(let j=i+1;j<items.length;j++){
    if(items[i].sourceIndex===items[j].sourceIndex) continue;
    const score=similarity(items[i],items[j]);
    if(score>=72) pairs.push({a:items[i].id,b:items[j].id,score});
  }
  return pairs.sort((x,y)=>y.score-x.score).slice(0,8);
}

window.AIWardrobeSegmentation = {
  modelId: MODEL_ID,
  version:'4.1',
  async segmentFiles(files, onProgress){
    const all=[];
    for(let i=0;i<files.length;i++){
      onProgress?.({type:'file',index:i,total:files.length,name:files[i].name});
      all.push(...await segmentOne(files[i],i,onProgress));
    }
    return {items:all,duplicates:findDuplicates(all)};
  }
};
window.dispatchEvent(new CustomEvent('aiwardrobe-segmenter-ready'));
