import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

const MODEL_ID = 'Xenova/segformer_b2_clothes';
let segmenterPromise = null;

const CLASS_MAP = {
  'Upper-clothes': { name: '上衣', cat: '上衣' },
  'Pants': { name: '長褲', cat: '下身' },
  'Skirt': { name: '裙子', cat: '下身' },
  'Dress': { name: '洋裝', cat: '洋裝' },
  'Bag': { name: '包包', cat: '鞋包' },
  'Hat': { name: '帽子', cat: '配件' },
  'Sunglasses': { name: '眼鏡', cat: '配件' },
  'Belt': { name: '腰帶', cat: '配件' },
  'Scarf': { name: '圍巾', cat: '配件' },
  'Left-shoe': { name: '鞋子', cat: '鞋包', group: 'Shoes' },
  'Right-shoe': { name: '鞋子', cat: '鞋包', group: 'Shoes' },
};

async function getSegmenter(onProgress) {
  if (!segmenterPromise) {
    segmenterPromise = pipeline('image-segmentation', MODEL_ID, {
      dtype: 'q8',
      progress_callback: (info) => {
        if (onProgress) onProgress({ type: 'model', info });
      },
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
  return { canvas, url: canvas.toDataURL('image/jpeg', 0.9), width, height };
}

function maskValue(mask, x, y, targetW, targetH) {
  const mw = mask.width, mh = mask.height;
  const mx = Math.min(mw - 1, Math.floor(x * mw / targetW));
  const my = Math.min(mh - 1, Math.floor(y * mh / targetH));
  const channels = mask.channels || Math.max(1, Math.round(mask.data.length / (mw * mh)));
  return mask.data[(my * mw + mx) * channels] || 0;
}

function buildCutout(sourceCanvas, masks) {
  const w = sourceCanvas.width, h = sourceCanvas.height;
  const src = sourceCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h);
  const out = new ImageData(new Uint8ClampedArray(src.data), w, h);
  let minX=w, minY=h, maxX=-1, maxY=-1, count=0;
  let sumR=0,sumG=0,sumB=0;
  for (let y=0; y<h; y++) {
    for (let x=0; x<w; x++) {
      const p = y*w+x;
      let a=0;
      for (const m of masks) a=Math.max(a, maskValue(m,x,y,w,h));
      if (a > 20) {
        out.data[p*4+3]=a;
        minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y); count++;
        sumR+=src.data[p*4]; sumG+=src.data[p*4+1]; sumB+=src.data[p*4+2];
      } else out.data[p*4+3]=0;
    }
  }
  if (!count || count/(w*h) < 0.0015) return null;
  const pad=Math.round(Math.max(w,h)*0.015);
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
    bbox:[minX,minY,cw,ch]
  };
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
    if (!grouped.has(key)) grouped.set(key,{...meta,labels:[],masks:[]});
    const g=grouped.get(key); g.labels.push(result.label); g.masks.push(result.mask);
  }
  const results=[];
  for (const [key,g] of grouped) {
    const cut=buildCutout(prepared.canvas,g.masks);
    if (!cut) continue;
    results.push({
      id:`${sourceIndex}-${key}-${crypto.randomUUID?.() || Math.random().toString(36).slice(2)}`,
      name:g.name,
      cat:g.cat,
      label:g.labels.join('+'),
      sourceIndex,
      sourceName:file.name,
      ...cut,
    });
  }
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
