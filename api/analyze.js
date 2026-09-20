const MODEL_VERSION='1bf97763d5dfd3a1584adca913a8ef4b43c684fca97e04e39e4c50a3a5e09650';

const OUTFIT=[
  ['outerwear','outerwear jacket coat blazer or cardigan worn by a person'],
  ['shirt','shirt or blouse worn by a person'],
  ['t-shirt','t-shirt worn by a person'],
  ['sweater','sweater knitwear or hoodie worn by a person'],
  ['pants','pants trousers jeans or shorts worn by a person'],
  ['skirt','skirt worn by a person'],
  ['dress','dress worn by a person'],
  ['shoes','shoes worn by a person'],
  ['bag','bag handbag backpack or shoulder bag carried or worn by a person'],
  ['belt','belt worn around the waist']
];
const SINGLE=[
  ['outerwear','a standalone jacket coat blazer cardigan or outerwear garment'],
  ['shirt','a standalone shirt or blouse garment'],
  ['t-shirt','a standalone t-shirt garment'],
  ['sweater','a standalone sweater knitwear or hoodie garment'],
  ['pants','a standalone pair of pants trousers jeans or shorts'],
  ['skirt','a standalone skirt garment'],
  ['dress','a standalone dress garment'],
  ['shoes','a standalone pair of shoes'],
  ['bag','a standalone bag handbag backpack or shoulder bag'],
  ['belt','a standalone belt']
];
const ACCESSORY=[
  ['watch','a wristwatch'],['glasses','eyeglasses'],['sunglasses','sunglasses'],
  ['necklace','a necklace'],['earring','earrings'],['bracelet','a bracelet'],
  ['ring','a ring'],['belt','a belt'],['scarf','a scarf'],['hat','a hat or cap']
];
const PERSON=[['person','a person or human body']];

function asArray(v){return Array.isArray(v)?v:[]}
function normBox(box){
  if(!Array.isArray(box)||box.length<4)return null;
  let a=box.map(Number);if(a.some(x=>!Number.isFinite(x)))return null;
  const max=Math.max(...a.map(Math.abs));if(max>1.5)return null;
  let [x1,y1,x2,y2]=a;
  if(x2>x1&&y2>y1)return [Math.max(0,x1),Math.max(0,y1),Math.min(1,x2)-Math.max(0,x1),Math.min(1,y2)-Math.max(0,y1)];
  let [cx,cy,w,h]=a;return [Math.max(0,cx-w/2),Math.max(0,cy-h/2),Math.min(1,w),Math.min(1,h)];
}
function firstUrl(obj,keys){if(!obj||typeof obj!=='object')return null;for(const k of keys){const v=obj[k];if(typeof v==='string'&&/^https?:\/\//.test(v))return v;if(Array.isArray(v)){const s=v.find(x=>typeof x==='string'&&/^https?:\/\//.test(x));if(s)return s}}return null}
function extract(json,label,category,visualization){
  const out=[];if(!json)return out;
  const boxes=asArray(json.boxes||json.bboxes||json.bounding_boxes),scores=asArray(json.scores||json.confidences),masks=asArray(json.masks||json.mask_urls||json.cutouts);
  if(boxes.length){for(let i=0;i<boxes.length;i++){const bbox=normBox(boxes[i]);if(!bbox)continue;const m=masks[i];out.push({label,category,confidence:Number(scores[i]??json.score??json.confidence??.5),bbox,cutout_url:typeof m==='string'&&/^https?:/.test(m)?m:firstUrl(m,['url','image','mask','cutout']),visualization_url:visualization})}return out}
  const preds=json.predictions||json.instances||json.objects||json.detections||json.results;
  if(Array.isArray(preds)){for(const p of preds){const bbox=normBox(p.bbox||p.box||p.bounding_box||p.boundingBox);if(!bbox)continue;out.push({label:p.label||label,category,confidence:Number(p.score??p.confidence??.5),bbox,cutout_url:firstUrl(p,['cutout_url','mask_url','image_url','url'])||firstUrl(p.mask,['url','image','cutout']),visualization_url:visualization})}return out}
  const bbox=normBox(json.bbox||json.box||json.bounding_box);if(bbox)out.push({label,category,confidence:Number(json.score??json.confidence??.5),bbox,cutout_url:firstUrl(json,['cutout_url','mask_url']),visualization_url:visualization});return out;
}
async function runReplicate(image,promptPairs,threshold=.36){
  const prompts=promptPairs.map(([,text])=>JSON.stringify({text}));
  const r=await fetch('https://api.replicate.com/v1/predictions',{method:'POST',headers:{Authorization:`Bearer ${process.env.REPLICATE_API_TOKEN}`,'Content-Type':'application/json','Prefer':'wait=60'},body:JSON.stringify({version:MODEL_VERSION,input:{image,prompts,confidence_threshold:threshold,visualize:true,offset_masks:true,split_output:true,concat_input:false,backbone_output:false}})});
  const pred=await r.json();if(!r.ok||pred.error)throw new Error(pred.error||`Replicate ${r.status}`);if(pred.status!=='succeeded'&&!pred.output)throw new Error(`Replicate status: ${pred.status}`);
  const urls=pred.output?.results||[],vis=pred.output?.visualizations||[];
  const jsons=await Promise.all(urls.map(async u=>{try{const rr=await fetch(u);return rr.ok?await rr.json():null}catch{return null}}));
  let items=[];for(let i=0;i<promptPairs.length;i++){const [category,label]=promptPairs[i];items.push(...extract(jsons[i],label,category,vis[i]||null))}
  return{items,predictionId:pred.id,rawShapes:jsons.map(j=>j?Object.keys(j):null)};
}
function iou(a,b){const ax2=a[0]+a[2],ay2=a[1]+a[3],bx2=b[0]+b[2],by2=b[1]+b[3],ix=Math.max(0,Math.min(ax2,bx2)-Math.max(a[0],b[0])),iy=Math.max(0,Math.min(ay2,by2)-Math.max(a[1],b[1])),inter=ix*iy,union=a[2]*a[3]+b[2]*b[3]-inter;return union?inter/union:0}
function dedupe(items,mode){
  const sorted=items.filter(x=>x.bbox&&x.confidence>=.28).sort((a,b)=>b.confidence-a.confidence),keep=[];
  for(const x of sorted){
    const dup=keep.find(k=>{
      const overlap=iou(k.bbox,x.bbox);
      if(k.category===x.category)return overlap>.72;
      if(mode==='single'||mode==='accessory')return overlap>.58;
      return false;
    });
    if(!dup)keep.push(x);
  }
  return keep.slice(0,18);
}
function resolveAuto(personItems){
  const candidates=personItems.filter(x=>x.bbox&&x.confidence>=.34);
  const best=candidates.sort((a,b)=>b.confidence-a.confidence)[0];
  if(!best)return {mode:'single',personConfidence:0,personArea:0};
  const area=best.bbox[2]*best.bbox[3];
  return {mode:(best.confidence>=.42&&area>=.08)?'outfit':'single',personConfidence:best.confidence,personArea:area};
}
export default async function handler(req,res){
  if(req.method==='GET')return res.status(200).json({ok:true,configured:!!process.env.REPLICATE_API_TOKEN,model:'vufinder/sam3',version:'4.4-routed-server',modes:['auto','outfit','single','accessory']});
  if(req.method!=='POST')return res.status(405).json({error:'Method not allowed'});
  if(!process.env.REPLICATE_API_TOKEN)return res.status(503).json({error:'REPLICATE_API_TOKEN 尚未設定'});
  try{
    const {image,mode='auto'}=req.body||{};
    if(typeof image!=='string'||!image.startsWith('data:image/'))return res.status(400).json({error:'Invalid image'});
    if(image.length>1400000)return res.status(413).json({error:'推論副本過大，請重新選擇或稍後重試'});
    let resolvedMode=mode,routerDebug=null;
    if(mode==='auto'){
      const routed=await runReplicate(image,PERSON,.34);
      const decision=resolveAuto(routed.items);
      resolvedMode=decision.mode;
      routerDebug={predictionId:routed.predictionId,personConfidence:decision.personConfidence,personArea:decision.personArea};
    }
    const pairs=resolvedMode==='accessory'?ACCESSORY:resolvedMode==='outfit'?OUTFIT:SINGLE;
    const result=await runReplicate(image,pairs,resolvedMode==='accessory'?.31:.35);
    const items=dedupe(result.items,resolvedMode);
    return res.status(200).json({items,debug:{predictionId:result.predictionId,router:routerDebug,requestedMode:mode,resolvedMode,rawShapes:result.rawShapes,count:items.length}});
  }catch(e){console.error(e);return res.status(500).json({error:e?.message||'Server AI failed'})}
}
