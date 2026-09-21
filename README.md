# AI Wardrobe V5.4 — Local Accuracy Pass

V5.4 keeps the iPhone memory-safe runtime proven stable in V5.3.2, and now targets recognition accuracy.

## Person-worn outfits
- Main parser: `Xenova/segformer_b2_clothes` (q8).
- Adds connected-component cleanup on semantic masks.
- Adds spatial plausibility gates for hat / sunglasses / belt / scarf / shoes.
- Example: dark floor objects or flip-flops labelled as `Hat` are rejected unless the mask is actually near the head region.
- Multi-layer outerwear + innerwear can still merge because this is a semantic human-clothes parser.

## Single / flat garments
- FashionPedia YOLOv8n is now treated primarily as a **garment locator**, not the final category authority.
- U2Netp creates the foreground cutout.
- `Xenova/mobileclip_s0` (q8) then performs zero-shot garment-type re-ranking on the isolated garment.
- Current prompts distinguish: button-up shirt, T-shirt, sweater, cardigan, jacket, vest, pants, shorts, skirt, coat, dress, jumpsuit, cape/shawl.
- Detector score is retained only as a weak prior. If the detector and semantic re-ranker disagree and the margin is small, the UI asks the user to confirm instead of presenting a wrong label as certain.

## Runtime / privacy
- All inference remains local in the browser.
- iPhone/iPad stays on WASM safety mode.
- 24MP photos are processed through reduced working copies; originals are not modified.
- Heavy models are released between stages / images on iOS, while downloaded weights stay cached.
- The first V5.4 single/flat run downloads the MobileCLIP quantized model once; later runs reuse browser cache.

## Benchmark goal
Use the same regression photos from V5.3.2. A release passes only if it improves category correctness without reintroducing Safari reloads.
