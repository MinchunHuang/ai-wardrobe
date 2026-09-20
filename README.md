# AI Wardrobe V5.2 — Local AI Stability + Mask Fix

V5.2 is a corrective benchmark release based on real iPhone tests.

## Confirmed V5.1 problems fixed

1. **Outfit parser mask bug**
   - Transformers.js semantic masks can render with an opaque canvas alpha.
   - V5.1 read that alpha directly, which could turn the whole source photo into the cutout.
   - V5.2 builds alpha from mask pixel values instead.

2. **U2Netp cutout was too aggressive**
   - V5.1 remapped saliency through a hard gate, which could leave only a logo/graphic on dark garments.
   - V5.2 follows soft min-max alpha and adds a completeness gate.
   - If the mask is implausibly tiny/huge, the UI shows the full detected crop instead of a misleading broken cutout.

3. **Classification honesty**
   - FashionPedia detection is still experimental for hand-held / messy-background garments.
   - V5.2 no longer presents a medium-confidence label as certain.
   - Low confidence or small top-1/top-2 margin becomes `單品（請確認）` with candidate labels.

4. **Mobile memory guard**
   - Model sessions are released after each batch on mobile (and in the current UI on all runs), while browser cache is retained.
   - This is intended to reduce iOS/Safari tab reloads caused by memory pressure.
   - The page also records whether a reload happened during an inference run, so the next test can distinguish a real browser reload from a UI state change.

5. **Mobile navigation**
   - Bottom navigation is now fixed to the bottom edge with safe-area padding.
   - Content has matching bottom padding, so the navigation remains visible without covering the last cards.

## Current model roles

- Person-worn outfits: `Xenova/segformer_b2_clothes` (q8)
- Single / flat garment locator: FashionPedia YOLOv8n ONNX
- Single / flat cutout: U2Netp ONNX

This is still a benchmark stage. The purpose of V5.2 is to separate implementation bugs from actual model limitations before replacing another model.
