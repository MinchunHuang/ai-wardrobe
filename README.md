# AI Wardrobe V4

V4 upgrades the import flow from prototype crops to real in-browser clothing semantic segmentation.

## Real AI segmentation
- Browser-side Transformers.js
- Model: Xenova/segformer_b2_clothes (q8 ONNX)
- Detects / segments upper-clothes, pants, skirt, dress, shoes, bag, hat, sunglasses, belt and scarf
- Produces transparent PNG cutouts from actual model masks
- Left/right shoes are merged into a single shoe item
- Multi-photo same-item suggestions use category + color + silhouette heuristics (not yet deep garment embeddings)

## Privacy
The V4 frontend performs segmentation locally in the browser. Model files are downloaded from Hugging Face; selected photos are not sent to an AI Wardrobe backend by this version.

## Notes
The clothing parser is strongest on people wearing clothes. Flat-lay images, wardrobes with many overlapping garments, and non-human product photos can be weaker. That is a known V4 limitation.
