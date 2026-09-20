# AI Wardrobe V4.1

V4.1 keeps the real browser-side clothing segmentation from V4 and adds a cleanup + human-review layer based on the first real user tests.

## V4.1 changes
- Connected-component cleanup removes detached segmentation speckles.
- Class-specific minimum/maximum area gates reject obvious tiny false positives.
- Pants/Skirt conflicts are post-processed per source image.
- Ambiguous lower-body results are marked for user confirmation instead of being presented as certain.
- Upper-clothes results warn that layered outerwear + innerwear can still be merged by this human-parsing model.
- Service-worker cache bumped to V4.1 and includes `ai-segmentation.js`.

## Still a known model limitation
The SegFormer human-parsing model is semantic, not garment-instance segmentation. It can merge a jacket and inner top into one Upper-clothes region, and cropped wide pants can be confused with skirts. Solving that requires V4.2 dual-model/instance segmentation rather than more UI heuristics.
