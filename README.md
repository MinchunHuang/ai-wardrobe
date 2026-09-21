# AI Wardrobe V5.4.1 — Cache-Safe Hotfix

這版針對 iPhone「人物穿搭第一次辨識會重新整理、第二次正常」做根因修正。

## 根因
舊版 Service Worker 每次部署升版時會刪除除了 app shell 以外的 Cache Storage；這會連 Transformers.js 預設的 `transformers-cache` 一起刪除。於是每次新部署後，人物 SegFormer 都被迫重新下載並同時建立 WASM session，造成 iOS Safari 冷啟動記憶體尖峰。

## V5.4.1 修正
- Service Worker 只刪舊的 `ai-wardrobe-v*` app-shell cache，不再刪 `transformers-cache`。
- iPhone 第一次使用人物穿搭時，先把 SegFormer q8 模型與設定檔逐一存進 `transformers-cache`，再建立 pipeline，避免下載與 session 建立重疊。
- 保留 V5.4 的 WASM、安全工作副本、逐張釋放與準確度流程。
- UI 會顯示「首次下載人物模型 x/3」與「人物模型檔已存入瀏覽器快取」。

## 驗收
1. 新部署後第一次就跑 1 張人物穿搭，頁面不得重整。
2. 第二次再跑人物穿搭，應直接命中快取，不再重新下載模型。
3. 單件／平拍三張仍可穩定完成。
