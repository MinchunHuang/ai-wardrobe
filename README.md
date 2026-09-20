# AI Wardrobe V5.0 — Local Core Benchmark

這一版正式暫停 Replicate / SAM 3，回到 local-first。

## 核心改動

- 主模型：`HasanKabir23/Yolov26s-DeepFashion2`
- 任務：13 類服飾 instance segmentation
- ONNX：約 11 MB
- 推論：瀏覽器 ONNX Runtime Web
- Chrome / Edge：優先 WebGPU，失敗時 fallback WASM
- Safari / iOS：以 WASM fallback 為主
- 不需要 `REPLICATE_API_TOKEN`
- 不需要 `/api/analyze`
- 照片不送到 Replicate 或 Wardrobe AI inference server
- 第一次會下載模型並存到 Browser Cache；後續直接讀快取

## V5.0 的目的

先回答一個關鍵問題：一顆約 11 MB、專門針對 DeepFashion2 訓練的 instance-segmentation 模型，能不能同時處理：

1. 人物穿搭
2. 單件 / 平拍 / 手持衣物
3. 混合批次

如果人物多層穿搭仍不夠，再把 SegFormer 降級為 outfit auxiliary model，而不是一開始就同時下載兩顆模型。

## 三種目前可測模式

- 一般衣物：本機批量，使用核心模型逐張處理
- 人物穿搭：class-aware NMS，允許不同衣物類別重疊
- 單件 / 平拍：cross-class NMS，避免同一件被褲 / 裙 / 洋裝重複命中

`配件近拍` 在 V5.0 暫不啟用。核心衣物 Gate 通過後再評估約 4.6 MB 的 U-2-Netp 或其他小型配件/去背方案。

## 部署

直接把本資料夾內容放到 GitHub repo 根目錄，Vercel 自動部署即可。

要刪除舊版：

- `server-ai.js`
- `api/analyze.js` / `api/` 資料夾

Vercel 上先前的 `REPLICATE_API_TOKEN` 可以保留但不會被 V5.0 使用；也可以自行刪除。

## 第一次測試

建議固定用同一組 Regression Test：

- 多層真人穿搭
- 白 T / 襯衫單件
- 米色寬褲近拍

畫面會顯示執行 backend、模型輸入尺寸與單張推論時間，方便比較 WebGPU / WASM。
