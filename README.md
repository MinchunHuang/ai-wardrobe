# AI Wardrobe V5.1 — Local Routed Multi-model

V5.0 的單顆 `Yolov26s-DeepFashion2` benchmark 經實機照片測試後不通過：同一件襯衫 / T-shirt 會被拆成多個狹長 instance，人物穿搭也會產生大量重複碎片。因此 V5.1 不再硬用一顆模型處理所有場景。

## V5.1 架構

### 人物穿搭
- `Xenova/segformer_b2_clothes`（q8）
- 用途：穿在人身上的上衣、下身、鞋、包、腰帶等 clothes parsing
- 優點：對人物穿搭整體區域比 V5.0 的 instance model 穩
- 已知限制：外套 + 內搭等多層上身仍可能合成 `Upper-clothes`

### 單件 / 平拍 / 手持衣服
- 分類定位：`louisJLN/yolo8-fashionpedia` 的 YOLOv8n ONNX（約 12.3 MB）
- 去背：`edgetools/u2netp`（約 4.6 MB）
- 此模式只取「主要單品」，不再把同一件衣服輸出成 5–7 張碎片
- YOLO 負責「這是襯衫 / 褲子 / 外套…」與主要區域；U2Netp 負責該區域去背

### 一般衣物
- 先用 clothes parser 的人體線索做 Local Router
- 有人體 → 人物穿搭 pipeline
- 無人體 → 單件 / 平拍 pipeline
- 此模式方便優先；若自動判錯，直接指定模式重跑

### 配件近拍
- 近拍限定
- 先支援 FashionPedia 有覆蓋的眼鏡、帽子、手錶、腰帶、鞋、包、圍巾等
- 不要求從全身照硬抓戒指 / 耳環等極小物件

## 成本與隱私
- 全部在瀏覽器本機推論
- 不呼叫 Replicate
- 不需要 `REPLICATE_API_TOKEN`
- 不需要 `/api/analyze`
- 模型第一次下載後會快取
- 照片不需要送到 Wardrobe AI 推論 Server

## Mobile UI 修正
手機底部功能列不再 `position: fixed` 浮在內容上方。V5.1 改成一般頁面流的最底部導覽列，所以不會遮住候選卡片。

## 部署
直接覆蓋 GitHub repo 根目錄：
- `index.html`
- `local-ai.js`
- `manifest.webmanifest`
- `sw.js`
- `vercel.json`
- `README.md`

舊的 `server-ai.js` 與 `api/` 資料夾仍然不需要。

## Regression Gate
請固定用同一批照片：
1. 手持襯衫 / 上衣 → 指定「單件 / 平拍」：每張應只出 1 個主要候選，不可再碎成多張。
2. 米色褲子 → 指定「單件 / 平拍」：應以 pants / trousers 為主要分類。
3. T-shirt + shorts 真人穿搭 → 指定「人物穿搭」：應接近 1 件上衣 + 1 件下身，而不是 5–10 個碎片。
4. 手機長頁面 → 底部 nav 不得遮住候選卡片。

> Prototype 注意：正式商業 App 上線前，仍要逐一完成模型 license / attribution audit。
