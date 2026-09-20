# AI Wardrobe V4.3 — Server AI Alpha

## 架構改動
- 主辨識不再在 iPhone Safari 裡跑 ONNX。
- 手機只建立 <=235KB 的推論副本，逐張 POST 到 `/api/analyze`。
- Vercel Serverless Function 以 `REPLICATE_API_TOKEN` 呼叫 Replicate 上的 SAM 3。
- SAM 3 以文字概念做 open-vocabulary instance segmentation；人物穿搭與單件平拍共用主模型。
- 細小配件不強迫從全身照辨識，改用「配件近拍」模式。

## 必要環境變數
在 Vercel Project -> Settings -> Environment Variables 新增：
`REPLICATE_API_TOKEN`

取得方式：Replicate 帳號的 API tokens 頁面。

## 目前 Alpha 注意
Replicate SAM3 wrapper 回傳 JSON 的 mask 格式可能隨模型版本變動。`api/analyze.js` 已對 boxes/scores/predictions/mask URL 做容錯解析；如果某次輸出沒有直接可用的 cutout URL，前端會暫時以模型 bbox 從原始照片裁切顯示，並保留 debug shape，方便下一版補齊 mask parser。
