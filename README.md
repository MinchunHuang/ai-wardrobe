# AI Wardrobe V4.4 — Routed Server AI

## 這版重點
V4.4 把「一般衣物 / 人物穿搭 / 單件平拍 / 配件近拍」從 UI 選項改成真正不同的 Server AI 路線，而不只是不同文案。

- **一般衣物（批量推薦）**：一次丟很多不同照片。Server 先用 person prompt 判斷每張是否為人物穿搭，再自動分流到 outfit 或 single pipeline。方便優先，會多一次 routing inference。
- **人物穿搭（效果優先）**：跳過場景判斷，直接使用 worn-outfit 專屬 prompts；跨類別重疊不會被 aggressive dedupe，較適合外套 / 內搭等重疊服飾。
- **單件 / 平拍（效果優先）**：跳過人物判斷，使用 standalone-garment prompts；對同一物件被 pants / dress / shirt 等多類別重複命中時採更積極跨類別去重。
- **配件近拍**：使用 accessory prompts；不要求從全身照硬抓細小配件。

## 推論解析度
- auto：約 1792 px 長邊，偏批量效率
- outfit / single：約 2048 px 長邊
- accessory：最高約 2560 px 長邊

所有重模型都在 Server 執行；瀏覽器只做影像前處理與顯示。

## 必要環境變數
在 Vercel Project -> Settings -> Environment Variables 新增：
`REPLICATE_API_TOKEN`

## Runtime 驗收
建議用固定 Regression Set：
1. 真人多層穿搭：外套 / 內搭 / 下身 / 鞋 / 包。
2. 單件上衣：雜背景但不應拆成多個互相衝突類別。
3. 單件寬褲：不應同時出現 Upper-clothes / Pants / Dress 三張碎片。
4. 同一批照片分別跑 auto 與指定模式，比較候選件數、分類、重複命中與邊界。

## Alpha 限制
目前仍以 hosted SAM 3 wrapper 回傳結構為基礎；若 provider 調整 mask JSON schema，需在 `api/analyze.js` 的 parser 對應更新。Same-item matching 仍是 heuristic，尚未使用 garment embedding。
