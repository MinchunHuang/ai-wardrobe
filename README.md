# AI Wardrobe V2 — Vercel Static Deploy

這是零建置（zero-build）的靜態 PWA 原型，可直接部署到 Vercel。

## 本版變更
- AI Import 支援一次選擇多張圖片（`multiple`）。
- 可同時預覽多張照片，超過 5 張以 +N 顯示。
- 桌機版只保留左側主選單；右側內容底部不再顯示重複橫向主選單。
- 手機版因沒有左側選單，仍保留底部主選單。
- 補齊 `manifest.webmanifest`、`sw.js`、`vercel.json`，可直接部署。

## Vercel Dashboard 部署
1. 解壓縮 ZIP。
2. 將資料夾放到 GitHub repository（推薦）。
3. Vercel Dashboard → Add New → Project。
4. Import 該 GitHub repository。
5. Framework Preset 選 Other。
6. Build Command 留空；Output Directory 留空；Install Command 留空。
7. Deploy。

## Vercel CLI（最快）
```bash
npm i -g vercel
cd wardrobe-ai-vercel-v2
vercel
```
第一次依提示建立新 Project；確認預覽正常後：
```bash
vercel --prod
```
