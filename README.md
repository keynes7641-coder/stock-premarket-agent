# 📈 台美股 AI 盤前情報與投資建議 Agent

全自動、零伺服器成本的台美股雙市場盤前投資情報系統。透過 GitHub Actions 在台股與美股開盤前自動執行，過濾休市日、彙整隔夜總經與籌碼/新聞數據，並透過 **Gemini 2.5 Flash / Pro（含 Google Search Grounding）** 生成結構化盤前報告寫入 Firebase Firestore，再由單頁式前端儀表板瀏覽報告與管理持股。

## 目錄結構

```text
.
├── .github/
│   └── workflows/
│       └── premarket.yml      # GitHub Actions 台美股雙時段自動排程
├── index.html                 # 單頁式前端儀表板 (Tailwind CSS + Firebase Web SDK)
├── main.py                    # Python 後端主程式 (休市判斷、數據抓取、Gemini 分析、寫入 DB)
├── requirements.txt           # Python 套件相依清單
├── .env.example               # 本地環境變數範本
├── .gitignore
└── README.md                  # 部署與操作指南
```

## 1. Firebase 設定

1. 前往 [Firebase Console](https://console.firebase.google.com/) 建立新專案。
2. **啟用 Authentication**：
   - 開啟 **Email/Password** 登入方式。
   - 手動新增一組管理員帳號與密碼（供登入 `index.html` 使用）。
3. **啟用 Firestore Database**：
   - 建立資料庫並套用以下 Security Rules，僅允許已登入的管理員讀寫：
     ```javascript
     rules_version = '2';
     service cloud.firestore {
       match /databases/{database}/documents {
         match /{document=**} {
           allow read, write: if request.auth != null;
         }
       }
     }
     ```
4. **取得後端服務帳戶金鑰 (Service Account)**：
   - 至 `Project Settings` -> `Service accounts` -> 點選 **Generate new private key** 下載 JSON 檔。
   - 將該 JSON 檔案完整內容設為環境變數 `FIREBASE_SERVICE_ACCOUNT`（本地測試亦可填寫該 JSON 檔路徑）。
5. **填入前端 Web Config**：
   - 至 `Project Settings` -> `Your apps` 新增 Web App，取得 `apiKey`、`authDomain`、`projectId`。
   - 更新 `index.html` 中的 `firebaseConfig` 物件。

## 2. Firestore 資料結構

### `portfolio`（持股清單）
- Document ID：`{market}_{ticker}`（例如 `TW_2308`、`US_RKLB`）
- 欄位：`market` (`"TW"` / `"US"`)、`ticker`、`yf_symbol`（上市加 `.TW`、上櫃加 `.TWO`、美股直接填代號）、`name`、`avg_cost` (number)、`focus`

### `reports`（每日盤前分析報告）
- Document ID：`{YYYY-MM-DD}_{market}`（例如 `2026-11-19_TW`）
- 欄位：`date`、`market`、`macro_summary`、`updated_at`、`stocks` (array of objects)

## 3. 本地開發與測試

```bash
# 1. 建立並啟用虛擬環境
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 2. 安裝相依套件
pip install -r requirements.txt

# 3. 複製並編輯環境變數
cp .env.example .env

# 4. 忽略休市日強制執行台股或美股分析測試
python main.py --market TW --force
python main.py --market US --force
```

> 提示：執行前請先開啟 `index.html` 登入並於「持股清單管理」新增至少一檔對應市場的持股。

## 4. GitHub Actions 自動排程部署

將專案推送至 GitHub 後，前往倉庫的 `Settings` -> `Secrets and variables` -> `Actions` 新增以下 Repository Secrets：

| Secret 名稱 | 必填 | 說明 |
| --- | :---: | --- |
| `GEMINI_API_KEY` | 是 | Google AI Studio API Key |
| `FIREBASE_SERVICE_ACCOUNT` | 是 | Firebase 服務帳戶完整 JSON 字串 |
| `FINNHUB_API_KEY` | 否 | Finnhub 免費版 API Key（美股新聞用，未填時改由 Google Search Grounding 檢索） |
| `TELEGRAM_BOT_TOKEN` | 否 | Telegram Bot Token（啟用盤前摘要推播） |
| `TELEGRAM_CHAT_ID` | 否 | Telegram 接收推播的 Chat ID |
| `DASHBOARD_URL` | 否 | 前端儀表板網址（推播通知內附點擊連結） |

### 排程時段（台灣時間 UTC+8）
- **台股**：週一至週五 `08:15`（UTC `00:15`）
- **美股**：週一至週五 `20:45`（UTC `12:45`）
- 亦可隨時於 GitHub Actions 頁面使用 **Run workflow (`workflow_dispatch`)** 手動指定市場與 `--force` 測試執行。

## 5. 前端儀表板部署 (GitHub Pages)

1. 確認已於 `index.html` 填入真實的 `firebaseConfig`。
2. 至 GitHub 倉庫 `Settings` -> `Pages`，將 Source 設為 `Deploy from a branch`，選擇 `main` 分支與 `/ (root)` 目錄。
3. 部署完成後即可透過 GitHub Pages 網址登入瀏覽每日盤前報告與維護持股清單。
