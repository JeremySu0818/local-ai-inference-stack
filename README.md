# Local AI Inference Stack

針對搭載 **NVIDIA RTX PRO 4500** 的單機推論環境設計。透過同一個 OpenAI 相容 API，可使用本地模型進行對話、文字補全、Responses、圖片生成與編輯、文字轉向量。

## 快速開始

安裝需要 NVIDIA 驅動、CUDA Toolkit（含 `nvcc`）、C/C++ 編譯工具和 `curl`；`uv`、CMake 和 Ninja 會由 `setup.sh` 自動安裝。首次安裝會下載依賴和已設定的模型，請預留磁碟空間（Edge0 145B 的權重約 107 GB，Inkling 2 Mini 約 18 GB）。

在專案目錄執行安裝，完成後啟動服務。

```bash
./setup.sh
./start.sh
```

服務預設監聽 `0.0.0.0:9000`。在本機連線時，將用戶端的 API Base URL 設為 `http://127.0.0.1:9000/v1`。從區網內其他裝置連線時，將 `127.0.0.1` 換成主機 IP。

執行以下指令確認服務已啟動，並列出可用的模型 ID。

```bash
curl http://127.0.0.1:9000/health
curl http://127.0.0.1:9000/v1/models
```

## 預設模型

下表列出 [config/models.yaml](config/models.yaml) 隨附的 9 個模型。API 請求的 `model` 欄位使用表中的 ID，自訂設定啟動後則以 `/v1/models` 回傳的清單為準。

| 用途 | 模型 ID |
| --- | --- |
| 對話、文字補全、Responses | `Qwen3.8-27B`、`gemma-4-E2B-it`、`gemma-4-E4B-it`、`NVIDIA-Nemotron-3.5-Lightning-30B-A3B`、`nsfwvision-qwen3-vl-8b-v3`、`Inkling-2-Mini` |
| 對話、Responses | `Edge0-145B` |
| 圖片生成與編輯 | `Qwen-Image-2.1` |
| 文字轉向量 | `embeddinggemma-2` |

## 文件

| 主題 | 內容 |
| --- | --- |
| [使用 API](docs/api.md) | 測試對話、文字轉向量與圖片輸入 |
| [服務維運](docs/operations.md) | 模型與 Router 設定、排程、紀錄檔及依賴更新 |
| [推論後端](docs/backends.md) | llama.cpp、Edge0、Qwen Image 與自訂 HTTP 服務的執行方式和設定 |
| [systemd 與 tmux](docs/systemd.md) | 開機啟動與查看服務輸出 |
