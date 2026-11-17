# Local AI Inference Stack

針對搭載 **NVIDIA RTX PRO 4500** 的單機推論環境設計。透過同一個 OpenAI 相容 API，可使用本地模型進行對話、文字補全、Responses、圖片生成與編輯。

## 排程管理

請求依到達順序排隊，並以模型 ID 指定要使用的模型。排程器按需載入模型，讓佇列中連續使用同一模型的請求共用已載入的模型。輪到不同模型時，排程器會等目前的推論完成，再卸載舊模型並載入下一個，讓多個模型輪流使用 GPU 顯存。

等待佇列的容量可調整。佇列已滿時，服務會拒絕新請求。

## 快速開始

安裝需要 `uv`、NVIDIA 驅動、CUDA Toolkit（含 `nvcc`）、CMake 和 C/C++ 編譯工具。首次安裝會下載依賴和已設定的模型，請預留磁碟空間。

在專案目錄執行安裝，完成後啟動服務。

```bash
./setup.sh
./start.sh
```

服務預設監聽 `0.0.0.0:9000`。在本機連線時，將用戶端的 API Base URL 設為 `http://127.0.0.1:9000/v1`。從區網內其他裝置連線時，將 `127.0.0.1` 換成主機 IP。

**需要開機自動啟動？** 請參閱 [systemd 與 tmux 設定指南](docs/systemd.md)。

## 使用 API

以下指令可查看服務狀態及可用的模型 ID。

```bash
curl http://127.0.0.1:9000/health
curl http://127.0.0.1:9000/v1/models
```

使用已設定的 Gemma 模型測試對話。

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-E2B-it","messages":[{"role":"user","content":"你好"}]}'
```

## 調整設定

- [config/models.yaml](config/models.yaml) 定義可用模型、模型 ID 和各模型支援的 API。新增模型或變更模型檔案後，重新執行 `./setup.sh` 以準備模型。
- [config/router.yaml](config/router.yaml) 設定佇列上限、請求逾時和狀態目錄。

修改設定後請重新啟動服務。
