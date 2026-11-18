# Local AI Inference Stack

針對搭載 **NVIDIA RTX PRO 4500** 的單機推論環境設計。透過同一個 OpenAI 相容 API，可使用本地模型進行對話、文字補全、Responses、圖片生成與編輯。

## 排程管理

請求依到達順序排隊，並以模型 ID 指定要使用的模型。排程器按需載入模型，讓佇列中連續使用同一模型的請求共用已載入的模型。輪到不同模型時，排程器會等目前的推論完成，再卸載舊模型並載入下一個，讓多個模型輪流使用 GPU 顯存。

等待佇列的容量可調整。佇列已滿時，服務會拒絕新請求。

## 快速開始

安裝需要 NVIDIA 驅動、CUDA Toolkit（含 `nvcc`）、C/C++ 編譯工具和 `curl`；`uv`、CMake 和 Ninja 會由 `setup.sh` 自動安裝。首次安裝會下載依賴和已設定的模型，請預留磁碟空間（Edge0 145B 的權重約 107 GB）。

在專案目錄執行安裝，完成後啟動服務。

```bash
./setup.sh
./start.sh
```

`setup.sh` 依照 `requirements/locked.txt` 安裝固定版本，鎖檔適用於 Linux x86-64、Python 3.12 和 CUDA 13.2。需要新增或調整 Python 依賴時，先修改 `requirements/base.txt` 或 `requirements/image.txt`，再於專案目錄更新鎖檔：

```bash
uv pip compile --torch-backend=cu132 --python-version 3.12 --python-platform x86_64-manylinux_2_28 --emit-index-url --no-annotate --no-header requirements/base.txt -o requirements/locked.txt
```

修改 requirements 後必須重新產生鎖檔，再執行 `./setup.sh` 安裝新的依賴版本。

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

## 推論後端

文字與多模態模型使用 llama.cpp 後端載入 GGUF 權重，並透過 OpenAI 相容 API 提供推論。可用模型和各模型支援的 API 請見 [config/models.yaml](config/models.yaml)。

### Edge0 後端

Edge0 145B 使用 Edge0 自己的推論引擎，不依賴 llama.cpp 或 vLLM：`edge0-infer` 是入口程式，負責啟動 `edge0-strata` 引擎、等待權重載入完成，再在本機埠提供 OpenAI 相容 API。排程器把它當成一般的伺服端後端管理，因此換用其他引擎時不需改動排程器。

- 引擎依本機 GPU 架構編譯（Blackwell SM 12.0），權重為 NVFP4 格式。
- 需要約 107 GB 磁碟空間存放權重（建議 NVMe），載入後佔用約 30 GB 顯存；載入需數分鐘，因此 `startup_timeout_seconds` 設得比預設值大。
- 僅支援 `chat/completions` 與 `responses`，不支援文字補全端點。
- 權重不由專案附帶，`setup.sh` 會從 Hugging Face 下載並校驗檔案。模型依 Edge0 Community Model License 授權（衍生自 Qwen 基礎模型），使用前請先閱讀權重目錄中的 `LICENSE` 與 `NOTICE`。

調整引擎行為的設定寫在模型條目中：

| 設定 | 用途 |
| --- | --- |
| `device` | 使用的 GPU，預設 `cuda:0` |
| `context_size` | 上下文長度上限 |
| `model_options` | 引擎選項：`du: q8` 或 `q4` 選專家投影權重的格式，`ple-io: direct` 從 SSD 直讀嵌入表、`ram` 則整表鎖進記憶體 |
| `engine_args` | 直接傳給引擎的參數，例如 `--vram-reserve-mib` 預留顯存 |

引擎會自行設定的參數（如 `--max-context`）不允許在 `engine_args` 中重複指定，設定載入時即會拒絕。

本機已下載過權重時，可以用 `model_path` 指向現有的權重目錄，讓 `setup.sh` 與排程器直接使用它而不另存一份；此時仍會檢查檔案是否完整。

Qwen Image 2.1 後端以 Diffusers 的 pipeline 和模型元件為基礎，提供圖片生成與編輯，並針對原生 checkpoint、Blackwell SM 12.0 和 PyTorch 2.14.0 啟用自訂推論路徑：

- 去噪時快取前綴 KV，後續步驟只處理目標影像 token。
- 使用自訂 Triton 核心執行投影、融合 SwiGLU、RMSNorm 與旋轉位置編碼；未符合最佳化條件的運算使用原有路徑。
- 載入時並行準備權重和初始化元件。預設依請求調整文字編碼器的 GPU 常駐量，顯存不足時回退至模型 CPU offload。

### Qwen Image 效能

在 RTX PRO 4500 32 GB、BF16 權重下，使用相同輸入、seed、取樣器和 40 步推論，從送出本機 HTTP 請求計時至解開回應的圖片：

| 請求                             | 修改前 | 修改後 |
| -------------------------------- | -----: | -----: |
| 1024×1024 文生圖，冷啟動         |     1× |  2.03× |
| 1024×1024 文生圖，模型常駐       |     1× |  2.03× |
| 自動尺寸文生圖，冷啟動           |     1× |  2.18× |
| 1024×1024 單參考圖編輯，模型常駐 |     1× |  1.94× |

前兩列各取三次中位數；自動尺寸的最佳化後取三次中位數、最佳化前測一次，參考圖編輯兩版各測一次。冷啟動從空閒 GPU 和新圖片 worker 開始，並對兩版權重要求釋放檔案快取。

載入階段透過 fastsafetensors 並行讀取權重、傳輸至 GPU 和初始化元件；自動尺寸請求可在背景載入時以 CPU 預選長寬比。去噪階段重用前綴 KV、位置編碼和注意力遮罩，並以自訂核心融合部分投影與正規化運算。較大的請求會在提示詞編碼後釋放部分或全部文字編碼器，以保留工作空間。

基準文生圖的兩版解碼像素雜湊一致；其他提示詞、尺寸、批次、負面提示詞、關閉 KV cache 和參考圖編輯案例也曾逐像素比對。這些結果反映上述測試環境與輸入，不代表所有設定的效能或輸出。
