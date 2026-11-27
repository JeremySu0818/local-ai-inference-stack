# 推論後端

Router 依模型條目的 `backend` 選擇執行方式，依 `endpoints` 決定該模型開放的 API。模型 ID 與端點見 [config/models.yaml](../config/models.yaml)，後端名稱見 [後端註冊表](../src/local_ai/backends/__init__.py)。

| `backend` | 執行方式 | 隨附設定中的用途 |
| --- | --- | --- |
| [`llama_cpp`](#llama_cpp) | 啟動 `llama-server` 並代理 HTTP 請求 | 對話、文字補全、Responses、文字轉向量，部分模型接受圖片輸入 |
| [`edge0`](#edge0) | 啟動 `edge0-infer` 並代理 HTTP 請求 | Edge0 145B 的對話、Responses 和圖片輸入 |
| [`qwen_image`](#qwen_image) | 在獨立 worker 中執行 Diffusers pipeline | Qwen Image 2.1 的圖片生成與編輯 |
| [`http`](#http) | 依設定啟動自訂 HTTP 服務並代理請求 | 自訂模型條目可使用 |

## llama_cpp

後端以固定版本的 llama.cpp 載入 GGUF 權重。Qwen3.8、Gemma 4、Nemotron、Inkling 2 Mini 和 EmbeddingGemma 2 等模型共用此後端。支援視覺的模型可用 `mmproj` 指定投影檔，並以 `mmproj_repo` 指定其他來源。

`llama-server` 固定使用 `--n-gpu-layers all` 和 `--kv-unified`。`parallel` 個併發槽位共用 `context_size` 的容量。若要讓單一會話獨占上下文，可將 `parallel` 設為 1。

可用 `ubatch_size` 設定物理批次大小。縮小批次可降低長上下文所需的運算緩衝區，但可能讓提示詞預填變慢。隨附的 `Inkling-2-Mini` 設定在 262144 token 的上下文使用 `ubatch_size: 128`，可作為調整顯存用量的參考。

### 文字轉向量

`embeddinggemma-2` 使用 `ggml-org/embeddinggemma-2-GGUF` 的 Q8_0（310 MB），回傳 768 維、已 L2 正規化的向量，上下文 8192 token。

向量模型和對話模型共用同一塊顯存，由排程器按需載入：佇列輪到它時才載入，輪到別的模型時卸載。所以「問一句、搜一次、再生成一次」這種用法會反覆載入模型，搜尋密集的工作最好一次傳送整批文字。

EmbeddingGemma 2 使用雙向注意力，輸入超過批次大小會直接失敗。統一 KV 快取讓單一槽位能使用完整的設定上下文，因此 `--batch-size` 和 `--ubatch-size` 都設為 `context_size`。

模型訓練時帶有任務前綴，加上來品質更好，省略也能用：

| 用途 | 前綴 |
| --- | --- |
| 搜尋的查詢 | `task: search result \| query: {query}` |
| 搜尋的文件 | `title: {title} \| text: {content}` |
| 問答的查詢 | `task: question answering \| query: {question}` |
| 分類、聚類、相似度 | `task: classification \| query: {content}` |

請求帶有 `input` 時，可用 `dimensions` 指定輸出維度。`embeddinggemma-2` 接受 `768`、`512`、`256`、`128`。要求較短的向量時，router 會取向量的前 `n` 個值並重新正規化。使用 llama.cpp 原生的 `content` 欄位時，請省略 `dimensions`。

其他 llama.cpp 向量模型可用 `embedding_dimensions` 宣告原生維度與支援的縮減維度。第一個值是原生維度，例如固定 384 維的模型填 `[384]`。只有模型支援向量截斷時才列入較小的維度。省略設定仍可取得原生向量，帶有 `dimensions` 的請求會收到 400。

只有一處與官方文件不同：`input` 只接受文字或文字陣列，不接受圖片和音訊的內容項目（官方端點也不支援）。

## edge0

Router 啟動 `edge0-infer`，由它啟動 `edge0-strata` 推論引擎並提供本機 OpenAI 相容 API。隨附設定使用 `Edge0-145B`。

- 引擎依本機 GPU 架構編譯（Blackwell SM 12.0），權重為 NVFP4 格式。
- 需要約 107 GB 磁碟空間存放權重（建議 NVMe），載入後佔用約 30 GB 顯存；載入需數分鐘，因此 `startup_timeout_seconds` 設得比預設值大。
- 僅支援 `chat/completions` 與 `responses`，不支援文字補全端點。
- 支援圖片輸入：`setup.sh` 會另外編譯 `strata-vision`（llama.cpp 的 mtmd 讀取權重目錄的 `mmproj-BF16.gguf`），並把它寫入引擎啟動設定。圖片以 CPU 編碼，約 1 GB 記憶體、每張數秒，不佔顯存。

### 圖片輸入

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Edge0-145B","max_tokens":128,"messages":[{"role":"user","content":[{"type":"text","text":"圖里是什麼"},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}'
```

`responses` 端點接受 `input_image` 內容項目。圖片經 base64 後整組請求上限約 16 MB，超過時引擎會拒絕。

### 權重與引擎設定

權重不由專案附帶，`setup.sh` 會從 Hugging Face 下載並校驗檔案。模型依 Edge0 Community Model License 授權（衍生自 Qwen 基礎模型），使用前請先閱讀權重目錄中的 `LICENSE` 與 `NOTICE`。

調整引擎行為的設定寫在模型條目中：

| 設定 | 用途 |
| --- | --- |
| `device` | 使用的 GPU，預設 `cuda:0` |
| `context_size` | 上下文長度上限 |
| `model_options` | 引擎選項：`du: q8` 或 `q4` 選專家投影權重的格式，`ple-io: direct` 從 SSD 直讀嵌入表、`ram` 則整表鎖進記憶體 |
| `engine_args` | 直接傳給引擎的參數，例如 `--vram-reserve-mib` 預留顯存 |

引擎會自行設定的參數（如 `--max-context`、`--vision`）不允許在 `engine_args` 中重複指定，設定載入時即會拒絕。

本機已下載過權重時，可以用 `model_path` 指向現有的權重目錄，讓 `setup.sh` 與排程器直接使用它而不另存一份；此時仍會檢查檔案是否完整。

## qwen_image

Router 在獨立 worker 中載入 Diffusers pipeline，處理 `images/generations` 與 `images/edits`。隨附設定使用 `Qwen-Image-2.1`，針對原生 checkpoint、Blackwell SM 12.0 和 PyTorch 2.14.0 啟用自訂推論路徑。

- 去噪時快取前綴 KV，後續步驟只處理目標影像 token。
- 使用自訂 Triton 核心執行投影、融合 SwiGLU、RMSNorm 與旋轉位置編碼；未符合最佳化條件的運算使用原有路徑。
- 載入時並行準備權重和初始化元件。預設依請求調整文字編碼器的 GPU 常駐量，顯存不足時回退至模型 CPU offload。

### 效能測試

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

## http

自訂模型條目可使用 `backend: http`，省略 `backend` 時也會選用此後端。Router 按 `command` 啟動程序，可用 `cwd` 指定工作目錄。`base_url` 必須是帶埠號的本機 HTTP loopback URL，Router 透過 `health_path` 檢查就緒狀態，預設為 `/health`。隨附的模型設定目前沒有使用此後端的條目。
