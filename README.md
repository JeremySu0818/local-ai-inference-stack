# 動態 AI 路由器

單機、單 GPU 排程器：OpenAI 相容 HTTP 入口 `http://127.0.0.1:9000/v1`，根據請求的 `model` 載入內建模型或啟動前景服務。預設 180 分鐘無活動後停止。文字、圖像生成及圖像編輯共用同一個排程器。

## 支援範圍

- 模型登記：編輯 `config.yaml` 後重啟；`GET /v1/models` 列出已登記 ID。ID 原樣轉發，必須與後端接受的 ID 相同。
- JSON：`chat/completions`、`completions`、`embeddings`、`responses`、`images/generations`、`audio/speech`；multipart：`images/edits`、`audio/transcriptions`、`audio/translations`。每個模型須在 `endpoints` 清單啟用。實際參數和能力由後端決定。
- 透明轉發回應、狀態碼、SSE 串流和 multipart 檔案。內建 Qwen-Image-2.1 在 API 程序內直接執行 Diffusers，無需外部圖片服務。
- 網路監聽、佇列上限、請求大小上限、啟動期限及程序監控。服務不驗證 API key；監聽 0.0.0.0 時，任何能連到 9000 埠的裝置皆可使用。請以防火牆限制存取；需要透過不可信網路存取時，請使用具備驗證與 TLS 的反向代理或 VPN。

## 執行條件與安全契約

1. Linux；Python 3.10 以上。只能執行 **一個 worker、一個路由器實例**。同一個 state directory 以檔案鎖阻擋第二個實例。不要使用多 worker 或多台主機共用一張 GPU。
2. HTTP 後端都必須只允許路由器存取；不要讓其他程式直接呼叫後端，否則路由器看不到它們的任務。使用專屬 localhost port。`base_url` 指向後端的 `/v1`，`health_path` 從後端 origin 開始。
3. HTTP 後端啟動腳本必須在前景執行且最後使用 `exec`，不可 daemonize、`nohup`、`setsid`、移交 systemd、容器或其他程序管理器。子程序必須留在相同 process group。後端不可以把長任務送到另一個在 HTTP 回應後仍然執行的外部服務。這些約束是確保卸載的必要條件。
4. 後端必須在該次請求真正結束後才結束 HTTP 回應（路由器拒絕 background=pending 或 async=true 的背景任務）；若上游連線出錯，路由器將該後端標記不可信，等待其他在途請求結束後停止整個程序群，再啟動下一個模型。個別客戶端斷線時則繼續讀取後端直至任務結束，避免提前卸載。串流傳輸途中後端發生錯誤時，已送出的 HTTP 200 無法改為 502，客戶端會收到截斷的串流。
5. 接收 `/v1` 請求和等待期間客戶端要容許冷啟動及排隊所需時間。`request_timeout_seconds` 是 HTTP 讀取閒置逾時，不是佇列等待上限或總執行時間上限。
6. 狀態檔 `unclean` 代表路由器執行中或上次非正常結束；如異常中止，必須確認所有後端子程序均已停止，再手動刪除此檔，避免與遺留程序重疊。不要在服務執行中刪除。系統異常可能使程序群 SIGKILL 後暫時無法退出，路由器會停止排程並保留狀態檔。
7. 信任配置檔、啟動腳本及後端：它們可以執行本機程式。機密用環境變數提供，不要寫入 YAML、腳本或日誌。

## 安裝（Arch Linux）

```bash
mkdir -p ~/ai-router
# 將壓縮檔內容解開至 ~/ai-router
cd ~/ai-router
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
cp config.example.yaml config.yaml
nano config.yaml
chmod 600 config.yaml
```

設定每個模型的 `id`、`command`（以 YAML 字串陣列逐個參數給出，無 shell 字串展開）、`base_url`、`health_path` 及 `endpoints`。`cwd` 可指定絕對工作目錄。預設 TTL `10800` 秒，可逐模型調整。

例如 vLLM 腳本：保留自己的模型參數，只需要確認最後以前景 `exec` 執行，並綁定指定 port：

```bash
#!/usr/bin/env bash
set -euo pipefail
exec /home/jeremy/ai/vllm/.venv/bin/vllm serve google/gemma-4-E4B-it \
  --host 127.0.0.1 --port 8001 \
  --served-model-name google/gemma-4-E4B-it
```

路由器不會自動增添或改寫 vLLM 的模型參數。圖像服務也使用同樣模式；編輯功能需要服務本身實作相容的 multipart API。

手動啟動：

```bash
cd ~/ai-router
AI_ROUTER_CONFIG="$PWD/config.yaml" .venv/bin/uvicorn router.app:app_factory \
  --factory --host 0.0.0.0 --port 9000 --workers 1 --timeout-graceful-shutdown 3700
```

## systemd

編輯 `systemd/ai-router.service` 的使用者、工作目錄及路徑後：

```bash
sudo cp systemd/ai-router.service /etc/systemd/system/ai-router.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-router
journalctl -u ai-router -f
```

使用 `KillMode=mixed`：正常停止時讓主程序排空，逾期才強制結束整個 cgroup。`TimeoutStopSec` 與 Uvicorn 的 graceful timeout 應大於最長可接受任務時間。若服務由 SIGKILL、掉電或 systemd 逾時強制終止，檢查並清理後端後再刪除 `state/unclean` 才重新啟動。不要設定自動重啟繞過此檢查。

## 使用

```bash
curl -s http://127.0.0.1:9000/v1/models
curl -N http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E4B-it","messages":[{"role":"user","content":"你好"}],"stream":true}'
curl http://127.0.0.1:9000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-image-model","prompt":"a cat"}'
curl http://127.0.0.1:9000/v1/images/edits \
  -F model=my-image-model -F 'prompt=make it blue' -F image=@input.png
curl -s http://127.0.0.1:9000/health
```

`/health` 回報狀態、目前模型、執行數和排隊數。`STOPPED → STARTING → READY → DRAINING → STOPPING → STOPPED`；啟動失敗為 `FAILED`，後續請求可重試。FIFO 佇列將相鄰同模型請求一起送入，出現不同模型後新進的舊模型請求不插隊。TTL 從最後一個執行中請求的生命週期結束時計算；排隊、啟動、卸載、租約與切換均由同一個排程器序列化。

## 測試

```bash
uv pip install --python .venv/bin/python -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

整合測試以真實本機 HTTP 和模擬後端檢查 300 個請求、模型切換期間的排空、idle TTL、客戶端斷線後的圖像和 SSE、multipart、免驗證存取，後端崩潰、端口占用及取消排隊中的切換。測試不載入 GPU，也不代表針對實際 vLLM、圖像服務或所有程序異常已驗證；請先在目標機器以自己的模型及服務進行驗收。

## 內建 Qwen-Image-2.1（取代上述外部圖片服務）

```bash
uv pip install --python .venv/bin/python -r requirements-image.txt
.venv/bin/python scripts/prepare_qwen_image.py
```

模型集中在 `~/.cache/huggingface/downloads/Qwen-Image-2.1/`。準備腳本將官方 processor、scheduler、編碼器及 VAE 下載到子目錄 `diffusers/`，固定官方 revision；不重複下載主 DiT。主 DiT 使用 Diffusers 官方單檔轉換器載入，無需 ComfyUI。原本 INT8 ConvRot 編碼器及 PE 擴寫模型仍保留，但不載入，因為需要專用量化支援。

```yaml
- id: Qwen-Image-2.1
  backend: qwen_image
  model_path: ~/.cache/huggingface/downloads/Qwen-Image-2.1/diffusers
  cpu_offload: true
  endpoints: [images/generations, images/edits]
  idle_ttl_seconds: 10800
```

執行時只讀本機模型。`cpu_offload: true` 將編碼器、DiT、VAE 依序搬到 GPU，適合 32GB 顯示記憶體，需要足夠系統 RAM。首次請求載入模型，切換模型或 TTL 到期時卸載。同模型推論序列執行，客戶端斷線後仍等推論結束才釋放租約。請以單 worker 執行。

生圖接受 JSON；編輯接受 multipart 的 `image`、`image[]` 或 `images` 圖片欄位（共最多 10 張），也可用 JSON `images` 列表提供 base64 或 image data URL。API 不讀取請求提供的路徑，也不下載圖片網址。輸出固定為 PNG 的 `data[].b64_json`，保留 alpha，另回傳 `seed` 和實際 `size`。

| 參數 | 預設與限制 |
| --- | --- |
| `prompt` | 必填，最長 32000 字元 |
| `size` | 預設自動（也可傳 `auto`）：編輯保留最後一張參考圖比例；生圖由本機 Qwen3.5-0.8B 依提示詞選比例。指定 `寬x高` 或同時提供 `width`、`height` 可覆蓋 |
| 尺寸限制 | 各邊 256～4096，32 倍數，總面積最多 8388608 像素 |
| `n` | 1，最多 4 張 |
| `seed` | 隨機產生，回應內可取得 |
| `num_inference_steps` | 40，可設 1～100 |
| `true_cfg_scale` | 1.0，可設 1～20 |
| `negative_prompt` | 選填，只在 `true_cfg_scale > 1` 生效 |
| `use_kv_cache` | true，加速但增加記憶體用量 |
| `output_resolution` | 1024，自動尺寸的目標面積約為此值平方，也影響參考圖縮放 |
| `response_format` | 僅支援 `b64_json` |

上表範圍是此 API 的資源限制，不代表模型硬性上限。RGBA 生圖在 prompt 明確描述透明背景。沒有獨立 `mask`、`strength`、`quality`、`style` 參數；未知參數回報錯誤。局部修改可將標記圖或遮罩圖當作參考圖並搭配文字說明。

```bash
curl http://127.0.0.1:9000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1","prompt":"A cat sticker, RGBA image with a transparent background","size":"1024x1024","seed":42}'
curl http://127.0.0.1:9000/v1/images/edits \
  -F model=Qwen-Image-2.1 -F 'prompt=make it blue' -F image=@input.png
```


文字生圖未指定尺寸（或 `size: "auto"`）時，由 Qwen3.5-0.8B 在 CPU 上選擇 `1:1`、`4:3`、`3:4`、`3:2`、`2:3`、`16:9` 或 `9:16`，再以 `output_resolution` 平方作為目標面積換算長寬並對齊 32 像素。提示詞保持原樣。比例輸出無效時回退 1:1；小模型載入失敗則回報錯誤。明確指定尺寸及圖片編輯都跳過此模型。小模型首次需要時載入 CPU，隨圖像後端卸載，不額外啟動服務。
