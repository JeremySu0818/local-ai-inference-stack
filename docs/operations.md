# 服務維運

## 調整設定

[config/models.yaml](../config/models.yaml) 定義模型 ID、後端和各模型開放的 API。每個模型可用 `startup_timeout_seconds` 和 `stop_timeout_seconds` 設定啟停等待時間，預設分別為 600 秒和 30 秒。

| Router 設定 | 預設值 | 用途 |
| --- | --- | --- |
| `state_dir` | `state` | 日誌和 Router 鎖的目錄 |
| `queue_limit` | `1024` | 等候中的請求上限 |
| `request_timeout_seconds` | `3600` | 上游 HTTP 的讀取、寫入及連線池等待逾時，連線逾時固定 10 秒 |

以上預設來自 [config/router.yaml](../config/router.yaml)。模型設定檔可在與 `models` 同一層覆寫這三個值。`request_timeout_seconds` 的範圍是上游 HTTP 操作。等候佇列、模型啟動和 Qwen Image worker 推論各有自己的執行流程。

| 啟動環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `AI_ROUTER_CONFIG` | `config/models.yaml` | `setup.sh` 和 `start.sh` 讀取的模型設定檔 |
| `AI_ROUTER_HOST` | `0.0.0.0` | 監聽位址。設成 `127.0.0.1` 可僅接受本機連線 |
| `AI_ROUTER_PORT` | `9000` | 監聽埠 |
| `AI_ROUTER_GRACEFUL_SHUTDOWN_SECONDS` | `3700` | Uvicorn 關閉時等待請求結束的秒數 |

Router 的 API 路由沒有驗證機制。使用預設監聽位址時，能連到該埠的用戶端就能送出請求。修改設定後請重新啟動服務。

### 使用自訂模型清單

複製預設設定，將 `models` 清單縮減至所需的模型，再用同一份設定執行安裝和啟動。

```bash
cp config/models.yaml config/models.local.yaml
# 編輯 config/models.local.yaml，保留要使用的模型
export AI_ROUTER_CONFIG="$PWD/config/models.local.yaml"
./setup.sh
./start.sh
```

`setup.sh` 依清單決定是否編譯 llama.cpp 或 Edge0，並準備列出的模型。新增模型或變更權重後，重新執行同一個安裝指令。若模型下載失敗，可依終端機印出的 `hf download` 指令重試，再執行 `./setup.sh`。llama.cpp 會重用 Hugging Face 快取中的 GGUF 檔案。Edge0 也可透過 `model_path` 使用現有權重目錄。

## 排程管理

請求依到達順序排隊，並以模型 ID 指定要使用的模型。排程器按需載入模型，讓佇列中連續使用同一模型的請求共用已載入的模型。輪到不同模型時，排程器會等目前的推論完成，再卸載舊模型並載入下一個，讓多個模型輪流使用 GPU 顯存。

等待佇列的容量可調整。佇列已滿時，服務會拒絕新請求。

## 紀錄檔

Router 會將 API 存取紀錄同時印到終端機並寫入狀態目錄的 `access.log`（預設 `state/access.log`），包含時間、來源 IP、方法、路徑、狀態碼、耗時、模型 ID、請求 ID 和結果。每次啟動會清空前一次的存取紀錄。

透過 `./start.sh` 啟動時，終端機輸出也會寫入 `state/server.log`（使用自訂狀態目錄時會寫到該目錄），包含連線紀錄和後端診斷訊息。每次啟動清空，保留本次執行最近 100 萬行。

## 排查請求與啟動問題

`GET /health` 回傳 Router 的排程狀態。

| 欄位 | 意義 |
| --- | --- |
| `state` | `STOPPED`（未載入）、`STARTING`（載入中）、`READY`（就緒）、`DRAINING`（等待使用中的請求結束）、`STOPPING`（卸載中）或 `FAILED`（啟動或排程失敗） |
| `model` | 目前選中的模型，閒置且無模型時為 `null` |
| `active` | 已取得排程名額、尚未釋放的請求數 |
| `queued` | 等候中的請求數 |
| `accepting` | 排程器是否仍接受新請求。佇列剩餘容量可從 `queued` 和 `queue_limit` 判斷 |

用 `curl -i` 查看回應標頭，將 `X-Request-ID` 的值和 `access.log` 的 `request_id` 對照。收到 `503` 時，先讀取回應的 `error.message`。佇列已滿時可稍後重試，後端啟動失敗時可查 `server.log` 的啟動訊息，核對權重、顯存和啟動逾時設定。

同一個狀態目錄由 `router.lock` 保護。若啟動訊息指出目錄已被使用，先檢查既有 Router 是否仍在執行。`unclean` 記錄上次的 Router PID。該 PID 仍存活時 Router 會拒絕啟動，程序已退出時，Router 會在啟動期間清除過期記錄。

## 更新 Python 依賴

`setup.sh` 依照 `requirements/locked.txt` 安裝固定版本，鎖檔適用於 Linux x86-64、Python 3.12 和 CUDA 13.2。需要新增或調整 Python 依賴時，先修改 `requirements/base.txt` 或 `requirements/image.txt`，再於專案目錄更新鎖檔：

```bash
uv pip compile --torch-backend=cu132 --python-version 3.12 --python-platform x86_64-manylinux_2_28 --emit-index-url --no-annotate --no-header requirements/base.txt -o requirements/locked.txt
```

修改 requirements 後必須重新產生鎖檔，再執行 `./setup.sh` 安裝新的依賴版本。

## 開機自動啟動

請參閱 [systemd 與 tmux 設定指南](systemd.md)。
