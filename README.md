# Local AI Inference Stack

這是一套以 **RTX PRO 4500 單機推論環境**為目標設計的本地 AI 服務，提供 OpenAI 相容 API，讓文字和圖片模型共用同一個入口。用戶端以模型 ID 指定要使用的模型，服務會依序載入、執行並切換模型，適合在本機部署多種模型，再透過支援 OpenAI API 的應用程式或工具呼叫。

服務內建動態排程與請求佇列，集中管理模型載入、推論和切換。請求依序排隊；佇列中相鄰的同模型請求會一起送入目前模型處理，讓連續使用同一模型時不必反覆切換。輪到不同模型時，排程器會先等目前模型的在途請求完成，再卸載並載入下一個模型，避免切換期間新舊模型同時搶用 GPU，或在推論尚未完成時提前卸載。佇列容量可設定，佇列已滿時會拒絕新請求，讓系統負載保持可控。

預設監聽 `0.0.0.0:9000`。本機呼叫範例使用 `http://127.0.0.1:9000/v1`；同一網路上的其他裝置可改用這台主機的 IP 位址。

## 啟動

首次安裝或更新依賴：

```bash
./setup.sh
```

前景啟動：

```bash
./start.sh
```

啟動後可用 `http://127.0.0.1:9000/health` 查看狀態。

## 開機自動啟動

### 從零設定（首次設定一次）

以下步驟沿用本機的 user systemd + tmux 方式。請先確認已安裝 tmux 和 uv，並將專案放在 /home/jeremy/local-ai-inference-stack；若帳號或路徑不同，請同步修改步驟中的所有 /home/jeremy 路徑。

建立啟動腳本：

~~~bash
mkdir -p ~/.local/bin ~/.config/systemd/user
cat > ~/.local/bin/local-ai-session <<'EOF'
#!/usr/bin/bash
set -u

cd /home/jeremy/local-ai-inference-stack || exit 1
./setup.sh
status=$?
if [ "$status" -eq 0 ]; then
  exec ./start.sh
fi

printf '\nsetup.sh failed with exit code %s. Keeping this tmux pane open for inspection.\n' "$status" >&2
exec /usr/bin/bash
EOF

cat > ~/.local/bin/local-ai-tmux <<'EOF'
#!/usr/bin/bash
set -euo pipefail

SESSION=local-ai
TMUX=/usr/bin/tmux

if [ "$#" -ne 1 ]; then
  printf 'Usage: %s {start|stop}\n' "$0" >&2
  exit 2
fi

case "$1" in
  start)
    if "$TMUX" has-session -t "=$SESSION" 2>/dev/null; then
      exit 0
    fi
    exec "$TMUX" new-session -d -s "$SESSION" -c /home/jeremy/local-ai-inference-stack /home/jeremy/.local/bin/local-ai-session
    ;;
  stop)
    "$TMUX" kill-session -t "$SESSION" 2>/dev/null || true
    ;;
  *)
    printf 'Usage: %s {start|stop}\n' "$0" >&2
    exit 2
    ;;
esac
EOF
chmod +x ~/.local/bin/local-ai-session ~/.local/bin/local-ai-tmux
~~~

建立 user systemd 服務：

~~~bash
cat > ~/.config/systemd/user/local-ai.service <<'EOF'
[Unit]
Description=Local AI inference stack in tmux

[Service]
Type=oneshot
RemainAfterExit=yes
Environment=HOME=/home/jeremy
Environment=PATH=/home/jeremy/.local/bin:/usr/local/bin:/usr/bin
ExecStart=/home/jeremy/.local/bin/local-ai-tmux start
ExecStop=/home/jeremy/.local/bin/local-ai-tmux stop

[Install]
WantedBy=default.target
EOF
~~~

設定 user manager 開機啟動，重新載入服務並啟動：

~~~bash
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now local-ai.service
~~~

本機使用 user systemd 的 `local-ai.service`，並已設定開機啟動及 linger。管理服務：

```bash
systemctl --user status local-ai.service
systemctl --user restart local-ai.service
systemctl --user stop local-ai.service
```

服務會在 tmux 的 `local-ai` 工作階段執行。要查看該工作階段，可執行 `tmux attach -t local-ai`；按 `Ctrl+B` 再按 `d` 可離開而不停止服務。

## 設定

編輯 `config.yaml`：

- 在 `models` 清單新增或修改模型。`id` 是 API 請求使用的模型名稱；`command`、`base_url` 和 `endpoints` 設定模型啟動方式與可用功能。
- `state_dir`、`queue_limit` 和 `request_timeout_seconds` 分別設定狀態目錄、等待佇列上限和請求逾時。
- 內建圖片模型使用 `backend: qwen_image` 和本機 `model_path`。首次使用前準備模型：

  ```bash
  .venv/bin/python scripts/prepare_qwen_image.py
  ```

修改設定後重新啟動服務。

## 支援模型與用法

目前 `config.yaml` 已設定以下模型：

| 模型 ID | 支援功能 |
| --- | --- |
| `nvidia/Qwen3.8-27B-NVFP4` | 對話、文字補全、Responses |
| `cyankiwi/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit` | 對話、文字補全、Responses |
| `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4` | 對話、文字補全、Responses |
| `unsloth/GLM-4.7-Flash-NVFP4` | 對話、文字補全、Responses |
| `google/gemma-4-E2B-it` | 對話、文字補全、Responses |
| `google/gemma-4-E4B-it` | 對話、文字補全、Responses |
| `ibm-granite/granite-4.2-30b-nvfp4` | 對話、文字補全、Responses |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | 對話、文字補全、Responses |
| `GitMylo/nsfwvision-qwen3-vl-8b-v3-safetensors` | 對話、文字補全、Responses |
| `Qwen/Qwen-Image-2.1` | 圖片生成、圖片編輯 |

列出目前模型：

```bash
curl http://127.0.0.1:9000/v1/models
```

對話範例：

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E4B-it","messages":[{"role":"user","content":"你好"}]}'
```

圖片生成與編輯範例：

```bash
curl http://127.0.0.1:9000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen-Image-2.1","prompt":"一隻坐在窗邊的貓"}'

curl http://127.0.0.1:9000/v1/images/edits \
  -F model=Qwen/Qwen-Image-2.1 \
  -F 'prompt=把天空改成夕陽色' \
  -F image=@input.png
```
