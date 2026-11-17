# 用 systemd 設定開機自動啟動

## 首次設定

這份指南使用 user systemd 在開機時啟動服務，並透過 tmux 查看執行輸出。請先確認已安裝 tmux，並備妥 [README 列出的安裝需求](../README.md#快速開始)。範例使用 `/home/jeremy/local-ai-inference-stack`，請依實際帳號、專案位置及 Bash、tmux 的安裝路徑調整。

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

## 管理服務

啟用後，可使用以下指令查看狀態、重新啟動或停止服務。

```bash
systemctl --user status local-ai.service
systemctl --user restart local-ai.service
systemctl --user stop local-ai.service
```

服務會在 tmux 的 `local-ai` 工作階段執行。要查看該工作階段，可執行 `tmux attach -t local-ai`；按 `Ctrl+B` 再按 `d` 可離開而不停止服務。
