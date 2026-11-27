# 使用 API

服務預設監聽 `0.0.0.0:9000`。在本機連線時，將用戶端的 API Base URL 設為 `http://127.0.0.1:9000/v1`。從區網內其他裝置連線時，將 `127.0.0.1` 換成主機 IP。

## 服務狀態與模型清單

以下指令可查看服務狀態及可用的模型 ID。

```bash
curl http://127.0.0.1:9000/health
curl http://127.0.0.1:9000/v1/models
curl http://127.0.0.1:9000/v1/models/gemma-4-E2B-it
```

模型可用的端點由 [config/models.yaml](../config/models.yaml) 中各模型的 `endpoints` 決定。查詢單一模型的路由會回傳模型基本資料。

## 對話

使用已設定的 Gemma 模型測試對話。

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-E2B-it","messages":[{"role":"user","content":"你好"}]}'
```

## 文字補全與 Responses

`completions` 接受 `prompt`，`responses` 接受 `input`。指定的模型必須在模型設定中啟用該端點。

```bash
curl -fsS http://127.0.0.1:9000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-E2B-it","prompt":"請續寫：今天的天氣","max_tokens":64}'

curl -fsS http://127.0.0.1:9000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-E2B-it","input":"用一句話解釋什麼是向量。"}'
```

## 文字轉向量

生成文字向量。排程器會按需載入 `embeddinggemma-2`，載入模型時請求會花較長時間。

```bash
curl http://127.0.0.1:9000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":"如何重設密碼？"}'
```

## 圖片輸入

Edge0 145B 的請求範例與限制見 [Edge0 後端](backends.md#圖片輸入)。

## 圖片生成與編輯

使用 `Qwen-Image-2.1` 以 JSON 生成圖片，並將回應存成檔案。

```bash
curl -fsS http://127.0.0.1:9000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen-Image-2.1","prompt":"雨後的山間小路","size":"1024x1024","seed":42}' \
  -o image-response.json
```

編輯時可用 multipart 上傳參考圖。將 `reference.png` 換成手邊的圖片檔。

```bash
curl -fsS http://127.0.0.1:9000/v1/images/edits \
  -F 'model=Qwen-Image-2.1' \
  -F 'prompt=把天空改成日落色調' \
  -F 'image=@reference.png' \
  -o image-response.json
```

編輯端點也接受 JSON，將參考圖放在 `images` 陣列。每個項目可為原始 base64 字串，或 `data:image/png;base64,...` 格式的資料 URL。multipart 可重複送出 `image` 欄位，最多 10 張參考圖，每張最多 16,777,216 像素。

輸出尺寸可用 `size`（如 `1024x1024`）指定，也可一起指定 `width` 和 `height`。長寬各須為 256 到 4096 之間的 32 倍數，總面積上限為 8,388,608 像素。省略尺寸時，文生圖會依提示詞選長寬比，編輯會沿用最後一張參考圖的長寬比。`output_resolution` 可調整自動選尺寸的目標解析度，預設為 1024。

回應包含實際的 `seed`、`size` 和 `data`。用專案環境的 Python 將第一張圖片的 `data[0].b64_json` 解碼成 PNG。

```bash
.venv/bin/python - <<'PY'
import base64
import json
from pathlib import Path

response = json.loads(Path("image-response.json").read_text())
Path("output.png").write_bytes(base64.b64decode(response["data"][0]["b64_json"]))
PY
```
