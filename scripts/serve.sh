#!/bin/sh
# サーバー版の起動。環境ごとの設定（Ollama のモデル名など）はリポジトリに入れない .env に書く（.env.example を参照）
#   scripts/serve.sh            # 前面で起動
#   scripts/serve.sh --bg       # 裏で起動し、ログは logs/server.log
cd "$(dirname "$0")/.." || exit 1
[ -f .env ] && set -a && . ./.env && set +a
[ -x .venv/bin/python ] || { echo ".venv が無い。README の手順で作る"; exit 1; }
if [ "$1" = "--bg" ]; then
  mkdir -p logs
  pkill -f "python server.py" 2>/dev/null
  nohup .venv/bin/python server.py > logs/server.log 2>&1 &
  echo "started (pid $!, log: logs/server.log)"
else
  exec .venv/bin/python server.py
fi
