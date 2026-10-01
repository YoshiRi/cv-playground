"""インタラクトの出口（WebSocket）が流すフレームを受け取る最小例。

サーバー版（scripts/serve.sh）で画面を開き、「インタラクト」の「WebSocket に流す」を選んで実行すると、
server.py の /ws がフレームを配る。これを受け取って、一番大きく写っている物と、骨格があれば鼻の位置を出す。

    python3 tools/ws_receiver.py                      # ws://127.0.0.1:8010/ws
    python3 tools/ws_receiver.py ws://host:port/ws

フレームの形は docs/ADDING_UI.md の「5. インタラクト」（座標は画像の幅・高さで割った 0〜1）
"""
import asyncio
import json
import sys
import time

import websockets


async def main(url: str) -> None:
    async for ws in websockets.connect(url):  # 切れたらつなぎ直す
        print(f"つながった: {url}")
        try:
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") != "frame":
                    continue
                f = m["frame"]
                items = f["items"]
                line = f"#{f['seq']} {f.get('task', '')} {len(items)} 件 遅れ {time.time() * 1000 - f['wall']:.0f}ms"
                if items:
                    big = max(items, key=lambda it: (it["box"][2] - it["box"][0]) * (it["box"][3] - it["box"][1]))
                    line += f" ・ 一番大きい: {big['label']}" + (f" #{big['id']}" if "id" in big else "")
                    kp = big.get("keypoints")
                    if kp and len(kp) == 17 and kp[0][2] > 0.4:
                        line += f" 鼻 ({kp[0][0]:.2f}, {kp[0][1]:.2f})"
                    if "emotion" in big:
                        line += f" 表情 {big['emotion']['label']}"
                print(line)
        except websockets.ConnectionClosed:
            print("切れた。つなぎ直す")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8010/ws"))
