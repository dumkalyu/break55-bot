"""
Для GitHub: бот слушает Telegram и отвечает на кнопки сразу.
Работает ~55 минут, потом запускается заново по расписанию (раз в час).
"""
import os, sys, time, subprocess, traceback
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOKEN = os.environ["TELEGRAM_TOKEN"].strip()
API = f"https://api.telegram.org/bot{TOKEN}/"
MINUTES = float(os.environ.get("LISTEN_MINUTES", 55))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    import bot_listen as bl
    end = time.time() + MINUTES * 60
    offset = None
    log("Слушаю Telegram...")
    while time.time() < end:
        try:
            wait = max(1, min(50, int(end - time.time())))
            r = requests.get(API + "getUpdates", params={"timeout": wait, "offset": offset},
                             timeout=wait + 20).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                m = u.get("message") or {}
                if "text" not in m:
                    continue
                log("Команда:", m["text"][:30])
                subprocess.run(["git", "pull", "--rebase", "-q"], cwd=HERE)   # свежее состояние
                try:
                    bl.handle(m["text"], str(m["chat"]["id"]))
                except Exception as e:
                    log("Ошибка:", str(e).replace(TOKEN, "***")[:200])
                    log(traceback.format_exc()[:500].replace(TOKEN, "***"))
        except Exception as e:
            log("Сбой связи:", str(e).replace(TOKEN, "***")[:150])
            time.sleep(10)
    log("Смена окончена.")


if __name__ == "__main__":
    main()
