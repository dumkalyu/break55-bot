"""
Для GitHub: проверить, нажимали ли кнопки в Telegram, и ответить.
  python gh_poll.py --check   только узнать, есть ли новые команды (печатает has=1 / has=0)
  python gh_poll.py           ответить на все новые команды
Номер последней обработанной команды хранится в bot_offset.txt.
"""
import os, sys, requests

HERE = os.path.dirname(os.path.abspath(__file__))
OFF = os.path.join(HERE, "bot_offset.txt")
TOKEN = os.environ["TELEGRAM_TOKEN"].strip()
API = f"https://api.telegram.org/bot{TOKEN}/"


def updates(offset):
    r = requests.get(API + "getUpdates", params={"offset": offset, "timeout": 0}, timeout=30).json()
    return r.get("result", [])


def main():
    offset = int(open(OFF).read().strip()) if os.path.exists(OFF) else None
    ups = updates(offset)
    if "--check" in sys.argv:
        print(f"has={1 if ups else 0}")
        return
    if not ups:
        return
    sys.path.insert(0, HERE)
    import bot_listen as bl
    for u in ups:
        offset = u["update_id"] + 1
        with open(OFF, "w") as f:           # сохраняем сразу: команда не выполнится дважды
            f.write(str(offset))
        m = u.get("message") or {}
        if "text" in m:
            print("Команда:", m["text"][:30])
            try:
                bl.handle(m["text"], str(m["chat"]["id"]))
            except Exception as e:
                print("Ошибка:", str(e).replace(TOKEN, "***")[:200])
    updates(offset)                          # подтверждаем Telegram, что всё прочитано


if __name__ == "__main__":
    main()
