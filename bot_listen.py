"""
Постоянно работающий бот: отвечает на кнопки и сам делает ночную проверку.

Запуск:  python bot_listen.py
Работает, пока открыт (на компьютере или на сервере). Остановить: Ctrl+C.
Ночная проверка: в будни после 17:10 по Нью-Йорку (00:10-01:10 по Москве), один раз за день.
Отвечает только вашему чату (telegram_chat.txt).
"""
import os, sys, time, traceback, datetime as dt
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scanner as sc  # noqa: E402

TOKEN = sc.tg_token()
API = f"https://api.telegram.org/bot{TOKEN}/"
CHAT_FILE = os.path.join(HERE, "telegram_chat.txt")


def log(*a):
    print(dt.datetime.now().strftime("%d.%m %H:%M:%S"), *a, flush=True)


def my_chat():
    if os.environ.get("TELEGRAM_CHAT"):
        return os.environ["TELEGRAM_CHAT"].strip()
    return open(CHAT_FILE, encoding="utf-8").read().strip() if os.path.exists(CHAT_FILE) else None


def handle(text, chat):
    t = (text or "").strip().lower()
    if t.startswith("/start"):
        if not my_chat():
            with open(CHAT_FILE, "w", encoding="utf-8") as f:
                f.write(chat)
        sc.tg_send("✅ Бот на связи. Кнопки внизу.\n\n" + sc.HELP, chat)
        return
    if chat != my_chat():
        return  # чужие чаты игнорируем
    actions = {"анализ": sc.watch_message, "/watch": sc.watch_message,
               "позиции": sc.positions_message, "/positions": sc.positions_message,
               "рынок": sc.market_message, "/market": sc.market_message}
    for key, fn in actions.items():
        if key in t:
            if not sc._CACHE:
                sc.tg_send("⏳ Скачиваю цены, это до минуты…", chat)
            try:
                sc.tg_send(fn(), chat)
            except Exception as e:
                sc.tg_send(f"Не получилось: {str(e)[:200]}", chat)
                log(traceback.format_exc())
            return
    sc.tg_send(sc.HELP, chat)
    sc.tg_send(sc.HOWTO, chat)


def nightly_due(last_ny_date):
    now = dt.datetime.now(sc.NY)
    return now.weekday() < 5 and now.time() >= dt.time(17, 10) and last_ny_date != now.date()


def main():
    log("Бот запущен.")
    offset = None
    last_run = None
    while True:
        try:
            if nightly_due(last_run):
                log("Ночная проверка…")
                sc.run_nightly(send=True, save=True)
                last_run = dt.datetime.now(sc.NY).date()
            r = requests.get(API + "getUpdates", params={"timeout": 50, "offset": offset}, timeout=70).json()
            for u in r.get("result", []):
                offset = u["update_id"] + 1
                m = u.get("message") or {}
                if "text" in m:
                    log("Команда:", m["text"][:30])
                    handle(m["text"], str(m["chat"]["id"]))
        except KeyboardInterrupt:
            log("Остановлен.")
            break
        except Exception as e:
            log("Ошибка:", str(e).replace(TOKEN, "***")[:200])
            time.sleep(30)


if __name__ == "__main__":
    main()
