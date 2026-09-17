"""
Бот сигналов пробойной системы -> Telegram.

Запуск:   python scanner.py          обычный ночной запуск (после закрытия рынка США)
          python scanner.py --dry    показать сообщение здесь, ничего не отправлять и не сохранять
          python scanner.py --test   отправить пробное сообщение в Telegram
          python scanner.py --watch  показать список «за чем следить»

Правила те же, что в backtest.py (функции индикаторов берутся оттуда):
  сигнал на закрытии: close > max(high 55 пред. дней) и close > SMA200 -> покупка на следующем открытии;
  стоп = цена входа - 2*ATR(20) дня сигнала; выход: close < min(low 20 пред. дней) -> продажа на следующем открытии.
Бумаги: сегодняшний S&P 500, у которых есть бессрочный контракт на Bybit (ONLY_BYBIT).
Цены: Yahoo (бесплатно). Бот ведёт виртуальный журнал: bot_state.json и paper_journal.csv.

Файлы рядом со скриптом:
  telegram_token.txt  - токен бота от @BotFather (одна строка)
  telegram_chat.txt   - создаётся сам после того, как вы нажмёте Start у бота
"""
import os, sys, json, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _s in (sys.stdout, sys.stderr):          # Windows: не падать на русских буквах и знаках
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import backtest as bt  # noqa: E402

ONLY_BYBIT = True
STATE = os.path.join(HERE, "bot_state.json")
JOURNAL = os.path.join(HERE, "paper_journal.csv")
SP_FILE = os.path.join(HERE, "data", "sp500_hist.csv")
REPLAY_DAYS = 260          # при первом запуске: какие позиции система держала бы сейчас
FUND_WARN = 0.0015         # плата за удержание выше 0.15% в месяц - предупреждение
NY = ZoneInfo("America/New_York")


# ---------------- бумаги ----------------
def sp500_now():
    fresh = os.path.exists(SP_FILE) and (dt.datetime.now().timestamp() - os.path.getmtime(SP_FILE)) < 7 * 86400
    if not fresh:
        try:
            r = requests.get(bt.dl.SP_URL, timeout=120)
            r.raise_for_status()
            os.makedirs(os.path.dirname(SP_FILE), exist_ok=True)
            with open(SP_FILE, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print("Не обновился состав S&P 500, беру сохранённый:", str(e)[:100])
    d = pd.read_csv(SP_FILE)
    return d.sort_values("date").iloc[-1].tickers.split(",")


def bybit_stocks():
    B = "https://api.bybit.com/v5/market/"
    L = requests.get(B + "instruments-info", params={"category": "linear", "limit": 1000}, timeout=30).json()["result"]["list"]
    T = {t["symbol"]: t for t in requests.get(B + "tickers", params={"category": "linear"}, timeout=30).json()["result"]["list"]}
    out = {}
    for i in L:
        if i.get("symbolType") != "stock" or i.get("status") != "Trading":
            continue
        base = i["baseCoin"]
        t = T.get(i["symbol"], {})
        per_day = 1440 / float(i.get("fundingInterval") or 480)
        fr = float(t.get("fundingRate") or 0)
        out[base] = dict(symbol=i["symbol"], funding_month=fr * per_day * 30,
                         turnover=float(t.get("turnover24h") or 0))
    return out


def universe():
    sp = sp500_now()
    cache = os.path.join(HERE, "bybit_list.csv")
    try:
        bb = bybit_stocks()
        pd.DataFrame([dict(base=k, symbol=v["symbol"]) for k, v in bb.items()]).to_csv(cache, index=False)
    except Exception as e:
        print("Bybit не ответил (с серверов США он закрыт), беру сохранённый список:", str(e)[:80])
        bb = {}
        if os.path.exists(cache):
            for r in pd.read_csv(cache).itertuples():
                bb[r.base] = dict(symbol=r.symbol)
    uni = {}
    for t in sp:
        key = t.replace(".", "")
        info = bb.get(key) or bb.get(key + "STOCK")
        if ONLY_BYBIT and not info:
            continue
        uni[t] = info or {}
    if not uni:
        raise RuntimeError("Не удалось получить список акций Bybit.")
    return uni


# ---------------- цены ----------------
def prices(tickers):
    import yfinance as yf
    ymap = {t: t.replace(".", "-") for t in tickers}
    raw = yf.download(list(ymap.values()) + ["SPY"], period="2y", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    now = dt.datetime.now(NY)
    frames = {}
    for t, y in list(ymap.items()) + [("SPY", "SPY")]:
        try:
            d = raw[y].dropna(subset=["Open", "High", "Low", "Close"])
            if len(d):
                frames[t] = d
        except KeyError:
            pass
    # Yahoo иногда не отдаёт часть бумаг в общей загрузке - докачиваем по одной
    for t, y in list(ymap.items()) + [("SPY", "SPY")]:
        for _ in range(2):
            if t in frames:
                break
            try:
                d = yf.download(y, period="2y", auto_adjust=True, progress=False, multi_level_index=False)
                d = d.dropna(subset=["Open", "High", "Low", "Close"])
                if len(d):
                    frames[t] = d
            except Exception:
                pass
    missing = [t for t in ymap if t not in frames]
    if missing:
        print("Нет цен:", ", ".join(missing))
    if len(missing) > max(3, 0.05 * len(ymap)):
        raise RuntimeError(f"Yahoo не отдал цены по {len(missing)} из {len(ymap)} бумаг. Попробую позже.")
    out = {}
    for t, d in frames.items():
        d = d[d.Close > 0]
        # незакрытый сегодняшний день не берём
        if len(d) and d.index[-1].date() == now.date() and now.time() < dt.time(16, 30):
            d = d.iloc[:-1]
        if len(d) < 260:
            continue
        a = pd.DataFrame({"date": pd.to_datetime(d.index).tz_localize(None), "O": d.Open.values,
                          "H": d.High.values, "L": d.Low.values, "C": d.Close.values,
                          "V": d.Volume.values.astype(float)})
        a["rawC"] = a.C
        a["fac"] = 1.0
        out[t] = bt.indicators(a).set_index("date")
    return out


# ---------------- состояние ----------------
def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(s):
    tmp = STATE + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


def step(state, data, day, events, journal, live):
    """Обработать один торговый день для всех бумаг (порядок как в бэктесте)."""
    pos, pend, pexit = state["positions"], state["pending"], state["pending_exit"]
    ds = str(day.date())
    for t, d in data.items():
        if t == "SPY" or day not in d.index:
            continue
        b = d.loc[day]
        # 1) выход по каналу на открытии
        if t in pexit and t in pos:
            p = pos.pop(t); pexit.pop(t)
            r = (b.O - p["entry"]) / (p["entry"] - p["stop"]) - bt.COST * p["entry"] / (p["entry"] - p["stop"])
            journal.append(dict(ticker=t, entry_date=p["entry_date"], exit_date=ds, entry=p["entry"],
                                stop=p["stop"], exit=b.O, reason="канал", R=r, live=live and p.get("live", False)))
            events["closed"].append((t, "канал", r, b.O / p["entry"] - 1 - bt.COST))
        pexit.pop(t, None)
        entered = False
        # 2) вход на открытии
        if t in pend:
            s = pend.pop(t)
            pos[t] = dict(entry_date=ds, entry=float(b.O), stop=float(b.O - bt.STOP_K * s["atr"]),
                          sig_date=s["sig_date"], live=live)
            entered = True
            events["opened"].append(t)
        # 3) стоп
        if t in pos and b.L <= pos[t]["stop"]:
            p = pos.pop(t)
            px = b.O if (not entered and b.O <= p["stop"]) else p["stop"]
            risk = p["entry"] - p["stop"]
            r = (px - p["entry"]) / risk - bt.COST * p["entry"] / risk
            journal.append(dict(ticker=t, entry_date=p["entry_date"], exit_date=ds, entry=p["entry"],
                                stop=p["stop"], exit=px, reason="стоп", R=r, live=live and p.get("live", False)))
            events["stopped"].append((t, px, r, px / p["entry"] - 1 - bt.COST))
        # 4) выход по каналу на закрытии
        if t in pos and b.C < b.ll20:
            pexit[t] = ds
            events["exit_next"].append(t)
        # 5) сигнал входа
        if (t not in pos and t not in pend and pd.notna(b.hh55) and pd.notna(b.sma200) and pd.notna(b.atr)
                and b.C > b.hh55 and b.C > b.sma200 and b.rawC >= bt.MIN_PRICE):
            pend[t] = dict(sig_date=ds, atr=float(b.atr))
            events["entry_next"].append(t)
    state["last_day"] = ds


def new_events():
    return dict(opened=[], closed=[], stopped=[], exit_next=[], entry_next=[])


# ---------------- сообщение ----------------
def fmt_money(x):
    return f"{x:,.2f}".replace(",", " ")


def pc(x, sign=False):
    """Процент: для малых значений с одним знаком после запятой."""
    f = ("+" if sign else "") + (".1%" if abs(x) < 0.1 else ".0%")
    return format(x, f)


def position_lines(state, data, day):
    rows = []
    for t, p in state["positions"].items():
        d = data.get(t)
        if d is None or day not in d.index:
            continue
        dd = d.loc[:day]
        c = dd.C.iloc[-1]
        pct = c / p["entry"] - 1
        lvl = dd.L.iloc[-bt.CH_OUT:].min()     # закрытие ниже этого уровня = продать
        old = "" if p.get("live") else " 📜"
        if t in state.get("pending_exit", {}):
            tail = "ПРОДАТЬ завтра на открытии"
        elif p["stop"] > lvl:
            tail = f"стоп {fmt_money(p['stop'])} (на {pc(max(0, 1 - p['stop'] / c))} ниже цены)"
        else:
            tail = f"продать, если день закроется ниже {fmt_money(lvl)} (на {pc(max(0, 1 - lvl / c))} ниже цены)"
        rows.append((pct, f"• <b>{t}</b>{old}: {pc(pct, True)} с покупки · {tail}"))
    return [s for _, s in sorted(rows, reverse=True)]


def build_message(state, data, uni, day, ev, journal):
    L = [f"<b>📊 Итоги дня {day.date():%d.%m.%Y}</b> (биржа США закрылась)"]
    spy = data.get("SPY")
    if spy is not None and day in spy.index:
        up = spy.loc[day].C > spy.loc[day].sma200
        L.append("Рынок в целом: " + ("растёт 📈" if up else "падает 📉") + " (просто для справки)")
    L.append(f"Проверено акций: {len(data) - 1}")
    L.append("")

    L.append("🟢 <b>Купить завтра на открытии биржи (16:30 мск):</b>")
    if ev["entry_next"]:
        for t in sorted(ev["entry_next"]):
            b = data[t].loc[day]
            gap = bt.STOP_K * b.atr
            u = uni.get(t, {})
            fm = u.get("funding_month")
            fs = "" if fm is None else f" Bybit берёт за удержание {fm:.2%} в месяц" + (" ⚠️ дорого." if fm > FUND_WARN else ".")
            L.append(f"• <b>{t}</b> — цена {fmt_money(b.C)}. Стоп = цена покупки − {fmt_money(gap)} "
                     f"(≈ {fmt_money(b.C - gap)}, убыток до {gap / b.C:.0%}).{fs}")
    else:
        L.append("• нет")
    L.append("")
    L.append("🔴 <b>Продать завтра на открытии:</b>")
    if ev["exit_next"]:
        for t in sorted(ev["exit_next"]):
            p = state["positions"][t]
            c = data[t].loc[day].C
            L.append(f"• <b>{t}</b> — цена ушла ниже минимума за 20 дней. С покупки {c / p['entry'] - 1:+.0%}")
    else:
        L.append("• нет")
    if ev["stopped"]:
        L.append("")
        L.append("⛔ <b>Сработал стоп — продано сегодня:</b>")
        for t, px, r, pct in ev["stopped"]:
            L.append(f"• {t} — по {fmt_money(px)}, {pct:+.1%} с покупки")
    if ev["closed"]:
        L.append("")
        L.append("💰 <b>Продано сегодня утром:</b>")
        for t, _, r, pct in ev["closed"]:
            L.append(f"• {t} — {pct:+.1%} с покупки")
    L.append("")

    pl = position_lines(state, data, day)
    if pl:
        L.append(f"📂 <b>Что сейчас куплено по системе ({len(pl)}):</b>")
        L += pl
        if any("📜" in s for s in pl):
            L.append("📜 — куплено по расчёту на истории, ещё до запуска бота; в итоги не идёт.")
        L.append("")

    j = pd.DataFrame(journal)
    if len(j) and "live" in j:
        j = j[j.live == True]  # noqa: E712
    if len(j):
        pc = j.exit / j.entry - 1 - bt.COST
        L.append(f"📒 <b>Итоги с запуска бота:</b> сделок {len(j)}, в плюс {np.mean(pc > 0):.0%}, "
                 f"в среднем {pc.mean():+.1%} на сделку")
    else:
        L.append("📒 Итоги с запуска бота: закрытых сделок пока нет.")
    L.append("ℹ️ Так устроена система: большинство сделок закрываются с небольшим убытком, "
             "а весь доход дают редкие сильные росты.")
    return "\n".join(L)



# ---------------- Telegram ----------------
def tg_token():
    if os.environ.get("TELEGRAM_TOKEN"):
        return os.environ["TELEGRAM_TOKEN"].strip()
    p = os.path.join(HERE, "telegram_token.txt")
    if not os.path.exists(p):
        sys.exit("Нет файла telegram_token.txt (токен от @BotFather).")
    return open(p, encoding="utf-8-sig").read().strip()


def tg_chat(token):
    if os.environ.get("TELEGRAM_CHAT"):
        return os.environ["TELEGRAM_CHAT"].strip()
    p = os.path.join(HERE, "telegram_chat.txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read().strip()
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30).json()
    chats = [u["message"]["chat"]["id"] for u in r.get("result", []) if "message" in u]
    if not chats:
        sys.exit("Бот ещё не знает ваш чат: откройте бота в Telegram, нажмите Start и запустите снова.")
    cid = str(chats[-1])
    with open(p, "w", encoding="utf-8") as f:
        f.write(cid)
    return cid


KEYBOARD = json.dumps({"keyboard": [[{"text": "🔎 Анализ"}, {"text": "📋 Позиции"}],
                                    [{"text": "🧭 Рынок"}, {"text": "❓ Помощь"}]],
                       "resize_keyboard": True})


def tg_send(text, chat=None):
    token = tg_token()
    chat = chat or tg_chat(token)
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > 3800:
            parts.append(cur); cur = ""
        cur += line + "\n"
    parts.append(cur)
    for p in parts:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data=dict(chat_id=chat, text=p, parse_mode="HTML", reply_markup=KEYBOARD,
                                    disable_web_page_preview=True), timeout=30)
        if r.status_code != 200:
            print("Telegram не принял сообщение:", r.status_code, r.text[:200].replace(token, "***"))


# ---------------- главное ----------------
_CACHE = {}


def market_data(max_age_min=60):
    """Бумаги и цены; держим в памяти час, чтобы кнопки отвечали быстро."""
    now = dt.datetime.now().timestamp()
    if max_age_min > 0 and _CACHE.get("t", 0) > now - max_age_min * 60:
        return _CACHE["uni"], _CACHE["data"]
    uni = universe()
    data = prices(list(uni))
    if data.get("SPY") is None:
        raise RuntimeError("Не скачались цены SPY - попробуйте позже.")
    _CACHE.update(t=now, uni=uni, data=data)
    return uni, data


def run_nightly(send=True, save=True):
    try:
        uni, data = market_data(max_age_min=0)
    except RuntimeError as e:
        print(e)
        if send:
            tg_send(f"⚠️ Итоги дня не посчитаны: {e}")
        return None
    print(f"Бумаг в работе: {len(uni)}")
    days = data["SPY"].index
    journal = pd.read_csv(JOURNAL).to_dict("records") if os.path.exists(JOURNAL) else []
    state = load_state()
    ev = new_events()
    if state is None:
        print("Первый запуск: восстанавливаю, какие позиции система держала бы сейчас...")
        state = dict(positions={}, pending={}, pending_exit={}, last_day=None,
                     started=str(days[-1].date()))
        for d in days[-REPLAY_DAYS:-1]:
            step(state, data, d, new_events(), [], live=False)
        todo = [days[-1]]
    else:
        todo = [d for d in days if str(d.date()) > state["last_day"]]
    if not todo:
        print("Новых закрытых дней нет.")
        return None
    for d in todo:
        e = new_events() if d != todo[-1] else ev
        step(state, data, d, e, journal, live=True)
        if d != todo[-1]:
            for k in ev:
                ev[k] += e[k]
    msg = build_message(state, data, uni, todo[-1], ev, journal)
    if len(todo) > 1:
        msg = f"(обработано дней: {len(todo)} — были пропущены запуски)\n" + msg
    print(msg)
    if send:
        tg_send(msg)
    if save:
        save_state(state)
        if journal:
            pd.DataFrame(journal).to_csv(JOURNAL, index=False)
    return msg


# ---------------- кнопки ----------------
def earnings_soon(ticker, days_ahead=14):
    """Отчёт компании в ближайшие дни (только для информации: отчёт = риск прыжка цены через стоп)."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker.replace(".", "-")).calendar
        ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not ds:
            return ""
        d0 = pd.Timestamp(ds[0]).date()
        left = (d0 - dt.date.today()).days
        if 0 <= left <= days_ahead:
            return f" · отчёт {d0:%d.%m} ⚠️"
    except Exception:
        pass
    return ""


def watch_message(top=15, max_dist=0.05):
    uni, data = market_data()
    state = load_state() or dict(positions={}, pending={}, pending_exit={})
    day = data["SPY"].index[-1]
    rows = []
    for t, d in data.items():
        if t == "SPY" or t in state["positions"] or t in state["pending"] or d.index[-1] != day:
            continue
        b = d.iloc[-1]
        if not (b.C > b.sma200) or pd.isna(b.atr):
            continue
        lvl = d.H.iloc[-bt.CH_IN:].max()   # уровень, который должно пробить следующее закрытие
        dist = lvl / b.C - 1
        if dist <= max_dist:
            rows.append((dist, t, b, lvl))
    rows.sort(key=lambda x: x[0])
    L = [f"<b>🔎 За чем следить · закрытие {day.date():%d.%m.%Y}</b>",
         f"Акции в росте, которым до сигнала «купить» осталось меньше {max_dist:.0%}. "
         f"Сигнал будет, если цена ЗАКРОЕТ день выше своего максимума за {bt.CH_IN} дней (уровень).", ""]
    if not rows:
        L.append("Близких к пробою бумаг нет.")
    for dist, t, b, lvl in rows[:top]:
        atrs = (lvl - b.C) / b.atr
        u = uni.get(t, {})
        fm = u.get("funding_month")
        fs = "" if fm is None else f" · Bybit {fm:.2%}/мес" + (" ⚠️" if fm > FUND_WARN else "")
        risk = bt.STOP_K * b.atr / b.C
        L.append(f"• <b>{t}</b> {fmt_money(b.C)} → уровень {fmt_money(lvl)}: "
                 f"ещё {dist:.1%} · убыток при стопе до {risk:.0%}{fs}{earnings_soon(t)}")
    if len(rows) > top:
        L.append(f"…и ещё {len(rows) - top}.")
    L += ["", "«отчёт ⚠️» — скоро отчётность компании, цена может резко прыгнуть.",
          "«Bybit …/мес» — сколько биржа берёт за удержание позиции в месяц.",
          "Это список для наблюдения, а не совет покупать."]
    return "\n".join(L)


def positions_message():
    uni, data = market_data()
    state = load_state()
    if not state or not state["positions"]:
        return "Сейчас по системе ничего не куплено."
    day = data["SPY"].index[-1]
    L = [f"<b>📋 Что куплено по системе · {day.date():%d.%m.%Y}</b>"]
    lines = position_lines(state, data, day)
    L += [s + earnings_soon(s.split("<b>")[1].split("</b>")[0]) for s in lines]
    if any("📜" in s for s in lines):
        L.append("📜 — куплено по расчёту на истории, ещё до запуска бота.")
    if state.get("pending"):
        L.append("🟢 Купить завтра на открытии: " + ", ".join(sorted(state["pending"])))
    L.append("⚠️ отчёт — скоро отчётность компании, цена может резко прыгнуть.")
    return "\n".join(L)


def market_message():
    uni, data = market_data()
    s = data["SPY"]
    b = s.iloc[-1]
    up = b.C > b.sma200
    diff = b.C / b.sma200 - 1
    return "\n".join([
        f"<b>🧭 Рынок США в целом · {s.index[-1].date():%d.%m.%Y}</b>",
        f"Фонд SPY (500 крупнейших компаний) стоит {fmt_money(b.C)}.",
        f"Его средняя цена за 200 дней — {fmt_money(b.sma200)}.",
        ("📈 Сейчас рынок ВЫШЕ средней на " if up else "📉 Сейчас рынок НИЖЕ средней на ") + f"{abs(diff):.1%}"
        + (" — это спокойный, растущий рынок." if up else " — это падающий, опасный рынок."),
        "",
        "Что показала проверка 2007–2026:",
        "• просто держать SPY — +11% в год, но в худший момент минус 55%;",
        "• держать SPY только пока он выше средней — +7% в год, худший момент минус 25%;",
        "• пробой по отдельным акциям рынок не обогнал.",
    ])


HELP = ("<b>❓ Что умеет бот</b>\n"
        "Каждую ночь, после закрытия биржи США, бот присылает итоги дня: что купить и что продать утром.\n\n"
        "🔎 <b>Анализ</b> — акции, которые скоро могут дать сигнал «купить». За ними стоит понаблюдать.\n"
        "📋 <b>Позиции</b> — что сейчас куплено по системе и при какой цене продавать.\n"
        "🧭 <b>Рынок</b> — растёт или падает рынок США в целом.\n\n"
        "<b>Словарик</b>\n"
        "• <b>Стоп</b> — цена, при которой продаём, чтобы не потерять больше.\n"
        "• <b>Продать, если день закроется ниже …</b> — цена опустилась ниже минимума за 20 дней, рост закончился.\n"
        "• <b>Уровень</b> — самая высокая цена за 55 дней. Закрылись выше — сигнал «купить».\n\n"
        "Бот ведёт учебный журнал сделок: это проверка системы, а не совет покупать.")


HOWTO = ("<b>📖 Как читать итоги дня и что делать</b>\n\n"
         "<b>1. Рынок в целом.</b> Просто посмотрите. На решения не влияет.\n\n"
         "<b>2. 🟢 Купить завтра.</b> По каждой акции:\n"
         "  а) Есть ли ⚠️ у платы Bybit? Дорогое удержание съедает доход — лучше пропустить.\n"
         "  б) Нет ли скоро отчёта компании (кнопка 📋 / 🔎 покажет ⚠️)? Отчёт — риск резкого прыжка через стоп.\n"
         "  в) Сколько покупать. Правило: на стопе теряем не больше 1% всего счёта.\n"
         "     Сумма = счёт × 1% ÷ «убыток до …%».\n"
         "     Пример: счёт 1000$, убыток до 10% → покупаем на 100$ (на стопе потеряем 10$ = 1%).\n"
         "  г) В 16:30 мск купить по рынку и СРАЗУ поставить стоп: цена покупки минус число из сообщения.\n\n"
         "<b>3. 🔴 Продать завтра.</b> В 16:30 мск продать по рынку. Не ждать и не надеяться.\n\n"
         "<b>4. ⛔ Стоп / 💰 Продано.</b> Уже случилось — ничего делать не нужно, только отметить.\n\n"
         "<b>5. 📂 Что куплено.</b> Проверьте, что у вас на бирже стоят все стопы.\n"
         "  Если до уровня «продать» осталось 1–2% — завтра вечером, скорее всего, придёт сигнал продать.\n"
         "  📜 — это старые покупки по истории. Сейчас их НЕ покупать: момент входа давно прошёл.\n\n"
         "<b>Чего не делать</b>\n"
         "• Не покупать заранее из списка 🔎 — ждать сигнала по закрытию дня.\n"
         "• Не опускать стоп ниже и не отменять его.\n"
         "• Не продавать раньше сигнала из страха. Весь доход дают редкие большие росты (как DELL +286%), "
         "а мелкие убытки — нормальная часть системы.\n\n"
         "<b>Раз в неделю</b> смотрите «Итоги с запуска бота». Через 1–2 месяца решаем, стоит ли торговать "
         "на реальные деньги. Напоминание: в проверке на истории система не обогнала простое держание SPY.")


def main():
    if "--test" in sys.argv:
        tg_send("✅ Бот Пробой55 на связи. Кнопки внизу.")
        print("Пробное сообщение отправлено.")
        return
    if "--watch" in sys.argv:
        print(watch_message())
        return
    dry = "--dry" in sys.argv
    run_nightly(send=not dry, save=not dry)


if __name__ == "__main__":
    main()
