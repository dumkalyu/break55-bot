"""
Загрузка цен из EODHD для всех бумаг, входивших в S&P 500 с 2007 года.

Запуск:  python dl_eodhd.py
Ключ читается из eodhd_key.txt (лежит рядом со скриптом) и нигде не печатается.
Повторный запуск докачивает только то, чего ещё нет (файлы в data/prices/).

Результат:
  data/sp500_hist.csv        - состав индекса по датам (скачанный как есть)
  data/symbols_active.csv    - список живых бумаг США в EODHD
  data/symbols_delisted.csv  - список ушедших бумаг США в EODHD
  data/prices/<CODE>.csv     - сырые цены по каждому коду
  data/mapping.csv           - какой код EODHD выбран для каждого тикера индекса
  data/report_download.txt   - отчёт: найдено / не найдено / покрытие по годам
"""
import os, sys, time, threading, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PRICES = os.path.join(DATA, "prices")
os.makedirs(PRICES, exist_ok=True)

SP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
          "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
BASE = "https://eodhd.com/api"
START = "2005-01-01"        # история для прогрева SMA200
TEST_FROM = "2007-01-01"    # с этой даты считаем членство в индексе
MIN_COVER = 0.30            # меньше - считаем, что это чужая бумага / не найдено
EXTRA = ["SPY", "QQQ"]
MIN_PART = 60               # второй и следующие коды тикера берутся, если закрывают >= 60 дней членства

# Тикеры из файла состава, чья история в EODHD лежит под другим кодом.
# Каждое соответствие сверено по названию компании в списке EODHD.
ALIASES = {
    "LEHMQ": ["LEH"], "ABKFQ": ["ABK"],  # ABK проверяется по цене в backtest.py
    "ANRZQ": ["ANR", "ANRZ"], "ATGE": ["DV_old"],
    "CBH": ["CBH1"], "EQ": ["EQ1"], "HSH": ["SLE_old"], "LIFE": ["LIFE2"],
    "MTLQQ": ["GM_old"], "RSHCQ": ["RSH"], "SE": ["SE1"], "SUN": ["SUN1"],
    "SUNEQ": ["SUNE_old", "WFR_old", "SDSNQ"], "TSG": ["TSG1"], "WYND": ["WYN"],
    "CDAY": ["DAY"], "PEAK": ["DOC"], "RE": ["EG"], "ADS": ["BFH"], "BHGE": ["BKR"],
    "ARNC": ["HWM"], "H": ["HET"],
}


def read_key():
    p = os.path.join(HERE, "eodhd_key.txt")
    if not os.path.exists(p):
        sys.exit("Нет файла eodhd_key.txt рядом со скриптом. Положите туда ключ EODHD одной строкой.")
    with open(p, encoding="utf-8-sig") as f:
        k = f.read().strip()
    if not k:
        sys.exit("Файл eodhd_key.txt пустой.")
    return k


KEY = None
_lock = threading.Lock()
_times = []


def _limit(per_min=900):
    """Не больше per_min запросов в минуту (у тарифа лимит 1000)."""
    while True:
        with _lock:
            now = time.time()
            while _times and now - _times[0] > 60:
                _times.pop(0)
            if len(_times) < per_min:
                _times.append(now)
                return
        time.sleep(0.2)


def api(path, **params):
    """Запрос к EODHD. Ключ никогда не попадает в сообщения об ошибках."""
    params.update(api_token=KEY, fmt="json")
    for attempt in range(4):
        _limit()
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=60)
        except Exception as e:
            msg = str(e).replace(KEY, "***")
            print(f"  сеть: {path}: {msg[:150]}")
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (401, 403):
            sys.exit(f"EODHD отказал в доступе ({r.status_code}) на {path}. Проверьте ключ и тариф.")
        if r.status_code == 402:
            sys.exit("EODHD: закончился дневной лимит запросов (402). Запустите завтра - скачанное сохранится.")
        print(f"  {path}: ответ {r.status_code}, повтор")
        time.sleep(5 * (attempt + 1))
    return None


# ---------- состав индекса ----------
def load_membership():
    p = os.path.join(DATA, "sp500_hist.csv")
    if not os.path.exists(p):
        print("Скачиваю состав S&P 500 по датам...")
        r = requests.get(SP_URL, timeout=120)
        r.raise_for_status()
        with open(p, "wb") as f:
            f.write(r.content)
    return pd.read_csv(p, parse_dates=["date"])


def membership_days(hist, days):
    """Словарь тикер -> DatetimeIndex торговых дней, когда бумага была в индексе.
    Состав на день d = последняя строка файла с датой <= d."""
    hist = hist.sort_values("date").reset_index(drop=True)
    pos = hist["date"].searchsorted(days, side="right") - 1
    acc = {}
    for row_i in sorted(set(pos)):
        if row_i < 0:
            continue
        dd = days[pos == row_i]
        for t in hist.at[row_i, "tickers"].split(","):
            acc.setdefault(t.strip(), []).append(dd.values)
    import numpy as np
    return {t: pd.DatetimeIndex(np.unique(np.concatenate(v))) for t, v in acc.items()}


# ---------- список символов ----------
def symbol_lists():
    res = {}
    for name, flag in (("active", 0), ("delisted", 1)):
        p = os.path.join(DATA, f"symbols_{name}.csv")
        if not os.path.exists(p):
            print(f"Скачиваю список бумаг США ({name})...")
            js = api("exchange-symbol-list/US", delisted=flag)
            if not js:
                sys.exit(f"Не удалось получить список бумаг ({name}).")
            pd.DataFrame(js).to_csv(p, index=False)
        df = pd.read_csv(p, keep_default_na=False)
        res[name] = df
        print(f"  {name}: {len(df)} символов")
    return res


def eod_code(t):
    return t.replace(".", "-")


def candidates(t, active_codes, delisted_codes):
    base = eod_code(t)
    c = [base]
    pref = base + "_"
    c += sorted(x for x in delisted_codes if x.startswith(pref))
    c += sorted(x for x in active_codes if x.startswith(pref))
    # у ушедших бумаг иногда нет точки/дефиса в классе акций
    if "-" in base:
        c.append(base.replace("-", ""))
    c += ALIASES.get(t, [])
    return list(dict.fromkeys(c))


# ---------- цены ----------
def price_path(code):
    return os.path.join(PRICES, re.sub(r"[^A-Za-z0-9_\-]", "_", code) + ".csv")


def get_prices(code, refresh=False):
    p = price_path(code)
    if refresh or not os.path.exists(p):
        js = api(f"eod/{code}.US", **{"from": START})
        df = pd.DataFrame(js or [], columns=["date", "open", "high", "low", "close", "adjusted_close", "volume"])
        tmp = p + ".part"
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def coverage(df, mdays):
    if len(df) == 0 or len(mdays) == 0:
        return 0.0
    ok = df.dropna(subset=["open", "high", "low", "close", "adjusted_close"])
    ok = ok[(ok.close > 0) & (ok.adjusted_close > 0)]
    return float(mdays.isin(ok["date"]).mean())


def main():
    global KEY
    KEY = read_key()
    hist = load_membership()
    syms = symbol_lists()
    active = set(syms["active"]["Code"].astype(str))
    delisted = set(syms["delisted"]["Code"].astype(str))

    spy = get_prices("SPY")
    if len(spy) == 0:
        sys.exit("Не скачался SPY - дальше нет смысла.")
    get_prices("QQQ")
    days = pd.DatetimeIndex(spy["date"])
    days = days[days >= TEST_FROM]
    mem = membership_days(hist, days)
    mem = {t: d for t, d in mem.items() if len(d)}
    print(f"Тикеров в индексе с {TEST_FROM}: {len(mem)}")

    tasks = {t: candidates(t, active, delisted) for t in mem}
    all_codes = sorted(set(c for v in tasks.values() for c in v))
    todo = [c for c in all_codes if not os.path.exists(price_path(c))]
    print(f"Вариантов кодов: {len(all_codes)}, осталось скачать: {len(todo)}")
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(get_prices, c) for c in todo]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 100 == 0:
                print(f"  скачано {done}/{len(todo)}")

    # Для каждого тикера: лучший по покрытию код берёт свои дни членства,
    # следующие коды добирают оставшиеся дни (тикер мог принадлежать разным компаниям).
    # живые бумаги, чей файл обрывается раньше времени, - скачать заново (EODHD иногда отдаёт неполный ответ)
    last_day = days.max()
    for c in all_codes:
        if c not in active:
            continue
        p = price_path(c)
        if os.path.exists(p):
            d = pd.read_csv(p, usecols=["date"])
            if len(d) and pd.Timestamp(d.date.iloc[-1]) < last_day - pd.Timedelta(days=20):
                for _ in range(3):
                    d2 = get_prices(c, refresh=True)
                    if len(d2) and d2.date.max() >= last_day - pd.Timedelta(days=20):
                        break
                print(f"  перекачан обрезанный файл {c}: теперь до {d2.date.max().date() if len(d2) else '-'}")

    rows = []
    cache = {}
    def dates_of(c):
        if c not in cache:
            df = get_prices(c)
            ok = df.dropna(subset=["open", "high", "low", "close", "adjusted_close"])
            cache[c] = pd.DatetimeIndex(ok[(ok.close > 0) & (ok.adjusted_close > 0)]["date"])
        return cache[c]
    for t, md in sorted(mem.items()):
        cov = {c: md.isin(dates_of(c)) for c in tasks[t]}
        tried = " ".join(f"{c}:{v.mean():.2f}" for c, v in cov.items())
        left = pd.Series(True, index=md)
        parts = []
        for c in sorted(cov, key=lambda c: -cov[c].mean()):
            take = left.values & cov[c]
            need = MIN_COVER * len(md) if not parts else MIN_PART
            if take.sum() == 0 or take.sum() < need:
                continue
            own = md[take]
            parts.append((c, own))
            left[own] = False
        if not parts:
            rows.append(dict(ticker=t, key=t, code="", coverage=round(max(v.mean() for v in cov.values()), 4),
                             found=False, is_delisted=False, member_days=len(md),
                             first_member=md.min().date(), last_member=md.max().date(),
                             assign_from="", assign_to="", tried=tried))
            continue
        for c, own in parts:
            rows.append(dict(ticker=t, key=t if len(parts) == 1 else f"{t}|{c}", code=c,
                             coverage=round(len(own) / len(md), 4), found=True,
                             is_delisted=c not in active, member_days=len(md),
                             first_member=md.min().date(), last_member=md.max().date(),
                             assign_from=own.min().date(), assign_to=own.max().date(), tried=tried))
    mp = pd.DataFrame(rows)
    mp.to_csv(os.path.join(DATA, "mapping.csv"), index=False)

    # покрытие по годам: доля "бумаго-дней в индексе", для которых есть цены
    cov_rows = []
    have = {}
    for t, md in mem.items():
        h = pd.Series(False, index=md)
        for r in mp[(mp.ticker == t) & (mp.found)].itertuples():
            own = md[(md >= pd.Timestamp(r.assign_from)) & (md <= pd.Timestamp(r.assign_to))]
            h[own[own.isin(dates_of(r.code))]] = True
        have[t] = h.values
    for y in sorted(set(days.year)):
        tot = hit = 0
        for t, md in mem.items():
            m = md.year == y
            tot += m.sum()
            hit += have[t][m].sum()
        cov_rows.append((y, tot, hit, hit / tot if tot else 0))

    pd.DataFrame(cov_rows, columns=["year", "member_days", "with_prices", "coverage"]).to_csv(
        os.path.join(DATA, "coverage_by_year.csv"), index=False)
    lines = []
    lines.append(f"Тикеров в S&P 500 с {TEST_FROM}: {len(mp)}")
    f = mp[mp.found]
    lines.append(f"Найдено тикеров: {f.ticker.nunique()}  (кодов EODHD: {len(f)}; тикеров из 2+ компаний: {(f.groupby('ticker').size() > 1).sum()})")
    lines.append(f"  кодов ушедших с биржи (delisted): {f.is_delisted.sum()}")
    lines.append(f"Не найдено: {(~mp.found).sum()}")
    lines.append("")
    lines.append("Покрытие состава по годам (доля бумаго-дней в индексе, где есть цены):")
    for y, tot, hit, fr in cov_rows:
        lines.append(f"  {y}: {fr:6.1%}   ({hit}/{tot})")
    lines.append("")
    lines.append("Не найдены (тикер, годы в индексе, что пробовали):")
    for r in mp[~mp.found].itertuples():
        lines.append(f"  {r.ticker:8s} {r.first_member}..{r.last_member}  {r.tried}")
    txt = "\n".join(lines)
    with open(os.path.join(DATA, "report_download.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print()
    print(txt)
    print("\nОтчёт сохранён: data/report_download.txt")


if __name__ == "__main__":
    main()
