"""
Бэктест пробойной системы на акциях S&P 500 (цены EODHD).

Запуск:  python backtest.py          (всё целиком: подготовка, прогон, варианты стопа, тест, отчёт)
         python backtest.py prep     (или run / stress / test / analyze - по этапам)

Правила (не менять без проверки):
  сигнал на закрытии дня N: close > max(high за 55 предыдущих дней) и close > SMA200;
  бумага в S&P 500 на день N; вход по открытию N+1;
  стоп = вход - 2*ATR(20) дня N (фиксированный), при гэпе - по открытию;
  выход по каналу: close < min(low за 20 предыдущих дней) -> продажа на следующем открытии;
  одна позиция на бумагу; издержки 0.25% туда-обратно, переводятся в R.
Все признаки берутся по дню N. Отдельный тест пересчитывает их на данных, обрезанных по день N.

Результат: data/trades.csv (+ trades_stop1.5.csv, trades_stop2.5.csv), data/lookahead_test.txt,
затем запускается analyze.py -> report.txt
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dl_eodhd as dl

DATA = os.path.join(HERE, "data")
TEST_FROM = pd.Timestamp("2007-01-01")
COST = 0.0025
CH_IN, CH_OUT, SMA_N, ATR_N, STOP_K = 55, 20, 200, 20, 2.0


# ---------------- подготовка цен ----------------
def load_raw(code):
    df = pd.read_csv(dl.price_path(code), parse_dates=["date"])
    df = df.dropna(subset=["open", "high", "low", "close", "adjusted_close"])
    df = df[(df.close > 0) & (df.adjusted_close > 0) & (df.open > 0) & (df.low > 0)]
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


MIN_PRICE, MIN_DVOL = 3.0, 1e6   # проверка качества данных: у настоящей бумаги S&P 500 всегда выполняется


def adjust(df):
    # "замороженные" дни: объём 0 и все цены равны - это заглушки, а не торги
    stale = (df.volume.fillna(0) <= 0) & (df.open == df.high) & (df.high == df.low) & (df.low == df.close)
    df = df[~stale].reset_index(drop=True)
    f = df.adjusted_close / df.close
    o, h, l, c = df.open * f, df.high * f, df.low * f, df.adjusted_close
    h = np.maximum.reduce([h, o, c])
    l = np.minimum.reduce([l, o, c])
    return pd.DataFrame({"date": df.date, "O": o, "H": h, "L": l, "C": c,
                         "V": df.volume.astype(float), "rawC": df.close, "fac": f}).reset_index(drop=True)


def rma(x, n):
    """Сглаживание Уайлдера, как ta.atr в TradingView (старт - простое среднее первых n)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    x2 = x.copy()
    x2[:n - 1] = np.nan
    x2[n - 1] = x[:n].mean()
    y = pd.Series(x2).ewm(alpha=1.0 / n, adjust=False, ignore_na=True).mean().values
    out[n - 1:] = y[n - 1:]
    return out


def rma_slow(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def indicators(a):
    """Все индикаторы бумаги. Только прошлое и день N - ничего из будущего."""
    d = a.copy()
    d["hh55"] = d.H.rolling(CH_IN).max().shift(1)
    d["ll20"] = d.L.rolling(CH_OUT).min().shift(1)
    d["sma200"] = d.C.rolling(SMA_N).mean()
    d["sma50"] = d.C.rolling(50).mean()
    pc = d.C.shift(1)
    tr = np.maximum.reduce([d.H - d.L, (d.H - pc).abs().fillna(0), (d.L - pc).abs().fillna(0)])
    d["atr"] = rma(tr, ATR_N)
    # признаки качества бумаги (день N)
    d["atr_pct"] = d.atr / d.C
    d["atr_ratio"] = d.atr / pd.Series(d.atr).rolling(100).mean()      # сжатие ATR
    d["dist52"] = d.C / d.H.rolling(252).max()                          # близость к 52-нед. максимуму
    d["depth"] = d.C / d.hh55 - 1                                       # глубина пробоя на дне N
    d["ret126"] = d.C / d.C.shift(126) - 1
    d["dv20"] = (d.rawC * d.V).rolling(20).median()                     # оборот в $ (качество данных)
    return d


# ---------------- режим рынка ----------------
def distribution_days(c, v, window=25, inval=0.05):
    """Число активных дней распределения (скилл ibd-distribution-day-monitor):
    close упал >= 0.2% при объёме выше вчерашнего; живёт 25 сессий,
    снимается, если индекс поднялся на 5% от закрытия того дня (по максимуму закрытий после)."""
    c = np.asarray(c, float); v = np.asarray(v, float)
    out = np.zeros(len(c))
    active = []   # [индекс, close, макс. close после]
    for i in range(len(c)):
        active = [x for x in active if i - x[0] <= window and x[2] < x[1] * (1 + inval)]
        for x in active:
            x[2] = max(x[2], c[i])
        active = [x for x in active if x[2] < x[1] * (1 + inval)]
        if i > 0 and c[i] <= c[i - 1] * (1 - 0.002) and v[i] > v[i - 1]:
            active.append([i, c[i], -np.inf])
        out[i] = len(active)
    return out


def ftd_recent(c, l, v, corr=0.08, window=25):
    """Упрощённый день подтверждения (скилл ftd-detector): коррекция = SPY ниже
    максимума за 252 дня на 8%+; отсчёт от минимума; день 4..25 с ростом >= 1.25%
    и объёмом выше вчерашнего = FTD. Признак = FTD был за последние 25 сессий
    и минимум коррекции с тех пор не пробит."""
    c = np.asarray(c, float); l = np.asarray(l, float); v = np.asarray(v, float)
    n = len(c); out = np.zeros(n)
    in_corr, low, low_i, ftd_i, ftd_low = False, np.inf, -1, -10**9, 0.0
    for i in range(n):
        hi = c[max(0, i - 251):i + 1].max()
        if not in_corr and c[i] <= hi * (1 - corr):
            in_corr, low, low_i = True, l[i], i
        if in_corr:
            if l[i] < low:
                low, low_i = l[i], i
            day = i - low_i
            if day > 25:                       # подтверждения не было: либо коррекция кончилась, либо новая попытка
                if c[i] > hi * (1 - corr):
                    in_corr = False
                else:
                    low, low_i = l[i], i
                day = i - low_i
            if in_corr and 4 <= day <= 25 and i > 0 and c[i] >= c[i - 1] * 1.0125 and v[i] > v[i - 1]:
                ftd_i, ftd_low, in_corr = i, low, False
        if i - ftd_i <= window and l[i] >= ftd_low:
            out[i] = 1
        elif i - ftd_i <= window:
            ftd_i = -10**9
    return out


def market_features(spy, qqq):
    s = spy.set_index("date")
    m = pd.DataFrame(index=s.index)
    m["spy_above200"] = (s.C > s.C.rolling(200).mean()).astype(float)
    m.loc[s.C.rolling(200).mean().isna(), "spy_above200"] = np.nan
    m["dd25"] = distribution_days(s.C, s.V)
    m["ftd_recent"] = ftd_recent(s.C, s.L, s.V)
    m["spy_ret126"] = s.C / s.C.shift(126) - 1
    q = qqq.set_index("date")
    qa = (q.C > q.C.rolling(200).mean()).astype(float)
    qa[q.C.rolling(200).mean().isna()] = np.nan
    m["qqq_above200"] = qa.reindex(m.index)
    return m


def breadth(ind, mem, dates):
    """Доля бумаг индекса выше SMA50 / SMA200 на каждый день (только члены на этот день)."""
    up50 = pd.Series(0.0, index=dates); up200 = up50.copy(); cnt50 = up50.copy(); cnt200 = up50.copy()
    for t, d in ind.items():
        x = d.set_index("date")
        md = mem[t]
        x = x[x.index.isin(md)]
        for col, up, cnt in (("sma50", up50, cnt50), ("sma200", up200, cnt200)):
            ok = x[col].notna()
            idx = x.index[ok]
            cnt.loc[idx] += 1
            up.loc[idx] += (x.C[ok] > x[col][ok]).astype(float).values
    return pd.DataFrame({"breadth50": up50 / cnt50.replace(0, np.nan),
                         "breadth200": up200 / cnt200.replace(0, np.nan)})


# ---------------- сделки ----------------
def simulate(t, d, md, stop_k=STOP_K, last_date=None, end="end", g=1.0):
    n = len(d)
    O, L, C = d.O.values, d.L.values, d.C.values
    hh, ll, sma, atr = d.hh55.values, d.ll20.values, d.sma200.values, d.atr.values
    dates = d.date.values
    member = d.date.isin(md).values
    with np.errstate(invalid="ignore"):
        sig = (C > hh) & (C > sma) & member & (dates >= np.datetime64(TEST_FROM)) & (atr > 0) \
            & (d.rawC.values >= MIN_PRICE) & (d.dv20.values >= MIN_DVOL)
    out, free = [], 0
    for i in np.flatnonzero(sig):
        if i < free or i + 1 >= n:
            continue
        e = i + 1
        entry = O[e]
        stop = entry - stop_k * atr[i]
        risk = entry - stop
        with np.errstate(invalid="ignore"):
            hs = np.flatnonzero(L[e:] <= stop)
            cs = np.flatnonzero(C[e:] < ll[e:])
        j = e + hs[0] if len(hs) else None
        k = e + cs[0] if len(cs) else None
        if k is not None and k + 1 >= n:      # выход по каналу, но следующего дня в данных нет
            if j is None or j > k:
                j, k = None, None
                kend = True
            else:
                k = None; kend = False
        else:
            kend = False
        if j is not None and (k is None or j <= k + 1):
            if k is not None and j == k + 1:
                x, px, why = j, O[j], "channel"
            else:
                x = j
                px, why = (O[j], "stop_gap") if (j > e and O[j] <= stop) else (stop, "stop")
        elif k is not None:
            x, px, why = k + 1, O[k + 1], "channel"
        else:
            x, px = n - 1, C[n - 1]
            gone = last_date is not None and (last_date - pd.Timestamp(dates[-1])).days > 10
            if end == "break":
                why = "data_break"
            elif end == "gap":
                why = "data_gap"
            else:
                why = "delisted" if gone else ("open" if not kend else "data_end")
        rg = (px - entry) / risk
        cr = COST * entry / risk
        # если разрыв был НАСТОЯЩИМ обвалом: выход по открытию после него (не лучше стопа)
        rg_real = rg
        if why == "data_break" and g < 1:
            rg_real = min(rg, (C[n - 1] * g - entry) / risk)
        out.append(dict(ticker=t, sig_i=i, sig_date=dates[i], entry_date=dates[e], exit_date=dates[x],
                        entry=entry, stop=stop, exit=px, risk_pct=risk / entry, R_gross=rg,
                        cost_R=cr, R=rg - cr, R_if_real=rg_real - cr, reason=why, bars=x - e + 1,
                        atr_pct=d.atr_pct.values[i], atr_ratio=d.atr_ratio.values[i],
                        dist52=d.dist52.values[i], depth=d.depth.values[i],
                        ret126=d.ret126.values[i],
                        # ЛОВУШКА: глубина пробоя по закрытию N+1 - известна только после входа
                        depth_next=C[e] / hh[i] - 1))
        free = x if why in ("channel", "stop", "stop_gap") else n
    return out


GAP = 20   # разрыв в данных больше 20 торговых дней SPY = новый кусок истории (другая компания / дыра)


FRACS = np.array([1/10, 1/8, 1/5, 1/4, 1/3, 1/2, 2/3, 3/2, 2, 3, 4, 5, 8, 10])


def breaks(a):
    """Индексы дней, с которых начинается новый кусок из-за ошибки данных:
    прыжок в "сплитовую" пропорцию (>40%) или выброс туда-обратно.
    Возвращает dict {индекс: отношение цены (открытие / прошлое закрытие)}."""
    O, C = a.O.values, a.C.values
    out = {}
    if len(a) < 3:
        return out
    g = O[1:] / C[:-1]
    cr = C[1:] / C[:-1]
    for j in np.flatnonzero((g < 1 / 1.4) | (g > 1.4)):
        near = np.min(np.abs(g[j] / FRACS - 1))
        if near < 0.03:
            out[j + 1] = g[j]
    fr = a.fac.values[1:] / a.fac.values[:-1]
    for j in np.flatnonzero((np.abs(np.log(fr)) > np.log(1.25)) & (np.abs(np.log(g)) > np.log(1.3))):
        out.setdefault(j + 1, g[j])
    lr = np.log(cr)
    for j in np.flatnonzero((np.abs(lr[:-1]) > 0.4) & (np.abs(lr[1:]) > 0.4) & (np.sign(lr[:-1]) != np.sign(lr[1:]))):
        out.setdefault(j + 1, g[j])
        out.setdefault(j + 2, g[j + 1] if j + 1 < len(g) else 1.0)
    return out


def segments(code, spy_days):
    """Скорректированные цены кода, разрезанные на куски по разрывам в датах и по ошибкам данных.
    Список (кусок, как_закончился, отношение_цены_на_разрыве)."""
    a = adjust(load_raw(code))
    if len(a) == 0:
        return []
    pos = spy_days.searchsorted(a.date.values)
    cut = np.zeros(len(a), bool)
    cut[1:] = np.diff(pos) > GAP
    kind = {}
    for i in np.flatnonzero(cut):
        kind[i] = ("gap", 1.0)
    for i, g in breaks(a).items():
        if not cut[i]:
            cut[i] = True
            kind[i] = ("break", g)
    seg = np.cumsum(cut)
    starts = list(np.flatnonzero(cut))
    res = []
    for k, (_, gdf) in enumerate(a.groupby(seg)):
        nxt = starts[k] if k < len(starts) else None
        end, g = kind.get(nxt, ("end", 1.0)) if nxt is not None else ("end", 1.0)
        res.append((gdf.reset_index(drop=True), end, g))
    return res


def load_all():
    mp = pd.read_csv(os.path.join(DATA, "mapping.csv"), keep_default_na=False)
    mp = mp[mp.found.astype(str) == "True"]
    spy = adjust(load_raw("SPY")); qqq = adjust(load_raw("QQQ"))
    hist = dl.load_membership()
    days = pd.DatetimeIndex(spy.date)
    mem0 = dl.membership_days(hist, days[days >= TEST_FROM])
    ind, mem, meta = {}, {}, {}
    for r in mp.itertuples():
        md = mem0[r.ticker]
        md = md[(md >= pd.Timestamp(r.assign_from)) & (md <= pd.Timestamp(r.assign_to))]
        for k, (seg, end, g) in enumerate(segments(r.code, days)):
            if len(seg) < 60:
                continue
            sm = md[(md >= seg.date.iloc[0]) & (md <= seg.date.iloc[-1])]
            if len(sm) == 0:
                continue
            px = seg.rawC[seg.date.isin(sm)].median()
            if not px >= 2.0:
                print(f"  ПОДОЗРИТЕЛЬНЫЙ КОД, пропущен: {r.ticker} -> {r.code} (цена в период членства ~{px:.2f}$)")
                continue
            sid = f"{r.key}#{k}"
            ind[sid] = indicators(seg)
            mem[sid] = sm
            meta[sid] = dict(ticker=r.ticker, code=r.code, seg=k, end=end, g=g,
                             is_delisted=str(r.is_delisted) == "True")
    return mp, spy, qqq, mem, ind, meta


def run(stop_k, meta, spy, mem, ind, mkt, br):
    last = pd.Timestamp(spy.date.iloc[-1])
    rows = []
    for sid, d in ind.items():
        rows += simulate(sid, d, mem[sid], stop_k, last, meta[sid]["end"], meta[sid]["g"])
    tr = pd.DataFrame(rows).rename(columns={"ticker": "sid"})
    tr["ticker"] = tr.sid.map(lambda x: meta[x]["ticker"])
    tr["code"] = tr.sid.map(lambda x: meta[x]["code"])
    tr["seg"] = tr.sid.map(lambda x: meta[x]["seg"])
    tr["is_delisted"] = tr.sid.map(lambda x: meta[x]["is_delisted"])
    tr = tr.join(mkt, on="sig_date").join(br, on="sig_date")
    spy_r = spy.set_index("date").C
    tr["rs126"] = tr.ret126 - tr.sig_date.map(spy_r / spy_r.shift(126) - 1)
    return tr.sort_values(["entry_date", "ticker"]).reset_index(drop=True)


# ---------------- тест на заглядывание вперёд ----------------
FEATS_T = ["atr_pct", "atr_ratio", "dist52", "depth", "ret126", "depth_next"]
FEATS_M = ["spy_above200", "dd25", "ftd_recent", "spy_ret126", "qqq_above200"]


def same(a, b):
    if (a is None or pd.isna(a)) and (b is None or pd.isna(b)):
        return True
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= 1e-9 * max(1.0, abs(a))


def lookahead_test(tr, meta, spy, qqq, mem, br, n_trades=300, n_dates=15, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.DatetimeIndex(spy.date)
    segc = {}
    def seg_of(sid):
        m = meta[sid]
        if m["code"] not in segc:
            segc[m["code"]] = segments(m["code"], days)
        return segc[m["code"]][m["seg"]][0]
    smp = tr.iloc[rng.choice(len(tr), min(n_trades, len(tr)), replace=False)]
    bad = {f: 0 for f in FEATS_T + FEATS_M + ["signal", "stop", "entry_is_next_open", "breadth50", "breadth200"]}
    for r in smp.itertuples():
        N = pd.Timestamp(r.sig_date)
        full = seg_of(r.sid)
        cut = indicators(full[full.date <= N].reset_index(drop=True))
        last = cut.iloc[-1]
        sig_ok = (last.C > last.hh55) and (last.C > last.sma200) and (N in mem[r.sid]) \
            and last.rawC >= MIN_PRICE and last.dv20 >= MIN_DVOL
        bad["signal"] += not sig_ok
        nxt = full[full.date > N].iloc[0]
        bad["entry_is_next_open"] += not (same(nxt.O, r.entry) and pd.Timestamp(r.entry_date) == nxt.date)
        bad["stop"] += not same(r.entry - STOP_K * last.atr, r.stop)
        for f in FEATS_T:
            v = last[f] if f in cut.columns else np.nan     # depth_next на обрезанных данных не посчитать
            bad[f] += not same(v, getattr(r, f))
        # режим рынка на данных до N
        mk = market_features(spy[spy.date <= N].reset_index(drop=True),
                             qqq[qqq.date <= N].reset_index(drop=True)).iloc[-1]
        for f in FEATS_M:
            bad[f] += not same(mk[f], getattr(r, f))
    # широта: полный пересчёт на обрезанных данных для нескольких дат
    dts = sorted(rng.choice(pd.DatetimeIndex(tr.sig_date.unique()), n_dates, replace=False))
    for N in dts:
        N = pd.Timestamp(N)
        up = {50: [0, 0], 200: [0, 0]}
        for t in mem:
            if N not in mem[t]:
                continue
            a = seg_of(t)
            a = a[a.date <= N]
            if len(a) == 0 or a.date.iloc[-1] != N:
                continue
            for k in (50, 200):
                s = a.C.rolling(k).mean().iloc[-1]
                if pd.notna(s):
                    up[k][1] += 1; up[k][0] += a.C.iloc[-1] > s
        bad["breadth50"] += not same(up[50][0] / up[50][1], br.loc[N, "breadth50"])
        bad["breadth200"] += not same(up[200][0] / up[200][1], br.loc[N, "breadth200"])
    lines = [f"Проверка заглядывания вперёд: {len(smp)} случайных сделок, {len(dts)} дат для широты.",
             "Каждый признак пересчитан на данных, ОБРЕЗАННЫХ по день сигнала N, и сравнён с тем, что в бэктесте.",
             ""]
    for f, b in bad.items():
        tot = len(dts) if f.startswith("breadth") else len(smp)
        if f == "depth_next":
            verdict = ("ЛОВУШКА ПОЙМАНА (так и должно быть: признак нельзя посчитать до N+1)"
                       if b == tot else "ОШИБКА ТЕСТА: ловушка не поймана")
        else:
            verdict = "ок" if b == 0 else "ЗАГЛЯДЫВАЕТ ВПЕРЁД / НЕ СОВПАДАЕТ"
        lines.append(f"  {f:20s} расхождений {b:4d} из {tot:4d}  -> {verdict}")
    passed = all(b == 0 for f, b in bad.items() if f != "depth_next") and bad["depth_next"] == len(smp)
    lines += ["", "ИТОГ ТЕСТА: " + ("ПРОЙДЕН" if passed else "НЕ ПРОЙДЕН")]
    txt = "\n".join(lines)
    with open(os.path.join(DATA, "lookahead_test.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return passed


CACHE = os.path.join(DATA, "cache_prep.pkl")


def stage_prep():
    import pickle
    print("Загружаю цены и считаю индикаторы...")
    mp, spy, qqq, mem, ind, meta = load_all()
    print(f"  кусков истории: {len(ind)} (бумаг: {len(set(m['ticker'] for m in meta.values()))})")
    mkt = market_features(spy, qqq)
    print("Считаю широту рынка...")
    br = breadth(ind, mem, pd.DatetimeIndex(spy.date))
    with open(CACHE, "wb") as f:
        pickle.dump(dict(spy=spy, qqq=qqq, mem=mem, ind=ind, meta=meta, mkt=mkt, br=br), f)


def load_cache():
    import pickle
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def stage_run(ks):
    c = load_cache()
    for k in ks:
        print(f"Прогон: стоп {k} ATR...")
        tr = run(k, c["meta"], c["spy"], c["mem"], c["ind"], c["mkt"], c["br"])
        name = "trades.csv" if k == STOP_K else f"trades_stop{k}.csv"
        tr.to_csv(os.path.join(DATA, name), index=False)
        print(f"  сделок: {len(tr)} -> data/{name}")


def stage_test():
    c = load_cache()
    tr = pd.read_csv(os.path.join(DATA, "trades.csv"), parse_dates=["sig_date", "entry_date", "exit_date"])
    print("Тест на заглядывание вперёд...")
    lookahead_test(tr, c["meta"], c["spy"], c["qqq"], c["mem"], c["br"])


def main():
    """python backtest.py            - всё целиком
       python backtest.py prep|run|stress|test|analyze  - по этапам"""
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("prep", "all"):
        stage_prep()
    if st in ("run", "all"):
        stage_run([STOP_K])
    if st in ("stress", "all"):
        stage_run([1.5, 2.5])
    if st in ("test", "all"):
        stage_test()
    if st in ("analyze", "all"):
        import analyze
        analyze.main()


if __name__ == "__main__":
    main()
