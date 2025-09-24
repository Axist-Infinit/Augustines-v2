# industry_ranker.py
from __future__ import annotations

import json, math, os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Plug-in providers (wire these up from your runner)
#   get_bars_fn(symbol, start_dt, end_dt) -> daily OHLCV DataFrame (DatetimeIndex)
#   components_by_industry_fn() -> dict[str, list[str]]
#   Optional: get_signal_df_fn(symbol) -> your Infinit-Strategies.compute_signals(df)
# ─────────────────────────────────────────────────────────────────────────────
_get_bars: Optional[Callable[[str, datetime, datetime], Optional[pd.DataFrame]]] = None
_components_by_industry: Optional[Callable[[], Dict[str, List[str]]]] = None
_get_signals: Optional[Callable[[str], Optional[pd.DataFrame]]] = None

def register_data_providers(get_bars_fn,
                            components_by_industry_fn,
                            get_signal_df_fn=None) -> None:
    global _get_bars, _components_by_industry, _get_signals
    _get_bars = get_bars_fn
    _components_by_industry = components_by_industry_fn
    _get_signals = get_signal_df_fn

# ─────────────────────────────────────────────────────────────────────────────
# Small indicator pack (local; avoids tight coupling)
# ─────────────────────────────────────────────────────────────────────────────
def _to_f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def ema(s: pd.Series, span: int) -> pd.Series:
    return _to_f(s).ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return _to_f(s).rolling(n).mean()

def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    h, l, c, v = (_to_f(df["high"]), _to_f(df["low"]), _to_f(df["close"]), _to_f(df["volume"]))
    mf_mult = ((c - l) - (h - c)) / (h - l + 1e-12)
    mf_vol = mf_mult * v
    return mf_vol.rolling(period).sum() / (v.rolling(period).sum() + 1e-12)

def obv(df: pd.DataFrame) -> pd.Series:
    c, v = _to_f(df["close"]), _to_f(df["volume"])
    return (np.sign(c.diff().fillna(0.0)) * v).cumsum()

def bb_width(c: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    m = sma(c, n)
    sd = _to_f(c).rolling(n).std()
    return (m + k*sd - (m - k*sd)) / (m.abs() + 1e-12)

# ─────────────────────────────────────────────────────────────────────────────
# Market regime (Ivanhoff): Uptrend / Range / Downtrend → risk scalars
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Regime:
    label: str
    risk_mult: float  # for position sizing

def _regime_for_index(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "Unknown"
    c = _to_f(df["close"])
    e10, s50, s200 = ema(c, 10), sma(c, 50), sma(c, 200)
    last = c.iloc[-1]; e10l, s50l, s200l = e10.iloc[-1], s50.iloc[-1], s200.iloc[-1]
    if (last > s50l) and (s50l > s200l) and (e10l > s50l): return "Up"
    if (last < s50l) and (e10l < s50l) and (s50l < s200l): return "Down"
    return "Range"

def compute_market_regime(get_bars=None) -> Regime:
    """Uses SPY/QQQ/IWM/MDY daily to assign regime and risk multiplier."""
    gb = get_bars or _get_bars
    end = datetime.utcnow(); start = end - timedelta(days=260)
    idx = {}
    for sym in ("SPY","QQQ","IWM","MDY"):
        try: idx[sym] = gb(sym, start, end)
        except Exception: idx[sym] = None
    states = [_regime_for_index(idx.get(k)) for k in ("SPY","QQQ","IWM","MDY")]
    if all(s == "Up" for s in states):   return Regime("Uptrend", 1.00)   # full risk  (Ivanhoff) :contentReference[oaicite:14]{index=14}
    if all(s == "Down" for s in states): return Regime("Downtrend", 0.35) # defensive (Ivanhoff) :contentReference[oaicite:15]{index=15}
    return Regime("Range", 0.60)                                          # half risk  (Ivanhoff)

# ─────────────────────────────────────────────────────────────────────────────
# Component-level features → industry aggregates
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IndMetrics:
    n: int
    ret_1w_med: float
    ret_1m_med: float
    ret_3m_med: float
    pct_above_20: float
    pct_above_50: float
    pct_above_200: float
    pct_near_20d_hi: float
    g4pct_last5: float
    squeeze_density: float
    pre_signal_density: float
    acc_minus_dist_20: float
    cmf_pos_share: float
    obv_up_share: float

def _returns(c: pd.Series, nd: int) -> float:
    if len(c) <= nd or c.iloc[-nd-1] <= 0: return np.nan
    return float(c.iloc[-1]/c.iloc[-nd-1]-1.0)

def _safe(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty: return None
    need = {"open","high","low","close","volume"}
    if not need.issubset(df.columns): return None
    return df.sort_index()

def _gainers_4pct_last5(df: pd.DataFrame) -> int:
    c = _to_f(df["close"])
    r = c.pct_change().tail(5)
    return int((r >= 0.04).sum())

def _acc_dist_flags(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Daily accumulation/distribution flags (up on HV in top/bot of range)."""
    h, l, c, v = (_to_f(df["high"]), _to_f(df["low"]), _to_f(df["close"]), _to_f(df["volume"]))
    rng = (h - l).replace(0, np.nan)
    pos_pos = ((c - l) / rng)   # 0..1
    hv = v >= 1.5 * v.rolling(20).mean()
    up = (c > c.shift(1))
    down = (c < c.shift(1))
    acc = hv & up   & (pos_pos >= 0.65)
    dist= hv & down & (pos_pos <= 0.35)
    return acc.fillna(False), dist.fillna(False)

def _squeeze_flag(c: pd.Series) -> bool:
    w = bb_width(c, 20, 2.0)
    q20 = w.rolling(120, min_periods=40).quantile(0.20)
    return bool((w.iloc[-1] <= q20.iloc[-1]) if (len(w) and not np.isnan(q20.iloc[-1])) else False)

def _near_20d_high(c: pd.Series, pct=0.01) -> bool:
    hi20 = _to_f(c).rolling(20).max()
    if len(c) < 21: return False
    pivot = hi20.iloc[-2]
    return bool(((pivot - c.iloc[-1]) / max(pivot, 1e-6)) <= pct)

def compute_industry_metrics(tickers: List[str],
                             start: datetime,
                             end: datetime) -> Optional[IndMetrics]:
    frames = []
    for t in tickers:
        df = _safe(_get_bars(t, start, end))
        if df is None: 
            continue
        frames.append((t, df))
    if not frames: return None

    # Component stats
    r1w, r1m, r3m = [], [], []
    above20, above50, above200 = 0,0,0
    near20hi = 0
    g4 = 0
    squeeze, pre_sig = 0, 0
    acc_m_d, cmf_pos, obv_up = [], 0, 0

    for t, df in frames:
        c = _to_f(df["close"])
        r1w.append(_returns(c, 5)); r1m.append(_returns(c, 21)); r3m.append(_returns(c, 63))
        s20 = sma(c,20); s50 = sma(c,50); s200 = sma(c,200)
        try:
            last = c.iloc[-1]
            if last > s20.iloc[-1]:  above20 += 1
            if last > s50.iloc[-1]:  above50 += 1
            if last > s200.iloc[-1]: above200 += 1
        except Exception:
            pass
        if _near_20d_high(c, 0.01): near20hi += 1
        g4 += _gainers_4pct_last5(df)

        # Squeeze / pre-signal (near pivot + compression), matches Ivanhoff’s “range contraction” idea
        if _squeeze_flag(c): squeeze += 1
        # Simple pre-signal proxy: squeeze + within 1% of 20d high
        pre_sig += int(_squeeze_flag(c) and _near_20d_high(c, 0.01))

        # Targeted buying proxies
        acc, dist = _acc_dist_flags(df)
        acc_m_d.append(int(acc.tail(20).sum() - dist.tail(20).sum()))
        cmf_pos += int(float(cmf(df).iloc[-1]) > 0)
        # OBV slope up over last 10 bars
        obv_up += int(float(obv(df).diff().tail(10).sum()) > 0)

    n = len(frames)
    if n == 0: return None
    return IndMetrics(
        n=n,
        ret_1w_med=float(np.nanmedian(r1w)),
        ret_1m_med=float(np.nanmedian(r1m)),
        ret_3m_med=float(np.nanmedian(r3m)),
        pct_above_20=above20/n, pct_above_50=above50/n, pct_above_200=above200/n,
        pct_near_20d_hi=near20hi/max(n,1),
        g4pct_last5=g4/max(n,1),
        squeeze_density=squeeze/max(n,1),
        pre_signal_density=pre_sig/max(n,1),
        acc_minus_dist_20=float(np.median(acc_m_d)),
        cmf_pos_share=cmf_pos/n,
        obv_up_share=obv_up/n,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Ranking engine  (weights chosen for 2–3 week horizon)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IndScore:
    industry: str
    sector: Optional[str]
    score: float
    reasons: Dict[str, float]
    metrics: IndMetrics

def _z(x: float, mean: float, std: float) -> float:
    if not np.isfinite(x) or std <= 1e-12: return 0.0
    return (x - mean) / std

def _persistence_bonus(industry: str, top10_json_path: str, lookback_days=14) -> float:
    """Small additive bonus for industries that persist in your daily top10 (Ivanhoff: momentum persists)."""
    if not os.path.exists(top10_json_path): return 0.0
    try:
        arr = json.load(open(top10_json_path, "r"))
    except Exception:
        return 0.0
    # Count appearances in last N snapshots
    count = 0; cutoff_dt = datetime.utcnow() - timedelta(days=lookback_days+2)
    for snap in arr[-60:]:
        dt = datetime.fromisoformat(snap.get("run_at","").split("T")[0] + "T00:00:00") if snap.get("run_at") else None
        if (dt is None) or (dt < cutoff_dt): continue
        for row in snap.get("leaders", []):
            if row.get("Name","") == industry:
                count += 1; break
    # Convert to small 0..+0.05 bonus
    return min(0.05, 0.01 * count)

def rank_industries(components_map: Dict[str, List[str]],
                    sector_map: Optional[Dict[str, str]] = None,
                    top10_json_path: str = "top10_leaders.json",
                    as_of_days: int = 260) -> pd.DataFrame:
    """Return ranked industries with composite score and diagnostics."""
    end = datetime.utcnow(); start = end - timedelta(days=as_of_days)
    rows = []
    # First pass: compute metrics for each industry
    m_by_ind: Dict[str, IndMetrics] = {}
    for ind, tickers in components_map.items():
        met = compute_industry_metrics(tickers, start, end)
        if met: m_by_ind[ind] = met

    if not m_by_ind:
        return pd.DataFrame(columns=["Industry","Sector","Score"])

    # Build arrays for z-scoring
    arr = list(m_by_ind.values())
    def col(fn): return np.array([getattr(x, fn) for x in arr], dtype=float)
    # Core momentum (1m, 3m) + breadth (above 50/200)
    M1, M3 = col("ret_1m_med"), col("ret_3m_med")
    B50, B200 = col("pct_above_50"), col("pct_above_200")
    # Targeted buying proxies
    Amd, CMF, OBVU = col("acc_minus_dist_20"), col("cmf_pos_share"), col("obv_up_share")
    # Setup density
    D_pre, D_sqz, NearHi, G4 = col("pre_signal_density"), col("squeeze_density"), col("pct_near_20d_hi"), col("g4pct_last5")

    # Means/stds
    def ms(x): return float(np.nanmean(x)), float(np.nanstd(x))
    stats = {name: ms(vals) for name, vals in {
        "M1":M1,"M3":M3,"B50":B50,"B200":B200,"Amd":Amd,"CMF":CMF,"OBVU":OBVU,"Dpre":D_pre,"Dsqz":D_sqz,"NearHi":NearHi,"G4":G4
    }.items()}

    regime = compute_market_regime()  # label + risk_mult (Ivanhoff) :contentReference[oaicite:16]{index=16}

    for ind, met in m_by_ind.items():
        # Z-scores
        z = lambda k, val: _z(val, stats[k][0], stats[k][1])
        z_m1, z_m3 = z("M1", met.ret_1m_med), z("M3", met.ret_3m_med)
        z_b50, z_b200 = z("B50", met.pct_above_50), z("B200", met.pct_above_200)
        z_amd, z_cmf, z_obv = z("Amd", met.acc_minus_dist_20), z("CMF", met.cmf_pos_share), z("OBVU", met.obv_up_share)
        z_pre, z_sqz = z("Dpre", met.pre_signal_density), z("Dsqz", met.squeeze_density)
        z_near, z_g4 = z("NearHi", met.pct_near_20d_hi), z("G4", met.g4pct_last5)

        # Weights (swing horizon 2–3 weeks): Core momentum 40%, breadth 20%, targeted buying 25%, setup density 12%, persistence 3%
        core = 0.40 * (0.6*z_m3 + 0.4*z_m1)
        breadth = 0.20 * (0.6*z_b50 + 0.4*z_b200)
        targeted = 0.25 * (0.50*z_amd + 0.25*z_cmf + 0.25*z_obv)
        setups = 0.12 * (0.5*z_pre + 0.3*z_sqz + 0.2*z_near + 0.2*z_g4)
        persist = _persistence_bonus(ind, top10_json_path)  # small additive tilt from your JSON history
        score = (core + breadth + targeted + setups) * regime.risk_mult + persist

        rows.append(dict(
            Industry=ind,
            Sector=(sector_map or {}).get(ind),
            Score=float(score),
            Regime=regime.label,
            reasons=dict(core=float(core), breadth=float(breadth), targeted=float(targeted),
                         setups=float(setups), persistence=float(persist)),
            n_components=met.n,
            metrics=met,
        ))
    out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# JSON writer (append snapshot in your existing shape)
# ─────────────────────────────────────────────────────────────────────────────
def append_topk_to_json(ranked_df: pd.DataFrame,
                        top_k: int = 10,
                        path: str = "top10_leaders.json") -> None:
    leaders = []
    for _, r in ranked_df.head(top_k).iterrows():
        leaders.append({"Name": r["Industry"],
                        "Sector": r.get("Sector"),
                        "RankScore": float(r["Score"])})
    payload = {
        "snapshot_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "run_at": datetime.utcnow().isoformat(timespec="seconds"),
        "top_k": top_k,
        "leaders": leaders
    }
    arr = []
    if os.path.exists(path):
        try: arr = json.load(open(path,"r"))
        except Exception: arr = []
    arr.append(payload)
    with open(path,"w") as f:
        json.dump(arr, f, indent=2)
