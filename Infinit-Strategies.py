# Infinit-Strategies.py
# Minimal, expert-guided OBV / MFI / CMF integration for conviction and dynamic stops/targets
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple
from datetime import date

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Public API (imported by the main runner)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Public API (imported by the main runner)
# ─────────────────────────────────────────────────────────────────────────────
TOS_MOMENTUM      = "tos_momentum"
BREAKOUT_CONSOL   = "breakout_consolidation"
PULLBACK_HG       = "pullback_holy_grail"
SQUEEZE_BB        = "squeeze_bollinger"
PATTERN_FORECAST  = "pattern_forecast"

STRATEGY_SLUGS = [
    TOS_MOMENTUM,
    BREAKOUT_CONSOL,
    PULLBACK_HG,
    SQUEEZE_BB,
    PATTERN_FORECAST,
]

STRATEGY_LABELS = {
    TOS_MOMENTUM:       "ToS Momentum",
    BREAKOUT_CONSOL:    "Breakout (Trend + Tight Base)",
    PULLBACK_HG:        "Pullback (Holy Grail)",
    SQUEEZE_BB:         "Volatility Squeeze (BB)",
    PATTERN_FORECAST:   "Pattern + Forecast (Triangle/Flag)",
}

DEFAULT_STRATEGY = "all"
STRATEGY_LABELS[PATTERN_FORECAST] = "Pattern + Forecast (Triangle/Flag)"


# Data providers are registered by the main script
_get_bars: Optional[Callable[[str, pd.Timestamp, pd.Timestamp, int], Optional[pd.DataFrame]]] = None
_fred_get: Optional[Callable[[str, int], pd.Series]] = None

def register_data_providers(get_bars_fn, fred_get_fn) -> None:
    """
    Register data providers supplied by the main script.
    """
    global _get_bars, _fred_get
    _get_bars = get_bars_fn
    _fred_get = fred_get_fn


# —— Tunables (env-var overridable) ──────────────────────────────────────────
MIN_PRICE               = float(os.getenv("MIN_PRICE",               "5"))
MIN_AVG_SHARES_20D      = int(float(os.getenv("MIN_AVG_SHARES_20D",  "500000")))
MIN_AVG_DOLLAR_20D      = float(os.getenv("MIN_AVG_DOLLAR_20D",      "5000000"))  # $5M

MIN_VOL_SCORE_BY_STRAT  = {
    BREAKOUT_CONSOL: int(os.getenv("MIN_VOL_SCORE_BREAKOUT", "1")),
    PULLBACK_HG:     int(os.getenv("MIN_VOL_SCORE_HG",       "1")),
    SQUEEZE_BB:      int(os.getenv("MIN_VOL_SCORE_SQUEEZE",  "1")),
    TOS_MOMENTUM:    int(os.getenv("MIN_VOL_SCORE_TOS",      "1")),
}
MIN_RR_BY_STRAT         = {
    BREAKOUT_CONSOL: float(os.getenv("MIN_RR_BREAKOUT", "1.8")),
    PULLBACK_HG:     float(os.getenv("MIN_RR_HG",       "1.6")),
    SQUEEZE_BB:      float(os.getenv("MIN_RR_SQUEEZE",  "1.8")),
    TOS_MOMENTUM:    float(os.getenv("MIN_RR_TOS",      "1.6")),
}
MIN_ADX_BY_STRAT        = {
    BREAKOUT_CONSOL: float(os.getenv("MIN_ADX_BREAKOUT", "18")),
    PULLBACK_HG:     float(os.getenv("MIN_ADX_HG",       "20")),
    SQUEEZE_BB:      float(os.getenv("MIN_ADX_SQUEEZE",  "18")),
    TOS_MOMENTUM:    float(os.getenv("MIN_ADX_TOS",      "12")),
}


PATTERN_FORECAST  = "pattern_forecast"
STRATEGY_SLUGS    = [TOS_MOMENTUM, BREAKOUT_CONSOL, PULLBACK_HG, SQUEEZE_BB, PATTERN_FORECAST]

MIN_VOL_SCORE_BY_STRAT[PATTERN_FORECAST] = int(os.getenv("MIN_VOL_SCORE_PATTERN", "1"))
MIN_RR_BY_STRAT[PATTERN_FORECAST]        = float(os.getenv("MIN_RR_PATTERN", "1.8"))
MIN_ADX_BY_STRAT[PATTERN_FORECAST]       = float(os.getenv("MIN_ADX_PATTERN", "16"))
PATTERN_NEAR_PCT = float(os.getenv("PATTERN_NEAR_PCT", "0.006"))   # within 0.6% of pivot
PATTERN_BUY_EPS  = float(os.getenv("PATTERN_BUY_EPS",  "0.003"))   # ±0.3% around pivot
FLAG_MIN_RUN21   = float(os.getenv("FLAG_MIN_RUN21",   "0.12"))    # prior 21d run ≥ 12%

# ── Multi-timeframe gating (weekly → daily) ----------------------------------
MTA_REQUIRE_BY_STRAT = {
    BREAKOUT_CONSOL: os.getenv("MTA_REQ_BREAKOUT", "1").lower() in ("1","true","yes"),
    SQUEEZE_BB:      os.getenv("MTA_REQ_SQUEEZE",  "1").lower() in ("1","true","yes"),
    PULLBACK_HG:     os.getenv("MTA_REQ_HG",       "0").lower() in ("1","true","yes"),
    TOS_MOMENTUM:    os.getenv("MTA_REQ_TOS",      "0").lower() in ("1","true","yes"),
    PATTERN_FORECAST:os.getenv("MTA_REQ_PATTERN",  "1").lower() in ("1","true","yes"),
}

def enforce_min_rr(entry: pd.Series | float,
                   stop0: pd.Series | float,
                   target0: pd.Series | float,
                   min_rr: float = 2.0):
    """
    Lift target0 where needed so that (target0 - entry) / (entry - stop0) >= min_rr.
    Works elementwise for Series.
    """
    e = pd.Series(entry)  if not isinstance(entry,  pd.Series) else entry
    s = pd.Series(stop0)  if not isinstance(stop0,  pd.Series) else stop0
    t = pd.Series(target0)if not isinstance(target0,pd.Series) else target0
    risk = (e - s).clip(lower=1e-6)
    req = e + (min_rr * risk)
    return pd.Series(np.maximum(t.values, req.values), index=t.index)

def _compute_weekly_context(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Resample daily OHLCV to weekly (Fri) and compute weekly trend/momentum pack.
    Returns dict aligned (ffill) to daily index.
    """
    if df.empty:
        idx = df.index
        return {k: pd.Series(index=idx, dtype=float) for k in
                ("w_close","w_sma20","w_sma50","w_hma20","w_adx","weekly_ok")}

    wk = (df[["open","high","low","close","volume"]]
            .resample("W-FRI")
            .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
            .dropna(how="any"))

    # Weekly indicators
    def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
    def _wma(s, length):
        w = np.arange(1, length+1, dtype=float)
        return s.rolling(length).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    def _hma(s, length):
        L = int(length); 
        if L < 2: return s*0.0*np.nan
        return _wma(2*_wma(s, L//2) - _wma(s, L), int(np.sqrt(L)))

    w_close = wk["close"]
    w_sma20 = w_close.rolling(20, min_periods=10).mean()
    w_sma50 = w_close.rolling(50, min_periods=25).mean()
    w_hma20 = _hma(w_close, 20)
    # reuse DMI on weekly bars
    pdi, mdi, w_adx = dmi_components(wk, 14)

    w_trend   = (w_close > w_sma20) & (w_sma20 > w_sma50)
    w_momentum= (w_hma20 > w_hma20.shift(1))
    w_adx_ok  = (w_adx >= 18)

    weekly_ok_w = (w_trend & w_momentum & w_adx_ok).astype(float)  # float→Series for merge ease

    # Align to daily index
    out = {
        "w_close":  w_close.reindex(df.index, method="ffill"),
        "w_sma20":  w_sma20.reindex(df.index, method="ffill"),
        "w_sma50":  w_sma50.reindex(df.index, method="ffill"),
        "w_hma20":  w_hma20.reindex(df.index, method="ffill"),
        "w_adx":    w_adx.reindex(df.index, method="ffill"),
        "weekly_ok":weekly_ok_w.reindex(df.index, method="ffill").fillna(0.0).astype(bool)
    }
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / Indicators
# ─────────────────────────────────────────────────────────────────────────────
def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def ema(s: pd.Series, span: int) -> pd.Series:
    return _to_float_series(s).ewm(span=span, adjust=False).mean()

def wma(s: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length+1, dtype=float)
    return _to_float_series(s).rolling(length).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def hma(s: pd.Series, length: int) -> pd.Series:
    L = int(length)
    if L < 2: 
        return _to_float_series(s) * np.nan
    w1 = wma(s, max(L//2, 1))
    w2 = wma(s, L)
    diff = 2*w1 - w2
    return wma(diff, max(int(math.sqrt(L)), 1))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _to_float_series(df["high"])
    low  = _to_float_series(df["low"])
    close= _to_float_series(df["close"])
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def dmi_components(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (+DI, -DI, ADX)
    """
    high = _to_float_series(df["high"])
    low  = _to_float_series(df["low"])
    close= _to_float_series(df["close"])

    up = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0),  up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

    # Wilder's smoothing
    def rma(x: pd.Series, n: int) -> pd.Series:
        return x.ewm(alpha=1/n, adjust=False).mean()

    atr_r = rma(tr, period)
    pdi = 100 * (rma(pd.Series(plus_dm, index=df.index), period) / (atr_r + 1e-12))
    mdi = 100 * (rma(pd.Series(minus_dm, index=df.index), period) / (atr_r + 1e-12))
    dx  = 100 * (abs(pdi - mdi) / (pdi + mdi + 1e-12))
    adx = rma(dx, period)
    return pdi, mdi, adx

def bollinger_bands(s: pd.Series, length: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = _to_float_series(s).rolling(length).mean()
    sd = _to_float_series(s).rolling(length).std()
    upper = m + std_mult*sd
    lower = m - std_mult*sd
    return lower, m, upper

def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    c = _to_float_series(close); v = _to_float_series(volume)
    dir_ = np.sign(c.diff().fillna(0.0))
    obv = (dir_ * v).cumsum()
    return obv

def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    h = _to_float_series(high); l = _to_float_series(low); c = _to_float_series(close); v = _to_float_series(volume)
    tp = (h + l + c) / 3.0
    pos_flow = np.where(tp > tp.shift(1), tp * v, 0.0)
    neg_flow = np.where(tp < tp.shift(1), tp * v, 0.0)
    pos_sum = pd.Series(pos_flow, index=tp.index).rolling(period).sum()
    neg_sum = pd.Series(neg_flow, index=tp.index).rolling(period).sum()
    # MFI = 100 - 100/(1 + money_flow_ratio)
    mfr = pos_sum / (neg_sum + 1e-12)
    mfi = 100 - 100 / (1 + mfr)
    return mfi

def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    h = _to_float_series(high); l = _to_float_series(low); c = _to_float_series(close); v = _to_float_series(volume)
    mf_mult = ((c - l) - (h - c)) / (h - l + 1e-12)
    mf_vol  = mf_mult * v
    cmf = mf_vol.rolling(period).sum() / (v.rolling(period).sum() + 1e-12)
    return cmf

def _volume_conviction_pack(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute OBV, MFI(14), CMF(20) and simple bullish/rising flags.
    Returns dict with:
      - obv, mfi, cmf
      - obv_rising, mfi_bullish, cmf_bullish
      - vol_score (0..3)
    """
    close = df["close"]; high = df["high"]; low = df["low"]; vol = df["volume"]

    obv = on_balance_volume(close, vol)
    obv_ema10 = ema(obv, 10)
    obv_rising = (obv > obv_ema10) & (obv.diff() > 0)

    mfi = money_flow_index(high, low, close, vol, period=14)
    mfi_bullish = (mfi > 50) & (mfi.diff() > 0)

    cmf = chaikin_money_flow(high, low, close, vol, period=20)
    cmf_bullish = (cmf > 0) & (cmf.diff() > 0)

    score = (obv_rising.astype(int) + mfi_bullish.astype(int) + cmf_bullish.astype(int)).clip(0, 3)

    return dict(
        obv=obv, mfi=mfi, cmf=cmf,
        obv_rising=obv_rising, mfi_bullish=mfi_bullish, cmf_bullish=cmf_bullish,
        vol_score=score
    )

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    need = ["open","high","low","close","volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"OHLCV columns missing: {miss}")
    out = df.copy()
    for c in need:
        out[c] = _to_float_series(out[c])
    if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("DataFrame index must be datetime-like.")
    return out.sort_index()

def _safe_min(a: float, b: float) -> float:
    if a is None or not np.isfinite(a): return b
    if b is None or not np.isfinite(b): return a
    return min(a, b)

def _safe_max(a: float, b: float) -> float:
    if a is None or not np.isfinite(a): return b
    if b is None or not np.isfinite(b): return a
    return max(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Core signal engine
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StrategyPack:
    sig: pd.Series           # boolean signal series
    stop0: pd.Series         # initial stop (per bar)
    target0: pd.Series       # initial target (per bar)
    rr: pd.Series            # (target - entry) / (entry - stop)
    label: str
    slug: str

def _compute_common(df: pd.DataFrame) -> Dict[str, pd.Series]:
    close = df["close"]; high = df["high"]; low = df["low"]; vol = df["volume"]

    ema10 = ema(close, 10)
    ema20 = ema(close, 20)
    sma50 = close.rolling(50).mean()
    sma200= close.rolling(200).mean()
    atr14 = atr(df, 14)

    # Bands & width
    bb_lo, bb_mid, bb_up = bollinger_bands(close, length=20, std_mult=2.0)
    bb_width = (bb_up - bb_lo) / (bb_mid.replace(0, np.nan).abs() + 1e-12)

    # Vol averages
    vol_sma20 = vol.rolling(20).mean()
    vol_sma50 = vol.rolling(50).mean()

    # HMA20, MACD hist
    hma20 = hma(close, 20)
    macd_line = ema(close, 12) - ema(close, 26)
    macd_sig  = ema(macd_line, 9)
    macd_hist = macd_line - macd_sig

    # DMI / ADX
    pdi, mdi, adx = dmi_components(df, 14)
    dmi_diff = pdi - mdi

    # 20/50 day highs/lows for quick pivots
    hi10  = high.rolling(10).max()
    lo10  = low.rolling(10).min()
    hi20  = high.rolling(20).max()
    lo20  = low.rolling(20).min()
    hi50  = high.rolling(50).max()

    return dict(
        ema10=ema10, ema20=ema20, sma50=sma50, sma200=sma200, atr14=atr14,
        bb_lo=bb_lo, bb_mid=bb_mid, bb_up=bb_up, bb_width=bb_width,
        vol_sma20=vol_sma20, vol_sma50=vol_sma50,
        hma20=hma20, macd_hist=macd_hist, dmi_diff=dmi_diff, adx=adx,
        hi10=hi10, lo10=lo10, hi20=hi20, lo20=lo20, hi50=hi50
    )

def _cap_pos(x: pd.Series) -> pd.Series:
    # Replace non-sensical or zero/negative risk cases with NaN
    return x.where(x > 0, np.nan)

def _build_pack(df: pd.DataFrame,
                sig: pd.Series,
                stop0: pd.Series,
                target0: pd.Series,
                label: str,
                slug: str) -> StrategyPack:
    close = df["close"]
    risk = _cap_pos(close - stop0)
    reward = (target0 - close)
    rr = (reward / risk).where((reward > 0) & risk.notna())
    return StrategyPack(sig=sig.fillna(False), stop0=stop0, target0=target0, rr=rr, label=label, slug=slug)

def _strategy_breakout(df: pd.DataFrame, C: Dict[str, pd.Series], V: Dict[str, pd.Series]) -> StrategyPack:
    """
    Trend-confirmed breakout from tight consolidation.
    Minimal volume gate: require (vol_score >= 1) OR (volume_surge).
    Stops/targets slightly adapted by V['vol_score'].
    """
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    ema10, ema20, sma50, sma200 = C["ema10"], C["ema20"], C["sma50"], C["sma200"]
    atr14, bb_width = C["atr14"], C["bb_width"]
    hi20, lo20 = C["hi20"], C["lo20"]
    vol_sma20 = C["vol_sma20"]

    uptrend = (close > sma50) & (sma50 > sma200) & (ema20 > ema20.shift(1))
    # Tightness proxy: BB width in the lower 20% of last 120 sessions
    bw_q20 = bb_width.rolling(120, min_periods=40).quantile(0.20)
    tight  = bb_width <= bw_q20

    # Breakout: new 20d high; prefer >4% day or volume surge
    new_hi = close >= hi20.shift(1)
    pct_day = (close / close.shift(1) - 1.0)
    vol_surge = vol >= (1.5 * vol_sma20)

    # Volume conviction
    score = V["vol_score"]
    vol_gate = (score >= 1) | vol_surge

    sig = uptrend & tight & new_hi & ( (pct_day >= 0.04) | vol_gate )

    # Initial dynamic levels
    entry = close
    # stop ~ just below today's low minus ~1.0 ATR, slightly tighter with volume conviction
    stop_k = (1.00 - 0.06 * score.clip(0, 3))  # 1.00, 0.94, 0.88, 0.82
    stop0 = (low - (stop_k * atr14)).clip(upper=entry - 1e-4)

    # target ~ measured move vs. ATR; base width = recent 20d range
    width20 = (hi20 - lo20).clip(lower=0)
    tgt_mult = (2.00 + 0.25 * score.clip(0, 3))  # 2.00, 2.25, 2.50, 2.75 ATR
    target0 = entry + np.maximum(tgt_mult * atr14, 0.75 * width20)

    return _build_pack(df, sig, stop0, target0, STRATEGY_LABELS[BREAKOUT_CONSOL], BREAKOUT_CONSOL)

def _strategy_pullback_hg(df: pd.DataFrame, C: Dict[str, pd.Series], V: Dict[str, pd.Series]) -> StrategyPack:
    """
    Linda Raschke "Holy Grail": strong uptrend, touch 20EMA, then turn up.
    Volume conviction refines stop0/target0 but does not hard‑gate the signal.
    """
    close, high, low = df["close"], df["high"], df["low"]
    ema10, ema20, sma50, sma200 = C["ema10"], C["ema20"], C["sma50"], C["sma200"]
    atr14, dmi_diff, adx = C["atr14"], C["dmi_diff"], C["adx"]
    hi20 = C["hi20"]

    strong_trend = (close > sma50) & (sma50 > sma200) & (adx > 25) & (dmi_diff > 0)
    touched_20   = (low <= ema20) | (close <= ema20 * 1.01)
    # Confirmation: close back above prior day's high OR back above EMA10
    confirm_up   = (close > high.shift(1)) | (close > ema10)

    sig = strong_trend & touched_20.rolling(3, min_periods=1).max().astype(bool) & confirm_up

    # Initial levels: stop below pullback swing low with 0.5 ATR buffer (modulated)
    entry = close
    recent_pullback_low = low.rolling(5).min()
    stop_k = (0.55 - 0.05 * V["vol_score"].clip(0, 3)).clip(0.35, 0.60)  # 0.55 → 0.40 as score rises
    stop0 = (recent_pullback_low - stop_k * atr14).clip(upper=entry - 1e-4)

    # Target: prior swing high (pre‑pullback area) or +ATR multiple
    base_tgt = hi20.shift(1)  # conservative: previous 20d high
    tgt_mult = (1.75 + 0.25 * V["vol_score"].clip(0, 3))  # 1.75 → 2.50 ATR
    target0 = np.maximum(base_tgt, entry + tgt_mult * atr14)
    target0 = enforce_min_rr(entry, stop0, target0, MIN_RR_BY_STRAT.get(PULLBACK_HG, 2.0)) 

    return _build_pack(df, sig, stop0, target0, STRATEGY_LABELS[PULLBACK_HG], PULLBACK_HG)

def _strategy_squeeze_bb(df: pd.DataFrame, C: Dict[str, pd.Series], V: Dict[str, pd.Series]) -> StrategyPack:
    """
    Bollinger Band squeeze → expansion breakout.
    Minimal volume gate: require (vol_score >= 1) OR (volume_surge) on breakout.
    """
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    bb_lo, bb_mid, bb_up, bb_width = C["bb_lo"], C["bb_mid"], C["bb_up"], C["bb_width"]
    atr14, hi10, lo10 = C["atr14"], C["hi10"], C["lo10"]
    vol_sma20 = C["vol_sma20"]

    # Squeeze: width in lowest 20% of last 120d
    bw_q20 = bb_width.rolling(120, min_periods=40).quantile(0.20)
    squeeze = bb_width <= bw_q20

    # Breakout: close > upper band OR new 10d high
    breakout = (close > bb_up) | (close >= hi10.shift(1))

    vol_surge = vol >= (1.5 * vol_sma20)
    score = V["vol_score"]
    vol_gate = (score >= 1) | vol_surge

    sig = squeeze & breakout & vol_gate

    entry = close
    # Stop: below lower band or range low with ATR buffer; slightly tighter with score
    stop_k = (1.00 - 0.08 * score.clip(0, 3))  # 1.00 → 0.76 ATR
    base_stop = np.minimum(bb_lo, lo10)
    stop0 = (base_stop - stop_k * atr14).clip(upper=entry - 1e-4)

    # Target: measured move of the recent 10–20 bar range or ATR multiple
    width = (hi10 - lo10).clip(lower=0)
    tgt_mult = (1.80 + 0.30 * score.clip(0, 3))  # 1.80 → 2.70 ATR
    target0 = entry + np.maximum(tgt_mult * atr14, 0.80 * width)
    target0 = enforce_min_rr(entry, stop0, target0, MIN_RR_BY_STRAT.get(SQUEEZE_BB, 2.0)) 

    return _build_pack(df, sig, stop0, target0, STRATEGY_LABELS[SQUEEZE_BB], SQUEEZE_BB)

def _strategy_tos_momentum(df: pd.DataFrame, C: Dict[str, pd.Series], V: Dict[str, pd.Series]) -> StrategyPack:
    """
    ToS-style momentum cluster: HMA rising, MACD hist rising/positive, +DMI bias,
    price above EMA20 & SMA50. Volume conviction refines stop/target but doesn't gate.
    """
    close = df["close"]
    hma20, macd_hist, dmi_diff = C["hma20"], C["macd_hist"], C["dmi_diff"]
    ema20, sma50 = C["ema20"], C["sma50"]
    atr14 = C["atr14"]

    trend_ok = (close > ema20) & (close > sma50)
    hma_up   = hma20 > hma20.shift(1)
    macd_ok  = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))
    dmi_ok   = (dmi_diff > 0)

    sig = trend_ok & hma_up & macd_ok & dmi_ok

    entry = close
    stop_mult = (1.25 - 0.10 * V["vol_score"].clip(0, 3)).clip(0.85, 1.25)  # 1.25 → 0.95 ATR
    stop0 = (entry - stop_mult * atr14)

    tgt_mult = (2.25 + 0.25 * V["vol_score"].clip(0, 3))  # 2.25 → 3.00 ATR
    target0 = entry + tgt_mult * atr14
    target0 = enforce_min_rr(entry, stop0, target0, MIN_RR_BY_STRAT.get(TOS_MOMENTUM, 2.0))

    return _build_pack(df, sig, stop0, target0, STRATEGY_LABELS[TOS_MOMENTUM], TOS_MOMENTUM)

def _strategy_pattern_forecast(df: pd.DataFrame, C: Dict[str, pd.Series], V: Dict[str, pd.Series]) -> StrategyPack:
    """
    Ascending triangle / tight flag detector with OBV/CMF confirmation.
    - Raw 'sig' is TRUE only on actual breakout above pivot (hi20) so that
      promotion to a trade still needs the global confluence gates.
    - "Pre-signal" fields (near-pivot + buy zone) are attached later in compute_signals.
    Heuristics (daily):
      • Triangle: price within PATTERN_NEAR_PCT below 20d high (pivot), rising short-term lows,
                  band width compressed (quantile≈35%), OBV rising or CMF > 0.
      • Flag: prior 21d run >= FLAG_MIN_RUN21; last ~7 bars drift sideways/down, squeeze,
              price near EMA20, OBV rising or CMF > 0.
    """
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    atr14, bb_lo, bb_mid, bb_up, bb_width = C["atr14"], C["bb_lo"], C["bb_mid"], C["bb_up"], C["bb_width"]
    ema10, ema20, sma50, sma200 = C["ema10"], C["ema20"], C["sma50"], C["sma200"]
    hi10, lo10, hi20, lo20 = C["hi10"], C["lo10"], C["hi20"], C["lo20"]

    # Compression context
    bw_q35 = bb_width.rolling(120, min_periods=40).quantile(0.35)
    compressed = bb_width <= bw_q35

    # OBV/CMF confirmation (already computed in V)
    smart_money_ok = (V.get("obv_rising", pd.Series(False, index=df.index)) | 
                      V.get("cmf_bullish", pd.Series(False, index=df.index)))

    # Triangle "near" conditions
    near_pivot = ((hi20 - close) / (hi20.replace(0, np.nan))).clip(lower=0).fillna(1.0) <= PATTERN_NEAR_PCT
    rising_lows = lo10 > lo10.shift(5)

    # Flag "near" conditions
    run21 = (close / close.shift(21) - 1.0).fillna(0.0)
    mild_drift = (close <= (ema20 * 1.02)) & (close >= (ema20 * 0.98))
    flag_near  = (run21 >= FLAG_MIN_RUN21) & compressed & mild_drift

    pre_near = (compressed & smart_money_ok & ((near_pivot & rising_lows) | flag_near))

    # Pivot for breakout = yesterday's 20d high (avoid lookahead)
    pivot = hi20.shift(1)

    # Raw breakout (actual signal) – requires close >= pivot
    sig_raw = (close >= pivot) & compressed

    # Initial levels on raw breakout (refined later by confluence)
    entry = close
    # Stop: below last swing (lo10) with ~0.6*ATR buffer
    stop0 = (lo10 - 0.60 * atr14).clip(upper=entry - 1e-4)
    # Target: measured move vs ATR or recent range
    width  = (hi20 - lo20).clip(lower=0)
    target0 = entry + np.maximum(2.20 * atr14, 0.80 * width)
    target0 = enforce_min_rr(entry, stop0, target0, MIN_RR_BY_STRAT.get(PATTERN_FORECAST, 2.0))


    return _build_pack(df, sig_raw, stop0, target0, STRATEGY_LABELS[PATTERN_FORECAST], PATTERN_FORECAST)

def _liquidity_gate(df: pd.DataFrame) -> pd.Series:
    """
    Tight liquidity filter:
      • price ≥ MIN_PRICE
      • 20d avg shares ≥ MIN_AVG_SHARES_20D
      • 20d avg $ turnover ≥ MIN_AVG_DOLLAR_20D
    """
    close = df["close"]
    vol   = df["volume"]
    avg_vol20   = vol.rolling(20).mean()
    avg_turn20  = (close * vol).rolling(20).mean()
    return (close >= MIN_PRICE) & (avg_vol20 >= MIN_AVG_SHARES_20D) & (avg_turn20 >= MIN_AVG_DOLLAR_20D)

def _apply_quality_confluence(df: pd.DataFrame,
                              packs: Dict[str, StrategyPack],
                              C: Dict[str, pd.Series],
                              V: Dict[str, pd.Series],
                              W: Dict[str, pd.Series]) -> Dict[str, StrategyPack]:
    """
    Enforce: base trend (daily) + min vol_score + min ADX + min RR
             + optional WEEKLY multi-timeframe gate.
    """
    close = df["close"]
    base_trend = (close > C["sma50"]) & (C["sma50"] > C["sma200"])
    adx = C["adx"]; vs = V["vol_score"]
    weekly_ok = W.get("weekly_ok", pd.Series(True, index=df.index))

    out: Dict[str, StrategyPack] = {}
    for slug, P in packs.items():
        min_rr  = MIN_RR_BY_STRAT.get(slug, 1.5)
        min_vs  = MIN_VOL_SCORE_BY_STRAT.get(slug, 1)
        min_adx = MIN_ADX_BY_STRAT.get(slug, 0)

        sig = P.sig & base_trend & (vs >= min_vs) & (adx >= min_adx) & (P.rr >= min_rr)
        if MTA_REQUIRE_BY_STRAT.get(slug, False):
            sig = sig & weekly_ok

        out[slug] = StrategyPack(sig=sig.fillna(False), stop0=P.stop0, target0=P.target0, rr=P.rr, label=P.label, slug=P.slug)
    return out


def _choose_best(rr_map: Dict[str, pd.Series], stop_map: Dict[str, pd.Series], tgt_map: Dict[str, pd.Series],
                 sig_map: Dict[str, pd.Series], close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Among strategies that fired on a bar, pick the one with max RR.
    Returns tuple: (best_slug, stop0, target0, best_rr)
    """
    # Build a DataFrame of RR for comparison; mask to only firing strategies
    rr_df = pd.DataFrame({slug: rr_map[slug] for slug in rr_map})
    fire_df = pd.DataFrame({slug: sig_map[slug] for slug in sig_map})
    rr_df = rr_df.where(fire_df, np.nan)

    # idx of max RR per row
    best_slug = rr_df.idxmax(axis=1)
    # Fallback when none fired
    best_slug = best_slug.fillna("")

    # Gather chosen stop/target/rr per row
    stop0 = pd.Series(index=close.index, dtype=float)
    target0 = pd.Series(index=close.index, dtype=float)
    best_rr = pd.Series(index=close.index, dtype=float)

    for slug in rr_map.keys():
        mask = (best_slug == slug)
        stop0.loc[mask]   = stop_map[slug].loc[mask]
        target0.loc[mask] = tgt_map[slug].loc[mask]
        best_rr.loc[mask] = rr_map[slug].loc[mask]

    return best_slug, stop0, target0, best_rr

def compute_signals(df: pd.DataFrame,
                    strategy: str = DEFAULT_STRATEGY,
                    as_of: Optional[date | pd.Timestamp] = None) -> pd.DataFrame:
    """
    Compute indicator packs and strategy signals with confluence and quality gates.
    Returns df with added columns:
      Common (reference features exposed for scanners/backtests):
        - ema10, ema20, sma50, sma200, hi10, hi20, lo10
        - bb_lo, bb_mid, bb_up, bb_width, vol_sma20
        - atr14, adx, dmi_diff, macd_hist, hma20
        - obv, mfi, cmf, vol_score
        - liq_pass (bool), trend_base (bool)
      Per-strategy (for all slugs in STRATEGY_SLUGS):
        - sig_<slug>         (final signal with liquidity + confluence gates)
        - rr_<slug>, stop0_<slug>, target0_<slug>
        - sig_raw_<slug>     (pattern logic only; before liquidity + confluence)
        - gate_rr_pass_<slug>, gate_adx_pass_<slug>, gate_vol_pass_<slug>  (booleans)
      Best-of bar:
        - signal (bool), strategy_slug_best, strategy, stop0, target, rr_best
        - strategy_slugs  (pipe-joined list of slugs that fired)
      Pattern+Forecast pre-signal (read-only helpers consumed by the scanner):
        - pre_sig_pattern (bool), pattern_pivot (float), buy_z_lo (float), buy_z_hi (float)
    """
    if df is None or df.empty:
        return df

    df = _ensure_ohlcv(df)

    # Common indicators + volume conviction
    C = _compute_common(df)
    V = _volume_conviction_pack(df)
    liq = _liquidity_gate(df)

    # Base trend (used both for gates and transparency)
    trend_base = (df["close"] > C["sma50"]) & (C["sma50"] > C["sma200"])

    # Build raw strategy packs (pattern logic only)
    packs: Dict[str, StrategyPack] = {}
    packs[BREAKOUT_CONSOL] = _strategy_breakout(df, C, V)
    packs[PULLBACK_HG]     = _strategy_pullback_hg(df, C, V)
    packs[SQUEEZE_BB]      = _strategy_squeeze_bb(df, C, V)
    packs[TOS_MOMENTUM]    = _strategy_tos_momentum(df, C, V)
    packs[PATTERN_FORECAST]= _strategy_pattern_forecast(df, C, V)

    # Preserve raw signals/rr for audits (pre‑liquidity / pre‑confluence)
    raw_sig_map: Dict[str, pd.Series] = {slug: packs[slug].sig.copy() for slug in packs.keys()}
    raw_rr_map:  Dict[str, pd.Series] = {slug: packs[slug].rr.copy()  for slug in packs.keys()}

    # Apply liquidity gate first
    for slug, p in packs.items():
        packs[slug] = StrategyPack(
            sig=(p.sig & liq).fillna(False),
            stop0=p.stop0, target0=p.target0, rr=p.rr, label=p.label, slug=p.slug
        )

    # Apply confluence/quality gates (trend, vol_score, adx, rr)
    W = _compute_weekly_context(df)

    # Apply confluence/quality gates (trend, vol_score, adx, rr)
    packs = _apply_quality_confluence(df, packs, C, V, W)  # ← pass W

    # Compose output
    out = df.copy()

    # --- Reference indicators exposed for scanners / predictive logic ---
    out["ema10"]      = C["ema10"]
    out["ema20"]      = C["ema20"]
    out["sma50"]      = C["sma50"]
    out["sma200"]     = C["sma200"]
    out["hi10"]       = C["hi10"]
    out["hi20"]       = C["hi20"]
    out["lo10"]       = C["lo10"]
    out["bb_lo"]      = C["bb_lo"]
    out["bb_mid"]     = C["bb_mid"]
    out["bb_up"]      = C["bb_up"]
    out["bb_width"]   = C["bb_width"]
    out["vol_sma20"]  = C["vol_sma20"]
    out["macd_hist"]  = C["macd_hist"]
    out["hma20"]      = C["hma20"]
    out["dmi_diff"]   = C["dmi_diff"]

    # Core transparency fields
    out["atr14"]      = C["atr14"]
    out["obv"]        = V["obv"]
    out["mfi"]        = V["mfi"]
    out["cmf"]        = V["cmf"]
    out["vol_score"]  = V["vol_score"]
    out["adx"]        = C["adx"]
    out["liq_pass"]   = liq.fillna(False)
    out["trend_base"] = trend_base.fillna(False)

    rr_map:   Dict[str, pd.Series] = {}
    stop_map: Dict[str, pd.Series] = {}
    tgt_map:  Dict[str, pd.Series] = {}
    sig_map:  Dict[str, pd.Series] = {}

    # Gate audit columns per-strategy (using RAW pattern + common floors)
    for slug in STRATEGY_SLUGS:
        # raw features (pattern-only)
        out[f"sig_raw_{slug}"] = raw_sig_map[slug].fillna(False)

        min_rr  = MIN_RR_BY_STRAT.get(slug, 1.5)
        min_vs  = MIN_VOL_SCORE_BY_STRAT.get(slug, 1)
        min_adx = MIN_ADX_BY_STRAT.get(slug, 0)

        out[f"gate_rr_pass_{slug}"]   = (raw_rr_map[slug] >= min_rr)
        out[f"gate_adx_pass_{slug}"]  = (C["adx"] >= min_adx)
        out[f"gate_vol_pass_{slug}"]  = (V["vol_score"] >= min_vs)

    # Final per-strategy outputs (after confluence gates)
    for slug in STRATEGY_SLUGS:
        P = packs[slug]
        out[f"sig_{slug}"]     = P.sig
        out[f"stop0_{slug}"]   = P.stop0
        out[f"target0_{slug}"] = P.target0
        out[f"rr_{slug}"]      = P.rr
        rr_map[slug]   = P.rr
        stop_map[slug] = P.stop0
        tgt_map[slug]  = P.target0
        sig_map[slug]  = P.sig

    # Pattern+Forecast pre-signal helpers (read-only; not trade gates)
    if PATTERN_FORECAST in STRATEGY_SLUGS:
        # pivot = yesterday's 20d high (as used inside the strategy)
        pivot = C["hi20"].shift(1)
        out["pattern_pivot"] = pivot

        # 'near' = compressed + within PATTERN_NEAR_PCT below pivot; must NOT have broken out yet
        out["pre_sig_pattern"] = (
            (out.get(f"sig_raw_{PATTERN_FORECAST}", pd.Series(False, index=out.index)).astype(bool) == False) &
            (
                (C["bb_width"] <= C["bb_width"].rolling(120, min_periods=40).quantile(0.35)) &
                (((C["hi20"] - df["close"]) / C["hi20"].replace(0, np.nan)).clip(lower=0) <= PATTERN_NEAR_PCT)
            )
        ).fillna(False)

        eps = PATTERN_BUY_EPS
        out["buy_z_lo"] = pivot * (1.0 - eps)
        out["buy_z_hi"] = pivot * (1.0 + 0.5 * eps)

    # Aggregate best-of-bar selection
    if strategy in ("all", "auto", "*", "", None):
        # Which slugs fired on each row
        fired = []
        for idx in out.index:
            slugs = [s for s in STRATEGY_SLUGS if bool(out.at[idx, f"sig_{s}"])]
            fired.append("|".join(slugs))
        out["strategy_slugs"] = fired

        any_sig = np.column_stack([sig_map[s].fillna(False).values for s in STRATEGY_SLUGS]).any(axis=1)
        out["signal"] = pd.Series(any_sig, index=out.index)

        # choose best by RR among firing strategies
        rr_df = pd.DataFrame({slug: rr_map[slug] for slug in STRATEGY_SLUGS})
        fire_df = pd.DataFrame({slug: sig_map[slug] for slug in STRATEGY_SLUGS})
        rr_df = rr_df.where(fire_df, np.nan)
        best_slug = rr_df.idxmax(axis=1).fillna("")

        stop0   = pd.Series(index=df.index, dtype=float)
        target0 = pd.Series(index=df.index, dtype=float)
        rr_best = pd.Series(index=df.index, dtype=float)
        for slug in STRATEGY_SLUGS:
            mask = (best_slug == slug)
            stop0.loc[mask]   = stop_map[slug].loc[mask]
            target0.loc[mask] = tgt_map[slug].loc[mask]
            rr_best.loc[mask] = rr_map[slug].loc[mask]

        out["strategy_slug_best"] = best_slug
        out["strategy"] = out["strategy_slug_best"].map(lambda s: STRATEGY_LABELS.get(s, "") if s else "")
        out["stop0"]    = stop0
        out["target"]   = target0
        out["rr_best"]  = rr_best

    else:
        slug = strategy
        if slug not in packs:
            return compute_signals(df, strategy="all", as_of=as_of)
        out["strategy_slugs"]    = slug
        out["strategy_slug_best"] = slug
        out["strategy"]          = STRATEGY_LABELS.get(slug, slug)
        out["signal"]            = out[f"sig_{slug}"]
        out["stop0"]             = out[f"stop0_{slug}"]
        out["target"]            = out[f"target0_{slug}"]
        out["rr_best"]           = out[f"rr_{slug}"]

    # Optional time cut
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        out = out.loc[out.index <= as_of_ts]

    return out



