# trade_management.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Provider hooks (the host app registers these at runtime)
# ─────────────────────────────────────────────────────────────────────────────
_get_intraday: Optional[Callable[[str, datetime, datetime], Optional[pd.DataFrame]]] = None
_get_daily:    Optional[Callable[[str, date, date, int], Optional[pd.DataFrame]]]   = None

def register_intraday_provider(fn: Callable) -> None:
    """Host app should call this with its get_intraday_bars(symbol, start, end, ...)"""
    global _get_intraday
    _get_intraday = fn

def register_daily_bars_provider(fn: Callable) -> None:
    """Host app should call this with its get_bars(symbol, start_date, end_date, min_history_days)"""
    global _get_daily
    _get_daily = fn


# ─────────────────────────────────────────────────────────────────────────────
# Indicators / helpers
# ─────────────────────────────────────────────────────────────────────────────
def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _rolling_ma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=max(2, length // 2)).mean()

def _swing_low_before(df: pd.DataFrame, entry_idx: int, lookback: int) -> Optional[float]:
    """Lowest low in [entry_idx-lookback, entry_idx-1]."""
    i0 = max(0, entry_idx - lookback)
    if entry_idx <= i0:
        return None
    window = df["low"].iloc[i0:entry_idx]
    if window.empty:
        return None
    return float(window.min())

def _chandelier_stop(highest_close: float, atr: float, mult: float) -> float:
    return float(highest_close - mult * atr)


# ─────────────────────────────────────────────────────────────────────────────
# Risk rules (moved out of Combined-Script.py)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RiskRules:
    # initial stop: structure + ATR
    atr_mult_stop: float = 2.0
    swing_lookback: int = 10
    swing_pad_atr: float = 0.25
    stop_hunt_buffer_atr: float = 0.10   # extra ATR cushion beyond obvious swing to avoid stop hunts

    # partial & transitions
    first_scale_R: float = 1.0
    first_scale_fraction: float = 0.5
    move_to_breakeven_on_first_scale: bool = True

    # trailing (kicks in after first scale)
    chandelier_mult: float = 2.5
    trail_use_ma_fail_safe: bool = True
    trail_ma_len: int = 10

    # “exceptional run” tightening
    accel_bars: int = 3
    accel_R: float = 2.0
    accel_chandelier_mult: float = 1.75

    # time stop
    max_bars_in_trade: int = 10

    # pyramiding on proof (+1R/+2R)
    pyramid_on: bool = True
    pyramid_triggers_R: tuple[float, ...] = (1.0, 2.0)
    pyramid_risk_fracs: tuple[float, ...] = (0.5, 0.5)  # of initial dollar risk
    pyramid_max_adds: int = 2

    # event blackouts ( host app applies gates; kept here for visibility )
    earnings_blackout_days: int = 3
    macro_blackout_window: int = 0

# default rule set
RULES = RiskRules()


# ─────────────────────────────────────────────────────────────────────────────
# Initial levels planner (anchored to structure + ATR, min‑RR aware)
# ─────────────────────────────────────────────────────────────────────────────
def _plan_initial_levels(entry: float,
                         atr: float,
                         swing_low: Optional[float],
                         rules: RiskRules,
                         *,
                         min_rr: float = 2.0) -> Tuple[float, float, float]:
    """
    Returns: (stop_price, R_per_share, first_scale_price)
    Stop anchoring priority:
      1) prior swing low − (swing_pad + stop_hunt_buffer)*ATR
      2) ATR fallback: entry − atr_mult_stop*ATR
    Enforces min risk:reward by lifting the first target to ≥ entry + min_rr*R.
    """
    stop_by_atr   = entry - rules.atr_mult_stop * atr
    stop_by_swing = None
    if swing_low is not None and np.isfinite(swing_low):
        pad = (rules.swing_pad_atr + rules.stop_hunt_buffer_atr) * atr
        stop_by_swing = float(swing_low) - float(pad)

    # For longs we pick the FURTHER stop (lower), i.e., the more conservative one
    if stop_by_swing is None:
        stop = stop_by_atr
    else:
        stop = min(stop_by_atr, stop_by_swing)

    # Risk per share
    R = max(1e-6, entry - stop)

    # First scale (“T1”) starts at +1R; lift to satisfy min_rr when needed
    t1 = entry + rules.first_scale_R * R
    t_req = entry + float(min_rr) * R
    t1 = max(t1, t_req)

    return float(stop), float(R), float(t1)


# ─────────────────────────────────────────────────────────────────────────────
# Intraday refinement (same signature as your previous function)
# ─────────────────────────────────────────────────────────────────────────────
def refine_stop_with_intraday(symbol: str,
                              session_day: date,
                              entry_px: float,
                              stop_fallback: float,
                              day_row: pd.Series,
                              lookback_minutes: int = 90) -> float:
    """
    Tighten the stop for same-day entries using the last `lookback_minutes`
    of intraday lows, buffered by ~0.35*ATR(14) (daily). Never raises stop above entry
    and never loosens beyond the fallback.
    """
    if _get_intraday is None:
        return float(stop_fallback)

    try:
        start_dt = datetime.combine(session_day, datetime.min.time())
        end_dt   = datetime.combine(session_day, datetime.max.time())
        intra = _get_intraday(symbol, start_dt, end_dt,
                              interval_minutes=1, minute_granularity=1, include_extended=False)
        if intra is None or intra.empty:
            return float(stop_fallback)
        recent = intra.tail(max(lookback_minutes, 10))
        intralow = float(recent["low"].min())
        atrv = float(day_row.get("atr14", np.nan))
        pad  = (0.35 * atrv) if np.isfinite(atrv) else 0.0
        cand = intralow - pad
        cand = min(cand, entry_px - 1e-4)                # never above entry for longs
        return float(max(stop_fallback, cand))           # never looser than fallback
    except Exception:
        return float(stop_fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Trade path simulator (moved from Combined-Script.py; minor cleanups)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trade_path(indf: pd.DataFrame,
                        entry_idx: int,
                        rules: Optional[RiskRules] = None,
                        stop0_override: Optional[float] = None,
                        target0_override: Optional[float] = None) -> dict:
    """
    indf: OHLCV (+ indicators). Entry at bar 'entry_idx' close.
    Stops/targets:
      - T1 = entry + 1R (partial scale per rules)
      - T2 = final target (if target0_override provided, treat it as T2; otherwise T2 = entry + 2R)
    Exits:
      - STOP if low touches stop
      - TARGET if high touches T2 (after STOP check)
      - MA_EXIT on trail fail-safe (next bar open)
      - TIME after max bars
      - OPEN if none above
    Returns dict: entry_date, exit_date, entry, exit_px, outcome, R, days_held, stop, target, t1
      where 'target' is T2 (final) and 't1' is the first scale price.
    """
    rules = rules or RULES
    n = len(indf)
    if entry_idx < 0 or entry_idx >= n:
        return {}

    closes = indf["close"].astype(float).reset_index(drop=True)
    highs  = indf["high"].astype(float).reset_index(drop=True)
    lows   = indf["low"].astype(float).reset_index(drop=True)
    opens  = indf["open"].astype(float).reset_index(drop=True) if "open" in indf.columns else closes
    atr14  = atr_series(indf, 14).reset_index(drop=True)
    ma10   = _rolling_ma(closes, rules.trail_ma_len)

    entry_price = float(closes.iloc[entry_idx])
    atr_e = float(atr14.iloc[entry_idx]) if np.isfinite(atr14.iloc[entry_idx]) else \
            (float(atr14.dropna().iloc[0]) if atr14.dropna().size else 0.0)
    swing_low = _swing_low_before(indf, entry_idx, rules.swing_lookback)

    # Initial levels (anchored planner) or overrides for stop only
    if (stop0_override is not None) and np.isfinite(stop0_override):
        stop0 = float(stop0_override)
        if not (stop0 < entry_price):  # safety
            stop0 = entry_price - max(1e-3, rules.atr_mult_stop * max(atr_e, 1e-3))
        R = max(1e-6, entry_price - stop0)
        t1 = entry_price + rules.first_scale_R * R
    else:
        stop0, R, t1 = _plan_initial_levels(entry_price, atr_e, swing_low, rules)

    # Final target (T2)
    if (target0_override is not None) and np.isfinite(target0_override):
        t2 = float(target0_override)
    else:
        t2 = entry_price + 2.0 * R
    # Ensure T2 is not below T1
    t2 = max(t2, t1)

    pos_open_frac = 1.0
    partials: List[dict] = []
    highest_close = entry_price
    trail_mult = rules.chandelier_mult
    trail_active = False
    stop = float(stop0)

    pyramid_positions: list[dict] = []
    triggers_hit: set[float] = set()

    def _exceptional_run(i_now: int) -> bool:
        j = min(n - 1, entry_idx + rules.accel_bars)
        if j <= entry_idx:
            return False
        max_close = float(closes.iloc[entry_idx + 1:j + 1].max())
        return (max_close - entry_price) >= (rules.accel_R * R)

    def _try_add_pyramid(i: int, kR: float):
        if (not rules.pyramid_on) or (len(pyramid_positions) >= rules.pyramid_max_adds) or (kR in triggers_hit):
            return
        level = entry_price + kR * R
        if highs.iloc[i] < level:
            return
        entry_add = max(level, float(opens.iloc[i]))
        entry_add = min(entry_add, float(highs.iloc[i]))
        risk_add_ps = max(entry_add - stop, 1e-6)
        idx = list(rules.pyramid_triggers_R).index(kR) if kR in rules.pyramid_triggers_R else 0
        risk_frac = (rules.pyramid_risk_fracs[idx] if idx < len(rules.pyramid_risk_fracs) else rules.pyramid_risk_fracs[-1])
        shares_equiv = float(risk_frac * (R / risk_add_ps))
        if shares_equiv <= 0:
            return
        pyramid_positions.append(dict(entry=entry_add, shares_frac=shares_equiv,
                                      date=str(indf.index[i].date()), trigger_R=kR))
        triggers_hit.add(kR)

    for i in range(entry_idx + 1, n):
        highest_close = max(highest_close, float(closes.iloc[i - 1]))
        atr_now = float(atr14.iloc[i]) if np.isfinite(atr14.iloc[i]) else atr_e

        # Update trailing stop when active
        if trail_active:
            ch_stop = _chandelier_stop(highest_close, atr_now, trail_mult)
            stop = max(stop, ch_stop)
            # Fail-safe: if prior close < MA, exit at next open
            if rules.trail_use_ma_fail_safe and (not np.isnan(ma10.iloc[i - 1])) and closes.iloc[i - 1] < ma10.iloc[i - 1]:
                exit_px = float(opens.iloc[i]) if "open" in indf.columns else float(closes.iloc[i])
                adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
                r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
                return dict(
                    entry_date=str(indf.index[entry_idx].date()),
                    exit_date=str(indf.index[i].date()),
                    entry=round(entry_price, 2),
                    exit_px=round(exit_px, 2),
                    outcome="MA_EXIT",
                    R=round(r_mult, 2),
                    days_held=(i - entry_idx),
                    stop=round(stop0, 2),
                    target=round(t2, 2),
                    t1=round(t1, 2),
                )

        # 1) STOP check
        if lows.iloc[i] <= stop:
            exit_px = stop
            adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
            r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
            return dict(
                entry_date=str(indf.index[entry_idx].date()),
                exit_date=str(indf.index[i].date()),
                entry=round(entry_price, 2),
                exit_px=round(exit_px, 2),
                outcome="STOP",
                R=round(r_mult, 2),
                days_held=(i - entry_idx),
                stop=round(stop0, 2),
                target=round(t2, 2),
                t1=round(t1, 2),
            )

        # 2) FINAL TARGET (T2) check
        if highs.iloc[i] >= t2:
            exit_px = t2
            adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
            r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
            return dict(
                entry_date=str(indf.index[entry_idx].date()),
                exit_date=str(indf.index[i].date()),
                entry=round(entry_price, 2),
                exit_px=round(exit_px, 2),
                outcome="TARGET",
                R=round(r_mult, 2),
                days_held=(i - entry_idx),
                stop=round(stop0, 2),
                target=round(t2, 2),
                t1=round(t1, 2),
            )

        # 3) First scale (partial) at +1R (only once on the core)
        if (pos_open_frac > (1.0 - rules.first_scale_fraction)) and (highs.iloc[i] >= t1):
            pf = rules.first_scale_fraction
            realized = (t1 - entry_price) / R * pf
            partials.append(dict(date=str(indf.index[i].date()),
                                 price=round(t1, 2),
                                 frac=pf, R_realized=round(realized, 2)))
            pos_open_frac = max(0.0, pos_open_frac - pf)
            if rules.move_to_breakeven_on_first_scale:
                stop = max(stop, entry_price)
            trail_active = True
            if _exceptional_run(i):
                trail_mult = min(trail_mult, rules.accel_chandelier_mult)

        # 4) Pyramiding after partial logic
        if rules.pyramid_on:
            for kR in rules.pyramid_triggers_R:
                _try_add_pyramid(i, kR)

        # 5) Time stop
        held_bars = i - entry_idx
        if held_bars >= rules.max_bars_in_trade:
            exit_px = float(closes.iloc[i])
            adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
            r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
            return dict(
                entry_date=str(indf.index[entry_idx].date()),
                exit_date=str(indf.index[i].date()),
                entry=round(entry_price, 2),
                exit_px=round(exit_px, 2),
                outcome="TIME",
                R=round(r_mult, 2),
                days_held=held_bars,
                stop=round(stop0, 2),
                target=round(t2, 2),
                t1=round(t1, 2),
            )

    # Still open on last bar
    exit_px = float(closes.iloc[-1])
    adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
    r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
    return dict(
        entry_date=str(indf.index[entry_idx].date()),
        exit_date=None,
        entry=round(entry_price, 2),
        exit_px=round(exit_px, 2),
        outcome="OPEN",
        R=round(r_mult, 2),
        days_held=(n - 1 - entry_idx),
        stop=round(stop0, 2),
        target=round(t2, 2),
        t1=round(t1, 2),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio View dynamic levels (moved)
# ─────────────────────────────────────────────────────────────────────────────
def compute_dynamic_levels_for_holding(holding: dict,
                                       as_of: Optional[date] = None,
                                       rules: Optional[RiskRules] = None) -> dict:
    """
    Compute dynamic stop/target for a single holding using daily bars (via registered provider).
    Expects: symbol, entry_date (YYYY-MM-DD), price_paid, stop_loss_initial/profit_target_initial (optional).
    """
    rules = rules or RULES
    if _get_daily is None:
        return {"status": "No provider"}

    sym = holding.get("symbol") or holding.get("ticker")
    if not sym:
        return {"status": "Invalid holding"}

    as_of = as_of or date.today()
    try:
        entry_date = datetime.strptime(str(holding.get("entry_date")), "%Y-%m-%d").date()
    except Exception:
        entry_date = as_of - timedelta(days=20)

    entry_price = float(holding.get("price_paid") or holding.get("entry") or 0.0)
    stop0 = holding.get("stop_loss_initial") or holding.get("stop_loss")
    tgt0  = holding.get("profit_target_initial") or holding.get("profit_target")
    stop0 = float(stop0) if stop0 not in (None, "") else None
    tgt0  = float(tgt0)  if tgt0  not in (None, "") else None

    bars = _get_daily(sym, start=entry_date - timedelta(days=90), end=as_of, min_history_days=260)
    if bars is None or bars.empty:
        return {
            "last_price": None, "updated_stop": stop0,
            "updated_target": tgt0 if tgt0 is not None else "—",
            "status": "No data", "partial_hit": False
        }

    closes = bars["close"].astype(float)
    highs  = bars["high"].astype(float)
    lows   = bars["low"].astype(float)

    entry_ts = pd.Timestamp(entry_date)
    mask_ge  = bars.index.normalize() >= entry_ts
    mask_le  = bars.index.normalize() <= entry_ts
    if mask_ge.any():
        entry_idx = bars.index.get_loc(bars.index[mask_ge][0])
    elif mask_le.any():
        entry_idx = bars.index.get_loc(bars.index[mask_le][-1])
    else:
        entry_idx = 0

    if not np.isfinite(entry_price) or entry_price <= 0:
        entry_price = float(closes.iloc[entry_idx])

    atr14 = atr_series(bars, 14)
    atr_e = float(atr14.iloc[entry_idx]) if np.isfinite(atr14.iloc[entry_idx]) else float(atr14.dropna().iloc[0]) if atr14.dropna().size else 0.0
    swing_low = _swing_low_before(bars, entry_idx, rules.swing_lookback)

    if tgt0 is None or stop0 is None:
        stop_calc, R, first_scale = _plan_initial_levels(entry_price, atr_e, swing_low, rules)
        if stop0 is None: stop0 = stop_calc
        if tgt0  is None: tgt0  = first_scale
    else:
        R = max(1e-6, entry_price - float(stop0))

    partial_hit = bool((highs.iloc[entry_idx+1:] >= float(tgt0)).any()) if entry_idx + 1 < len(bars) else False

    if entry_idx + 1 < len(bars):
        highest_close = float(closes.iloc[entry_idx:len(bars)].cummax().iloc[-2]) if len(bars) > (entry_idx+1) else float(closes.iloc[entry_idx])
    else:
        highest_close = float(closes.iloc[entry_idx])

    updated_stop = float(stop0)
    if partial_hit:
        if rules.move_to_breakeven_on_first_scale:
            updated_stop = max(updated_stop, entry_price)
        atr_now = float(atr14.iloc[-1]) if np.isfinite(atr14.iloc[-1]) else atr_e
        chand   = _chandelier_stop(highest_close, atr_now, rules.chandelier_mult)
        updated_stop = max(updated_stop, chand)

    updated_target = float(tgt0) if not partial_hit else "TRAIL"
    last_price = float(closes.iloc[-1])

    status = "Holding"
    if last_price <= updated_stop:
        status = "Stop Loss Hit"
    elif (not partial_hit) and last_price >= float(tgt0):
        status = "Profit Target Hit"

    return {
        "last_price": round(last_price, 2),
        "updated_stop": round(updated_stop, 2) if np.isfinite(updated_stop) else None,
        "updated_target": (round(updated_target, 2) if isinstance(updated_target, (int, float)) else updated_target),
        "status": status,
        "partial_hit": partial_hit,
        "initial_stop": round(float(stop0), 2) if stop0 is not None else None,
        "initial_target": round(float(tgt0), 2) if tgt0 is not None else None,
        "entry": round(float(entry_price), 2),
        "entry_idx": entry_idx
    }
