#!/usr/bin/env python3
"""
Enhanced Industry Momentum Model + Interactive ToS Scanner
──────────────────────────────────────────────────────────
Adds to your original model:
  • Schwab OAuth via easy_client (token.json; no env refresh token required)
  • Menu:
       1) Stock Watchlist  (persisted in watchlist.json)
       2) Current Signals   (ToS-style rules; choose Watchlist / Industry / Manual)
       3) Backtest Signals  (single date or range; includes End-of-Week close)
       4) Rank Industries   (your XGBoost-based model + drill down)
  • Universe picker for 2 & 3: pick industries (from industries-tickers.json),
    use Watchlist, or enter manual tickers.

This file preserves your industry feature engineering & ranking pipeline and
adds a separate signal-scanning workflow. For Schwab API access it uses the
same no-paste OAuth flow as our older script: `schwab.auth.easy_client`.

Dependencies:
    pandas, numpy, requests, xgboost

"""

from __future__ import annotations
import os, re, json, sys, time, pickle, math, urllib.parse
from pathlib import Path
from datetime import datetime, timedelta, date, timezone
from typing import List, Dict, Tuple, Optional, Iterable
import calendar
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import importlib.util, sys
_SESSION = None

from trade_management import (
    register_intraday_provider,
    register_daily_bars_provider,
    RULES as TM_RULES,
    simulate_trade_path as TM_simulate_trade_path,
    refine_stop_with_intraday as TM_refine_stop,
    compute_dynamic_levels_for_holding as TM_compute_dynamic_levels_for_holding,
)

# ── Dynamic import for Infinit-Strategies.py (hyphenated filename) ───────────
DEBUG = os.environ.get("INFINIT_DEBUG", "0").lower() in ("1", "true", "yes", "y")
def _wsl_interop_enabled() -> bool:
    """True only when actually inside WSL with interop enabled."""
    try:
        return bool(os.environ.get("WSL_DISTRO_NAME")) and os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
    except Exception:
        return False


def _ms_to_utc_naive(ms: int | float) -> datetime:
    """
    Convert epoch milliseconds to a tz-naive UTC datetime (forward compatible).
    Replaces deprecated datetime.utcfromtimestamp().
    """
    return datetime.fromtimestamp(float(ms) / 1000.0, timezone.utc).replace(tzinfo=None)

def _resolve_token_path() -> Path:
    """
    Determine where to store/read the Schwab OAuth token.
    Priority:
      1) INFINIT_SCHWAB_TOKEN_PATH or SCHWAB_TOKEN_PATH env var (explicit override)
      2) If running under sudo, use the invoking user's home, not /root
      3) Default to Path.home()/.config/schwab/token.json
    """
    env = os.environ.get("INFINIT_SCHWAB_TOKEN_PATH") or os.environ.get("SCHWAB_TOKEN_PATH")
    if env:
        return Path(env).expanduser()

    try:
        if os.geteuid() == 0 and os.environ.get("SUDO_USER"):
            sudo_user = os.environ["SUDO_USER"]
            home = Path(os.path.expanduser(f"~{sudo_user}"))
            return home / ".config" / "schwab" / "token.json"
    except Exception:
        pass

    return Path.home() / ".config" / "schwab" / "token.json"


def _detect_browser_for_oauth() -> Optional[str]:
    """
    Pick a browser launcher for OAuth.
      • INFINIT_BROWSER env overrides everything (e.g. 'xdg-open', 'open', 'wslview', or 'none')
      • Use 'wslview' only if WSL interop looks enabled
      • Fallback: 'xdg-open' on Linux, 'open' on macOS, default on Windows
    """
    b = os.environ.get("INFINIT_BROWSER", "").strip()
    if b:
        if b.lower() in ("none", "off", "disabled"):
            return None
        return b

    # WSL detection + interop check
    if os.environ.get("WSL_DISTRO_NAME"):
        try:
            if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
                return "wslview"
        except Exception:
            pass
        # fall through if interop is not available

    if sys.platform.startswith("linux"):
        return "xdg-open"
    if sys.platform == "darwin":
        return "open"
    # On Windows, let the stdlib webbrowser decide
    return None



STRAT_PATH = Path(__file__).resolve().parent / "Infinit-Strategies.py"
_spec = importlib.util.spec_from_file_location("infinit_strategies", STRAT_PATH)
_infmod = importlib.util.module_from_spec(_spec)
sys.modules["infinit_strategies"] = _infmod
assert _spec and _spec.loader, "Cannot load Infinit-Strategies.py"
_spec.loader.exec_module(_infmod)

# Pull the API we need
PATTERN_FORECAST        = _infmod.PATTERN_FORECAST  
register_data_providers = _infmod.register_data_providers
compute_signals         = _infmod.compute_signals
STRATEGY_SLUGS          = _infmod.STRATEGY_SLUGS
STRATEGY_LABELS         = _infmod.STRATEGY_LABELS
DEFAULT_STRATEGY        = _infmod.DEFAULT_STRATEGY
TOS_MOMENTUM            = _infmod.TOS_MOMENTUM
BREAKOUT_CONSOL         = _infmod.BREAKOUT_CONSOL
PULLBACK_HG             = _infmod.PULLBACK_HG
SQUEEZE_BB              = _infmod.SQUEEZE_BB
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_token_path() -> Path:
    """
    Determine where to store/read the Schwab OAuth token.
    Priority:
      1) INFINIT_SCHWAB_TOKEN_PATH or SCHWAB_TOKEN_PATH env var (explicit override)
      2) If running under sudo, use the invoking user's home, not /root
      3) Default to Path.home()/.config/schwab/token.json
    """
    # 1) explicit override
    env = os.environ.get("INFINIT_SCHWAB_TOKEN_PATH") or os.environ.get("SCHWAB_TOKEN_PATH")
    if env:
        return Path(env).expanduser()

    # 2) map sudo -> original user
    try:
        if os.geteuid() == 0 and os.environ.get("SUDO_USER"):
            sudo_user = os.environ["SUDO_USER"]
            home = Path(os.path.expanduser(f"~{sudo_user}"))
            return home / ".config" / "schwab" / "token.json"
    except Exception:
        pass

    # 3) default
    return Path.home() / ".config" / "schwab" / "token.json"


# ─────────────────────────────────────────────────────────────────────────────
# SCHWAB OAUTH (no env refresh token; persists ~/.config/schwab/token.json)
# ─────────────────────────────────────────────────────────────────────────────
SCHWAB_KEY        = "wZWNoHzFYx5U85qs1B4dE04236pk7hWU"   # <- your real values
SCHWAB_SECRET     = "fpvBemXih4NChuGn"                   # <- your real values
CALLBACK_URL      = "https://127.0.0.1:8182/"            # MUST match app settings exactly
SCHWAB_TOKEN_PATH = _resolve_token_path()
SCHWAB_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
SCHWAB_BASE_URL   = "https://api.schwabapi.com"
schwab_client = None

# best-effort: secure the directory on *nix
try:
    os.chmod(SCHWAB_TOKEN_PATH.parent, 0o700)
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS (kept from your model; added local paths for menu pieces)
# ─────────────────────────────────────────────────────────────────────────────
# Keep your original project layout defaults
ROOT_DIR_MODEL = Path(__file__).resolve().parent.parent
DATA_DIR       = (ROOT_DIR_MODEL / "Data")
MAP_PATH       = (ROOT_DIR_MODEL / "industry_to_sector.json")
CACHE_DIR      = (ROOT_DIR_MODEL / ".cache"); CACHE_DIR.mkdir(exist_ok=True)
HOLDINGS_PATH      = ROOT_DIR_MODEL / "holdings.json"
TRADE_HISTORY_PATH = ROOT_DIR_MODEL / "trade_history.json"
# Place this next to other local paths (e.g., under WATCHLIST_PATH)
# Add a local dir for interactive files (watchlist.json, industries-tickers.json)
LOCAL_DIR      = Path(__file__).resolve().parent
WATCHLIST_PATH = LOCAL_DIR / "watchlist.json"

# Macro blackout dates file (same folder)
MACRO_BLACKOUT_PATH = LOCAL_DIR / "macro_blackout_days.json"

# Industry-tickers map candidates (your original + local + /mnt/data)
UPLOADED_TICKERS_FILE = Path("/home/axis/Augustines/industries-tickers.json")
IND_TICKERS_FILE_PATH = ROOT_DIR_MODEL / "industries-tickers.json"
IND_TICKER_PATHS      = [
    UPLOADED_TICKERS_FILE,
    Path("/mnt/data/industries-tickers.json"),
    IND_TICKERS_FILE_PATH,
    LOCAL_DIR / "industries-tickers.json",
    Path.cwd() / "industries-tickers.json"
]
TOP10_LEADERS_PATH = LOCAL_DIR / "top10_leaders.json"

# Other constants from your model
YEAR, FWD_DAYS, TOP_K, RNG_SEED = datetime.now().year, 21, 10, 42
date_pat = re.compile(r"(\d{1,2})\.(\d{1,2})\.json$")

# FRED key (optional); your code already no-ops when absent
FRED_KEY = os.getenv("FRED_API_KEY")
DEFAULT_FORECAST_HOURS = int(float(os.getenv("FORECAST_HOURS", "4")))  # X hours horizon for nowcasts
INFINIT_EXPLAIN = os.getenv("INFINIT_EXPLAIN", "0").lower() in ("1","true","yes","y")


ANSI_RESET  = "\x1b[0m"
ANSI_BOLD   = "\x1b[1m"
FG_RED      = "\x1b[31m"
FG_GREEN    = "\x1b[32m"
FG_YELLOW   = "\x1b[33m"
FG_CYAN     = "\x1b[36m"
FG_MAGENTA  = "\x1b[35m"
FG_GRAY     = "\x1b[90m"

def _ansi_on() -> bool:
    # Respect NO_COLOR env; only color when stdout is a TTY.
    return sys.stdout.isatty() and os.getenv("NO_COLOR", "").lower() not in ("1", "true", "yes")

def _c(s: str, color: str) -> str:
    return f"{color}{s}{ANSI_RESET}" if _ansi_on() else s

def assert_data_provider_ok() -> None:
    """
    Sanity check: try to fetch SPY daily bars and a simple quote after ensuring OAuth is set.
    Retries once with force re-auth on failure.
    """
    try:
        if schwab_client is None:
            init_schwab_client()
    except Exception as e:
        print(_c("[ERROR] Schwab OAuth init failed.", FG_RED))
        if DEBUG:
            print(f"[DEBUG] OAuth init error: {e}")
        return

    def _now_utc_naive():
        return datetime.now(timezone.utc).replace(tzinfo=None)

    start = _now_utc_naive() - timedelta(days=30)
    end   = _now_utc_naive()

    probe = None
    try:
        probe = _schwab_daily("SPY", start, end)
    except Exception:
        probe = None

    # If daily fails, try a lightweight quotes check to distinguish 404 vs token issues
    if (probe is None) or probe.empty:
        q = schwab_get("marketdata/v1/quotes", params={"symbols": "SPY"})
        if not q:
            # try forced re-auth once
            if DEBUG:
                print("[DEBUG] healthcheck: retrying after forced re-auth …")
            try:
                init_schwab_client(force_reauth=True)
                probe = _schwab_daily("SPY", start, end)
            except Exception:
                probe = None

    if probe is None or probe.empty:
        print(_c("[ERROR] Could not retrieve daily bars via Schwab.", FG_RED))
        print("        Signals, pre‑signals, and backtests will be empty until this is fixed.")
        print("        Most common causes:")
        print("          • Wrong endpoint shape (use /marketdata/v1/pricehistory?symbol=… not …/pricehistory/{symbol})")
        print("          • Expired/invalid tokens (re‑auth) or callback URL mismatch")
        print("          • Running with sudo (token path differs)\n")
    elif DEBUG:
        print("[DEBUG] SPY probe OK — Schwab data provider looks healthy.")

def init_schwab_client(*, force_reauth: bool = False) -> None:
    """
    Create / refresh Schwab client using easy_client (opens browser once).
    - Forces non-WSL environments to use xdg-open (never wslview).
    - interactive=False to skip the "Press ENTER" prompt.
    - Deletes old token on force_reauth.
    """
    global schwab_client

    try:
        from schwab import auth as schwab_auth  # lazy import
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: schwab-py\n  pip install 'schwab-py>=0.5'\n"
            f"(import error: {exc})"
        )

    if force_reauth:
        try:
            SCHWAB_TOKEN_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    # Pick a browser safely
    requested_browser = None
    if _wsl_interop_enabled():
        requested_browser = "wslview"
    else:
        # Avoid accidental wslview via $BROWSER
        if "BROWSER" in os.environ and "wslview" in os.environ["BROWSER"]:
            os.environ["BROWSER"] = "xdg-open"
        # Prefer xdg-open on Linux
        if sys.platform.startswith("linux"):
            requested_browser = "xdg-open"

    if DEBUG:
        who = os.environ.get("SUDO_USER") or os.environ.get("USER") or os.getlogin()
        print(f"[DEBUG] Schwab token path: {SCHWAB_TOKEN_PATH}")
        print(f"[DEBUG] Running as user: {who}")
        print(f"[DEBUG] requested_browser -> {requested_browser or 'default'}")

    def _client():
        return schwab_auth.easy_client(
            api_key      = SCHWAB_KEY,
            app_secret   = SCHWAB_SECRET,
            callback_url = CALLBACK_URL,
            token_path   = SCHWAB_TOKEN_PATH,
            requested_browser = requested_browser,
            interactive  = False  # skip the ENTER prompt
        )

    try:
        schwab_client = _client()
    except ValueError as e:
        # compatibility with older token formats
        if "token format has changed" in str(e).lower():
            print("[INFO] Deleting obsolete token & restarting OAuth …")
            try:
                SCHWAB_TOKEN_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            schwab_client = _client()
        else:
            raise

def _schwab_search(symbol: str, projection: str = "fundamental"):
    """
    Instruments API wrapper that returns a requests.Response-like object.
    Follows the same resolution order as the older script.
    """
    if schwab_client is None:
        raise RuntimeError("schwab_client not initialised")

    # v ≥ 0.5.x
    if hasattr(schwab_client, "search_instruments"):
        return schwab_client.search_instruments(symbol=symbol, projection=projection)

    # v 0.4.x – use authenticated session
    url = (f"{SCHWAB_BASE_URL}/marketdata/v1/instruments?"
           f"symbol={urllib.parse.quote(symbol)}&projection={projection}")
    if hasattr(schwab_client, "session"):
        return schwab_client.session.get(url, timeout=10)

    # last-ditch with bearer
    hdrs = {"Authorization": f"Bearer {getattr(schwab_client, 'access_token', '')}"}
    return requests.get(url, headers=hdrs, timeout=10)

def schwab_get(endpoint: str,
               params: dict | None = None,
               cache: str | None = None,
               ttl: int = 86_400,
               *,
               _retry_auth_once: bool = True):
    """
    Generic Schwab GET using the OAuth'd session (plus simple on-disk cache).
    Retries once with forced re-auth if we see token/401/403 issues.
    """
    if cache:
        cached = cache_load(cache, ttl)
        if cached is not None:
            return cached

    if schwab_client is None:
        return {}

    url = f"{SCHWAB_BASE_URL}/{endpoint.lstrip('/')}"
    try:
        if hasattr(schwab_client, "session"):
            r = schwab_client.session.get(url, params=params or {}, timeout=30)
        else:
            hdrs = {"Authorization": f"Bearer {getattr(schwab_client, 'access_token', '')}"}
            r = _session_with_retries().get(url, headers=hdrs, params=params or {}, timeout=30)

        # Handle token problems
        token_bad = False
        try:
            if r.status_code in (401, 403):
                token_bad = True
            else:
                js = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
                # Some endpoints report 'token_invalid' in JSON error payload
                if isinstance(js, dict):
                    msg = (json.dumps(js)[:256]).lower()
                    if "token_invalid" in msg or "invalid token" in msg:
                        token_bad = True
        except Exception:
            pass

        if token_bad and _retry_auth_once:
            if DEBUG:
                print("[DEBUG] schwab_get: token invalid → forced re-auth and retry …")
            init_schwab_client(force_reauth=True)
            return schwab_get(endpoint, params=params, cache=cache, ttl=ttl, _retry_auth_once=False)

        if r.status_code != 200:
            return {}

        data = r.json()
        if cache:
            cache_save(cache, data)
        return data

    except Exception as e:
        print(f"[WARN] schwab_get failed ({endpoint}): {e}")
        return {}


def _session_with_retries() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    s = requests.Session()
    try:
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"])
        )
    except TypeError:  # urllib3<1.26
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            method_whitelist=frozenset(["GET", "POST"])
        )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    _SESSION = s
    return s



def _status_bar(i: int, n: int, *, prefix: str = "", width: int = 32) -> None:
    """In-place progress bar: i completed out of n (1-based OK)."""
    i = max(0, min(i, n))
    done = int(width * (i / n)) if n > 0 else width
    bar = "█" * done + " " * (width - done)
    pct = (100.0 * i / n) if n else 100.0
    msg = f"{prefix} [{bar}] {i}/{n} ({pct:5.1f}%)"
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()
    if i >= n:
        sys.stdout.write("\n")
        sys.stdout.flush()

# ── Event-gate helpers (earnings & macro blackout) ───────────────────────────
def _load_macro_blackout_dates() -> set[date]:
    """Read macro blackout ISO dates from macro_blackout_days.json (if present)."""
    try:
        if not MACRO_BLACKOUT_PATH.exists():
            return set()
        arr = json.loads(MACRO_BLACKOUT_PATH.read_text()) or []
        out = set()
        for s in arr:
            try:
                out.add(datetime.fromisoformat(str(s)[:10]).date())
            except Exception:
                pass
        return out
    except Exception:
        return set()

def _is_macro_blackout(d: date, window: int) -> bool:
    """
    True if 'd' is within ±window days of any user-specified macro blackout date.
    window=0 → only the exact date.
    """
    if window < 0:
        window = 0
    days = _load_macro_blackout_dates()
    return any(abs((d - m).days) <= window for m in days)

def _extract_iso_dates(obj) -> list[date]:
    """Best-effort: dig out ISO 'YYYY-MM-DD' strings from nested Schwab JSON."""
    out: list[date] = []
    if obj is None:
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_extract_iso_dates(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_extract_iso_dates(it))
    elif isinstance(obj, str):
        s = obj.strip()
        try:
            out.append(datetime.fromisoformat(s[:10]).date())
        except Exception:
            pass
    return out

def next_earnings_date(symbol: str, ref_day: date) -> Optional[date]:
    """
    Query Schwab endpoints and return the earliest earnings date >= ref_day.
    Returns None if not available.
    """
    look_to = ref_day + timedelta(days=90)
    candidates = [
        ("marketdata/earnings", {"symbol": symbol, "startDate": ref_day.isoformat(), "endDate": look_to.isoformat()}),
        (f"fundamentals/earnings/{symbol}", {"start": ref_day.isoformat(), "end": look_to.isoformat()}),
        (f"fundamentals/calendar/{symbol}", {"type": "earnings", "start": ref_day.isoformat(), "end": look_to.isoformat()}),
    ]
    best: Optional[date] = None
    for ep, params in candidates:
        try:
            data = schwab_get(ep, params=params or {}, cache=None, ttl=6*3600) or {}
        except Exception:
            data = {}
        if not data:
            continue
        for d in _extract_iso_dates(data):
            if d >= ref_day and (best is None or d < best):
                best = d
    return best

def within_earnings_blackout(symbol: str, d: date, days: int) -> bool:
    """True if symbol has earnings within 'days' ahead of d (inclusive)."""
    if days <= 0:
        return False
    nxt = next_earnings_date(symbol, d)
    return bool(nxt and 0 <= (nxt - d).days <= days)

def append_top10_leaders_log(snapshot_date, leaders_df: pd.DataFrame, path: Path = TOP10_LEADERS_PATH) -> None:
    """
    Append one record to top10_leaders.json containing:
      - snapshot_date: the data's snapshot date (df['Date'].max())
      - run_at: the wall-clock timestamp of this ranking run
      - top_k: number of leaders saved (usually 10)
      - leaders: [{Name, Sector, RankScore}, ...]
    """
    if not isinstance(leaders_df, pd.DataFrame) or leaders_df.empty:
        return

    try:
        snap_str = pd.to_datetime(snapshot_date).date().isoformat()
    except Exception:
        snap_str = date.today().isoformat()

    # Make leaders JSON-friendly
    rows = []
    for _, r in leaders_df.iterrows():
        rows.append({
            "Name":   str(r.get("Name", "")),
            "Sector": str(r.get("Sector", "")),
            "RankScore": (float(r.get("RankScore")) if pd.notna(r.get("RankScore")) else None),
        })

    payload = {
        "snapshot_date": snap_str,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "top_k": int(len(rows)),
        "leaders": rows,
    }

    # Load → append → save
    data = []
    if path.exists():
        try:
            data = json.loads(path.read_text()) or []
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

    data.append(payload)
    _atomic_write_json(path, data)
    print(f"[Saved Top-10 leaders] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TINY CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cache_load(name, ttl):
    fp = CACHE_DIR / f"{name}.pkl"
    if fp.exists() and time.time() - fp.stat().st_mtime < ttl:
        return pickle.loads(fp.read_bytes())
    return None

def cache_save(name, obj):
    (CACHE_DIR / f"{name}.pkl").write_bytes(pickle.dumps(obj))

# ─────────────────────────────────────────────────────────────────────────────
# ATOMIC JSON WRITE (fixes NameError in append_top10_leaders_log)
# ─────────────────────────────────────────────────────────────────────────────
def _atomic_write_json(path: Path, obj) -> None:
    """
    Write JSON atomically: write to a temp file in the same directory
    and replace the target path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    tmp.write_text(data)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES: industry map, watchlist, sanitizer
# ─────────────────────────────────────────────────────────────────────────────
_TICKER_OK = re.compile(r'^[A-Z]{1,6}(?:[.-][A-Z]{1,3})?$')

def load_holdings() -> list[dict]:
    if HOLDINGS_PATH.exists():
        try:
            return json.loads(HOLDINGS_PATH.read_text())
        except Exception:
            pass
    return []

def save_holdings(holdings: list[dict]) -> None:
    HOLDINGS_PATH.write_text(json.dumps(holdings, indent=2, default=str))

def load_trade_history() -> list[dict]:
    if TRADE_HISTORY_PATH.exists():
        try:
            return json.loads(TRADE_HISTORY_PATH.read_text())
        except Exception:
            pass
    return []

def save_trade_history(trades: list[dict]) -> None:
    TRADE_HISTORY_PATH.write_text(json.dumps(trades, indent=2, default=str))

def _predicted_signals_path() -> Path:
    """
    Location for the staging file (next to watchlist.json for easy inspection).
    """
    try:
        base = LOCAL_DIR  # defined near WATCHLIST_PATH
    except NameError:
        base = Path(__file__).resolve().parent
    return base / "predicted_signals.json"

def load_predicted_signals() -> list[dict]:
    p = _predicted_signals_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return []

def save_predicted_signals(items: list[dict]) -> None:
    _atomic_write_json(_predicted_signals_path(), items)

def _dedupe_predicted(items: list[dict]) -> list[dict]:
    """
    Dedupe by (symbol, strategy, buy_z_lo, buy_z_hi, plan_entry, plan_stop).
    """
    seen = set()
    out = []
    for r in items:
        key = (
            str(r.get("symbol","")).upper(),
            str(r.get("strategy","")).strip(),
            float(r.get("buy_z_lo", float("nan"))),
            float(r.get("buy_z_hi", float("nan"))),
            float(r.get("plan_entry", float("nan"))),
            float(r.get("plan_stop", float("nan")))
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def _sanitize_tickers(seq) -> list[str]:
    out = []
    for t in (seq or []):
        t = str(t or "").strip().upper()
        if _TICKER_OK.match(t):
            out.append(t)
    return sorted(set(out))

def _norm_industry_key(s: str) -> str:
    s = re.sub(r'[\u2010-\u2015\u2212]', '-', str(s or ''))
    s = re.sub(r'[^a-z0-9]+', ' ', s.lower())
    return re.sub(r'\s+', ' ', s).strip()

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()

def _load_industry_ticker_map() -> dict[str, list[str]]:
    """
    Loads and normalizes { industry_key: [TICKER,...] }. First match wins.
    """
    for p in IND_TICKER_PATHS:
        if p and p.exists():
            try:
                raw = json.loads(p.read_text()) or {}
            except Exception as e:
                print(f"[WARN] Could not parse {p}: {e}")
                continue
            out: dict[str, list[str]] = {}
            for k, v in raw.items():
                if isinstance(v, (list, tuple, set)):
                    out[_norm_industry_key(k)] = _sanitize_tickers(v)
            if out:
                print(f"[INFO] Loaded {len(out)} industries from {p}")
                return out
    return {}

def load_watchlist() -> List[str]:
    if WATCHLIST_PATH.exists():
        try:
            return _sanitize_tickers(json.loads(WATCHLIST_PATH.read_text()))
        except Exception:
            pass
    return []

def save_watchlist(tickers: Iterable[str]) -> None:
    WATCHLIST_PATH.write_text(json.dumps(_sanitize_tickers(tickers), indent=2))

def _proposed_trades_path() -> Path:
    """
    Path for Proposed Trades staging file; lives next to holdings.json.
    """
    try:
        return HOLDINGS_PATH.with_name("proposed_trades.json")
    except Exception:
        # Fallback to local dir if HOLDINGS_PATH missing for some reason
        return Path(__file__).resolve().parent / "proposed_trades.json"


def load_proposed_trades() -> list[dict]:
    p = _proposed_trades_path()
    if p.exists():
        try:
            data = json.loads(p.read_text()) or []
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_proposed_trades(rows: list[dict]) -> None:
    """
    Atomic save to avoid file corruption.
    """
    _atomic_write_json(_proposed_trades_path(), rows)


def _in_zone(last: float, lo: float, hi: float) -> bool:
    if not (np.isfinite(last) and np.isfinite(lo) and np.isfinite(hi)):
        return False
    lo, hi = min(lo, hi), max(lo, hi)
    return (lo <= last <= hi)


def _zone_from_entry_and_atr(entry: float, atr: float, strategy_slug: str) -> tuple[float, float]:
    """
    Direction-aware buy zone from entry & ATR:
      • Breakout/Momo/Squeeze → zone ABOVE entry
      • Pullback HG → zone BELOW entry
    Uses your existing _entry_buffer() heuristics (percent band scaled by ATR/price).
    """
    if not np.isfinite(entry) or not np.isfinite(atr) or entry <= 0 or atr <= 0:
        # Conservative default ±0.25–0.75% band around entry
        pct_low, pct_high = 0.0025, 0.0075
        return entry * (1 + pct_low), entry * (1 + pct_high)

    pct_lo, pct_hi = _entry_buffer(entry, atr)  # returns fractions like 0.002 → 0.2%

    strat = (strategy_slug or "").lower()
    if ("pullback" in strat) or ("holy" in strat):
        # Buy a pullback: zone BELOW entry (wider first)
        lo = entry * (1 - max(pct_lo, pct_hi))
        hi = entry * (1 - min(pct_lo, pct_hi))
        return lo, hi

    # Breakout/momo/squeeze default: zone ABOVE entry
    lo = entry * (1 + min(pct_lo, pct_hi))
    hi = entry * (1 + max(pct_lo, pct_hi))
    return lo, hi


def _build_plan_from_signal_row(r: pd.Series) -> Optional[dict]:
    """
    Build a proposed plan from a CONFIRMED signal row (Current Signals table).
    Expected columns: date, ticker, entry, atr14 (optional), stop, target, strategy
    """
    try:
        sym   = str(r.get("ticker", "")).upper()
        if not sym:
            return None
        entry = float(r.get("entry", np.nan))
        stop  = float(r.get("stop",  np.nan))
        targ  = float(r.get("target", np.nan)) if pd.notna(r.get("target")) else np.nan
        atr   = float(r.get("atr14", np.nan)) if pd.notna(r.get("atr14")) else np.nan
        strat = str(r.get("strategy", "Current")).strip()

        if not np.isfinite(entry):
            return None

        # If stop/target missing, derive via simple RR levels around a midpoint entry
        zone_lo, zone_hi = _zone_from_entry_and_atr(entry, atr if np.isfinite(atr) else max(entry*0.01, 0.1), strat)
        mid_entry = round((zone_lo + zone_hi) / 2.0, 2)

        if not np.isfinite(stop):
            s, t = rr_levels(mid_entry, (atr if np.isfinite(atr) else max(mid_entry*0.01, 0.1)))
            stop_calc, t2_calc = float(s), float(t)
        else:
            stop_calc = float(stop)
            t2_calc   = float(targ) if np.isfinite(targ) else float(round(entry + 2.0 * (entry - stop_calc), 2))

        t1_calc = round(mid_entry + 1.0 * (mid_entry - stop_calc), 2)
        R_val   = round((mid_entry - stop_calc), 2)
        rr_t2   = round((t2_calc - mid_entry) / max(1e-6, (mid_entry - stop_calc)), 2)

        last = _last_close_price_schwab(sym)

        status = ("ENTERED" if _in_zone(last, zone_lo, zone_hi)
                  else ("MISSED" if (np.isfinite(last) and last > max(zone_lo, zone_hi)) else "WAITING"))

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "valid_through": (date.today() + timedelta(days=3)).isoformat(),  # 3 sessions default
            "type": "signal",
            "date": str(r.get("date")) if pd.notna(r.get("date")) else date.today().isoformat(),
            "ticker": sym,
            "strategy": strat,
            "pivot": float(round(entry, 2)),
            "plan_entry": float(mid_entry),
            "buy_z_lo": float(round(zone_lo, 2)),
            "buy_z_hi": float(round(zone_hi, 2)),
            "plan_stop": float(round(stop_calc, 2)),
            "t1": float(round(t1_calc, 2)),
            "t2": float(round(t2_calc, 2)),
            "R": float(round(R_val, 2)),
            "rr_to_t2": float(rr_t2),
            "last": (float(last) if last is not None else None),
            "status": status,
            "notes": "Auto‑derived from Confirmed Signal"
        }
    except Exception:
        return None


def _build_plan_from_pre_row(r: pd.Series) -> Optional[dict]:
    """
    Build a proposed plan from a Forecast Pre‑Signal row.
    Expected columns: date,time,ticker,buy_z_lo,buy_z_hi,pivot,plan_entry,plan_stop,R,t1,t2,rr_to_t2,atr14,strategy
    """
    try:
        sym   = str(r.get("ticker", "")).upper()
        if not sym:
            return None
        pivot = float(r.get("pivot", np.nan))
        plan_entry = float(r.get("plan_entry", np.nan))
        buy_lo = float(r.get("buy_z_lo", np.nan))
        buy_hi = float(r.get("buy_z_hi", np.nan))
        stop0  = float(r.get("plan_stop", np.nan))
        t1     = float(r.get("t1", np.nan))
        t2     = float(r.get("t2", np.nan))
        rr_t2  = float(r.get("rr_to_t2", np.nan)) if pd.notna(r.get("rr_to_t2")) else np.nan
        R_val  = float(r.get("R", np.nan)) if pd.notna(r.get("R")) else np.nan
        strat  = str(r.get("strategy", "Pattern + Forecast (pre)")).strip()

        if not (np.isfinite(buy_lo) and np.isfinite(buy_hi)):
            # If range absent, synthesize around pivot
            atr = float(r.get("atr14", np.nan))
            buy_lo, buy_hi = _zone_from_entry_and_atr((pivot if np.isfinite(pivot) else plan_entry), (atr if np.isfinite(atr) else max((plan_entry or 1.0)*0.01, 0.1)), strat)
        if not np.isfinite(plan_entry):
            plan_entry = round((buy_lo + buy_hi) / 2.0, 2)
        if not np.isfinite(stop0):
            stop0 = round(plan_entry - max(0.10, abs(plan_entry - buy_lo)), 2)
        if not np.isfinite(t1):
            t1 = round(plan_entry + (plan_entry - stop0), 2)
        if not np.isfinite(t2):
            t2 = round(plan_entry + 2.0 * (plan_entry - stop0), 2)
        if not np.isfinite(R_val):
            R_val = round(plan_entry - stop0, 2)
        if not np.isfinite(rr_t2) and np.isfinite(t2):
            rr_t2 = round((t2 - plan_entry) / max(1e-6, (plan_entry - stop0)), 2)

        last = _last_close_price_schwab(sym)
        status = ("ENTERED" if _in_zone(last, buy_lo, buy_hi)
                  else ("MISSED" if (np.isfinite(last) and last > max(buy_lo, buy_hi)) else "WAITING"))

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "valid_through": (date.today() + timedelta(days=3)).isoformat(),
            "type": "pre",
            "date": str(r.get("date")) if pd.notna(r.get("date")) else date.today().isoformat(),
            "ticker": sym,
            "strategy": strat,
            "pivot": (float(round(pivot, 2)) if np.isfinite(pivot) else float(round(plan_entry, 2))),
            "plan_entry": float(round(plan_entry, 2)),
            "buy_z_lo": float(round(min(buy_lo, buy_hi), 2)),
            "buy_z_hi": float(round(max(buy_lo, buy_hi), 2)),
            "plan_stop": float(round(stop0, 2)),
            "t1": float(round(t1, 2)),
            "t2": float(round(t2, 2)),
            "R": float(round(R_val, 2)),
            "rr_to_t2": float(round(rr_t2, 2)) if np.isfinite(rr_t2) else None,
            "last": (float(last) if last is not None else None),
            "status": status,
            "notes": "Auto‑derived from Forecast Pre‑Signal"
        }
    except Exception:
        return None


def stage_proposals_from_scan(sig_df: pd.DataFrame, pre_df: pd.DataFrame, *, valid_days: int = 3) -> int:
    """
    Merge new proposals from scan outputs into proposed_trades.json.
    De‑dupe by (ticker, strategy, type); keep the most recent created_at.
    Returns number of new/updated proposals persisted.
    """
    existing = load_proposed_trades()
    # Index existing by key
    by_key = {}
    for row in existing:
        k = (row.get("ticker","").upper(), row.get("strategy",""), row.get("type","signal"))
        by_key[k] = row

    added_or_updated = 0

    if isinstance(sig_df, pd.DataFrame) and not sig_df.empty:
        for _, r in sig_df.iterrows():
            plan = _build_plan_from_signal_row(r)
            if not plan:
                continue
            k = (plan["ticker"], plan["strategy"], plan["type"])
            prev = by_key.get(k)
            if not prev or plan["created_at"] > prev.get("created_at",""):
                by_key[k] = plan
                added_or_updated += 1

    if isinstance(pre_df, pd.DataFrame) and not pre_df.empty:
        for _, r in pre_df.iterrows():
            plan = _build_plan_from_pre_row(r)
            if not plan:
                continue
            k = (plan["ticker"], plan["strategy"], plan["type"])
            prev = by_key.get(k)
            if not prev or plan["created_at"] > prev.get("created_at",""):
                by_key[k] = plan
                added_or_updated += 1

    if added_or_updated:
        save_proposed_trades(list(by_key.values()))
    return added_or_updated


def _evaluate_trade_outcome(
    bars: pd.DataFrame,
    entry_date: date,
    entry_px: float,
    stop_px: float,
    target_px: float,
    horizon_end: date,
    stop_first_on_same_day: bool = True
) -> dict:
    """
    Evaluate outcome from the session AFTER 'entry_date' through 'horizon_end' (inclusive).
    Uses daily OHLC to detect intraday touches. If both levels are touched the same day,
    assumes STOP occurs first when 'stop_first_on_same_day' is True (conservative).
    Returns dict with keys: outcome ('TARGET'|'STOP'|'OPEN'), exit_date, exit_px, R, days_held.
    """
    if bars is None or bars.empty:
        return {"outcome": "OPEN", "exit_date": None, "exit_px": None, "R": None, "days_held": 0}

    idx = bars.index.normalize()
    start_mask = idx > pd.Timestamp(entry_date)                    # start the next session after entry close
    end_mask   = idx <= pd.Timestamp(horizon_end)
    fwd = bars.loc[start_mask & end_mask].copy()

    if fwd.empty:
        # No forward data → keep open at last known close
        last_close = float(bars.loc[idx == pd.Timestamp(entry_date), "close"].iloc[-1]) \
                     if (idx == pd.Timestamp(entry_date)).any() else None
        risk = (entry_px - stop_px) if (entry_px is not None and stop_px is not None) else None
        r = ((last_close - entry_px) / risk) if (risk and risk > 0 and last_close is not None) else None
        return {"outcome": "OPEN", "exit_date": None, "exit_px": last_close, "R": r, "days_held": 0}

    result = {"outcome": "OPEN", "exit_date": fwd.index[-1].date(), "exit_px": float(fwd["close"].iloc[-1])}
    for ts, row in fwd.iterrows():
        lo = float(row["low"]); hi = float(row["high"])
        if lo <= stop_px and hi >= target_px:
            if stop_first_on_same_day:
                result = {"outcome": "STOP", "exit_date": ts.date(), "exit_px": stop_px}
            else:
                result = {"outcome": "TARGET", "exit_date": ts.date(), "exit_px": target_px}
            break
        elif lo <= stop_px:
            result = {"outcome": "STOP", "exit_date": ts.date(), "exit_px": stop_px}
            break
        elif hi >= target_px:
            result = {"outcome": "TARGET", "exit_date": ts.date(), "exit_px": target_px}
            break

    # Compute R and days_held
    risk = (entry_px - stop_px)
    R = ((result["exit_px"] - entry_px) / risk) if (risk and risk > 0 and result["exit_px"] is not None) else None
    days_held = (pd.Timestamp(result["exit_date"]) - pd.Timestamp(entry_date)).days if result["exit_date"] else 0
    result["R"] = float(R) if R is not None and np.isfinite(R) else None
    result["days_held"] = int(days_held) if days_held is not None else 0
    return result

def _simulate_trade_path(indf: pd.DataFrame,
                         entry_idx: int,
                         rules: Optional[RiskRules] = None,
                         stop0_override: Optional[float] = None,
                         target0_override: Optional[float] = None) -> dict:
    """
    indf: OHLCV (+ indicators). Entry at bar 'entry_idx' close.
    If stop0_override/target0_override provided, they become the initial levels.
    Returns: entry_date, exit_date, entry, exit_px, outcome, R, days_held, stop, target
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
    atr_e       = float(atr14.iloc[entry_idx]) if np.isfinite(atr14.iloc[entry_idx]) else float(atr14.dropna().iloc[0]) if atr14.dropna().size else 0.0
    swing_low   = _swing_low_before(indf, entry_idx, rules.swing_lookback)

    # Initial levels
    if (stop0_override is not None) and np.isfinite(stop0_override):
        stop0 = float(stop0_override)
        if not (stop0 < entry_price):  # safety
            stop0 = entry_price - max(1e-3, rules.atr_mult_stop * max(atr_e, 1e-3))
        R = max(1e-6, entry_price - stop0)
        if (target0_override is not None) and np.isfinite(target0_override):
            first_scale = float(target0_override)
            if not (first_scale > entry_price):  # safety
                first_scale = entry_price + rules.first_scale_R * R
        else:
            first_scale = entry_price + rules.first_scale_R * R
    else:
        stop0, R, first_scale = _plan_initial_levels(entry_price, atr_e, swing_low, rules)

    pos_open_frac = 1.0
    partials = []
    highest_close = entry_price
    trail_mult = rules.chandelier_mult
    trail_active = False
    stop = stop0

    # ── NEW: pyramiding legs (fractions measured in "original-share equivalents")
    pyramid_positions: list[dict] = []
    triggers_hit: set[float] = set()

    def _exceptional_run(i_now: int) -> bool:
        j = min(n-1, entry_idx + rules.accel_bars)
        if j <= entry_idx: return False
        max_close = float(closes.iloc[entry_idx+1:j+1].max())
        return (max_close - entry_price) >= (rules.accel_R * R)

    # Add a pyramid leg on bar i at trigger kR (fill at max(level, open), capped by high)
    def _try_add_pyramid(i: int, kR: float):
        if (not rules.pyramid_on) or (len(pyramid_positions) >= rules.pyramid_max_adds) or (kR in triggers_hit):
            return
        level = entry_price + kR * R
        if highs.iloc[i] < level:
            return
        entry_add = max(level, float(opens.iloc[i]))
        entry_add = min(entry_add, float(highs.iloc[i]))
        risk_add_ps = max(entry_add - stop, 1e-6)
        # choose risk fraction for this trigger (fallback to last provided if needed)
        idx = list(rules.pyramid_triggers_R).index(kR) if kR in rules.pyramid_triggers_R else 0
        risk_frac = (rules.pyramid_risk_fracs[idx] if idx < len(rules.pyramid_risk_fracs) else rules.pyramid_risk_fracs[-1])
        shares_equiv = float(risk_frac * (R / risk_add_ps))
        if shares_equiv <= 0:
            return
        pyramid_positions.append(dict(entry=entry_add, shares_frac=shares_equiv, date=str(indf.index[i].date()), trigger_R=kR))
        triggers_hit.add(kR)

    for i in range(entry_idx + 1, n):
        highest_close = max(highest_close, float(closes.iloc[i-1]))
        atr_now = float(atr14.iloc[i]) if np.isfinite(atr14.iloc[i]) else atr_e

        if trail_active:
            ch_stop = _chandelier_stop(highest_close, atr_now, trail_mult)
            stop = max(stop, ch_stop)
            if rules.trail_use_ma_fail_safe and (not np.isnan(ma10.iloc[i-1])) and closes.iloc[i-1] < ma10.iloc[i-1]:
                exit_px = float(opens.iloc[i]) if "open" in indf.columns else float(closes.iloc[i])
                adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
                r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
                return dict(
                    entry_date=str(indf.index[entry_idx].date()),
                    exit_date=str(indf.index[i].date()),
                    entry=round(entry_price,2),
                    exit_px=round(exit_px,2),
                    outcome="MA_EXIT",
                    R=round(r_mult, 2),
                    days_held=(i - entry_idx),
                    stop=round(stop0,2),
                    target=round(first_scale,2)
                )

        # STOP check (core + pyramid legs exit together)
        if lows.iloc[i] <= stop:
            exit_px = stop
            adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
            r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
            return dict(
                entry_date=str(indf.index[entry_idx].date()),
                exit_date=str(indf.index[i].date()),
                entry=round(entry_price,2),
                exit_px=round(exit_px,2),
                outcome="STOP",
                R=round(r_mult, 2),
                days_held=(i - entry_idx),
                stop=round(stop0,2),
                target=round(first_scale,2)
            )

        # First scale (partial) on core at +1R
        pf = float(rules.first_scale_fraction)
        if (pos_open_frac > (1.0 - rules.first_scale_fraction)) and (highs.iloc[i] >= first_scale):
            realized = (first_scale - entry_price) / R * pf
            partials.append(dict(date=str(indf.index[i].date()),
                                 price=round(first_scale,2),
                                 frac=pf, R_realized=round(realized,2)))
            pos_open_frac = max(0.0, pos_open_frac - pf)
            if rules.move_to_breakeven_on_first_scale:
                stop = max(stop, entry_price)
            trail_active = True
            if _exceptional_run(i):
                trail_mult = min(trail_mult, rules.accel_chandelier_mult)

        # ── NEW: pyramiding triggers (+1R, +2R) after partial logic
        if rules.pyramid_on:
            for kR in rules.pyramid_triggers_R:
                _try_add_pyramid(i, kR)

        # Time stop
        held_bars = i - entry_idx
        if held_bars >= rules.max_bars_in_trade:
            exit_px = float(closes.iloc[i])
            adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
            r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
            return dict(
                entry_date=str(indf.index[entry_idx].date()),
                exit_date=str(indf.index[i].date()),
                entry=round(entry_price,2),
                exit_px=round(exit_px,2),
                outcome="TIME",
                R=round(r_mult, 2),
                days_held=held_bars,
                stop=round(stop0,2),
                target=round(first_scale,2)
            )

    # Still open on last bar
    exit_px = float(closes.iloc[-1])
    adds_realized = sum(((exit_px - p["entry"]) / R) * p["shares_frac"] for p in pyramid_positions)
    r_mult = (exit_px - entry_price) / R * pos_open_frac + sum(p["R_realized"] for p in partials) + adds_realized
    return dict(
        entry_date=str(indf.index[entry_idx].date()),
        exit_date=None,
        entry=round(entry_price,2),
        exit_px=round(exit_px,2),
        outcome="OPEN",
        R=round(r_mult, 2),
        days_held=(n - 1 - entry_idx),
        stop=round(stop0,2),
        target=round(first_scale,2)
    )

def scan_on_date_backtest(tickers: List[str],
                          as_of: date,
                          * ,
                          strategies: Optional[List[str]] = None,
                          forecast_hours: Optional[int] = None) -> pd.DataFrame:
    if not strategies:
        strategies = STRATEGY_SLUGS

    rows = []
    dbg = DEBUG

    uniq = sorted(set(tickers))
    print("\nSchwab: backtest (single date) — pulling bars & simulating …")
    n = len(uniq)

    for i, sym in enumerate(uniq, start=1):
        _status_bar(i, n, prefix="  backtest")
        try:
            bars = get_bars(sym, as_of - timedelta(days=140), as_of + timedelta(days=40), min_history_days=260)
            if bars is None or len(bars) < 60:
                if dbg: print(f"[DEBUG] backtest({as_of}): {sym} insufficient bars")
                continue

            indf = compute_signals_for_ticker(bars, strategy="all", as_of=None)

            # entry index on 'as_of' (or last <= as_of)
            mask_day = indf.index.normalize() == pd.Timestamp(as_of)
            if not mask_day.any():
                elig = indf.loc[indf.index.normalize() <= pd.Timestamp(as_of)]
                if elig.empty:
                    if dbg: print(f"[DEBUG] backtest({as_of}): {sym} no row <= as_of")
                    continue
                entry_idx = indf.index.get_loc(elig.index[-1])
            else:
                entry_idx = indf.index.get_loc(indf.loc[mask_day].index[-1])

            row = indf.iloc[entry_idx]

            # detect signal for any strategy slug present
            sig = False
            for strat in strategies:
                s, _ = _detect_signal_or_raw(row, strat, dbg=dbg)
                if s:
                    sig = True
                    break
            if not sig:
                continue

            if within_earnings_blackout(sym, as_of, RULES.earnings_blackout_days):
                continue
            if _is_macro_blackout(as_of, RULES.macro_blackout_window):
                continue

            best_slug = str(row.get("strategy_slug_best", "") or "")
            def _pick(names: List[str]) -> Optional[float]:
                for nm in names:
                    if nm and nm in row.index:
                        v = row.get(nm, np.nan)
                        if pd.notna(v) and np.isfinite(v):
                            return float(v)
                return None

            stop_override = _pick([f"stop0_{best_slug}", f"stop_{best_slug}", "stop0", "stop"])
            target_override = _pick([f"target_{best_slug}", f"target0_{best_slug}", "target", "target0"])

            res = TM_simulate_trade_path(indf, entry_idx, RULES,
                                         stop0_override=stop_override,
                                         target0_override=target_override)
            if res:
                atrv = float(indf["atr14"].iloc[entry_idx]) if "atr14" in indf.columns and pd.notna(indf["atr14"].iloc[entry_idx]) \
                       else float(atr_series(indf, 14).iloc[entry_idx])

                p1r = None
                if forecast_hours and isinstance(forecast_hours, int) and forecast_hours > 0:
                    entry_px = float(res["entry"])
                    R = float(res.get("entry", np.nan) - float(res.get("stop", np.nan))) if np.isfinite(res.get("stop", np.nan)) else (atrv if np.isfinite(atrv) else None)
                    if np.isfinite(entry_px) and np.isfinite(R):
                        start_dt = datetime.combine(as_of - timedelta(days=14), datetime.min.time())
                        end_dt   = datetime.combine(as_of, datetime.max.time())
                        intra = get_intraday_bars(sym, start_dt, end_dt, interval_minutes=5, minute_granularity=1, include_extended=False)
                        p1r = estimate_prob_hit_plus1R(intra, entry_px=entry_px, R_per_share=R, horizon_hours=forecast_hours, sessions_lookback=10)

                rows.append({
                    "date": res["entry_date"],
                    "ticker": sym,
                    "entry": res["entry"],
                    "atr14": round(atrv, 2),
                    "stop": res.get("stop"),
                    "target": res.get("target"),
                    "exit_date": res.get("exit_date"),
                    "exit_px": res.get("exit_px"),
                    "outcome": res.get("outcome"),
                    "R": res.get("R"),
                    "days_held": res.get("days_held"),
                    "strategy": str(row.get("strategy","—")),
                    "strategy_slug_best": best_slug,
                    "strategy_slugs": str(row.get("strategy_slugs","")),
                    "vol_score": (float(row.get("vol_score")) if pd.notna(row.get("vol_score", np.nan)) else None),
                    "adx": (float(row.get("adx")) if pd.notna(row.get("adx", np.nan)) else None),
                    "rr_best": (float(row.get("rr_best")) if pd.notna(row.get("rr_best", np.nan)) else None),
                    **({"p_hit_1R": (float(p1r) if p1r is not None else None)} if forecast_hours else {})
                })
        except Exception as e:
            if dbg: print(f"[DEBUG] scan_on_date_backtest: sym={sym} error: {e}")
            continue

    cols = ["date","ticker","entry","atr14","stop","target","exit_date","exit_px","outcome","R","days_held",
            "strategy","strategy_slug_best","strategy_slugs","vol_score","adx","rr_best"]
    if forecast_hours:
        cols.append("p_hit_1R")
    return pd.DataFrame(rows, columns=cols)

def scan_range_backtest(tickers: List[str],
                        start_d: date,
                        end_d: date,
                        * ,
                        strategies: Optional[List[str]] = None,
                        forecast_hours: Optional[int] = None) -> pd.DataFrame:
    if not strategies:
        strategies = STRATEGY_SLUGS

    results = []
    open_until: dict[str, pd.Timestamp] = {}
    cache: dict[str, pd.DataFrame] = {}
    dbg = DEBUG

    uniq = sorted(set(tickers))

    # Preload per symbol
    print("\nSchwab: backtest (range) — preloading bars per symbol …")
    n = len(uniq)
    for i, sym in enumerate(uniq, start=1):
        _status_bar(i, n, prefix="  preload")
        try:
            bars = get_bars(sym, start_d - timedelta(days=140), end_d + timedelta(days=40), min_history_days=260)
            if bars is None or bars.empty:
                if dbg: print(f"[DEBUG] range_backtest preload: {sym} no bars")
                continue
            cache[sym] = compute_signals_for_ticker(bars, strategy="all", as_of=None)
        except Exception as e:
            if dbg: print(f"[DEBUG] preload bars: sym={sym} error: {e}")
            continue

    # Per-day simulation
    total_days = (end_d - start_d).days + 1
    print("\nSchwab: backtest (range) — scanning days …")
    d = start_d
    day_idx = 0
    while d <= end_d:
        day_idx += 1
        _status_bar(day_idx, total_days, prefix="  days")
        ts_d = pd.Timestamp(d)
        for sym, indf in list(cache.items()):
            try:
                last_exit = open_until.get(sym)
                if last_exit is not None and ts_d <= last_exit:
                    continue

                mask_day = indf.index.normalize() == ts_d
                if not mask_day.any():
                    continue
                idx = indf.index.get_loc(indf.loc[mask_day].index[-1])
                row = indf.iloc[idx]

                sig = False
                for strat in strategies:
                    s, _ = _detect_signal_or_raw(row, strat, dbg=dbg)
                    if s:
                        sig = True
                        break
                if not sig:
                    continue

                if within_earnings_blackout(sym, d, RULES.earnings_blackout_days):
                    continue
                if _is_macro_blackout(d, RULES.macro_blackout_window):
                    continue

                best_slug = str(row.get("strategy_slug_best", "") or "")
                def _pick(names: List[str]) -> Optional[float]:
                    for nm in names:
                        if nm and nm in row.index:
                            v = row.get(nm, np.nan)
                            if pd.notna(v) and np.isfinite(v):
                                return float(v)
                    return None

                stop_override   = _pick([f"stop0_{best_slug}", f"stop_{best_slug}", "stop0", "stop"])
                target_override = _pick([f"target_{best_slug}", f"target0_{best_slug}", "target", "target0"])

                res = TM_simulate_trade_path(indf, idx, RULES,
                                             stop0_override=stop_override,
                                             target0_override=target_override)
                if res:
                    exit_ts = pd.Timestamp(res["exit_date"]) if res.get("exit_date") else pd.Timestamp(indf.index[-1])
                    open_until[sym] = exit_ts

                    atrv = float(indf["atr14"].iloc[idx]) if "atr14" in indf.columns and pd.notna(indf["atr14"].iloc[idx]) \
                           else float(atr_series(indf, 14).iloc[idx])

                    p1r = None
                    if forecast_hours and isinstance(forecast_hours, int) and forecast_hours > 0:
                        entry_px = float(res["entry"])
                        R = float(res.get("entry", np.nan) - float(res.get("stop", np.nan))) if np.isfinite(res.get("stop", np.nan)) else (atrv if np.isfinite(atrv) else None)
                        if np.isfinite(entry_px) and np.isfinite(R):
                            start_dt = datetime.combine(d - timedelta(days=14), datetime.min.time())
                            end_dt   = datetime.combine(d, datetime.max.time())
                            intra = get_intraday_bars(sym, start_dt, end_dt, interval_minutes=5, minute_granularity=1, include_extended=False)
                            p1r = estimate_prob_hit_plus1R(intra, entry_px=entry_px, R_per_share=R, horizon_hours=forecast_hours, sessions_lookback=10)

                    results.append({
                        "date": res["entry_date"],
                        "ticker": sym,
                        "entry": res["entry"],
                        "atr14": round(atrv, 2),
                        "stop": res.get("stop"),
                        "target": res.get("target"),
                        "exit_date": res.get("exit_date"),
                        "exit_px": res.get("exit_px"),
                        "outcome": res.get("outcome"),
                        "R": res.get("R"),
                        "days_held": res.get("days_held"),
                        "strategy": str(row.get("strategy","—")),
                        "strategy_slug_best": best_slug,
                        "strategy_slugs": str(row.get("strategy_slugs","")),
                        "vol_score": (float(row.get("vol_score")) if pd.notna(row.get("vol_score", np.nan)) else None),
                        "adx": (float(row.get("adx")) if pd.notna(row.get("adx", np.nan)) else None),
                        "rr_best": (float(row.get("rr_best")) if pd.notna(row.get("rr_best", np.nan)) else None),
                        **({"p_hit_1R": (float(p1r) if p1r is not None else None)} if forecast_hours else {})
                    })
            except Exception as e:
                if dbg: print(f"[DEBUG] scan_range_backtest: sym={sym} day={d} error: {e}")
                continue
        d += timedelta(days=1)

    cols = ["date","ticker","entry","atr14","stop","target","exit_date","exit_px","outcome","R","days_held",
            "strategy","strategy_slug_best","strategy_slugs","vol_score","adx","rr_best"]
    if forecast_hours:
        cols.append("p_hit_1R")
    return pd.DataFrame(results, columns=cols) if results else pd.DataFrame(columns=cols)


def compute_regime_risk_on(as_of: date) -> Optional[bool]:
    """
    Lightweight macro regime score (same ingredients as the ranker’s overlay):
    Yield curve change, M2 YoY, Fed Funds trend, USD trend, Utilities vs SPY, VIX.
    Returns True (Risk ON) / False (Risk OFF) / None (not enough data).
    """
    try:
        idx = pd.date_range(end=pd.to_datetime(as_of), periods=400, freq="D")
        t10 = fred_get("DGS10"); t3m = fred_get("DGS3MO"); m2 = fred_get("M2SL")
        dff = fred_get("DFF");   usd = fred_get("DTWEXBGS"); vix = fred_get("VIXCLS")

        def fx(s): return s.reindex(idx).ffill() if s is not None and not s.empty else pd.Series(index=idx, dtype=float)
        t10, t3m, m2, dff, usd, vix = map(fx, (t10,t3m,m2,dff,usd,vix))

        yc_slope = (t10 - t3m)
        yc_chg21 = yc_slope - yc_slope.shift(21)
        m2_yoy   = m2.pct_change(365, fill_method=None) * 100.0
        dff_tr21 = dff - dff.shift(21)
        usd_tr21 = (usd / usd.shift(21) - 1.0) * 100.0

        close_us, _, _ = _schwab_panel(["XLU","SPY"], idx.min()-pd.Timedelta(days=1), idx.max())
        if not close_us.empty and {"XLU","SPY"} <= set(close_us.columns):
            spy = close_us["SPY"].reindex(idx).ffill(); xlu = close_us["XLU"].reindex(idx).ffill()
            rs_utl = (xlu / xlu.iloc[0]) / (spy / spy.iloc[0])
            utl_tr21 = rs_utl - rs_utl.shift(21)
        else:
            utl_tr21 = pd.Series(index=idx, dtype=float)

        def z(s):
            m, st = s.mean(skipna=True), s.std(skipna=True)
            st = st if (st and st > 0) else 1.0
            return (s - m) / st

        score = z(yc_chg21) + z(m2_yoy) - z(dff_tr21) - z(usd_tr21) - z(utl_tr21) - z(vix)
        last = score.dropna()
        if last.empty:
            return None
        return bool(last.iloc[-1] > 0.0)
    except Exception:
        return None

def _schwab_daily(ticker: str,
                  start_dt: datetime,
                  end_dt: datetime) -> Optional[pd.DataFrame]:
    """
    Daily OHLCV via Schwab.
    Uses library method when available; else REST:
      GET /marketdata/v1/pricehistory?symbol=SPY
          &startDate=<ms>&endDate=<ms>&frequencyType=daily&frequency=1
    Returns normalized OHLCV or None.
    """
    if schwab_client is None:
        if DEBUG:
            print(f"[DEBUG] _schwab_daily({ticker}) skipped — schwab_client is None")
        return None

    # Normalize to UTC-aware datetimes
    s_utc = start_dt.astimezone(timezone.utc) if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
    e_utc = end_dt.astimezone(timezone.utc)   if end_dt.tzinfo   else end_dt.replace(tzinfo=timezone.utc)
    start_ms = int(round(s_utc.timestamp() * 1000))
    end_ms   = int(round(e_utc.timestamp() * 1000))

    data = None

    # 1) Try official client method (schwab-py >= 0.5)
    try:
        if hasattr(schwab_client, "get_price_history_every_day"):
            resp = schwab_client.get_price_history_every_day(
                ticker,
                start_datetime=s_utc,
                end_datetime=e_utc
            )
            if isinstance(resp, dict):
                data = resp
            elif hasattr(resp, "json"):
                data = resp.json()
            elif hasattr(resp, "__dict__"):
                d = getattr(resp, "__dict__", None) or {}
                data = d.get("priceHistory") or d
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] get_price_history_every_day failed: {e}")

    # 2) Raw REST (correct 'symbol' query parameter)
    def _fetch_once() -> Optional[requests.Response]:
        url = f"{SCHWAB_BASE_URL}/marketdata/v1/pricehistory"
        params = dict(
            symbol=ticker,
            startDate=start_ms,
            endDate=end_ms,
            frequencyType="daily",
            frequency=1
        )
        if DEBUG:
            print(f"[DEBUG] GET {url} params={params}")
        if hasattr(schwab_client, "session") and schwab_client.session:
            return schwab_client.session.get(url, params=params, timeout=30)
        hdrs = {}
        token = getattr(schwab_client, "access_token", None)
        if not token and hasattr(schwab_client, "access_tokens"):
            token = getattr(schwab_client.access_tokens, "access_token", None)
        if token:
            hdrs["Authorization"] = f"Bearer {token}"
        return _session_with_retries().get(url, headers=hdrs, params=params, timeout=30)

    if not data:
        for attempt in (0, 1):
            r = None
            try:
                r = _fetch_once()
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] _schwab_daily fetch error {ticker}: {e}")

            if r is not None and r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    data = None
                break

            # token refresh once on 401/403
            if r is not None and r.status_code in (401, 403) and attempt == 0:
                try:
                    if hasattr(schwab_client, "refresh_access_token"):
                        schwab_client.refresh_access_token()
                    else:
                        init_schwab_client(force_reauth=True)
                except Exception as e:
                    if DEBUG:
                        print(f"[DEBUG] _schwab_daily token refresh failed: {e}")
                continue

            if DEBUG and r is not None:
                body = (r.text or "")[:200].replace("\n", " ")
                print(f"[DEBUG] _schwab_daily {ticker} HTTP {r.status_code} {body}")
            break

    if not data:
        return None

    # “candles” may be nested
    candles = (data.get("candles") if isinstance(data, dict) else None) \
              or (data.get("priceHistory", {}).get("candles") if isinstance(data, dict) else None) \
              or []
    if not isinstance(candles, list) or not candles:
        return None

    df = pd.DataFrame(candles)
    # Parse datetime column → naive UTC
    dtcol = None
    for key in ("datetime", "time", "timestamp", "DateTime", "date"):
        if key in df.columns:
            try:
                if key.lower() in ("datetime", "time", "timestamp"):
                    dtcol = pd.to_datetime(df[key], unit="ms", utc=True, errors="coerce")
                else:
                    dtcol = pd.to_datetime(df[key], utc=True, errors="coerce")
                break
            except Exception:
                pass
    if dtcol is None:
        return None

    df = df.assign(datetime=dtcol.dt.tz_localize(None)).dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # Map OHLCV
    cols_lower = {str(c).lower(): c for c in df.columns}
    def _pick(*names):
        for nm in names:
            if nm.lower() in cols_lower:
                return cols_lower[nm.lower()]
        return None

    c_open  = _pick("open")
    c_high  = _pick("high")
    c_low   = _pick("low")
    c_close = _pick("close", "closeprice")
    c_vol   = _pick("volume", "totalvolume")
    if any(x is None for x in (c_open, c_high, c_low, c_close, c_vol)):
        return None

    out = df[[c_open, c_high, c_low, c_close, c_vol]].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Range filter with non-deprecated conversion (naive UTC bounds)
    s_naive = _ms_to_utc_naive(start_ms)
    e_naive = _ms_to_utc_naive(end_ms)
    out = out.loc[(out.index >= s_naive) & (out.index <= e_naive)]

    return out if not out.empty else None


def get_bars(ticker: str, start: date, end: date, min_history_days: int = 260) -> Optional[pd.DataFrame]:
    # Pull extra lookback for indicators
    start_dt = datetime.combine(start, datetime.min.time()) - timedelta(days=int(min_history_days * 1.1))
    # include the whole end date (23:59:59), then cap by "now - 16m" in UTC
    end_dt   = datetime.combine(end, datetime.max.time())
    now_cut  = datetime.now(timezone.utc) - timedelta(minutes=16)
    # make naive timestamps for request builder (we normalize to naive later)
    end_dt   = min(end_dt, now_cut.replace(tzinfo=None))
    return _schwab_daily(ticker, start_dt, end_dt)

def fred_get(series_id: str, ttl: int = 86_400) -> pd.Series:
    cache_key = f"fred_{series_id}"
    if (cached := cache_load(cache_key, ttl)) is not None:
        return cached
    if not FRED_KEY:
        return pd.Series(dtype=float)
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = _session_with_retries().get(
        url,
        params=dict(series_id=series_id, api_key=FRED_KEY, file_type="json"),
        timeout=15
    )
    r.raise_for_status()
    df = (pd.DataFrame(r.json()["observations"])
            .assign(date=lambda d: pd.to_datetime(d["date"]))
            .set_index("date")["value"]
            .pipe(pd.to_numeric, errors="coerce"))
    cache_save(cache_key, df)
    return df

register_data_providers(get_bars, fred_get)

def _schwab_intraday(ticker: str,
                     start_dt: datetime,
                     end_dt: datetime,
                     *,
                     minute_granularity: int = 1,
                     include_extended: bool = False) -> Optional[pd.DataFrame]:
    """
    Intraday OHLCV via Schwab minute endpoint.
      GET /marketdata/v1/pricehistory?symbol=SPY
          &startDate=<ms>&endDate=<ms>&frequencyType=minute&frequency=<N>
          &needExtendedHoursData=true|false
    """
    if schwab_client is None:
        if DEBUG:
            print(f"[DEBUG] _schwab_intraday({ticker}) skipped — schwab_client is None")
        return None

    s_utc = start_dt.astimezone(timezone.utc) if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
    e_utc = end_dt.astimezone(timezone.utc)   if end_dt.tzinfo   else end_dt.replace(tzinfo=timezone.utc)
    start_ms = int(round(s_utc.timestamp() * 1000))
    end_ms   = int(round(e_utc.timestamp() * 1000))

    data = None

    # 1) Library method if available
    try:
        if hasattr(schwab_client, "get_price_history_every_minute"):
            resp = schwab_client.get_price_history_every_minute(
                ticker,
                start_datetime=s_utc,
                end_datetime=e_utc,
                need_extended_hours_data=bool(include_extended)
            )
            if isinstance(resp, dict):
                data = resp
            elif hasattr(resp, "json"):
                data = resp.json()
            elif hasattr(resp, "__dict__"):
                d = getattr(resp, "__dict__", None) or {}
                data = d.get("priceHistory") or d
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] get_price_history_every_minute failed: {e}")

    # 2) Raw REST
    def _fetch_once() -> Optional[requests.Response]:
        url = f"{SCHWAB_BASE_URL}/marketdata/v1/pricehistory"
        params = dict(
            symbol=ticker,
            startDate=start_ms,
            endDate=end_ms,
            frequencyType="minute",
            frequency=int(max(1, minute_granularity)),
            needExtendedHoursData="true" if include_extended else "false",
        )
        if DEBUG:
            print(f"[DEBUG] GET {url} params={params}")
        if hasattr(schwab_client, "session") and schwab_client.session:
            return schwab_client.session.get(url, params=params, timeout=30)
        hdrs = {}
        token = getattr(schwab_client, "access_token", None)
        if not token and hasattr(schwab_client, "access_tokens"):
            token = getattr(schwab_client.access_tokens, "access_token", None)
        if token:
            hdrs["Authorization"] = f"Bearer {token}"
        return _session_with_retries().get(url, headers=hdrs, params=params, timeout=30)

    if not data:
        for attempt in (0, 1):
            r = None
            try:
                r = _fetch_once()
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] _schwab_intraday fetch error {ticker}: {e}")

            if r is not None and r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    data = None
                break

            if r is not None and r.status_code in (401, 403) and attempt == 0:
                try:
                    if hasattr(schwab_client, "refresh_access_token"):
                        schwab_client.refresh_access_token()
                    else:
                        init_schwab_client(force_reauth=True)
                except Exception as e:
                    if DEBUG:
                        print(f"[DEBUG] _schwab_intraday token refresh failed: {e}")
                continue

            if DEBUG and r is not None:
                body = (r.text or "")[:200].replace("\n", " ")
                print(f"[DEBUG] _schwab_intraday {ticker} HTTP {r.status_code} {body}")
            break

    if not data:
        return None

    candles = (data.get("candles") if isinstance(data, dict) else None) \
              or (data.get("priceHistory", {}).get("candles") if isinstance(data, dict) else None) \
              or []
    if not isinstance(candles, list) or not candles:
        return None

    df = pd.DataFrame(candles)
    dtcol = None
    for key in ("datetime", "time", "timestamp", "DateTime", "date"):
        if key in df.columns:
            try:
                if key.lower() in ("datetime", "time", "timestamp"):
                    dtcol = pd.to_datetime(df[key], unit="ms", utc=True, errors="coerce")
                else:
                    dtcol = pd.to_datetime(df[key], utc=True, errors="coerce")
                break
            except Exception:
                pass
    if dtcol is None:
        return None

    df = df.assign(datetime=dtcol.dt.tz_localize(None)).dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    cols_lower = {str(c).lower(): c for c in df.columns}
    def _pick(*names):
        for nm in names:
            if nm.lower() in cols_lower:
                return cols_lower[nm.lower()]
        return None
    c_open  = _pick("open")
    c_high  = _pick("high")
    c_low   = _pick("low")
    c_close = _pick("close", "closeprice")
    c_vol   = _pick("volume", "totalvolume")
    if any(x is None for x in (c_open, c_high, c_low, c_close, c_vol)):
        return None

    out = df[[c_open, c_high, c_low, c_close, c_vol]].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Non-deprecated range masking (naive UTC bounds)
    s_naive = _ms_to_utc_naive(start_ms)
    e_naive = _ms_to_utc_naive(end_ms)
    out = out.loc[(out.index >= s_naive) & (out.index <= e_naive)]

    return out if not out.empty else None


def get_intraday_bars(ticker: str,
                      start: datetime,
                      end: datetime,
                      *,
                      interval_minutes: int = 30,
                      minute_granularity: int = 1,
                      include_extended: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch minute bars and resample to `interval_minutes`.
    Returns OHLCV dataframe indexed by datetime. None on failure.
    """
    raw = _schwab_intraday(
        ticker, start, end,
        minute_granularity=minute_granularity,
        include_extended=include_extended
    )
    if raw is None or raw.empty:
        return None

    # Resample to uniform interval
    rule = f"{int(max(1, interval_minutes))}T"
    agg = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume":"sum"
    }
    try:
        res = raw.resample(rule).agg(agg).dropna(how="any")
    except Exception:
        return None
    return res if not res.empty else None


def refine_stop_with_intraday(symbol: str,
                              session_day: date,
                              entry_px: float,
                              stop_fallback: float,
                              day_row: pd.Series,
                              lookback_minutes: int = 90) -> float:
    """
    Wrapper that delegates to trade_management.refine_stop_with_intraday.
    Requires register_intraday_provider(...) to have been called.
    """
    try:
        return float(
            TM_refine_stop(
                symbol, session_day, entry_px, stop_fallback, day_row,
                lookback_minutes=lookback_minutes
            )
        )
    except Exception:
        return float(stop_fallback)


def estimate_prob_hit_plus1R(intra: pd.DataFrame,
                             *,
                             entry_px: float,
                             R_per_share: float,
                             horizon_hours: int = 4,
                             sessions_lookback: int = 10) -> Optional[float]:
    """
    Estimate P(+1R hit within next `horizon_hours`) from recent intraday excursions.
    For each of the last `sessions_lookback` sessions, measure up/down excursion
    from the first bar to horizon X hours. Approximate first-touch ordering via:
       p ≈ (up_only + 0.5 * both) / total
    Returns float in [0,1] or None if insufficient data.
    """
    if intra is None or intra.empty or not np.isfinite(entry_px) or not np.isfinite(R_per_share):
        return None
    try:
      horizon = pd.Timedelta(hours=max(1, int(horizon_hours)))
      dates = sorted({ts.date() for ts in intra.index})
      if len(dates) < 2:
          return None
      dates = dates[-int(max(2, sessions_lookback)):]
      up_only = down_only = both = neither = 0

      thr_frac = (R_per_share / entry_px)
      for d in dates:
          day = intra.loc[intra.index.date == d]
          if day.empty: 
              continue
          t0 = day.index.min(); t1 = t0 + horizon
          win = day.loc[(day.index >= t0) & (day.index <= t1)]
          if win.empty or len(win) < 2:
              continue
          px0 = float(win["open"].iloc[0]) if "open" in win.columns else float(win["close"].iloc[0])
          hi = float(win["high"].max()); lo = float(win["low"].min())
          up = (hi - px0) / px0; dn = (px0 - lo) / px0
          hit_up   = (up >= thr_frac)
          hit_down = (dn >= thr_frac)
          if  hit_up and not hit_down: up_only   += 1
          elif hit_down and not hit_up: down_only += 1
          elif hit_up and hit_down:     both      += 1
          else:                         neither   += 1

      n = up_only + down_only + both + neither
      if n == 0:
          return None
      return float(max(0.0, min(1.0, (up_only + 0.5 * both) / n)))
    except Exception:
      return None

def _schwab_panel(tickers: list[str],
                  start: pd.Timestamp | None = None,
                  end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build (date × ticker) wide panels for close, open, volume via Schwab.
    Avoids pandas fragmentation by constructing frames in one shot.
    """
    frames: dict[str, pd.DataFrame] = {}
    ustart = (start.to_pydatetime() if isinstance(start, pd.Timestamp) else (datetime.utcnow() - timedelta(days=800)))
    uend   = (end.to_pydatetime()   if isinstance(end,   pd.Timestamp) else datetime.utcnow())

    for t in sorted(set(tickers)):
        bars = _schwab_daily(t, ustart, uend)
        if bars is None or bars.empty:
            continue
        f = (bars.copy()
                 .reset_index()
                 .rename(columns={"datetime": "date"}))
        f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None)
        if start is not None:
            f = f[f["date"] >= pd.to_datetime(start)]
        if end is not None:
            f = f[f["date"] <= pd.to_datetime(end)]
        frames[t] = (f.set_index("date")[["close", "open", "volume"]]
                       .astype(float))

    if not frames:
        idx = pd.DatetimeIndex([], name="date")
        empty = pd.DataFrame(index=idx)
        return empty, empty, empty

    # unified index
    idx = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))), name="date")

    # build in one shot to avoid fragmentation
    close_cols = {t: frames[t]["close"].reindex(idx)  for t in frames}
    open_cols  = {t: frames[t]["open"].reindex(idx)   for t in frames}
    vol_cols   = {t: frames[t]["volume"].reindex(idx) for t in frames}

    close = pd.DataFrame(close_cols, index=idx)
    open_ = pd.DataFrame(open_cols,  index=idx)
    vol   = pd.DataFrame(vol_cols,   index=idx)

    return close.sort_index(), open_.sort_index(), vol.sort_index()

def _last_close_price_schwab(symbol: str) -> Optional[float]:
    bars = _schwab_daily(symbol, datetime.utcnow() - timedelta(days=7), datetime.utcnow())
    if bars is None or bars.empty:
        return None
    return float(bars["close"].iloc[-1])


def proposed_trades_view():
    """
    CLI: Proposed Trades – Staging
      • Shows per-plan: Last, Pivot, Buy Zone, Stop, T1/T2, RR, Status (WAITING/ENTERED/MISSED), Strategy
      • R: refresh (re‑query Schwab for last and update status)
      • A: accept all IN‑ZONE → moves to active Portfolio (entry = current last)
      • S: accept selected indices (comma‑separated) → Portfolio
      • D: delete selected indices / 'missed' / 'expired' / 'all'
      • Q: back to main
    """
    def _print_table(rows: list[dict]) -> None:
        if not rows:
            print("\nProposed Trades (staging) — none.\n")
            return
        print("\n" + _c(" Proposed Trades — Staging ", ANSI_BOLD))
        hdr = (f"{'#':>3} {'Ticker':<7} {'Last':>8} {'Pivot':>8} "
               f"{'BuyLo':>8} {'BuyHi':>8} {'Stop':>8} {'T1':>8} {'T2':>8} "
               f"{'RR':>5} {'Status':<9} {'Strategy':<30} {'ValidThru':<10}")
        print(_c(hdr, FG_CYAN))
        for i, r in enumerate(rows, 1):
            last = r.get("last", None)
            lo, hi = r.get("buy_z_lo"), r.get("buy_z_hi")
            status = str(r.get("status",""))
            color = FG_YELLOW
            if status == "ENTERED": color = FG_GREEN
            elif status == "MISSED": color = FG_RED
            line = (
                f"{i:>3} "
                f"{str(r.get('ticker','')):<7} "
                f"{(f'{last:.2f}' if last is not None and np.isfinite(last) else '—'):>8} "
                f"{(f'{float(r.get('pivot',np.nan)):.2f}' if np.isfinite(r.get('pivot',np.nan)) else '—'):>8} "
                f"{(f'{float(lo):.2f}' if np.isfinite(lo) else '—'):>8} "
                f"{(f'{float(hi):.2f}' if np.isfinite(hi) else '—'):>8} "
                f"{(f'{float(r.get('plan_stop',np.nan)):.2f}' if np.isfinite(r.get('plan_stop',np.nan)) else '—'):>8} "
                f"{(f'{float(r.get('t1',np.nan)):.2f}' if np.isfinite(r.get('t1',np.nan)) else '—'):>8} "
                f"{(f'{float(r.get('t2',np.nan)):.2f}' if np.isfinite(r.get('t2',np.nan)) else '—'):>8} "
                f"{(f'{float(r.get('rr_to_t2',np.nan)):.2f}' if np.isfinite(r.get('rr_to_t2',np.nan)) else '—'):>5} "
                f"{_c(status, color):<9} "
                f"{str(r.get('strategy','')):<30} "
                f"{str(r.get('valid_through','')):<10}"
            )
            print(line)
        print()

    def _refresh(rows: list[dict]) -> None:
        changed = 0
        for r in rows:
            sym = r.get("ticker","")
            last = _last_close_price_schwab(sym)
            if last is not None:
                r["last"] = float(last)
            lo, hi = r.get("buy_z_lo"), r.get("buy_z_hi")
            if last is None or not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(last)):
                r["status"] = "WAITING"
            else:
                if lo > hi: lo, hi = hi, lo
                if last < lo: r["status"] = "WAITING"
                elif lo <= last <= hi: r["status"] = "ENTERED"
                else: r["status"] = "MISSED"
            changed += 1
        if changed:
            save_proposed_trades(rows)

    def _accept_rows(rows: list[dict], idxs: list[int]) -> None:
        if not idxs:
            print("Nothing selected.")
            return
        sel = []
        for i in idxs:
            if 1 <= i <= len(rows):
                sel.append(rows[i-1])
        if not sel:
            print("Nothing selected.")
            return
        # Build DF for portfolio adder; use current LAST as entry to reflect actual fill
        df = pd.DataFrame([{
            "date": date.today().isoformat(),
            "time": datetime.now().strftime("%H:%M:%S"),
            "ticker": r.get("ticker"),
            "entry": (r.get("last") if np.isfinite(r.get("last", np.nan)) else r.get("plan_entry")),
            "stop":  r.get("plan_stop"),
            "target": r.get("t2"),
            "strategy": str(r.get("strategy","Proposed")).strip()
        } for r in sel if r.get("ticker")])
        # Filter out rows without usable entry/stop
        df = df[pd.to_numeric(df["entry"], errors="coerce").notna() & pd.to_numeric(df["stop"], errors="coerce").notna()]
        if df.empty:
            print("No valid rows to accept.")
            return
        add_signals_to_portfolio_prompt(df, strategy=None)
        # Remove accepted from staging (by exact ticker+strategy+type match)
        keep = []
        accepted_keys = {(str(r.get("ticker","")).upper(), str(r.get("strategy","")), str(r.get("type",""))) for r in sel}
        for r in rows:
            k = (str(r.get("ticker","")).upper(), str(r.get("strategy","")), str(r.get("type","")))
            if k not in accepted_keys:
                keep.append(r)
        save_proposed_trades(keep)
        print("Accepted → moved to Portfolio.")

    rows = load_proposed_trades()
    rows.sort(key=lambda r: r.get("created_at",""), reverse=True)

    while True:
        if not rows:
            print("\nProposed Trades (staging) — none.\n")
            return
        _print_table(rows)
        cmd = input("Proposed: [R]efresh, [A]ccept in‑zone, [S]elect accept, [D]elete, [Q]uit: ").strip().lower()
        if cmd in ("q",""):
            return
        elif cmd == "r":
            _refresh(rows)
            rows = load_proposed_trades()
            rows.sort(key=lambda r: r.get("created_at",""), reverse=True)
        elif cmd == "a":
            # Accept all ENTERED
            entered = [i+1 for i, r in enumerate(rows) if str(r.get("status")) == "ENTERED"]
            if not entered:
                print("No IN‑ZONE (ENTERED) proposals.")
                continue
            _accept_rows(rows, entered)
            rows = load_proposed_trades()
            rows.sort(key=lambda r: r.get("created_at",""), reverse=True)
        elif cmd == "s":
            raw = input("Enter indices (e.g. 1,3,7) or 'all': ").strip().lower()
            if raw == "all":
                idxs = list(range(1, len(rows)+1))
            else:
                idxs = []
                for tok in raw.replace(",", " ").split():
                    if tok.isdigit():
                        idx = int(tok)
                        if 1 <= idx <= len(rows): idxs.append(idx)
            _accept_rows(rows, idxs)
            rows = load_proposed_trades()
            rows.sort(key=lambda r: r.get("created_at",""), reverse=True)
        elif cmd == "d":
            raw = input("Delete [indices | missed | expired | all]: ").strip().lower()
            keep = rows
            if raw == "all":
                keep = []
            elif raw == "missed":
                keep = [r for r in rows if str(r.get("status")) != "MISSED"]
            elif raw == "expired":
                today_s = date.today().isoformat()
                keep = [r for r in rows if str(r.get("valid_through","")) >= today_s]
            else:
                idxs = []
                for tok in raw.replace(",", " ").split():
                    if tok.isdigit():
                        i = int(tok)
                        if 1 <= i <= len(rows):
                            idxs.append(i)
                kill = set(id(rows[i-1]) for i in idxs)
                keep = [r for r in rows if id(r) not in kill]
            save_proposed_trades(keep)
            rows = load_proposed_trades()
            rows.sort(key=lambda r: r.get("created_at",""), reverse=True)
        else:
            print("Invalid command.")

# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT TABLE (kept from your drilldown)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_prices_table(tickers: List[str]) -> pd.DataFrame:
    cols = ["Ticker","Last","High52W","DistFrom52WHigh_%","Ret_1W_%","Ret_1M_%","Ret_3M_%"]
    if not tickers:
        return pd.DataFrame(columns=cols)

    rows = []
    for t in sorted({x.strip().upper() for x in tickers if x and isinstance(x, str)}):
        bars = _schwab_daily(t, datetime.utcnow() - timedelta(days=365), datetime.utcnow())
        if bars is None or bars.empty:
            continue
        close = bars["close"].astype(float)
        high  = bars["high"].astype(float)
        last = float(close.iloc[-1])
        high52 = float(high.tail(min(252, len(high))).max()) if not high.empty else np.nan
        dist = (high52 - last) / high52 * 100.0 if high52 and high52 > 0 else np.nan
        def _ret(nd): return (last / float(close.iloc[-nd]) - 1.0) * 100.0 if len(close) > nd else np.nan
        rows.append({
            "Ticker": t, "Last": last, "High52W": high52, "DistFrom52WHigh_%": dist,
            "Ret_1W_%": _ret(5), "Ret_1M_%": _ret(21), "Ret_3M_%": _ret(63),
        })
    out = pd.DataFrame(rows, columns=cols)
    if out.empty: return out
    return out.sort_values("DistFrom52WHigh_%", ascending=True).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# RISK / EXIT RULES (initial stop, partials, trailing, time stop)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RiskRules:
    # initial stop
    atr_mult_stop: float = 2.0
    swing_lookback: int = 10
    swing_pad_atr: float = 0.25

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

    # ── NEW: pyramiding on proof (+1R/+2R), each add risks 0.5× the initial $ risk
    pyramid_on: bool = True
    pyramid_triggers_R: tuple[float, ...] = (1.0, 2.0)
    pyramid_risk_fracs: tuple[float, ...] = (0.5, 0.5)  # of initial dollar risk
    pyramid_max_adds: int = 2

    # ── NEW: event blackouts
    earnings_blackout_days: int = 3      # block entries X days before earnings
    macro_blackout_window: int = 0       # blackout window around macro dates (±N days)

RULES = RiskRules()



def _rolling_ma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=max(2, length//2)).mean()

def _swing_low_before(df: pd.DataFrame, entry_idx: int, lookback: int) -> Optional[float]:
    """Lowest low in [entry_idx - lookback, entry_idx-1]."""
    i0 = max(0, entry_idx - lookback)
    if entry_idx <= i0: return None
    window = df["low"].iloc[i0:entry_idx]
    if window.empty: return None
    return float(window.min())

def _plan_initial_levels(entry: float, atr: float, swing_low: Optional[float], rules: RiskRules) -> tuple[float, float, float]:
    """
    Returns: (stop_price, R, first_scale_price)
    R = entry - stop (per-share risk)
    first_scale at entry + first_scale_R * R
    """
    stop_by_atr   = entry - rules.atr_mult_stop * atr
    stop_by_swing = (swing_low - rules.swing_pad_atr * atr) if swing_low is not None else None
    if stop_by_swing is None:
        stop = stop_by_atr
    else:
        # for longs we want the FURTHER stop (lower), i.e. min()
        stop = min(stop_by_atr, stop_by_swing)
    R = max(1e-6, entry - stop)
    first_scale = entry + rules.first_scale_R * R
    return float(stop), float(R), float(first_scale)

def _chandelier_stop(highest_close: float, atr: float, mult: float) -> float:
    return float(highest_close - mult * atr)


def _compute_dynamic_levels_for_holding(holding: dict,
                                        as_of: Optional[date] = None,
                                        rules: Optional[RiskRules] = None) -> dict:
    """
    Thin wrapper over trade_management.compute_dynamic_levels_for_holding.
    Requires register_daily_bars_provider(...) to have been called.
    """
    return TM_compute_dynamic_levels_for_holding(holding, as_of=as_of, rules=rules or RULES)


# ─────────────────────────────────────────────────────────────────────────────
# TECH / INDICATORS (ToS-style signal engine)
# ─────────────────────────────────────────────────────────────────────────────
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def wma(s: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length+1, dtype=float)
    return s.rolling(length).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def hma(s: pd.Series, length: int) -> pd.Series:
    L = int(length)
    if L < 2: return s.copy()*np.nan
    w1 = wma(s, max(L//2,1)); w2 = wma(s, L)
    diff = 2*w1 - w2
    return wma(diff, max(int(math.sqrt(L)),1))

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    sig_line  = ema(macd_line, signal)
    hist      = macd_line - sig_line
    return macd_line, sig_line, hist

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def dmi_diff(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    def rma(x: pd.Series, n: int) -> pd.Series: return x.ewm(alpha=1/n, adjust=False).mean()
    atr = rma(tr, period)
    pdm = rma(pd.Series(plus_dm, index=df.index), period)
    mdm = rma(pd.Series(minus_dm,index=df.index), period)
    di_plus  = 100 * (pdm / (atr + 1e-9))
    di_minus = 100 * (mdm / (atr + 1e-9))
    return di_plus - di_minus

# ToS-style conditions and thresholds
VOL_MIN_ABS   = 100_000
VOL_SMA50_MIN = 750_000
ATR_MIN       = 0.5
DMI_LEN       = 10
HMA_LEN       = 20
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9

def _small_upper_shadow(row) -> bool:
    rng = row["high"] - row["low"]
    if not np.isfinite(rng) or rng <= 0:
        return True
    ratio = (row["high"] - row["close"]) / rng
    return ratio < 0.20

def compute_signals_for_ticker(df: pd.DataFrame, *, strategy: str = DEFAULT_STRATEGY, as_of: Optional[date] = None) -> pd.DataFrame:
    indf = compute_signals(df, strategy=strategy, as_of=as_of)

    if not indf.index.is_monotonic_increasing:
        indf = indf.sort_index()

    cols = list(indf.columns)
    need_signal = ("signal" not in indf.columns) or indf["signal"].isna().all()

    strat_name = str(strategy or "").strip()
    spec_names = [f"signal_{strat_name}", f"signal_{strat_name.lower()}", f"signal_{strat_name.upper()}"]

    if need_signal:
        chosen = next((c for c in spec_names if c in indf.columns), None)
        if chosen is not None:
            indf["signal"] = indf[chosen].astype(bool)
        else:
            bad_prefixes = ("pre_", "raw_", "gate_", "valid_", "near_", "cand_", "dbg_")
            cand = []
            for c in cols:
                if not isinstance(c, str):
                    continue
                lc = c.lower()
                if any(lc.startswith(p) for p in bad_prefixes):
                    continue
                # Positive patterns (expanded)
                if (
                    lc == "signal" or
                    lc.startswith("signal_") or lc.endswith("_signal") or
                    lc.endswith("_trigger") or "trigger_long" in lc or
                    "enter_long" in lc or "long_entry" in lc or "entry_long" in lc or
                    "buy_signal" in lc or "go_long" in lc or
                    (lc.startswith("sig_") and not lc.startswith("sig_raw_"))   # ← NEW
                ):
                    cand.append(c)

            if cand:
                indf["signal"] = indf[cand].astype(bool).any(axis=1)
            else:
                sig_cols_all = [c for c in cols if isinstance(c, str) and c.startswith("signal_")]
                indf["signal"] = indf[sig_cols_all].astype(bool).any(axis=1) if sig_cols_all else False

    if "strategy" not in indf.columns:
        indf["strategy"] = STRATEGY_LABELS.get(strategy, strategy)

    return indf

def _detect_signal_or_raw(row: pd.Series, strat: str, *, dbg: bool=False) -> tuple[bool, str]:
    """
    Decide whether a row is a valid signal for strategy 'strat'.
    Order:
      1) final 'signal' or strategy-specific 'signal_<slug>'
      2) raw hits 'sig_raw_*' + gates (liq/trend plus per-strategy if present)
      3) compatibility: 'trigger_*' / 'setup_*' + basic gates
    Returns (True/False, reason).
    """
    # 1) final signal
    if bool(row.get("signal", False)):
        return True, "signal"

    for nm in (f"signal_{strat}", f"signal_{str(strat).lower()}", f"signal_{str(strat).upper()}"):
        if nm in row.index and bool(row.get(nm, False)):
            return True, nm

    # 2) raw + gates
    raw_cols = [c for c in row.index if isinstance(c, str) and c.startswith("sig_raw_")]
    raw_hit = any(bool(row.get(c, False)) for c in raw_cols)
    if raw_hit:
        liq_ok   = bool(row.get("liq_pass", True))
        trend_ok = bool(row.get("trend_base", True))
        vol_ok   = bool(row.get(f"gate_vol_pass_{strat}", True))
        adx_ok   = bool(row.get(f"gate_adx_pass_{strat}", True))
        rr_ok    = bool(row.get(f"gate_rr_pass_{strat}",  True))
        if liq_ok and trend_ok and vol_ok and adx_ok and rr_ok:
            if dbg: print(f"[DEBUG] promoted raw→signal for strat={strat}")
            return True, "raw+gates"

    # 3) compatibility: trigger/setup + basic gates
    trig_cols = [c for c in row.index if isinstance(c, str) and (c.startswith("trigger_") or c.startswith("setup_"))]
    trig_hit = any(bool(row.get(c, False)) for c in trig_cols)
    if trig_hit:
        liq_ok   = bool(row.get("liq_pass", True))
        trend_ok = bool(row.get("trend_base", True))
        if liq_ok and trend_ok:
            if dbg: print(f"[DEBUG] promoted trigger/setup→signal for strat={strat}")
            return True, "trigger/setup+gates"

    return False, ""


def next_friday(d: date) -> date:
    wd = d.weekday()
    return d + timedelta(days=(4 - wd) % 7)

def end_of_week_close(df: pd.DataFrame, ref_day: date) -> Optional[float]:
    if df is None or df.empty: return None
    tgt = pd.Timestamp(next_friday(ref_day))
    i  = df.index.normalize()
    elig = df.loc[i <= tgt]
    if elig.empty: return None
    return float(elig["close"].iloc[-1])

def rr_levels(entry: float, atr: float) -> Tuple[float, float]:
    stop   = round(entry - atr, 2)
    target = round(entry + 2*atr, 2)
    return stop, target

def _entry_buffer(close: float | pd.Series,
                  atr: float  | pd.Series) -> tuple[float, float]:
    """
    Convert ATR into a small % band ABOVE a breakout pivot for stop‑buy orders.
    Returns (low_pct, high_pct) as fractions (0.002 = 0.20%).

    Heuristic:
      - base = clip(ATR / price, 0.30%, 1.00%)
      - low  ≈ 0.5 * base  (clipped to 0.10%..0.60%)
      - high ≈ 1.25 * base (clipped to [low+0.05%]..1.20%)
    This keeps ranges modest for low‑vol names and roomier for high‑vol names.
    """
    # accept Series or floats
    def _scalar(x):
        try:
            return float(x.iloc[-1]) if isinstance(x, pd.Series) else float(x)
        except Exception:
            return float('nan')

    c = _scalar(close)
    a = _scalar(atr)

    if not np.isfinite(c) or c <= 0 or not np.isfinite(a) or a <= 0:
        # sensible default band: 0.15%..0.60%
        return (0.0015, 0.0060)

    r = a / c                              # volatility as a fraction of price
    base = float(np.clip(r, 0.0030, 0.0100))  # 0.30%..1.00%

    lo = float(np.clip(0.50 * base, 0.0010, 0.0060))              # 0.10%..0.60%
    hi = float(np.clip(1.25 * base, lo + 0.0005, 0.0120))         # ≥(lo+0.05%)..1.20%
    return lo, hi

def scan_on_date(tickers: List[str], as_of: date, *, strategies: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Run strategies on 'as_of' and return one row per (ticker, strategy) that fires.
    Now prints a compact status bar while pulling Schwab data & evaluating.
    """
    if not strategies:
        strategies = STRATEGY_SLUGS

    rows = []
    now_ts = datetime.now().strftime("%H:%M:%S")

    def _first_finite(row_like, names: List[str]) -> float:
        for nm in names:
            if nm in row_like.index:
                v = row_like.get(nm, np.nan)
                if pd.notna(v) and np.isfinite(v):
                    return float(v)
        return float("nan")

    uniq = sorted(set(tickers))
    print("\nSchwab: fetching bars & computing signals …")
    n = len(uniq)

    for i, sym in enumerate(uniq, start=1):
        _status_bar(i, n, prefix="  signals")
        try:
            bars = get_bars(sym, as_of - timedelta(days=90), as_of, min_history_days=260)
            if bars is None or len(bars) < 60:
                continue

            # Select the session row for 'as_of' (or last <= as_of)
            if pd.Timestamp(as_of) in bars.index.normalize().unique():
                day_ts = bars.loc[bars.index.normalize() == pd.Timestamp(as_of)].index[-1]
            else:
                elig = bars.loc[bars.index.normalize() <= pd.Timestamp(as_of)]
                if elig.empty:
                    continue
                day_ts = elig.index[-1]

            for strat in strategies:
                indf = compute_signals_for_ticker(bars, strategy=strat, as_of=as_of)
                if day_ts not in indf.index:
                    continue
                day_row = indf.loc[day_ts]

                # Hit detection: generic OR strategy-specific
                spec_names = [f"signal_{strat}", f"signal_{str(strat).lower()}", f"signal_{str(strat).upper()}"]
                sig = bool(day_row.get("signal", False))
                for nm in spec_names:
                    if nm in day_row.index:
                        sig = sig or bool(day_row.get(nm, False))
                if not sig:
                    continue

                # Event gates
                if within_earnings_blackout(sym, day_ts.date(), RULES.earnings_blackout_days):
                    continue
                if _is_macro_blackout(day_ts.date(), RULES.macro_blackout_window):
                    continue

                entry = float(day_row.get("close", np.nan))
                if not np.isfinite(entry):
                    continue

                # Prefer per-strategy levels; robust fallbacks
                stop_names   = [f"stop0_{strat}", f"stop_{strat}", "stop0", "stop"]
                target_names = [f"target_{strat}", f"target0_{strat}", "target", "target0"]
                stop   = _first_finite(day_row, stop_names)
                target = _first_finite(day_row, target_names)

                if (not np.isfinite(stop)) or (not np.isfinite(target)):
                    try:
                        atrv = float(day_row.get("atr14", np.nan))
                        if not np.isfinite(atrv):
                            atrv = float(atr_series(indf, 14).iloc[indf.index.get_loc(day_ts)])
                    except Exception:
                        atrv = np.nan
                    s, t = rr_levels(entry, atrv if np.isfinite(atrv) else 0.0)
                    if not np.isfinite(stop):
                        stop = s
                    if not np.isfinite(target):
                        target = t

                # End-of-Week close
                bars_fwd = get_bars(sym, as_of - timedelta(days=15), as_of + timedelta(days=7), min_history_days=260)
                eow = end_of_week_close(bars_fwd, day_ts.date()) if bars_fwd is not None else None

                rows.append({
                    "date": day_ts.date().isoformat(),
                    "time": now_ts,
                    "ticker": sym,
                    "entry": round(entry, 2),
                    "atr14": (round(float(day_row.get("atr14", np.nan)), 2) if pd.notna(day_row.get("atr14", np.nan)) else None),
                    "stop":  round(float(stop), 2) if np.isfinite(stop) else None,
                    "target":round(float(target), 2) if np.isfinite(target) else None,
                    "eow_close": (round(float(eow), 2) if eow is not None else None),
                    "strategy": STRATEGY_LABELS.get(strat, strat),
                })
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] scan_on_date: sym={sym} error: {e}")
            continue

    cols = ["date","time","ticker","entry","atr14","stop","target","eow_close","strategy"]
    out = pd.DataFrame(rows, columns=cols)
    if DEBUG:
        print(f"[DEBUG] scan_on_date: signals_found={len(out)}")
    return out

def scan_current_signals(tickers: List[str], *, strategies: Optional[List[str]] = None) -> pd.DataFrame:
    from collections import Counter

    uniq = sorted(set(tickers))
    last_by_ticker: dict[str, date] = {}
    missing, have = [], []

    # Status: show Schwab pulls for latest session date detection
    print("\nSchwab: probing latest available dates …")
    n = len(uniq)
    for i, sym in enumerate(uniq, start=1):
        _status_bar(i, n, prefix="  daily bars")
        bars = get_bars(sym, date.today() - timedelta(days=10), date.today(), 200)
        if bars is None or bars.empty:
            missing.append(sym)
            continue
        last_by_ticker[sym] = bars.index[-1].date()
        have.append(sym)

    if DEBUG:
        print(f"[DEBUG] scan_current_signals: universe={len(uniq)} have_data={len(have)} missing={len(missing)}")
        if missing:
            print("[DEBUG] first 20 no‑data symbols:", ", ".join(missing[:20]))

    if not last_by_ticker:
        return pd.DataFrame()

    counts = Counter(last_by_ticker.values())
    common_day = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    eligible = [sym for sym, d in last_by_ticker.items() if d == common_day]

    print(f"[OK] Schwab daily bars OK: {len(have)}/{len(uniq)} tickers | common day = {common_day}")

    if not eligible:
        return pd.DataFrame()

    # Let scan_on_date do the heavy lifting; that function now shows its own status bar too.
    return scan_on_date(eligible, common_day, strategies=strategies)


def fmt_num(x, width=8, prec=2, signed=False):
        if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
            s = "—"
        else:
            s = f"{x:+.{prec}f}" if signed else f"{x:.{prec}f}"
        return s.rjust(width)

def render_table(df: pd.DataFrame, title: str) -> None:
    if df is None or df.empty:
        print(f"\n{title}: (no signals)\n")
        return

    # Detect backtest-like frame
    is_backtest = any(c in df.columns for c in ("outcome","result","exit_date","exit_px","R","R_mult"))

    if is_backtest:
        # normalize columns (allow legacy names)
        f = df.copy()
        if "outcome" not in f.columns and "result" in f.columns:
            f["outcome"] = f["result"]
        if "exit_px" not in f.columns and "exit" in f.columns:
            f["exit_px"] = f["exit"]
        if "R" not in f.columns and "R_mult" in f.columns:
            f["R"] = f["R_mult"]
        if "days_held" not in f.columns:
            f["days_held"] = 0
        if "strategy" not in f.columns:
            f["strategy"] = "—"

        print("\n" + _c(title, FG_CYAN) + "\n")
        hdr = (
            f"{'Date':<10} {'Ticker':<7} {'Entry':>8} {'Stop':>8} {'Target':>8} "
            f"{'Exit Date':<10} {'Exit Px':>8} {'Outcome':<10} {'R':>6} {'Days':>5} {'Strategy':<30}"
        )
        print(_c(hdr, ANSI_BOLD))

        for _, r in f.iterrows():
            oc_raw = str(r.get("outcome", "")).upper()
            if oc_raw == "TARGET":
                color, oc_disp = FG_GREEN, "TARGET ✓"
            elif oc_raw == "STOP":
                color, oc_disp = FG_RED, "STOP  ✕"
            elif oc_raw == "MA_EXIT":
                color, oc_disp = FG_YELLOW, "MA EXIT"
            elif oc_raw == "TIME":
                color, oc_disp = FG_YELLOW, "TIME"
            else:
                color, oc_disp = FG_YELLOW, "OPEN  …"

            line = (
                f"{str(r.get('date','')):<10} "
                f"{str(r.get('ticker','')):<7} "
                f"{fmt_num(r.get('entry'),8)} "
                f"{fmt_num(r.get('stop'),8)} "
                f"{fmt_num(r.get('target'),8)} "
                f"{(str(r.get('exit_date')) if pd.notna(r.get('exit_date')) else '—'):<10} "
                f"{fmt_num(r.get('exit_px'),8)} "
                f"{oc_disp:<10} "
                f"{fmt_num(r.get('R'),6,prec=2,signed=True)} "
                f"{str(int(r.get('days_held',0))).rjust(5)} "
                f"{str(r.get('strategy','—')):<30}"
            )
            print(_c(line, color))

        # Summary
        total  = len(f)
        closed_mask = f["outcome"].isin(["TARGET","STOP","MA_EXIT","TIME"])
        wins   = int((f["outcome"] == "TARGET").sum())
        losses = int((f["outcome"] == "STOP").sum())
        other_closed = int(f["outcome"].isin(["MA_EXIT","TIME"]).sum())
        opens  = int((f["outcome"] == "OPEN").sum())
        closed = int(closed_mask.sum())
        win_rate = (wins / closed * 100.0) if closed > 0 else float("nan")
        avg_R_closed = f.loc[closed_mask, "R"].dropna().mean()
        net_R = f["R"].dropna().sum()
        avg_days = f.loc[closed_mask, "days_held"].dropna().mean()

        print()
        print(_c("Summary", FG_MAGENTA))
        summ = (
            f"Trades: {total}  |  "
            f"{_c('Hit Target', FG_GREEN)}: {wins}  |  "
            f"{_c('Hit Stop', FG_RED)}: {losses}  |  "
            f"Other Closed: {other_closed}  |  "
            f"{_c('Open', FG_YELLOW)}: {opens}  |  "
            f"Win rate(closed): {(f'{win_rate:.1f}%' if np.isfinite(win_rate) else '—')}  |  "
            f"Avg R(closed): {(f'{avg_R_closed:+.2f}' if pd.notna(avg_R_closed) else '—')}  |  "
            f"Net R: {(f'{net_R:+.2f}' if pd.notna(net_R) else '—')}  |  "
            f"Avg days: {(f'{avg_days:.1f}' if pd.notna(avg_days) else '—')}"
        )
        print(summ + "\n")
        return

    # ── Current Signals view (cleaned) ────────────────────────────────────────
    # Display: Date (MM-DD), Time, Ticker, Entry, Stop, Target, Strategy
    f = df.copy()
    run_ts = datetime.now().strftime("%H:%M:%S")

    def _mmdd(x):
        try:
            return pd.to_datetime(x).strftime("%m-%d")
        except Exception:
            return str(x)

    if "strategy" not in f.columns:
        f["strategy"] = "—"
    if "time" not in f.columns:
        f["time"] = run_ts

    disp = pd.DataFrame({
        "Date":     f["date"].map(_mmdd) if "date" in f.columns else "",
        "Time":     f["time"],
        "Ticker":   f["ticker"],
        "Entry":    pd.to_numeric(f.get("entry"), errors="coerce"),
        "Stop":     pd.to_numeric(f.get("stop"), errors="coerce"),
        "Target":   pd.to_numeric(f.get("target"), errors="coerce"),
        "Strategy": f["strategy"]
    })

    print("\n" + _c(f"{title} — run at {run_ts}", FG_CYAN) + "\n")
    print(disp.to_string(index=False, formatters={
        "Entry": "{:.2f}".format, "Stop": "{:.2f}".format, "Target": "{:.2f}".format
    }))
    print()

def render_predictive_table(df: pd.DataFrame, title: str) -> None:
    if df is None or df.empty:
        print(f"\n{title}: (none)\n"); return
    print("\n" + _c(title, FG_CYAN) + "\n")
    hdr = f"{'Date':<10} {'Ticker':<7} {'Strategy':<30} {'Setup':<26} {'Pivot':>8}  {'Buy Range':<23} {'near(ATR)':>9} {'near(%)':>8}"
    print(_c(hdr, ANSI_BOLD))
    def _fmt(x):
        try: return f"{float(x):.2f}"
        except Exception: return "—"
    for _, r in df.iterrows():
        rng = f"{_fmt(r.get('buy_low')):>8}–{_fmt(r.get('buy_high')):>8}"
        line = (
            f"{str(r.get('date','')):<10} "
            f"{str(r.get('ticker','')):<7} "
            f"{str(r.get('strategy','')):<30} "
            f"{str(r.get('setup_type','')):<26} "
            f"{_fmt(r.get('pivot')):>8}  "
            f"{rng:<23} "
            f"{_fmt(r.get('proximity_atr')):>9} "
            f"{_fmt(r.get('proximity_pct')):>8}"
        )
        print(line)
    print()


def _prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default
    
def render_pre_signals_table(df: pd.DataFrame, title: str, *, horizon_hours: int) -> None:
    """
    Pretty-print Forecast Pre‑signals with a concrete trading plan:
      • Entry plan (buy stop @ pivot + buy zone)
      • Stop (refined), R/share
      • Targets: T1=+1R, T2=max(pattern target, +2R)
      • P(+1R @ horizon)
    Output is a compact, human-friendly two-line block per ticker.
    """
    if df is None or df.empty:
        print(f"\n{title}: (none)\n")
        return

    tstamp = datetime.now().strftime("%H:%M:%S")
    print("\n" + _c(f"{title} — horizon {horizon_hours}h — run at {tstamp}", FG_CYAN) + "\n")

    # Friendly helpers
    def _fmt_px(x): 
        try:    return f"{float(x):.2f}"
        except: return "—"
    def _fmt_r(x):
        try:    return f"{float(x):.2f}R"
        except: return "—"
    def _fmt_p(x):
        try:
            v = float(x)
            if not np.isfinite(v): return "—"
            return f"{100.0*v:.0f}%"
        except:
            return "—"

    # Sort for stable output
    f = df.copy().sort_values(["ticker","date"])

    # Header row (table-like but roomy)
    hdr = (
        f"{'Ticker':<8} {'Entry':>10} {'Stop':>10} {'R/share':>10} "
        f"{'T1(+1R)':>12} {'T2(final)':>12} {'RR→T2':>8} {'P(+1R)':>8}  {'Buy Zone':>20}"
    )
    print(_c(hdr, ANSI_BOLD))

    for _, r in f.iterrows():
        tkr   = str(r.get("ticker",""))
        ent   = _fmt_px(r.get("plan_entry"))
        stp   = _fmt_px(r.get("plan_stop"))
        R_    = _fmt_px(r.get("R"))
        t1    = _fmt_px(r.get("t1"))
        t2    = _fmt_px(r.get("t2"))
        rr2   = _fmt_px(r.get("rr_to_t2"))
        p1r   = _fmt_p(r.get("p_hit_1R"))
        zlo   = _fmt_px(r.get("buy_z_lo"))
        zhi   = _fmt_px(r.get("buy_z_hi"))
        pivot = _fmt_px(r.get("pivot"))

        # Line 1: table columns
        line1 = (
            f"{tkr:<8} {ent:>10} {stp:>10} {R_:>10} "
            f"{t1:>12} {t2:>12} {rr2:>8} {p1r:>8}  "
            f"[{zlo}..{zhi}]"
        )
        print(line1)

        # Line 2: compact narrative (friendlier scan)
        note = (
            f"Plan: Buy‑stop @ pivot {pivot}, zone [{zlo}..{zhi}]; "
            f"Stop {stp}; Targets → T1 {t1} (+1R), T2 {t2} (~{rr2})."
        )
        print(_c("      " + note, FG_GRAY))

    print()  # trailing newline

def gate_audit_for_universe(tickers: List[str],
                            as_of: date,
                            *,
                            strategies: Optional[List[str]] = None) -> dict:
    from collections import defaultdict
    if not strategies:
        strategies = STRATEGY_SLUGS

    counts = defaultdict(int)
    for sym in sorted(set(tickers)):
        try:
            bars = get_bars(sym, as_of - timedelta(days=90), as_of, min_history_days=260)
            if bars is None or bars.empty:
                continue
            indf = compute_signals_for_ticker(bars, strategy="all", as_of=as_of)
            if indf.empty:
                continue
            mask_day = indf.index.normalize() == pd.Timestamp(as_of)
            if not mask_day.any():
                elig = indf.loc[indf.index.normalize() <= pd.Timestamp(as_of)]
                if elig.empty:
                    continue
                row = elig.iloc[-1]
            else:
                row = indf.loc[mask_day].iloc[-1]

            earn_blk = within_earnings_blackout(sym, as_of, RULES.earnings_blackout_days)
            macr_blk = _is_macro_blackout(as_of, RULES.macro_blackout_window)

            for slug in strategies:
                # raw recognition (compat: consider trigger/setup too)
                raw_cols = [c for c in row.index if isinstance(c, str) and c.startswith("sig_raw_")]
                trig_cols = [c for c in row.index if isinstance(c, str) and (c.startswith("trigger_") or c.startswith("setup_"))]
                raw = any(bool(row.get(c, False)) for c in (raw_cols + trig_cols))
                if not raw:
                    continue

                counts["total_raw"] += 1

                liq_pass   = bool(row.get("liq_pass", True))
                vol_pass   = bool(row.get(f"gate_vol_pass_{slug}", True))
                adx_pass   = bool(row.get(f"gate_adx_pass_{slug}", True))
                rr_pass    = bool(row.get(f"gate_rr_pass_{slug}",  True))
                trend_pass = bool(row.get("trend_base", True))

                if not liq_pass:
                    counts["blocked_liquidity"] += 1
                    continue

                if earn_blk:
                    counts["blocked_earnings"] += 1
                if macr_blk:
                    counts["blocked_macro"] += 1
                if not vol_pass:
                    counts["blocked_vol"] += 1
                if not adx_pass:
                    counts["blocked_adx"] += 1
                if not rr_pass:
                    counts["blocked_rr"] += 1
                if not trend_pass:
                    counts["blocked_trend"] += 1
        except Exception:
            continue

    return counts


def print_gate_audit(counts: dict, *, regime_on: Optional[bool]) -> None:
    total = counts.get("total_raw", 0)
    liq   = counts.get("blocked_liquidity", 0)
    vol   = counts.get("blocked_vol", 0)
    adx   = counts.get("blocked_adx", 0)
    rr    = counts.get("blocked_rr", 0)
    tr    = counts.get("blocked_trend", 0)
    ear   = counts.get("blocked_earnings", 0)
    mac   = counts.get("blocked_macro", 0)
    regime = "ON" if regime_on else ("OFF" if regime_on is not None else "—")
    print(_c("\nContext / Gate Audit", FG_MAGENTA))
    print(f"Regime_RiskOn: {regime}  |  Raw candidates: {total}")
    if total > 0:
        print(f"Blocked — Liquidity:{liq}  Vol:{vol}  ADX:{adx}  RR:{rr}  Trend:{tr}  Earnings:{ear}  Macro:{mac}")
    print()


def add_signals_to_portfolio_prompt(df: pd.DataFrame, *, strategy: Optional[str] = None) -> None:
    """
    If df has a 'strategy' column, each row's strategy label is used.
    Dedupe by (symbol, strategy_label).
    """
    if df is None or df.empty:
        return
    ans = input("Add these signals to Portfolio View? (y/N): ").strip().lower()
    if ans not in ("y","yes"):
        return
    shares = _prompt_int("Shares per position", 100)

    holdings = load_holdings()
    existing = {(str(h.get("symbol","")).upper(), str(h.get("strategy","")).strip()) for h in holdings}
    added = []

    for _, r in df.iterrows():
        sym = str(r.get("ticker","")).upper()
        if not sym:
            continue
        entry  = float(r.get("entry", np.nan))
        stop   = float(r.get("stop",  np.nan))
        target = float(r.get("target", np.nan)) if pd.notna(r.get("target")) else None
        if not np.isfinite(entry) or not np.isfinite(stop):
            continue

        # prefer per-row label; fallback to provided param; then 'Current'
        strat_label = str(r.get("strategy", "")).strip() or (strategy or "Current")
        key = (sym, strat_label)
        if key in existing:
            continue

        holding = {
            "symbol": sym,
            "price_paid": float(entry),
            "shares": int(shares),
            "stop_loss": float(stop),
            "profit_target": (float(target) if target is not None and np.isfinite(target) else None),
            "stop_loss_initial": float(stop),
            "profit_target_initial": (float(target) if target is not None and np.isfinite(target) else None),
            "strategy": strat_label,
            "entry_date": str(r.get("date")) if pd.notna(r.get("date")) else date.today().isoformat()
        }
        holdings.append(holding)
        existing.add(key); added.append(f"{sym} [{strat_label}]")

    save_holdings(holdings)
    if added:
        print(f"Added to Portfolio: {', '.join(added)}")
    else:
        print("Nothing added (already present or invalid).")


def add_open_backtests_to_portfolio_prompt(df: pd.DataFrame) -> None:
    if df is None or df.empty or "outcome" not in df.columns:
        return
    open_df = df[df["outcome"] == "OPEN"].copy()
    if open_df.empty:
        print("No OPEN backtest trades to add.")
        return
    ans = input(f"Add {len(open_df)} OPEN backtest trades to Portfolio View? (y/N): ").strip().lower()
    if ans not in ("y","yes"):
        return
    add_signals_to_portfolio_prompt(open_df.rename(columns={"R":"R_"}), strategy="Backtest-OPEN")


# ─────────────────────────────────────────────────────────────────────────────
# YOUR INDUSTRY MODEL – feature builders & ranking (kept, lightly refactored)
# ─────────────────────────────────────────────────────────────────────────────
def rsi(series, length=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def smart_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(',', '')
    if s in ("", "-", "N/A"): return np.nan
    if s.endswith('%'):
        try: return float(s[:-1])
        except ValueError: return np.nan
    suf = s[-1].upper()
    if suf in "KMB":
        mult = {'K':1e3, 'M':1e6, 'B':1e9}[suf]
        try: return float(s[:-1]) * mult
        except ValueError: return np.nan
    try: return float(s)
    except ValueError: return np.nan

def force_numeric(series: pd.Series) -> pd.Series:
    return series.map(smart_float).astype(float)

def add_multi_horizon_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "price_idx" not in df.columns:
        df["price_idx"] = (1 + df["Change"].fillna(0).div(100)) \
                            .groupby(df["Name"]).transform("cumprod")
    for h in (5, 21, 63):
        df[f"Ret{h}"] = df.groupby("Name")["price_idx"].pct_change(h) * 100.0
    df["DailyRet"] = df.groupby("Name")["price_idx"].pct_change()
    df["RealizedVol21"] = df.groupby("Name")["DailyRet"].transform(lambda s: s.rolling(21).std()).mul(np.sqrt(252) * 100.0)
    if "MA20" not in df.columns:
        df["MA20"] = df.groupby("Name")["price_idx"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    if "MA50" not in df.columns:
        df["MA50"] = df.groupby("Name")["price_idx"].transform(lambda x: x.rolling(50, min_periods=1).mean())
    df["Above20"] = (df["price_idx"] > df["MA20"]).astype(int)
    df["Above50"] = (df["price_idx"] > df["MA50"]).astype(int)
    df["GoldenCross20_50"] = (df["MA20"] > df["MA50"]).astype(int)
    df["MaxDD63"] = df.groupby("Name")["price_idx"].transform(lambda s: (s / s.rolling(63, min_periods=63).max() - 1) * 100.0)
    for h in (5, 21, 63):
        col = f"Ret{h}"
        df[f"{col}_PctileCS"] = df.groupby("Date")[col].rank(pct=True) * 100.0
    if "RSI14" not in df.columns:
        df["RSI14"] = df.groupby("Name")["price_idx"].transform(lambda x: rsi(x, 14))
    df.drop(columns=["DailyRet"], inplace=True)
    return df

def build_macro_features(df: pd.DataFrame, ind2sec: dict) -> pd.DataFrame:
    df = df.copy()
    t10   = fred_get("DGS10")
    oil   = fred_get("DCOILWTICO")
    dxy   = fred_get("DTWEXBGS")
    vix   = fred_get("VIXCLS")
    hyoas = fred_get("BAMLH0A0HYM2")
    def _ret21(s): return (s / s.shift(21) - 1.0) * 100.0
    rate_21d = (t10 - t10.shift(21)) * 100.0
    oil_21d  = _ret21(oil)
    usd_21d  = _ret21(dxy)
    df["Macro_Rate_21d"] = rate_21d.reindex(df["Date"]).ffill().values
    df["Macro_Oil_21d"]  = oil_21d .reindex(df["Date"]).ffill().values
    df["Macro_USD_21d"]  = usd_21d .reindex(df["Date"]).ffill().values
    df["VIX_Level"]      = vix     .reindex(df["Date"]).ffill().values
    df["HYOAS_Level"]    = hyoas   .reindex(df["Date"]).ffill().values
    SENS = {
        "Financial":{"rate":+1.0,"oil":0.0,"usd":0.0},
        "Real Estate":{"rate":-1.0,"oil":0.0,"usd":0.0},
        "Utilities":{"rate":-0.9,"oil":0.0,"usd":0.0},
        "Technology":{"rate":-0.6,"oil":0.0,"usd":-0.6},
        "Consumer Cyclical":{"rate":-0.4,"oil":0.0,"usd":-0.4},
        "Consumer Defensive":{"rate":-0.2,"oil":0.0,"usd":-0.2},
        "Industrials":{"rate":-0.3,"oil":0.1,"usd":-0.3},
        "Basic Materials":{"rate":-0.1,"oil":+0.3,"usd":-0.5},
        "Energy":{"rate":+0.2,"oil":+1.0,"usd":-0.3},
        "Communication Services":{"rate":-0.3,"oil":0.0,"usd":-0.3},
        "Healthcare":{"rate":-0.2,"oil":0.0,"usd":0.0},
        "Utilities - Renewable":{"rate":-0.8,"oil":0.0,"usd":0.0},
    }
    w_rate = df["Sector"].map(lambda s: SENS.get(s, {}).get("rate", 0.0))
    w_oil  = df["Sector"].map(lambda s: SENS.get(s, {}).get("oil",  0.0))
    w_usd  = df["Sector"].map(lambda s: SENS.get(s, {}).get("usd",  0.0))
    df["Rate_Context"] = w_rate * df["Macro_Rate_21d"]
    df["Oil_Context"]  = w_oil  * df["Macro_Oil_21d"]
    df["USD_Context"]  = w_usd  * df["Macro_USD_21d"]
    return df

def add_macro_regime_gate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    dates = pd.to_datetime(df["Date"].unique())
    start = dates.min() - pd.Timedelta(days=400); end = dates.max()
    idx   = pd.date_range(start=start, end=end, freq="D")
    t10, t3m, m2, dff, usd = map(fred_get, ("DGS10","DGS3MO","M2SL","DFF","DTWEXBGS"))
    def _fx(s): return s.reindex(idx).ffill() if not s.empty else pd.Series(index=idx, dtype=float)
    t10, t3m, m2, dff, usd = map(_fx, (t10, t3m, m2, dff, usd))
    yc_slope = (t10 - t3m); yc_chg21 = yc_slope - yc_slope.shift(21)
    m2_yoy   =  m2.pct_change(365, fill_method=None) * 100.0
    dff_tr21 = dff - dff.shift(21)
    usd_tr21 = (usd / usd.shift(21) - 1.0) * 100.0
    close_us, _, _ = _schwab_panel(["XLU","SPY"], start, end)
    if not close_us.empty and {"XLU","SPY"} <= set(close_us.columns):
        spy = close_us["SPY"].reindex(idx).ffill(); xlu = close_us["XLU"].reindex(idx).ffill()
        rs_utl = (xlu / xlu.iloc[0]) / (spy / spy.iloc[0])
        utl_tr21 = rs_utl - rs_utl.shift(21)
    else:
        utl_tr21 = pd.Series(index=idx, dtype=float)
    vix = _fx(fred_get("VIXCLS"))
    def _z(s):
        m, st = s.mean(skipna=True), s.std(skipna=True)
        return (s - m) / (st if st and st > 0 else 1.0)
    score  = _z(yc_chg21) + _z(m2_yoy) - _z(dff_tr21) - _z(usd_tr21) - _z(utl_tr21) - _z(vix)
    regime = (score > 0).astype(int)
    reg_df = pd.DataFrame({
        "Date": idx,
        "YC_Slope": yc_slope.reindex(idx).values,
        "YC_Slope_21dChg": yc_chg21.reindex(idx).values,
        "M2_YoY": m2_yoy.reindex(idx).values,
        "DFF_Trend_21d": dff_tr21.reindex(idx).values,
        "USD_Trend_21d": usd_tr21.reindex(idx).values,
        "XLU_SPY_RS_21dChg": utl_tr21.reindex(idx).values,
        "Risk_Score": score.reindex(idx).values,
        "Regime_RiskOn": regime.reindex(idx).values,
    })
    reg_df["Date"] = pd.to_datetime(reg_df["Date"])
    return df.merge(reg_df, on="Date", how="left")

def add_group_breadth_features(
    df: pd.DataFrame,
    ind2ticks: dict | None = None
) -> pd.DataFrame:
    if df.empty:
        return df

    if ind2ticks is None:
        ind2ticks = _load_industry_ticker_map()
    norm_map = {_norm_industry_key(k): _sanitize_tickers(v) for k, v in (ind2ticks or {}).items()}

    start = pd.to_datetime(df["Date"].min()) - pd.Timedelta(days=280)
    end   = pd.to_datetime(df["Date"].max())

    spy_close, _, _ = _schwab_panel(["SPY"], start, end)
    spy = spy_close["SPY"] if "SPY" in spy_close.columns else pd.Series(dtype=float)

    def _share(flag_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> pd.Series:
        if valid_df is None:
            valid_df = flag_df.notna()
        denom = valid_df.sum(axis=1).replace(0, np.nan)
        num = (flag_df & valid_df).sum(axis=1)
        return (num / denom) * 100.0

    out_rows: list[pd.DataFrame] = []

    inds = sorted(df["Name"].dropna().astype(str).unique())
    total = len(inds)
    if total == 0:
        return df

    # Progress bar start
    try:
        _status_bar(0, total, prefix="Breadth")
    except NameError:
        print("[Breadth] Computing member breadth features…")

    for i, ind in enumerate(inds, 1):
        try:
            _status_bar(i - 1, total, prefix="Breadth")
        except NameError:
            pass

        ticks = norm_map.get(_norm_industry_key(ind), [])
        if not ticks:
            try:
                _status_bar(i, total, prefix="Breadth")
            except NameError:
                pass
            continue

        close, open_, vol = _schwab_panel(ticks, start, end)
        if close.empty:
            try:
                _status_bar(i, total, prefix="Breadth")
            except NameError:
                pass
            continue

        spy_al = spy.reindex(close.index).ffill() if not spy.empty else pd.Series(index=close.index, dtype=float)

        # Rolling stats
        ma20  = close.rolling(20, min_periods=20).mean()
        ma50  = close.rolling(50, min_periods=50).mean()
        hi20  = close.rolling(20,  min_periods=20).max()
        hi50  = close.rolling(50,  min_periods=50).max()
        hi252 = close.rolling(252, min_periods=126).max()
        lo20  = close.rolling(20,  min_periods=20).min()
        vol20 = vol.rolling(20, min_periods=20).mean()

        newhi20   = close >= hi20
        newhi50   = close >= hi50
        newhi252  = close >= hi252
        newlo20   = close <= lo20
        up_day    = close > close.shift(1)
        vol_thrst = (vol >= (1.5 * vol20)) & up_day

        # Relative strength vs SPY (normalized)
        if spy_al.notna().any():
            s0 = close.apply(lambda s: s.ffill().bfill().iloc[0])
            spy0 = spy_al.ffill().bfill().iloc[0]
            s0 = s0.replace(0, np.nan)
            spy0 = np.nan if spy0 == 0 else spy0
            rs = close.divide(s0, axis=1).divide(spy_al / spy0, axis=0)
            rs_hi_base = rs.rolling(252, min_periods=126).max()
            rs_hi_252  = rs >= rs_hi_base
            rs_valid   = rs_hi_base.notna() & rs.notna()
        else:
            rs_hi_252 = close.astype(bool) & False
            rs_valid = rs_hi_252.notna() & False  # all False

        # 21d return for top-half concentration (silence FutureWarning)
        ret21   = close.pct_change(21, fill_method=None)
        ranks   = ret21.rank(axis=1, ascending=False, method="min")
        n_valid = (~ret21.isna()).sum(axis=1).clip(lower=1)
        cutoff  = np.floor(n_valid / 2)
        top_half = ranks.le(cutoff, axis=0)

        # Setups
        breakout50 = (close >= hi50)
        pb_to_20   = (close >= ma20) & (close <= (ma20 * 1.02)) & (ma20 > ma50)
        bbw        = (close.rolling(20).std() * 2.0) / ma20
        bbw_q20    = bbw.rolling(120, min_periods=40).quantile(0.20)
        squeeze    = bbw <= bbw_q20
        setup_any  = breakout50 | pb_to_20 | squeeze

        # Gaps & breakdowns
        gap_up        = ((open_ / close.shift(1) - 1.0) >= 0.03) & (vol >= (2.0 * vol20))
        breakdown_pen = (close < ma50) & (vol >= (1.5 * vol20))

        feat = pd.DataFrame({
            "Date": close.index,
            "Name": ind,
            "Pct_NewHigh_20":      _share(newhi20,   hi20.notna()),
            "Pct_NewHigh_50":      _share(newhi50,   hi50.notna()),
            "Pct_NewHigh_252":     _share(newhi252,  hi252.notna()),
            "NH_NL_Spread20":      _share(newhi20,   hi20.notna()) - _share(newlo20, lo20.notna()),
            "RS52W_High_Share":    _share((rs_hi_252 if spy_al.notna().any() else newhi252),
                                          (rs_valid if spy_al.notna().any() else hi252.notna())),
            "VolThrust_Share":     _share(vol_thrst, vol20.notna()),
            "TopHalf_Share":       _share(top_half,  ret21.notna()),
            "Setup_Density_Share": _share(setup_any, ma20.notna() & ma50.notna()),
            "GapUp_Share":         _share(gap_up,    (close.shift(1).notna() & open_.notna() & vol20.notna())),
            "Breakdown_Penalty":   (_share(breakdown_pen, (ma50.notna() & vol20.notna())) / 100.0),
        }).set_index("Date")

        # Rolling gap density
        daily_gap = (feat["GapUp_Share"].fillna(0) / 100.0)
        feat["GapUp_Share_10d"] = (daily_gap.rolling(10, min_periods=2).mean() * 100.0)
        feat["GapUp_Share_21d"] = (daily_gap.rolling(21, min_periods=5).mean() * 100.0)

        out_rows.append(feat.reset_index())

        try:
            _status_bar(i, total, prefix="Breadth")
        except NameError:
            pass

    # Ensure final newline after progress bar
    try:
        _status_bar(total, total, prefix="Breadth")
    except NameError:
        pass

    if not out_rows:
        return df

    gfeat = pd.concat(out_rows, ignore_index=True).rename(columns={"index": "Date"})
    gfeat["Date"] = pd.to_datetime(gfeat["Date"])
    return df.merge(gfeat, on=["Name", "Date"], how="left")


def add_supersector_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    start = pd.to_datetime(df["Date"].min()) - pd.Timedelta(days=280)
    end   = pd.to_datetime(df["Date"].max())
    spy_close, _, _ = _schwab_panel(["SPY"], start, end)
    spy = spy_close["SPY"] if "SPY" in spy_close.columns else pd.Series(dtype=float)
    def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
    adds: list[pd.DataFrame] = []
    for ind, g in df.sort_values("Date").groupby("Name"):
        s = g.set_index("Date")["price_idx"].astype(float)
        if s.isna().all(): continue
        macd_line = _ema(s, 12) - _ema(s, 26)
        macd_pos_rising = (macd_line > 0) & (macd_line.diff() > 0)
        rsi14 = g.set_index("Date")["RSI14"].astype(float)
        rsi50_rising = (rsi14 > 50) & (rsi14.diff(5) > 0)
        if spy.notna().any():
            spy_al = spy.reindex(s.index).ffill()
            s0 = s.ffill().bfill().iloc[0]; spy0 = spy_al.ffill().bfill().iloc[0]
            spy0 = np.nan if spy0 == 0 else spy0
            if not np.isfinite(s0) or not np.isfinite(spy0):
                rs_above_ma50 = pd.Series(np.nan, index=s.index)
            else:
                rs = (s / s0) / (spy_al / spy0)
                rs_ma50 = rs.rolling(50, min_periods=25).mean()
                rs_above_ma50 = (rs > rs_ma50).astype(float)
        else:
            rs_above_ma50 = pd.Series(np.nan, index=s.index)
        ma50  = s.rolling(50,  min_periods=25).mean()
        ma200 = s.rolling(200, min_periods=100).mean()
        golden = (ma50 > ma200)
        cross_change = (golden.astype(int).diff().fillna(0) != 0)
        last_cross_dates = cross_change[cross_change].index
        age = pd.Series(index=s.index, dtype=float); last = None
        for dt_ in s.index:
            if dt_ in last_cross_dates: last = dt_
            age.loc[dt_] = (dt_ - last).days if last is not None else np.nan
        golden_age = age.where(golden, 0.0)
        feat = pd.DataFrame({
            "Date": s.index, "Name": ind,
            "MACD_Pos_Rising": macd_pos_rising.astype(float).values,
            "RSI50_Rising":    rsi50_rising.astype(float).values,
            "RS_Above_MA50":   rs_above_ma50.values,
            "GoldenCross_50_200": golden.astype(float).values,
            "GoldenCross_Age": golden_age.values,
        })
        adds.append(feat)
    if not adds: return df
    addf = pd.concat(adds, ignore_index=True)
    df = df.merge(addf, on=["Name","Date"], how="left")
    if "Breakdown_Penalty" in df.columns:
        df["Exit_Penalty"] = df["Breakdown_Penalty"].clip(0, 1)
    return df

def add_schwab_ownership_insider(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); inds = df["Name"].unique()
    buckets = {
        "Inst_Own_Pct": {}, "Inst_Own_QoQ": {}, "Inst_Holders_QoQ": {},
        "Insider_BuySell_Ratio_30D": {}, "Insider_NetBuys_30D": {}
    }
    for ind in inds:
        own = schwab_get(f"ownership/institutional/{ind}", cache=f"own_{ind}", ttl=86400) or {}
        now  = own.get("instOwnershipPct"); prev = own.get("instOwnershipPctPrev") or own.get("instOwnershipPct_1Qago")
        hnow = own.get("institutionHoldersNow") or own.get("holdersNow")
        hpre = own.get("institutionHoldersPrev") or own.get("holdersPrev")
        buckets["Inst_Own_Pct"][ind] = smart_float(now)
        buckets["Inst_Own_QoQ"][ind] = (smart_float(now) - smart_float(prev)) if (now is not None and prev is not None) else np.nan
        buckets["Inst_Holders_QoQ"][ind] = (smart_float(hnow) - smart_float(hpre)) if (hnow is not None and hpre is not None) else np.nan

        ins = schwab_get(f"fundamentals/insideractivity/{ind}", cache=f"ins_{ind}", ttl=86400) or {}
        buys  = smart_float(ins.get("buys_30d")  or ins.get("buys30d"))
        sells = smart_float(ins.get("sells_30d") or ins.get("sells30d"))
        ratio = (buys / max(sells, 1.0)) if (buys is not None and sells is not None) else np.nan
        net   = (buys - sells) if (buys is not None and sells is not None) else np.nan
        buckets["Insider_BuySell_Ratio_30D"][ind] = ratio
        buckets["Insider_NetBuys_30D"][ind]       = net
    for k, m in buckets.items():
        df[k] = df["Name"].map(m)
    return df

def add_analyst_dispersion_revisions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); inds = df["Name"].unique()
    out = {
        "Analyst_Target_Dispersion": {}, "Revisions_UpDown_Ratio_30D": {},
        "Consensus_Rating": {}, "EPS_RevPct": {}
    }
    for ind in inds:
        anal = schwab_get(f"research/analyst/{ind}", cache=f"anal_{ind}", ttl=86400) or {}
        t_mean = smart_float(anal.get("targetMean")); t_std  = smart_float(anal.get("targetStdDev"))
        if t_std is not None and t_mean not in (None, 0):
            disp = abs(t_std / t_mean) * 100.0
        else:
            targets = anal.get("targets") or []
            vals = [smart_float(t.get("price")) for t in targets if t.get("price") is not None]
            disp = (np.std(vals) / np.mean(vals) * 100.0) if len(vals) >= 2 and np.mean(vals) else np.nan
        up   = smart_float(anal.get("upRevisions_30d")   or anal.get("up30d"))
        down = smart_float(anal.get("downRevisions_30d") or anal.get("down30d"))
        ratio = (up / max(down, 1.0)) if (up is not None and down is not None) else np.nan
        out["Analyst_Target_Dispersion"][ind]  = disp
        out["Revisions_UpDown_Ratio_30D"][ind] = ratio
        out["Consensus_Rating"][ind]           = smart_float(anal.get("consensusRating"))
        out["EPS_RevPct"][ind]                 = smart_float(anal.get("epsRevisionPct"))
    for k, m in out.items():
        if k in df.columns: df[k] = df[k].where(df[k].notna(), df["Name"].map(m))
        else:               df[k] = df["Name"].map(m)
    return df

def add_feature_interactions(df: pd.DataFrame, ind2sec: dict) -> pd.DataFrame:
    df = df.copy()
    rel_ma = df["Rel_MA_Spread"] if "Rel_MA_Spread" in df.columns else df["MA_Spread"]
    df["Mom21_x_EPSRev"]      = df["Rel_Ret21"] * df.get("EPS_RevPct", np.nan)
    df["MASpread_x_Breadth"]  = rel_ma           * df.get("Breadth50", np.nan)
    df["RSI14_x_Breadth"]     = df.get("RSI14", np.nan) * df.get("Breadth50", np.nan)
    df["InstOwnTrend_x_Mom"]  = df.get("Inst_Own_QoQ", np.nan) * df["Rel_Ret21"]
    df["InsiderRatio_x_Mom"]  = df.get("Insider_BuySell_Ratio_30D", np.nan) * df["Rel_Ret21"]
    df["RateDelta_x_Sens"]    = df.get("Macro_Rate_21d", np.nan) * df.get("Rate_Context", np.nan)
    df["OilDelta_x_Sens"]     = df.get("Macro_Oil_21d",  np.nan) * df.get("Oil_Context",  np.nan)
    df["USDDelta_x_Sens"]     = df.get("Macro_USD_21d",  np.nan) * df.get("USD_Context",  np.nan)
    df["Mom21_ShockAdj"]      = df["Rel_Ret21"] / (df.get("RealizedVol21", np.nan) + 1e-6)
    for c in ("Mom21_x_EPSRev","MASpread_x_Breadth","RSI14_x_Breadth",
              "InstOwnTrend_x_Mom","InsiderRatio_x_Mom",
              "RateDelta_x_Sens","OilDelta_x_Sens","USDDelta_x_Sens","Mom21_ShockAdj"):
        if c in df.columns: df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

def prepare_ranking_frame(df: pd.DataFrame,
                          fwd_days: int = 21,
                          target_col: str = "Future1M") -> tuple[pd.DataFrame, np.ndarray, list[int], list[str]]:
    df = df.copy()
    if target_col not in df.columns:
        future = {}
        for n, g in df.groupby("Name", sort=False):
            pct = g["Change"].fillna(0) / 100.0; idx = g.index
            for i in range(len(g)):
                future[idx[i]] = ((np.prod(1 + pct.iloc[i+1:i+1+fwd_days]) - 1) * 100.0
                                  if i + fwd_days < len(g) else np.nan)
        df[target_col] = pd.Series(future)
    use = df.dropna(subset=[target_col]).sort_values(["Date","Sector","Name"]).copy()
    grp_sizes = use.groupby("Date", sort=False).size()
    valid_dates = grp_sizes[grp_sizes >= 2].index
    use = use[use["Date"].isin(valid_dates)].copy()
    id_like = {"Name","Sector","Date",target_col}
    num = use.select_dtypes(include=[np.number]).copy()
    feat_cols = [c for c in num.columns if c not in id_like]
    med = num[feat_cols].median()
    X = num[feat_cols].fillna(med)
    y = use[target_col].astype(float).values
    groups = use.groupby("Date", sort=False).size().tolist()
    return X, y, groups, feat_cols

def fit_rank_model_and_predict(df: pd.DataFrame,
                               last_date: pd.Timestamp,
                               top_k: int = 10,
                               rng_seed: int = 42) -> pd.DataFrame:
    """
    GPU-first XGBoost ranker with robust preprocessing and stable display.
    - Drops ~empty features (>=95% NaN in training)
    - Imputes with training medians (0 if median is NaN)
    - Casts to float32 for CUDA stability
    - Trains on GPU (fallback to CPU)
    - Predicts via Booster + DMatrix to avoid device mismatch warnings
    - Returns RankScore normalized to snapshot percentile [0..1] for readability
      (ordering is identical to raw margins).
    """
    # --- Prepare training frame
    X, y, groups, feat_cols = prepare_ranking_frame(df, fwd_days=21, target_col="Future1M")

    # If training insufficient, fall back to heuristic ranking
    def _heuristic_snapshot_rank() -> pd.DataFrame:
        snap_mask = pd.to_datetime(df["Date"]) == pd.to_datetime(last_date)
        snap = df.loc[snap_mask].copy()
        if snap.empty:
            return pd.DataFrame(columns=["Name","Sector","RankScore"])
        snap = snap.drop_duplicates(subset=["Name"], keep="last")

        pos_feats = {
            "Rel_Ret21","Ret21","MA_Spread","Rel_MA_Spread","Breadth50",
            "Pct_NewHigh_20","Pct_NewHigh_50","Pct_NewHigh_252","NH_NL_Spread20",
            "RS52W_High_Share","VolThrust_Share","TopHalf_Share","Setup_Density_Share",
            "MACD_Pos_Rising","RSI50_Rising","RS_Above_MA50","GoldenCross_50_200","GoldenCross_Age",
            "Risk_Score","Regime_RiskOn"
        }
        neg_feats = {"RealizedVol21","MaxDD63","Exit_Penalty","Breakdown_Penalty",
                     "USD_Trend_21d","DFF_Trend_21d","VIX_Level","HYOAS_Level"}

        # Use whatever numeric features exist in the snapshot that are also in training
        snap_num = snap.select_dtypes(include=[np.number]).copy()
        use_feats = [c for c in feat_cols if c in snap_num.columns]
        if not use_feats:
            return pd.DataFrame(columns=["Name","Sector","RankScore"])

        eps = 1e-9
        score = pd.Series(0.0, index=snap.index, dtype=float)
        for c in use_feats:
            s = pd.to_numeric(snap_num[c], errors="coerce")
            med = s.median(skipna=True)
            mad = (s - med).abs().median(skipna=True)
            z = (s - med) / (mad + eps)
            if c in neg_feats:
                score = score - z.fillna(0.0)
            elif c in pos_feats:
                score = score + z.fillna(0.0)

        # Normalize to percentile for display
        pct = score.rank(pct=True)
        out = snap.assign(RankScore=pct.values)
        return (out.sort_values("RankScore", ascending=False)
                    .loc[:, ["Name","Sector","RankScore"]]
                    .head(min(top_k, len(out)))
                    .reset_index(drop=True))

    n_rows = len(X)
    if n_rows == 0 or len(groups) < 3 or sum(groups) != n_rows:
        if DEBUG: print("[DEBUG] Insufficient training rows/groups; using heuristic ranking.")
        return _heuristic_snapshot_rank()

    # --- Drop nearly-empty columns and impute medians
    # Keep only features with <= 95% NaN in training
    na_frac = X.isna().mean()
    keep_cols = [c for c in feat_cols if na_frac.get(c, 1.0) <= 0.95]
    if not keep_cols:
        if DEBUG: print("[DEBUG] All features are ~empty; using heuristic ranking.")
        return _heuristic_snapshot_rank()

    if DEBUG:
        dropped = [c for c in feat_cols if c not in keep_cols]
        print(f"[DEBUG] Training features: kept={len(keep_cols)} dropped={len(dropped)}")

    X = X[keep_cols]
    feat_cols = keep_cols

    med_train = X.median(numeric_only=True).fillna(0.0)
    X_f = X.fillna(med_train).astype(np.float32)
    y_f = np.asarray(y, dtype=np.float32)

    # --- Build monotone constraints string (aligned to feat_cols)
    pos_feats = {
        "Rel_Ret21","Ret21","MA_Spread","Rel_MA_Spread","Breadth50",
        "Pct_NewHigh_20","Pct_NewHigh_50","Pct_NewHigh_252","NH_NL_Spread20",
        "RS52W_High_Share","VolThrust_Share","TopHalf_Share","Setup_Density_Share",
        "MACD_Pos_Rising","RSI50_Rising","RS_Above_MA50","GoldenCross_50_200","GoldenCross_Age",
        "Risk_Score","Regime_RiskOn"
    }
    neg_feats = {"RealizedVol21","MaxDD63","Exit_Penalty","Breakdown_Penalty",
                 "USD_Trend_21d","DFF_Trend_21d","VIX_Level","HYOAS_Level"}
    constr_vec = [1 if c in pos_feats else (-1 if c in neg_feats else 0) for c in feat_cols]
    constr_str = "(" + ",".join(str(int(v)) for v in constr_vec) + ")"

    # --- Train (GPU first, CPU fallback)
    try:
        from xgboost import XGBRanker, DMatrix
    except Exception:
        if DEBUG: print("[DEBUG] xgboost not available; using heuristic ranking.")
        return _heuristic_snapshot_rank()

    params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20.0,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=1.0,
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        monotone_constraints=constr_str,
        random_state=rng_seed,
    )

    ranker = None
    device_used = "cuda"
    try:
        ranker = XGBRanker(**params, tree_method="hist", device="cuda")
        ranker.fit(X_f, y_f, group=groups)
    except Exception as e:
        if DEBUG: print(f"[DEBUG] GPU training failed ({e}); retrying on CPU …")
        try:
            device_used = "cpu"
            ranker = XGBRanker(**params, tree_method="hist", device="cpu", n_jobs=-1)
            ranker.fit(X_f, y_f, group=groups)
        except Exception as e2:
            if DEBUG: print(f"[DEBUG] CPU training also failed ({e2}); using heuristic ranking.")
            return _heuristic_snapshot_rank()

    # --- Build snapshot matrix (align columns, same imputation & dtype)
    snap_mask = pd.to_datetime(df["Date"]) == pd.to_datetime(last_date)
    snap = df.loc[snap_mask].copy()
    if snap.empty:
        return pd.DataFrame(columns=["Name","Sector","RankScore"])
    snap = snap.drop_duplicates(subset=["Name"], keep="last")

    snap_num = snap.select_dtypes(include=[np.number]).copy()
    for c in feat_cols:
        if c not in snap_num.columns:
            snap_num[c] = np.nan
    snapX = snap_num[feat_cols].fillna(med_train).astype(np.float32)

    # --- Predict with Booster + DMatrix (prevents device mismatch warning)
    try:
        booster = ranker.get_booster()
        dm = DMatrix(snapX)
        raw_scores = booster.predict(dm)
    except Exception as e:
        if DEBUG: print(f"[DEBUG] predict() failed ({e}); using heuristic ranking.")
        return _heuristic_snapshot_rank()

    if DEBUG:
        print(f"[DEBUG] XGBoost ranker used device: {device_used}")
        # Optional: show how many NaNs remained in snapshot
        if snapX.isna().any().any():
            nan_cols = snapX.columns[snapX.isna().any()].tolist()
            print(f"[DEBUG] Snapshot still has NaNs in: {nan_cols}")

    # --- Normalize scores for display (ordering preserved)
    s = pd.Series(raw_scores, index=snap.index)
    rank_pct = s.rank(pct=True)  # 0..1
    leaders = (snap.assign(RankScore=rank_pct.values)
                    .sort_values("RankScore", ascending=False)
                    .loc[:, ["Name","Sector","RankScore"]]
                    .head(min(top_k, len(snap)))
                    .reset_index(drop=True))
    return leaders


def _choose_industry_top(leaders_df: pd.DataFrame) -> str | None:
    if not isinstance(leaders_df, pd.DataFrame) or leaders_df.empty:
        print("\nNo Top-K industries available."); return None
    menu = leaders_df["Name"].astype(str).tolist()
    print("\nSelect a Top-10 industry to view its tickers (industries-tickers.json):\n")
    for i, name in enumerate(menu, 1):
        print(f"{i:2d}. {name}")
    choice = input("\nEnter number or type industry name (blank to skip): ").strip()
    if not choice: return None
    if choice.isdigit():
        idx = int(choice);  return menu[idx-1] if 1 <= idx <= len(menu) else None
    return choice

def _run_drilldown_block(df_obj: pd.DataFrame, leaders_obj: pd.DataFrame) -> None:
    if not isinstance(df_obj, pd.DataFrame) or df_obj.empty:
        print("\n[Drill‑down skipped] df is missing or empty."); return
    if not isinstance(leaders_obj, pd.DataFrame) or leaders_obj.empty:
        print("\n[Drill‑down skipped] leaders is missing or empty."); return
    gate = input(f"\nView tickers for one of the Top {len(leaders_obj)} industries? [y/N]: ").strip().lower()
    if gate not in ("y","yes"): return
    _selected = _choose_industry_top(leaders_obj)
    if not _selected: return
    m = _load_industry_ticker_map()
    _ticks = m.get(_norm_industry_key(_selected), [])
    if not _ticks:
        print(f"\nNo tickers found for '{_selected}' in industries-tickers.json.")
        return
    snap = _fetch_prices_table(_ticks)
    if snap.empty:
        print(f"\nNo price data via Schwab for: {', '.join(_ticks)}"); return
    print(f"\nTickers in '{_selected}' ranked by proximity to 52‑week high:\n")
    _fmt = {k:(lambda x: f"{x:.2f}") for k in ("Last","High52W","DistFrom52WHigh_%","Ret_1W_%","Ret_1M_%","Ret_3M_%")}
    print(snap.to_string(index=False, formatters=_fmt))
    out_csv = LOCAL_DIR / f"{_slug(_selected)}_ticker_snapshot.csv"
    snap.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS (industry ranker pipeline wrapped as a callable for the menu)
# ─────────────────────────────────────────────────────────────────────────────
def run_industry_ranker() -> None:
    # 1) load daily triplets (performance/value/sector)
    invalid, miss_val, miss_sec, trips = [], [], [], []
    if not DATA_DIR.exists():
        print(f"Data folder not found: {DATA_DIR}")
        return
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith(".json") and "-value" not in fn and "-sectors" not in fn:
            m = date_pat.match(fn)
            if not m:
                invalid.append(fn); continue
            mm, dd = map(int, m.groups())
            try:
                dt_ = datetime(YEAR, mm, dd)
            except ValueError:
                invalid.append(fn); continue
            base = fn[:-5]
            perf, val, sec = (DATA_DIR / fn, DATA_DIR / f"{base}-value.json", DATA_DIR / f"{base}-sectors.json")
            if not val.exists(): miss_val.append(val.name); continue
            if not sec.exists(): miss_sec.append(sec.name); continue
            trips.append((dt_, perf, val, sec))
    print("Triplets:", len(trips), "| invalid:", len(invalid),
          "| missing‑val:", len(miss_val), "| missing‑sec:", len(miss_sec))
    if not trips:
        print("No valid data."); return

    with open(MAP_PATH) as f:
        IND2SEC = json.load(f)

    rows = []
    for dt_, perf_fp, val_fp, sec_fp in trips:
        perf = pd.read_json(perf_fp); val = pd.read_json(val_fp)
        sec  = pd.read_json(sec_fp).rename(columns={"Name": "Sector"})
        for d in (perf, val, sec):
            for c in d.columns:
                if c not in {"Name","Sector"}:
                    d[c] = force_numeric(d[c])
        merged = perf.merge(val, on="Name", suffixes=('', '_val'))
        merged["Sector"] = merged["Name"].map(IND2SEC)
        merged = merged.merge(sec, on="Sector", suffixes=('', '_sector'))
        merged["Date"] = dt_
        rows.append(merged)

    df = (pd.concat(rows, ignore_index=True).sort_values(["Name","Date"]))

    # 2) macro overlays
    df = build_macro_features(df, IND2SEC)
    _cpi = fred_get("CPIAUCSL")
    if not _cpi.empty:
        df["FRED_CPI_YoY"] = (_cpi.pct_change(12) * 100).reindex(df["Date"]).ffill().values
    df = add_macro_regime_gate_features(df)

    # 3) technical momentum & sector‑relative packs
    df["price_idx"] = (1 + df["Change"].fillna(0).div(100)).groupby(df["Name"]).transform("cumprod")
    df["MA20"] = df.groupby("Name")["price_idx"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["MA50"] = df.groupby("Name")["price_idx"].transform(lambda x: x.rolling(50, min_periods=1).mean())
    df["MA_Spread"] = (df["price_idx"] / df["MA50"] - 1) * 100
    df["RSI14"] = df.groupby("Name")["price_idx"].transform(lambda x: rsi(x, 14))
    breadth = (df.assign(above50=lambda d: d["price_idx"] > d["MA50"])
                 .groupby(["Sector","Date"])["above50"].mean().mul(100).rename("Breadth50"))
    df = df.merge(breadth, on=["Sector","Date"], how="left")
    df = add_multi_horizon_technicals(df)
    df["Ret21"] = df.groupby("Name")["price_idx"].pct_change(21) * 100
    df["Sector_Ret21_Avg"] = df.groupby(["Sector","Date"])["Ret21"].transform("mean")
    df["Rel_Ret21"] = df["Ret21"] - df["Sector_Ret21_Avg"]
    df["RankInSector_Ret21"] = df.groupby(["Sector","Date"])["Ret21"].rank(method="dense", ascending=False)
    df["RankInSector_Change"] = df.groupby("Name")["RankInSector_Ret21"].diff()
    df["Sector_MA_Spread"] = df.groupby(["Sector","Date"])["MA_Spread"].transform("mean")
    df["Rel_MA_Spread"]    = df["MA_Spread"] - df["Sector_MA_Spread"]

    # 4) Schwab enrichers (best effort; gracefully NaN if APIs differ)
    last_date, win4 = df["Date"].max(), timedelta(days=28)
    add = {k: [] for k in ("Earn_Density","Avg_EPS_Surp","Net_UpDown","EPS_RevPct","ShortFloat","ETF_FlowPct")}
    for ind in df["Name"].unique():
        anal = schwab_get(f"research/analyst/{ind}", cache=f"anal_{ind}", ttl=86400) or {}
        msk = (df["Name"] == ind)
        add["Earn_Density"].extend([schwab_get(f"fundamentals/earningsdensity/{ind}",
                                               params=dict(start=last_date.strftime("%Y-%m-%d"),
                                                           end=(last_date + win4).strftime("%Y-%m-%d")),
                                               cache=f"earn_{ind}", ttl=86400).get("count", np.nan)] * msk.sum())
        add["Avg_EPS_Surp"].extend([anal.get("avgSurprise", np.nan)] * msk.sum())
        add["Net_UpDown"].extend([anal.get("netUpgrades", np.nan)] * msk.sum())
        add["EPS_RevPct"].extend([anal.get("epsRevisionPct", np.nan)] * msk.sum())
        add["ShortFloat"].extend([schwab_get(f"marketdata/shortinterest/{ind}", cache=f"short_{ind}", ttl=86400).get("avgShortFloat", np.nan)] * msk.sum())
        add["ETF_FlowPct"].extend([schwab_get(f"fundflows/etf/{ind}", cache=f"flow_{ind}", ttl=86400).get("flowPct", np.nan)] * msk.sum())
    for k, v in add.items():
        df[k] = v

    df = add_group_breadth_features(df)
    df = add_supersector_scorecard(df)
    df = add_schwab_ownership_insider(df)
    df = add_analyst_dispersion_revisions(df)
    df = add_feature_interactions(df, {})

    # 5) forward 1‑month % return (training target) and ranking
    future = {}
    for n, g in df.groupby("Name", sort=False):
        pct = g["Change"].fillna(0) / 100; idx = g.index
        for i in range(len(g)):
            future[idx[i]] = ((np.prod(1 + pct.iloc[i+1:i+1+FWD_DAYS]) - 1) * 100
                              if i + FWD_DAYS < len(g) else np.nan)
    df["Future1M"] = pd.Series(future)
    snapshot_dt = df["Date"].max()
    leaders = fit_rank_model_and_predict(df, last_date=snapshot_dt, top_k=TOP_K, rng_seed=RNG_SEED)

    print(f"\nTop {TOP_K} ranked leaders (snapshot {snapshot_dt:%Y-%m-%d})\n")
    print(leaders.to_string(index=False, formatters={"RankScore": "{:.4f}".format}))

    # >>> NEW: persist Top-10 leaders for this run
    append_top10_leaders_log(snapshot_dt, leaders, TOP10_LEADERS_PATH)

    # Existing drill-down prompt
    _view = input("\nView tickers inside one of the Top 10 industries from this run? [y/N]: ").strip().lower()
    if _view in {"y","yes"}:
        _run_drilldown_block(df, leaders)

    # Post-Top-10 options requested earlier
    while True:
        post = input("""
After Top-10:
  S) Scan Current Signals in selected Top industries
  B) Backtest Signals (submenu) on selected Top industries
  Q) Back to main
Selection: """).strip().lower()

        if post in ("q",""):
            break
        elif post == "s":
            inds = _choose_multiple_from_leaders(leaders)
            if not inds:
                continue
            ticks = _resolve_tickers_for_industries(inds)
            if not ticks:
                print("No tickers resolved from the selected industries.")
                continue

            sig_df, pre_df, audit, regime_flag = scan_current_signals_with_forecast(
                ticks,
                strategies=STRATEGY_SLUGS,
                horizon_hours=DEFAULT_FORECAST_HOURS,
                include_pre_signals=True
            )

            as_of = (sig_df["date"].max() if isinstance(sig_df, pd.DataFrame) and "date" in sig_df.columns and not sig_df.empty else "—")
            render_table(sig_df, f"Current Signals (Top selections; as of {as_of})")
            render_pre_signals_table(pre_df, "Forecast Pre‑Signals (read‑only; do not auto‑trade)", horizon_hours=DEFAULT_FORECAST_HOURS)
            print_gate_audit(audit, regime_on=regime_flag)

            if not sig_df.empty:
                add_signals_to_portfolio_prompt(sig_df, strategy="Current")
        elif post == "b":
            inds = _choose_multiple_from_leaders(leaders)
            if not inds:
                continue
            ticks = _resolve_tickers_for_industries(inds)
            if not ticks:
                print("No tickers resolved from the selected industries.")
                continue
            menu_backtest_signals(tickers=ticks)
        else:
            print("Invalid selection.")

def _resolve_tickers_for_industries(industry_names: List[str]) -> List[str]:
    """
    Given a list of industry display names, return the deduped list of member tickers
    using industries-tickers.json. Names are normalized before lookup.
    """
    ind_map = _load_industry_ticker_map()
    if not ind_map:
        return []
    out: set[str] = set()
    for name in industry_names:
        ticks = ind_map.get(_norm_industry_key(name), [])
        out.update(_sanitize_tickers(ticks))
    return sorted(out)


def _choose_multiple_from_leaders(leaders_df: pd.DataFrame) -> List[str]:
    """
    Let the user choose one/multiple/all industries from the Top-K leaders table.
    Accepts: 'all', comma-separated numbers, names.
    Returns a list of industry display names.
    """
    if not isinstance(leaders_df, pd.DataFrame) or leaders_df.empty:
        return []
    names = leaders_df["Name"].astype(str).tolist()
    print("\nSelect industries from the Top list:")
    for i, n in enumerate(names, 1):
        print(f"  {i:>2}) {n}")
    raw = input("\nEnter 'all', numbers (e.g. 1,3,7) or names (comma-separated). Blank to cancel: ").strip()
    if not raw:
        return []
    raw = raw.strip()
    if raw.lower() == "all":
        return names[:]
    picks: list[str] = []
    parts = [p.strip() for p in raw.split(",")]
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            idx = int(p)
            if 1 <= idx <= len(names):
                picks.append(names[idx - 1])
        else:
            # match by exact name first, else case-insensitive
            if p in names:
                picks.append(p)
            else:
                cand = [n for n in names if n.lower() == p.lower()]
                if cand:
                    picks.append(cand[0])
    # dedupe, preserve order
    seen = set(); out = []
    for x in picks:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def _last_full_week_bounds(today: date) -> tuple[date, date]:
    """
    Return (Mon..Sun) for the last *completed* calendar week.
    """
    cw_mon = today - timedelta(days=today.weekday())  # current week's Monday
    start = cw_mon - timedelta(days=7)                # last week's Monday
    end   = cw_mon - timedelta(days=1)                # last week's Sunday
    return start, end


def _list_last_n_weeks(n: int = 20, *, today: Optional[date] = None) -> List[tuple[date, date]]:
    """
    Produce a list of (Mon..Sun) tuples for the last n completed weeks, newest first.
    """
    if today is None:
        today = date.today()
    first_start, first_end = _last_full_week_bounds(today)
    out: list[tuple[date, date]] = []
    for i in range(n):
        s = first_start - timedelta(days=7 * i)
        e = s + timedelta(days=6)
        out.append((s, e))
    return out


def _prev_month_bounds(today: date) -> tuple[date, date]:
    """
    Previous calendar month (first..last day).
    """
    first_cur = today.replace(day=1)
    last_prev = first_cur - timedelta(days=1)
    first_prev = last_prev.replace(day=1)
    return first_prev, last_prev


def _month_bounds_for_date(d: date) -> tuple[date, date]:
    """
    Calendar month for the given date (first..last day).
    """
    first = d.replace(day=1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    last = d.replace(day=last_day)
    return first, last


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE PICKER & MENUS
# ─────────────────────────────────────────────────────────────────────────────
def pick_universe() -> List[str]:
    while True:
        print("""
Choose universe:
  1) Use Stock Watchlist
  2) Pick Industry (one or many) from industries-tickers.json
  3) Enter tickers manually
  0) Cancel
""")
        sel = input("Selection: ").strip()
        if sel in ("0",""): return []
        if sel == "1":
            wl = load_watchlist()
            if not wl: print("Watchlist is empty.")
            return wl
        if sel == "2":
            ind_map = _load_industry_ticker_map()
            if not ind_map:
                print("industries-tickers.json not found or empty."); continue
            names = sorted(ind_map.keys())
            for i, n in enumerate(names, 1):
                print(f"  {i:>3}) {n}")
            raw = input("Enter numbers or names (comma-separated): ").strip()
            if not raw: continue
            parts = [p.strip() for p in raw.split(",")]
            inds = []
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    if 1 <= idx <= len(names): inds.append(names[idx-1])
                elif p in ind_map:
                    inds.append(p)
            tickers = sorted({t for ind in inds for t in ind_map.get(ind, [])})
            if not tickers:
                print("No tickers resolved from selection."); continue
            return tickers
        if sel == "3":
            raw = input("Enter symbols separated by space or comma: ").strip()
            toks = [t.strip().upper() for t in raw.replace(","," ").split() if t.strip()]
            return _sanitize_tickers(toks)
        print("Invalid selection.")

def menu_watchlist():
    while True:
        wl = load_watchlist()
        print("\nStock Watchlist\nCurrent:", ", ".join(wl) if wl else "(empty)")
        print("  A) Add    R) Remove    C) Clear    Q) Back\n")
        sel = input("Choice: ").strip().upper()
        if sel in ("Q",""): break
        if sel == "A":
            raw = input("Enter symbols (space/comma): ").strip()
            add = _sanitize_tickers(raw.replace(","," ").split())
            save_watchlist(wl + add)
        elif sel == "R":
            raw = input("Enter symbols to remove: ").strip()
            rem = set(_sanitize_tickers(raw.replace(","," ").split()))
            save_watchlist([t for t in wl if t not in rem])
        elif sel == "C":
            if input("Clear watchlist? (yes/no): ").strip().lower() == "yes":
                save_watchlist([])
        else:
            print("Invalid.")

def menu_current_signals():
    tickers = pick_universe()
    if not tickers:
        return

    # Run scanner with Forecast layer (includes pre‑signals)
    sig_df, pre_df, audit, regime_flag = scan_current_signals_with_forecast(
        tickers,
        strategies=STRATEGY_SLUGS,
        horizon_hours=DEFAULT_FORECAST_HOURS,
        include_pre_signals=True
    )

    as_of = (sig_df["date"].max() if isinstance(sig_df, pd.DataFrame) and "date" in sig_df.columns and not sig_df.empty else "—")
    render_table(sig_df, f"Current Signals (ALL strategies; as of {as_of})")

    # Forecast pre‑signals rendered beneath Current Signals
    render_pre_signals_table(pre_df, "Forecast Pre‑Signals (staging → Predicted Signals)", horizon_hours=DEFAULT_FORECAST_HOURS)

    # Context / gate audits
    print_gate_audit(audit, regime_on=regime_flag)

    # Option to add *actual* signals straight to Active Portfolio
    if not sig_df.empty:
        add_signals_to_portfolio_prompt(sig_df)

    # NEW: Option to stage forecasted trade plans into "Predicted Signals"
    if isinstance(pre_df, pd.DataFrame) and not pre_df.empty:
        add_pre_signals_to_predicted_prompt(pre_df)

def add_pre_signals_to_predicted_prompt(pre_df: pd.DataFrame) -> None:
    """
    Stages forecasted setups (with buy zones) into a persistent 'predicted_signals.json'
    so they can be monitored and promoted later.
    """
    if pre_df is None or pre_df.empty:
        return
    ans = input(f"Add {len(pre_df)} forecasted setups to 'Portfolio View – Predicted Signals'? (y/N): ").strip().lower()
    if ans not in ("y","yes"):
        return

    items = load_predicted_signals()
    for _, r in pre_df.iterrows():
        sym = str(r.get("ticker","")).upper().strip()
        if not sym:
            continue
        def fnum(x):
            try:
                v = float(x); return None if not np.isfinite(v) else float(v)
            except Exception:
                return None
        rec = {
            "symbol":       sym,
            "date":         str(r.get("date")) if pd.notna(r.get("date")) else date.today().isoformat(),
            "strategy":     str(r.get("strategy","Pattern + Forecast (pre)")),
            "pivot":        fnum(r.get("pivot")),
            "buy_z_lo":     fnum(r.get("buy_z_lo")),
            "buy_z_hi":     fnum(r.get("buy_z_hi")),
            "plan_entry":   fnum(r.get("plan_entry")),
            "plan_stop":    fnum(r.get("plan_stop")),
            "t1":           fnum(r.get("t1")),
            "t2":           fnum(r.get("t2")),
            "R":            fnum(r.get("R")),
            "rr_to_t2":     fnum(r.get("rr_to_t2")),
            "p_hit_1R":     fnum(r.get("p_hit_1R")),
            "atr14":        fnum(r.get("atr14")),
            "notes":        "Staged from Current Signals → Pre‑Signals",
            "created_ts":   datetime.now().isoformat(timespec="seconds")
        }
        # sanity: need a buy zone and a stop
        if rec["buy_z_lo"] is None or rec["buy_z_hi"] is None or rec["plan_stop"] is None:
            continue
        items.append(rec)

    items = _dedupe_predicted(items)
    save_predicted_signals(items)
    print(f"[Saved Predicted Signals] {_predicted_signals_path()}")


def _eligible_and_common_day(tickers: List[str]) -> tuple[List[str], date]:
    from collections import Counter
    last_by_ticker: dict[str, date] = {}
    for sym in sorted(set(tickers)):
        bars = get_bars(sym, date.today() - timedelta(days=10), date.today(), 200)
        if bars is None or bars.empty:
            continue
        last_by_ticker[sym] = bars.index[-1].date()
    if not last_by_ticker:
        return [], date.today()
    counts = Counter(last_by_ticker.values())
    common_day = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    eligible = [sym for sym, d in last_by_ticker.items() if d == common_day]
    return eligible, common_day

def scan_predictive_setups(tickers: List[str], as_of: date, horizon_days: int = 3) -> pd.DataFrame:
    """
    Forward-looking scan: flag symbols CLOSE to a valid setup and emit a BUY RANGE
    for the next ~1–3 sessions. Uses indicators already exposed by compute_signals.
    Returns: date,ticker,strategy,setup_type,pivot,buy_low,buy_high,proximity_atr,proximity_pct,atr14,note
    """
    rows = []
    for sym in sorted(set(tickers)):
        try:
            bars = get_bars(sym, as_of - timedelta(days=140), as_of, min_history_days=260)
            if bars is None or len(bars) < 60:
                continue
            indf = compute_signals_for_ticker(bars, strategy="all", as_of=as_of)
            # locate the session row for 'as_of' (or last <= as_of)
            if pd.Timestamp(as_of) in indf.index.normalize().unique():
                day_ts = indf.loc[indf.index.normalize() == pd.Timestamp(as_of)].index[-1]
            else:
                elig = indf.loc[indf.index.normalize() <= pd.Timestamp(as_of)]
                if elig.empty:
                    continue
                day_ts = elig.index[-1]

            i = indf.index.get_loc(day_ts)
            if i < 1:
                continue
            row  = indf.iloc[i]
            prev = indf.iloc[i-1]

            def _add(slug_label, setup_type, pivot, buy_low, buy_high, prox_abs, note):
                close = float(row["close"]); atrv = float(row.get("atr14", np.nan))
                prox_pct = (prox_abs / max(close, 1e-6)) * 100.0
                rows.append({
                    "date": day_ts.date().isoformat(),
                    "ticker": sym,
                    "strategy": slug_label,
                    "setup_type": setup_type,
                    "pivot": round(pivot, 2) if np.isfinite(pivot) else None,
                    "buy_low": round(buy_low, 2) if np.isfinite(buy_low) else None,
                    "buy_high": round(buy_high, 2) if np.isfinite(buy_high) else None,
                    "proximity_atr": round(prox_abs / atrv, 2) if np.isfinite(atrv) else None,
                    "proximity_pct": round(prox_pct, 2),
                    "atr14": round(atrv, 2) if np.isfinite(atrv) else None,
                    "note": note
                })

            # Pull the reference fields (were exposed in Infinit-Strategies.compute_signals)
            close  = float(row.get("close", np.nan))
            atrv   = float(row.get("atr14", np.nan))
            ema20  = float(row.get("ema20", np.nan))
            sma50  = float(row.get("sma50", np.nan))
            sma200 = float(row.get("sma200", np.nan))
            adx    = float(row.get("adx", np.nan))
            dmi    = float(row.get("dmi_diff", np.nan))
            macdh  = float(row.get("macd_hist", np.nan))
            hma20  = float(row.get("hma20", np.nan))
            bb_up  = float(row.get("bb_up", np.nan))
            bb_lo  = float(row.get("bb_lo", np.nan))
            bb_w   = float(row.get("bb_width", np.nan))
            vol_sc = float(row.get("vol_score", 0.0))
            hi10_prev = float(indf["hi10"].iloc[i-1]) if pd.notna(indf["hi10"].iloc[i-1]) else np.nan
            hi20_prev = float(indf["hi20"].iloc[i-1]) if pd.notna(indf["hi20"].iloc[i-1]) else np.nan

            if not (np.isfinite(close) and np.isfinite(atrv) and atrv > 0):
                continue
            b_lo_pct, b_hi_pct = _entry_buffer(close, atrv)

            # ===== 1) Breakout (Trend + Tight Base) – near 20d high =====
            uptrend = (close > sma50) and (sma50 > sma200) and (ema20 > float(prev.get("ema20", ema20))) and (adx > 18)
            bw_q20 = indf["bb_width"].rolling(120, min_periods=40).quantile(0.20).iloc[i]
            tight  = np.isfinite(bb_w) and np.isfinite(bw_q20) and (bb_w <= bw_q20)
            if uptrend and tight and np.isfinite(hi20_prev):
                dist = hi20_prev - close
                if 0 < dist <= min(0.75*atrv, 0.02*close):
                    pivot = hi20_prev
                    _add("Breakout (Trend + Tight Base)", "Breakout (near 20d high)",
                         pivot, pivot*(1.0 + b_lo_pct), pivot*(1.0 + b_hi_pct), dist,
                         f"Within {dist/atrv:.2f} ATR of pivot; vol_score={int(vol_sc)}")

            # ===== 2) Squeeze (BB) – squeeze on, near upper band =====
            squeeze = np.isfinite(bb_w) and np.isfinite(bw_q20) and (bb_w <= bw_q20)
            pivot_sq = max(hi10_prev if np.isfinite(hi10_prev) else -np.inf,
                           bb_up      if np.isfinite(bb_up)      else -np.inf)
            if squeeze and np.isfinite(pivot_sq):
                dist = pivot_sq - close
                if 0 < dist <= min(0.75*atrv, 0.02*close):
                    _add("Volatility Squeeze (BB)", "Squeeze (near breakout)",
                         pivot_sq, pivot_sq*(1.0 + b_lo_pct), pivot_sq*(1.0 + b_hi_pct), dist,
                         f"Squeeze active; within {dist/atrv:.2f} ATR of trigger")

            # ===== 3) Pullback (Holy Grail) – approaching EMA20 =====
            strong_trend = (close > sma50) and (sma50 > sma200) and (adx > 25) and (dmi > 0)
            pulled = (close < float(prev.get("close", close))) or (indf["close"].iloc[max(0, i-3):i+1].pct_change().fillna(0).sum() < 0)
            if strong_trend and pulled and np.isfinite(ema20):
                dist = close - ema20
                if 0 <= dist <= min(0.50*atrv, 0.02*close):
                    buy_low  = ema20 - 0.15*atrv
                    buy_high = ema20 + 0.10*atrv
                    _add("Pullback (Holy Grail)", "Pullback (tag 20EMA)",
                         ema20, buy_low, buy_high, dist, "Near EMA20; watch for reversal day")

            # ===== 4) ToS Momentum – MACD hist rising toward zero =====
            trend_ok = (close > ema20) and (close > sma50) and (dmi > 0) and (hma20 > float(prev.get("hma20", hma20)))
            macd_rising = macdh > float(prev.get("macd_hist", macdh))
            macd_near = abs(macdh) <= max(0.0025*close, 0.10*atrv)  # near zero line
            if trend_ok and macd_rising and macd_near:
                buy_low  = close - 0.10*atrv
                buy_high = close + 0.20*atrv
                prox = max(0.0, min(0.20*atrv, buy_high - close))
                _add("ToS Momentum", "Momentum (MACD→0)", close, buy_low, buy_high, prox,
                     "MACD hist rising toward zero; HMA rising")
        except Exception:
            continue

    cols = ["date","ticker","strategy","setup_type","pivot","buy_low","buy_high","proximity_atr","proximity_pct","atr14","note"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["proximity_atr","proximity_pct"], ascending=[True, True]).reset_index(drop=True)
    return df

def scan_current_signals_with_forecast(tickers: List[str],
                                       *,
                                       strategies: Optional[List[str]] = None,
                                       horizon_hours: int = DEFAULT_FORECAST_HOURS,
                                       include_pre_signals: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict, Optional[bool]]:
    """
    Current signals + optional Forecast pre-signals.
    Improvements:
      • If strict common-day scan yields nothing, fall back to scanning each
        symbol on its own latest trading day.
    """
    if not strategies:
        strategies = STRATEGY_SLUGS
    dbg = DEBUG

    # Discover the most recent date for each symbol
    last_by_ticker: dict[str, date] = {}
    for sym in sorted(set(tickers)):
        bars = get_bars(sym, date.today() - timedelta(days=10), date.today(), 200)
        if bars is None or bars.empty:
            if dbg: print(f"[DEBUG] current: no recent bars for {sym}")
            continue
        last_by_ticker[sym] = bars.index[-1].date()

    if not last_by_ticker:
        if dbg: print("[DEBUG] current: empty universe after data fetch.")
        return pd.DataFrame(), pd.DataFrame(), {}, None

    # Strict common-day attempt
    from collections import Counter
    counts = Counter(last_by_ticker.values())
    common_day = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    eligible_common = [sym for sym, d in last_by_ticker.items() if d == common_day]

    if dbg:
        print(f"[DEBUG] current: {len(last_by_ticker)} symbols; common_day={common_day}; eligible_common={len(eligible_common)}")

    sig_df_common = scan_on_date(eligible_common, common_day, strategies=strategies) if eligible_common else pd.DataFrame()

    # Pre‑signals on the common day
    pre_rows_common = []
    if include_pre_signals and PATTERN_FORECAST in STRATEGY_SLUGS and eligible_common:
        for sym in eligible_common:
            try:
                bars = get_bars(sym, common_day - timedelta(days=150), common_day, min_history_days=260)
                if bars is None or len(bars) < 60:
                    continue

                indf = compute_signals_for_ticker(bars, strategy="all", as_of=common_day)
                ts = pd.Timestamp(common_day)
                if (indf.index.normalize() == ts).any():
                    row = indf.loc[indf.index.normalize() == ts].iloc[-1]
                else:
                    elig = indf.loc[indf.index.normalize() <= ts]
                    if elig.empty:
                        continue
                    row = elig.iloc[-1]

                if not bool(row.get("pre_sig_pattern", False)):
                    continue
                if within_earnings_blackout(sym, common_day, RULES.earnings_blackout_days):
                    continue
                if _is_macro_blackout(common_day, RULES.macro_blackout_window):
                    continue

                pivot   = float(row.get("pattern_pivot", np.nan))
                buy_lo  = float(row.get("buy_z_lo", np.nan))
                buy_hi  = float(row.get("buy_z_hi", np.nan))
                close_px= float(row.get("close", np.nan)) if "close" in row else float(bars["close"].iloc[-1])
                if not np.isfinite(pivot): pivot = close_px

                stop0_pat = float(row.get(f"stop0_{PATTERN_FORECAST}", np.nan))
                atrv = float(row.get("atr14", np.nan)) if pd.notna(row.get("atr14", np.nan)) else float(atr_series(bars, 14).iloc[-1])
                if not np.isfinite(stop0_pat): stop0_pat = pivot - max(1e-3, 1.2 * atrv)

                plan_entry = pivot
                plan_stop  = float(refine_stop_with_intraday(sym, common_day, plan_entry, stop0_pat, row, lookback_minutes=90))
                R = max(1e-6, plan_entry - plan_stop)

                pattern_tgt0 = float(row.get(f"target0_{PATTERN_FORECAST}", np.nan))
                if not np.isfinite(pattern_tgt0): pattern_tgt0 = plan_entry + 2.2 * atrv
                t1 = plan_entry + RULES.first_scale_R * R
                t2 = max(pattern_tgt0, plan_entry + 2.0 * R)

                start_dt = datetime.combine(common_day - timedelta(days=14), datetime.min.time())
                end_dt   = datetime.combine(common_day, datetime.max.time())
                intra = get_intraday_bars(sym, start_dt, end_dt, interval_minutes=5, minute_granularity=1, include_extended=False)
                p1r = estimate_prob_hit_plus1R(intra, entry_px=plan_entry, R_per_share=R, horizon_hours=horizon_hours, sessions_lookback=10)

                pre_rows_common.append({
                    "date":     common_day.isoformat(),
                    "time":     datetime.now().strftime("%H:%M:%S"),
                    "ticker":   sym,
                    "buy_z_lo": (None if not np.isfinite(buy_lo) else float(buy_lo)),
                    "buy_z_hi": (None if not np.isfinite(buy_hi) else float(buy_hi)),
                    "pivot":    float(pivot),
                    "plan_entry": round(float(plan_entry), 2),
                    "plan_stop":  round(float(plan_stop),  2),
                    "R":          round(float(R),          2),
                    "t1":         round(float(t1),         2),
                    "t2":         round(float(t2),         2),
                    "rr_to_t2":   round(float((t2 - plan_entry) / max(R, 1e-6)), 1),
                    "p_hit_1R":   (None if p1r is None else float(p1r)),
                    "atr14":      (round(float(atrv), 2) if np.isfinite(atrv) else None),
                    "strategy":  "Pattern + Forecast (pre)"
                })
            except Exception:
                continue

    pre_df_common = pd.DataFrame(
        pre_rows_common,
        columns=["date","time","ticker","buy_z_lo","buy_z_hi","pivot","plan_entry","plan_stop","R","t1","t2","rr_to_t2","p_hit_1R","atr14","strategy"]
    )

    # If we found nothing on the common day, fall back to each symbol's own last day
    if sig_df_common.empty and (not include_pre_signals or pre_df_common.empty):
        if dbg:
            print("[DEBUG] current: common-day produced no results; trying per-symbol latest-day scan.")

        sig_rows = []
        pre_rows = []
        for sym, d in last_by_ticker.items():
            try:
                s_one = scan_on_date([sym], d, strategies=strategies)
                if not s_one.empty:
                    sig_rows.append(s_one)

                if include_pre_signals and PATTERN_FORECAST in STRATEGY_SLUGS:
                    bars = get_bars(sym, d - timedelta(days=150), d, min_history_days=260)
                    if bars is None or len(bars) < 60:
                        continue

                    indf = compute_signals_for_ticker(bars, strategy="all", as_of=d)
                    ts = pd.Timestamp(d)
                    if (indf.index.normalize() == ts).any():
                        row = indf.loc[indf.index.normalize() == ts].iloc[-1]
                    else:
                        elig = indf.loc[indf.index.normalize() <= ts]
                        if elig.empty:
                            continue
                        row = elig.iloc[-1]
                    if not bool(row.get("pre_sig_pattern", False)):
                        continue
                    if within_earnings_blackout(sym, d, RULES.earnings_blackout_days):
                        continue
                    if _is_macro_blackout(d, RULES.macro_blackout_window):
                        continue

                    pivot   = float(row.get("pattern_pivot", np.nan))
                    buy_lo  = float(row.get("buy_z_lo", np.nan))
                    buy_hi  = float(row.get("buy_z_hi", np.nan))
                    close_px= float(row.get("close", np.nan)) if "close" in row else float(bars["close"].iloc[-1])
                    if not np.isfinite(pivot): pivot = close_px

                    stop0_pat = float(row.get(f"stop0_{PATTERN_FORECAST}", np.nan))
                    atrv = float(row.get("atr14", np.nan)) if pd.notna(row.get("atr14", np.nan)) else float(atr_series(bars, 14).iloc[-1])
                    if not np.isfinite(stop0_pat): stop0_pat = pivot - max(1e-3, 1.2 * atrv)

                    plan_entry = pivot
                    plan_stop  = float(refine_stop_with_intraday(sym, d, plan_entry, stop0_pat, row, lookback_minutes=90))
                    R = max(1e-6, plan_entry - plan_stop)

                    pattern_tgt0 = float(row.get(f"target0_{PATTERN_FORECAST}", np.nan))
                    if not np.isfinite(pattern_tgt0): pattern_tgt0 = plan_entry + 2.2 * atrv
                    t1 = plan_entry + RULES.first_scale_R * R
                    t2 = max(pattern_tgt0, plan_entry + 2.0 * R)

                    start_dt = datetime.combine(d - timedelta(days=14), datetime.min.time())
                    end_dt   = datetime.combine(d, datetime.max.time())
                    intra = get_intraday_bars(sym, start_dt, end_dt, interval_minutes=5, minute_granularity=1, include_extended=False)
                    p1r = estimate_prob_hit_plus1R(intra, entry_px=plan_entry, R_per_share=R, horizon_hours=horizon_hours, sessions_lookback=10)

                    pre_rows.append({
                        "date":     d.isoformat(),
                        "time":     datetime.now().strftime("%H:%M:%S"),
                        "ticker":   sym,
                        "buy_z_lo": (None if not np.isfinite(buy_lo) else float(buy_lo)),
                        "buy_z_hi": (None if not np.isfinite(buy_hi) else float(buy_hi)),
                        "pivot":    float(pivot),
                        "plan_entry": round(float(plan_entry), 2),
                        "plan_stop":  round(float(plan_stop),  2),
                        "R":          round(float(R),          2),
                        "t1":         round(float(t1),         2),
                        "t2":         round(float(t2),         2),
                        "rr_to_t2":   round(float((t2 - plan_entry) / max(R, 1e-6)), 1),
                        "p_hit_1R":   (None if p1r is None else float(p1r)),
                        "atr14":      (round(float(atrv), 2) if np.isfinite(atrv) else None),
                        "strategy":  "Pattern + Forecast (pre)"
                    })
            except Exception:
                continue

        sig_df = (pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame(columns=["date","time","ticker","entry","atr14","stop","target","eow_close","strategy"]))
        pre_df = pd.DataFrame(pre_rows, columns=["date","time","ticker","buy_z_lo","buy_z_hi","pivot","plan_entry","plan_stop","R","t1","t2","rr_to_t2","p_hit_1R","atr14","strategy"])
        audit = gate_audit_for_universe(list(last_by_ticker.keys()), max(last_by_ticker.values()), strategies=strategies)
        regime_flag = compute_regime_risk_on(max(last_by_ticker.values()))
        return sig_df, pre_df, audit, regime_flag

    # Common-day success path
    audit = gate_audit_for_universe(eligible_common, common_day, strategies=strategies)
    regime_flag = compute_regime_risk_on(common_day)
    return sig_df_common, pre_df_common, audit, regime_flag


def _parse_date_token(tok: str) -> date:
    tok = tok.strip()
    for fmt in ("%m-%d-%y", "%Y-%m-%d"):
        try: return datetime.strptime(tok, fmt).date()
        except ValueError: pass
    raise ValueError(f"Invalid date: {tok}")

def menu_backtest_signals(tickers: Optional[List[str]] = None):
    """
    Backtest sub-menu:
      • Specific date
      • Specific week (choose from last 20 completed weeks)
      • Yesterday
      • Last week (Mon–Sun)
      • Last two weeks (two full Mon–Sun weeks)
      • Last month (previous calendar month)
      • Last 90 days
      • Month from 90 days ago
    """
    if tickers is None:
        tickers = pick_universe()
    if not tickers:
        return

    SUBMENU = """
Backtest – choose a window:
  1) Specific date
  2) Specific week (pick from last 20 weeks)
  3) Yesterday
  4) Last week (Mon–Sun)
  5) Last two weeks (2× Mon–Sun)
  6) Last month (previous calendar month)
  7) Last 90 days
  8) Month from 90 days ago
  0) Cancel
Selection: """

    while True:
        sel = input(SUBMENU).strip()
        if sel in ("0", ""):
            return

        today = date.today()

        if sel == "1":
            raw = input("Enter a date (MM-DD-YY or YYYY-MM-DD): ").strip()
            try:
                d = _parse_date_token(raw)
            except ValueError as e:
                print(e)
                continue
            df = scan_on_date_backtest(tickers, d)
            render_table(df, f"Backtest Signals ({d})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        elif sel == "2":
            weeks = _list_last_n_weeks(20, today=today)
            print("\nSelect a week:")
            for i, (s, e) in enumerate(weeks, 1):
                iso = s.isocalendar()
                print(f"  {i:>2}) {s} → {e}  (ISO {iso.year}-W{iso.week:02d})")
            raw = input("Choice (1-20): ").strip()
            if not raw.isdigit() or not (1 <= int(raw) <= len(weeks)):
                print("Invalid selection.")
                continue
            s, e = weeks[int(raw) - 1]
            df = scan_range_backtest(tickers, s, e)
            render_table(df, f"Backtest Signals (Week {s} → {e})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        elif sel == "3":
            d = today - timedelta(days=1)
            df = scan_on_date_backtest(tickers, d)
            render_table(df, f"Backtest Signals (Yesterday {d})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        elif sel == "4":
            s, e = _last_full_week_bounds(today)
            df = scan_range_backtest(tickers, s, e)
            render_table(df, f"Backtest Signals (Last week {s} → {e})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        elif sel == "5":
            s1, e1 = _last_full_week_bounds(today)
            s0 = s1 - timedelta(days=7)
            df = scan_range_backtest(tickers, s0, e1)
            render_table(df, f"Backtest Signals (Last two weeks {s0} → {e1})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        elif sel == "6":
            s, e = _prev_month_bounds(today)
            df = scan_range_backtest(tickers, s, e)
            render_table(df, f"Backtest Signals (Previous month {s} → {e})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        elif sel == "7":
            s = today - timedelta(days=90)
            e = today - timedelta(days=1)
            df = scan_range_backtest(tickers, s, e)
            render_table(df, f"Backtest Signals (Last 90 days {s} → {e})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        elif sel == "8":
            anchor = today - timedelta(days=90)
            s, e = _month_bounds_for_date(anchor)
            df = scan_range_backtest(tickers, s, e)
            render_table(df, f"Backtest Signals (Month from 90 days ago: {s} → {e})")
            if not df.empty:
                add_open_backtests_to_portfolio_prompt(df)
            return

        else:
            print("Invalid selection.")



def _holding_status(holding: dict) -> tuple[str, Optional[float]]:
    sym = holding.get("symbol","")
    last = _last_close_price_schwab(sym)
    if last is None:
        return ("No Data", None)
    stop  = float(holding.get("stop_loss", np.nan))
    pt    = holding.get("profit_target", None)
    status = "Holding"
    if np.isfinite(stop) and last <= stop:
        status = "Stop Loss Hit"
    elif pt is not None and np.isfinite(pt) and last >= float(pt):
        status = "Profit Target Hit"
    return status, last

def portfolio_view_active():
    """
    Portfolio View – Active Trades (CLI):
      • Shows Last, Entry, Initial Stop/Target and Updated Stop/Target (dynamic)
      • R: refresh
      • D: delete holdings that hit (dynamic) Stop/Target → logs to trade_history.json
      • Q: back to main
    """
    while True:
        holdings = load_holdings()
        if not holdings:
            print("Active Trades portfolio is empty.")
            return

        print("\n" + _c(" Portfolio View – Active Trades ", ANSI_BOLD))
        hdr = (
            f"{'Symbol':<7} {'Last':>8} {'Entry':>8} "
            f"{'InitStop':>9} {'InitTgt':>9} {'UpdStop':>9} {'UpdTgt':>9} "
            f"{'%Gain':>8} {'Status':<16} {'Strategy':<30}"
        )
        print(_c(hdr, FG_CYAN))

        total_gain = 0.0

        for i, h in enumerate(holdings):
            sym = str(h.get("symbol","")).upper()
            dyn = _compute_dynamic_levels_for_holding(h)
            last = dyn.get("last_price")
            entry= dyn.get("entry") or float(h.get("price_paid", np.nan))
            init_stop = dyn.get("initial_stop")
            init_tgt  = dyn.get("initial_target")
            upd_stop  = dyn.get("updated_stop")
            upd_tgt   = dyn.get("updated_target")
            status    = dyn.get("status","Holding")
            strat     = str(h.get("strategy",""))

            if last is None or not np.isfinite(entry):
                line = (
                    f"{sym:<7} {'—':>8} {fmt_num(entry,8):>8} "
                    f"{(f'{init_stop:.2f}' if init_stop is not None else '—'):>9} "
                    f"{(f'{init_tgt:.2f}'  if init_tgt  is not None else '—'):>9} "
                    f"{(f'{upd_stop:.2f}'  if upd_stop  is not None else '—'):>9} "
                    f"{(str(upd_tgt) if isinstance(upd_tgt,str) else (f'{upd_tgt:.2f}' if upd_tgt is not None else '—')):>9} "
                    f"{'—':>8} {_c(status, FG_YELLOW):<16} {strat:<30}"
                )
                print(line)
                continue

            gain_per = last - entry
            shs  = int(h.get("shares", 0))
            gain = gain_per * shs
            total_gain += gain

            color = FG_GREEN if gain >= 0 else FG_RED
            stat_color = (FG_RED if status == "Stop Loss Hit"
                          else (FG_GREEN if status == "Profit Target Hit" else FG_YELLOW))

            line = (
                f"{sym:<7} "
                f"{last:>8.2f} {entry:>8.2f} "
                f"{(f'{init_stop:.2f}' if init_stop is not None else '—'):>9} "
                f"{(f'{init_tgt:.2f}'  if init_tgt  is not None else '—'):>9} "
                f"{(f'{upd_stop:.2f}'  if upd_stop  is not None else '—'):>9} "
                f"{(str(upd_tgt) if isinstance(upd_tgt,str) else (f'{upd_tgt:.2f}' if upd_tgt is not None else '—')):>9} "
                f"{(f'{(gain_per/entry*100):.2f}%' if entry else '—'):>8} "
                f"{_c(status, stat_color):<16} {strat:<30}"
            )
            print(_c(line, color))

        print(_c(f"\nTotal P/L: {total_gain:+.2f}\n", FG_MAGENTA))

        cmd = input("Portfolio: [R]efresh, [D]elete hit, [Q]uit: ").strip().lower()
        if cmd in ("q",""):
            return
        elif cmd == "r":
            continue
        elif cmd == "d":
            # Recompute & build removal list
            holdings = load_holdings()
            to_keep, removed = [], []
            for h in holdings:
                dyn = _compute_dynamic_levels_for_holding(h)
                status = dyn.get("status","Holding")
                last   = dyn.get("last_price")
                if status in ("Stop Loss Hit","Profit Target Hit") and last is not None:
                    h["exit_date"]  = date.today().isoformat()
                    h["exit_price"] = float(last)
                    h["result"]     = status
                    removed.append(h)
                else:
                    to_keep.append(h)
            if not removed:
                print("No holdings qualify for deletion.")
                continue
            syms = ", ".join(sorted({h["symbol"].upper() for h in removed}))
            ans = input(f"Remove hit holdings ({syms})? (yes/no): ").strip().lower()
            if ans == "yes":
                save_holdings(to_keep)
                hist = load_trade_history()
                hist.extend(removed)
                save_trade_history(hist)
                print(f"Removed: {syms}")
            else:
                print("Deletion canceled.")

def portfolio_view():
    # Backward‑compat: old name
    return portfolio_view_active()


def portfolio_view_predicted():
    """
    Staging area for forecasted trade plans (persistent across runs).
    Shows live last price via Schwab, flags 'In Zone', and lets you promote
    entries IN the buy zone into Active Trades.
    """
    while True:
        items = load_predicted_signals()
        if not items:
            print("Predicted Signals staging is empty.")
            return

        print("\n" + _c(" Portfolio View – Predicted Signals ", ANSI_BOLD))
        hdr = (
            f"{'#':>2}  {'Symbol':<7} {'Last':>8} "
            f"{'BuyLo':>8} {'BuyHi':>8} {'Plan':>8} {'Stop':>8} {'T1':>8} {'T2':>8} "
            f"{'InZone':<8} {'Dist%':>7} {'Strategy':<28} {'AsOf':<10}"
        )
        print(_c(hdr, FG_CYAN))

        eligible = []   # indices that are in buy zone
        display_rows = []

        for i, r in enumerate(items, start=1):
            sym   = str(r.get("symbol","")).upper()
            last  = _last_close_price_schwab(sym)
            blo   = r.get("buy_z_lo"); bhi = r.get("buy_z_hi")
            plan  = r.get("plan_entry"); stp = r.get("plan_stop")
            t1    = r.get("t1"); t2 = r.get("t2")
            strat = str(r.get("strategy",""))
            asof  = str(r.get("date",""))

            in_zone = (last is not None and
                       blo is not None and bhi is not None and
                       float(blo) <= float(last) <= float(bhi))
            if in_zone:
                eligible.append(i-1)

            # % distance to plan entry
            dist = None
            try:
                if last is not None and plan not in (None, 0):
                    dist = (float(last)/float(plan) - 1.0) * 100.0
            except Exception:
                dist = None

            color = (FG_GREEN if in_zone else (FG_YELLOW if last is not None else FG_GRAY))
            line = (
                f"{i:>2}  {sym:<7} "
                f"{(f'{last:8.2f}' if last is not None else '      — ')} "
                f"{(f'{float(blo):8.2f}' if blo is not None else '      — ')} "
                f"{(f'{float(bhi):8.2f}' if bhi is not None else '      — ')} "
                f"{(f'{float(plan):8.2f}' if plan is not None else '      — ')} "
                f"{(f'{float(stp):8.2f}'  if stp  is not None else '      — ')} "
                f"{(f'{float(t1):8.2f}'   if t1   is not None else '      — ')} "
                f"{(f'{float(t2):8.2f}'   if t2   is not None else '      — ')} "
                f"{('YES' if in_zone else 'NO '):<8} "
                f"{(f'{dist:6.2f}' if dist is not None and np.isfinite(dist) else '   —  ')} "
                f"{strat:<28} {asof:<10}"
            )
            print(_c(line, color))

        print()
        cmd = input("Predicted: [R]efresh, [M]ove in‑zone → Active, [D]elete, [Q]uit: ").strip().lower()
        if cmd in ("q",""):
            return
        elif cmd == "r":
            continue
        elif cmd == "d":
            raw = input("Delete which? indices (e.g. 1,3) or 'all': ").strip().lower()
            if raw == "all":
                save_predicted_signals([])
                print("Cleared Predicted Signals.")
                continue
            idxs = []
            for tok in raw.replace(",", " ").split():
                if tok.isdigit():
                    k = int(tok) - 1
                    if 0 <= k < len(items):
                        idxs.append(k)
            if not idxs:
                print("Nothing to delete.")
                continue
            keep = [r for j, r in enumerate(items) if j not in set(idxs)]
            save_predicted_signals(keep)
            print(f"Deleted {len(idxs)} item(s).")
            continue
        elif cmd == "m":
            if not eligible:
                print("No items currently IN the buy zone.")
                continue

            shares = _prompt_int("Shares per promoted position", 100)
            moved = 0
            holdings = load_holdings()
            to_remove = []

            for j in eligible:
                r = items[j]
                sym = str(r.get("symbol","")).upper()
                last = _last_close_price_schwab(sym)
                if last is None:
                    continue

                # Choose entry: default to Last when available
                plan = r.get("plan_entry")
                ch = input(f"{sym}: use [L]ast {last:.2f} or [P]lan {plan if plan is not None else '—'}? [L/p]: ").strip().lower()
                entry = float(last) if ch in ("","l") or plan is None else float(plan)

                stop  = float(r.get("plan_stop"))
                tgt   = r.get("t2") if r.get("t2") is not None else r.get("t1")
                tgt   = (float(tgt) if tgt is not None and np.isfinite(float(tgt)) else None)
                strat = str(r.get("strategy","Pattern + Forecast"))

                holding = {
                    "symbol": sym,
                    "price_paid": float(entry),
                    "shares": int(shares),
                    "stop_loss": float(stop),
                    "profit_target": (float(tgt) if tgt is not None else None),
                    "stop_loss_initial": float(stop),
                    "profit_target_initial": (float(tgt) if tgt is not None else None),
                    "strategy": strat,
                    "entry_date": date.today().isoformat()
                }
                holdings.append(holding)
                to_remove.append(j)
                moved += 1

            if moved > 0:
                save_holdings(holdings)
                # remove promoted items from staging
                keep = [r for idx, r in enumerate(items) if idx not in set(to_remove)]
                save_predicted_signals(keep)
                print(f"Moved {moved} item(s) to Active Trades.")
            else:
                print("Nothing moved.")
            continue
        else:
            print("Invalid selection.")

def menu():
    # 1) OAuth bootstrap (opens a browser on first run / when refresh is needed)
    init_schwab_client()

    # 2) Register providers that depend on live clients
    try:
        register_daily_bars_provider(get_bars)
        register_intraday_provider(get_intraday_bars)
        # Unify the risk rules across modules
        global RULES
        RULES = TM_RULES
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] trade_management provider registration failed: {e}")

    # 3) Only now it is meaningful to probe data access
    assert_data_provider_ok()

    MENU = """
═══════════════════════════════════════════════════════
Industry Momentum + ToS Strategy Scanner
  1) Stock Watchlist
  2) Current Signals
  3) Backtest Signals
  4) Rank Industries (XGBoost)
  5) Portfolio View - Active Trades
  6) Portfolio View - Predicted Signals
  0) Quit
═══════════════════════════════════════════════════════
Selection: """
    while True:
        try:
            sel = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye"); return
        if sel in ("0",""): print("Bye"); return
        if sel == "1": menu_watchlist()
        elif sel == "2": menu_current_signals()
        elif sel == "3": menu_backtest_signals()
        elif sel == "4": run_industry_ranker()
        elif sel == "5": portfolio_view_active()
        elif sel == "6": portfolio_view_predicted()
        else: print("Invalid selection.")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nBye")
        sys.exit(0)
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)