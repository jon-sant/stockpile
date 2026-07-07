"""Fetch option chains by scraping finance.yahoo.com with a headless
browser and return the same normalized DataFrame as chain.py.

Why this exists: Yahoo's JSON API (what yfinance calls) frequently
serves zeroed bid/ask quotes after hours and soft-throttles bursts,
while the *website* keeps rendering real chain tables. Driving the
site through Selenium sidesteps the API entirely — slower per page,
but it works when the API path returns nothing.

Returns the same 17-column shape as chain.py:fetch_chain() so all
downstream code (iv_surface, earnings, display, report) is unchanged.
Greeks are computed via Black-Scholes from the table's IV, exactly
like the yfinance path.

Concurrency model
-----------------
All page loads go through one process-wide ThreadPoolExecutor whose
workers each own a long-lived browser (thread-local, quit at exit).
A single-ticker fetch fans its expiration pages across the pool; the
Watchlist/Portfolio tabs additionally scan multiple tickers at once,
and their page loads interleave in the same pool — total browser
count stays bounded by `max_workers` no matter how many symbols are
in flight.

Prerequisites
-------------
Chrome (default) or Firefox installed locally. Selenium 4.6+ resolves
the matching driver binary automatically (Selenium Manager) — no
chromedriver install needed. Tune via `[yahoo_headless]` in
config.toml (browser, max_workers, headless, settle_seconds,
page_timeout).
"""

from __future__ import annotations

import atexit
import logging
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from options_scanner.chain_common import build_option_row, safe_float, safe_int

log = logging.getLogger(__name__)

# Same rate chain.py uses for its Black-Scholes Greeks.
_RISK_FREE_RATE = 0.045

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# OCC contract symbol: root + yymmdd + C/P + strike*1000, e.g.
# AAPL260918C00150000. Groups: (yymmdd, side letter).
_OCC_RE = re.compile(r"(\d{6})([CP])\d{7,8}")

# The placeholder Yahoo renders in place of a chain table when its
# option data is dark (observed in the overnight hours).
_NO_DATA_RE = re.compile(r"There are no (?:calls|puts)", re.IGNORECASE)

# ── Pure HTML parsers (no Selenium — unit-testable) ──────────────────────


def extract_spot(html: str, ticker: str) -> float | None:
    """Pull the underlying's price out of an options-page HTML dump.

    Tries the structured fin-streamer element first (attribute and
    text-content forms, either attribute order), then the qsp-price
    test id, then the loose "price ±chg (%) At close/After hours"
    text. Returns None when nothing matches — the caller decides on
    a fallback.
    """
    tkr = re.escape(ticker.upper())
    patterns = [
        # <fin-streamer data-symbol="X" ... data-field="regularMarketPrice"
        #               ... data-value="123.45">
        rf'<fin-streamer[^>]*data-symbol="{tkr}"[^>]*data-field='
        rf'"(?:regularMarketPrice|price)"[^>]*data-value="([\d,]+\.?\d*)"',
        rf'<fin-streamer[^>]*data-field="(?:regularMarketPrice|price)"'
        rf'[^>]*data-symbol="{tkr}"[^>]*data-value="([\d,]+\.?\d*)"',
        # Text-content form: <fin-streamer ...>123.45</fin-streamer>
        rf'<fin-streamer[^>]*data-symbol="{tkr}"[^>]*data-field='
        rf'"(?:regularMarketPrice|price)"[^>]*>([\d,]+\.?\d*)<',
        r'data-testid="qsp-price"[^>]*>([\d,]+\.?\d*)<',
        # Loose text: "123.45 +1.23 (+1.01%) At close" / "After hours"
        r'([\d,]+\.\d+)\s*(?:\+|-)\S+\s*\(\S+%\)\s*(?:At close|After hours)',
    ]
    for pattern in patterns:
        m = re.search(pattern, html, re.IGNORECASE)
        if m:
            try:
                price = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if price > 0:
                return price
    return None


def extract_expiration_timestamps(html: str) -> list[int]:
    """All expiration unix timestamps referenced by an options page.

    Yahoo has cycled through several renderings of the expiry picker
    (anchor links with ?date=, <option value=...>, data-value attrs —
    the current layout only materializes those after the date-select
    menu is clicked open, which `_load_page` does — and an embedded
    expirationDates JSON array), so match all of them and keep anything
    that decodes to a plausible expiry date (a week back through six
    years out). Sorted ascending. A stray non-expiry timestamp that
    sneaks through only costs a wasted page load — the fetcher verifies
    each page's OCC contract date before keeping its rows.
    """
    found: set[int] = set()
    for pattern in (
        r'options\?date=(\d{9,10})',
        r'"expirationDates"\s*:\s*\[([\d,\s]+)\]',
        r'<option[^>]*value="(\d{9,10})"',
        r'data-value="(\d{9,10})"',
    ):
        for m in re.finditer(pattern, html):
            for tok in m.group(1).split(","):
                tok = tok.strip()
                if tok.isdigit():
                    found.add(int(tok))
    now = datetime.now(timezone.utc)
    lo = (now - timedelta(days=7)).timestamp()
    hi = (now + timedelta(days=6 * 365)).timestamp()
    return sorted(ts for ts in found if lo <= ts <= hi)


def _side_of_table(df: pd.DataFrame) -> str | None:
    """'call' / 'put' from a chain table's Contract Name column; None
    when the column is missing or unrecognizable."""
    if "Contract Name" not in df.columns:
        return None
    for val in df["Contract Name"].astype(str).head(5):
        m = _OCC_RE.search(val.upper())
        if m:
            return "call" if m.group(2) == "C" else "put"
    return None


def chain_matches_expiration(chain: dict[str, pd.DataFrame | None],
                             exp_str: str) -> bool:
    """Whether a parsed page's contracts actually expire on `exp_str`.

    Yahoo silently serves the *nearest* expiration when ?date= names one
    it doesn't like (and a stray timestamp can slip through discovery),
    which would file rows under the wrong expiration/DTE. Checked via
    the yymmdd embedded in the OCC contract symbol; pages without a
    Contract Name column are trusted as-is.
    """
    want = exp_str.replace("-", "")[2:]  # "2026-09-18" → "260918"
    for df in (chain.get("calls"), chain.get("puts")):
        if df is None or "Contract Name" not in df.columns or df.empty:
            continue
        for val in df["Contract Name"].astype(str).head(5):
            m = _OCC_RE.search(val.upper())
            if m:
                return m.group(1) == want
    return True  # nothing to verify against — trust the page


def parse_chain_tables(html: str) -> dict[str, pd.DataFrame | None]:
    """Locate the calls/puts tables in an options page and clean them.

    Returns {'calls': df|None, 'puts': df|None} with numeric columns
    coerced (thousands separators stripped, '-' placeholders → NaN)
    and Implied Volatility as a ratio (0.28, not 28%). Rows missing
    Strike or IV are dropped.
    """
    import io

    chain: dict[str, pd.DataFrame | None] = {"calls": None, "puts": None}
    try:
        # Pinned to lxml (already a dependency) — on a table-less page the
        # default flavor chain falls through to html5lib, which isn't
        # installed and turns "no tables" into an ImportError.
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except ValueError:  # "No tables found"
        return chain

    detected: dict[str, list[pd.DataFrame]] = {"calls": [], "puts": []}
    positional: list[pd.DataFrame] = []
    for df in tables:
        df.columns = [str(c).strip() for c in df.columns]
        if not {"Strike", "Bid", "Implied Volatility"} <= set(df.columns):
            continue
        side = _side_of_table(df)
        if side == "call":
            detected["calls"].append(df)
        elif side == "put":
            detected["puts"].append(df)
        else:
            positional.append(df)
    # No Contract Name column (older/alternate layout): Yahoo renders
    # calls above puts, so assign leftovers to the empty slots in order.
    for df in positional:
        if not detected["calls"]:
            detected["calls"].append(df)
        elif not detected["puts"]:
            detected["puts"].append(df)
    for key, dfs in detected.items():
        if not dfs:
            continue
        # Some layouts split a side across tables (and a sticky-header
        # clone shows up as a duplicate) — concatenate rather than keep
        # only the first, then dedupe so OI can't double-count.
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if len(dfs) > 1:
            df = (df.drop_duplicates(subset=["Contract Name"])
                  if "Contract Name" in df.columns
                  else df.drop_duplicates())
        chain[key] = df

    for key, df in chain.items():
        if df is None:
            continue
        df = df.copy()
        for col in ("Strike", "Bid", "Ask", "Last Price",
                    "Volume", "Open Interest"):
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ""), errors="coerce")
        if "Implied Volatility" in df.columns:
            df["Implied Volatility"] = pd.to_numeric(
                df["Implied Volatility"].astype(str)
                  .str.replace("%", "").str.replace(",", "").str.strip(),
                errors="coerce") / 100.0
        df = df.dropna(subset=["Strike", "Implied Volatility"])
        chain[key] = df
    return chain


def _trade_age_days(raw) -> float:
    """Days since a 'Last Trade Date' cell like '6/27/2025 3:59 PM EDT'.

    The trailing timezone abbreviation is unparseable portably, so it
    is stripped and the comparison done in local naive time — hours of
    skew at worst, fine for a freshness filter measured in days. NaN
    when unusable.
    """
    try:
        s = re.sub(r"\s+[A-Z]{2,5}$", "", str(raw).strip())
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return float("nan")
        return max((pd.Timestamp.now() - ts).total_seconds(), 0.0) / 86400.0
    except (TypeError, ValueError):
        return float("nan")


def rows_from_tables(chain: dict[str, pd.DataFrame | None], *,
                     spot: float, exp_str: str, dte: int,
                     opt_type: str) -> list[dict]:
    """Canonical scanner rows from one expiration's cleaned tables."""
    from stocks_shared.black_scholes import (
        bs_delta, bs_gamma, bs_theta, bs_vega,
    )

    T = max(dte, 1) / 365.0
    sides = []
    if opt_type in ("both", "calls") and chain.get("calls") is not None:
        sides.append(("call", chain["calls"]))
    if opt_type in ("both", "puts") and chain.get("puts") is not None:
        sides.append(("put", chain["puts"]))

    rows: list[dict] = []
    for side, df in sides:
        # Header carries the zone in the current layout ("Last Trade
        # Date (EDT)"); older dumps put it in the cells instead.
        lt_col = next((c for c in df.columns
                       if str(c).startswith("Last Trade Date")), None)
        for _, row in df.iterrows():
            K = safe_float(row.get("Strike"))
            iv = safe_float(row.get("Implied Volatility"))
            if K <= 0 or iv <= 0:
                continue
            built = build_option_row(
                side=side, strike=K, expiration=exp_str, dte=dte,
                spot=spot,
                bid=safe_float(row.get("Bid")),
                ask=safe_float(row.get("Ask")),
                mid=0.0,
                last=safe_float(row.get("Last Price")),
                iv=iv,
                delta=bs_delta(spot, K, T, _RISK_FREE_RATE, iv, side),
                gamma=bs_gamma(spot, K, T, _RISK_FREE_RATE, iv),
                theta=bs_theta(spot, K, T, _RISK_FREE_RATE, iv, side),
                vega=bs_vega(spot, K, T, _RISK_FREE_RATE, iv),
                open_interest=safe_int(row.get("Open Interest")),
                volume=safe_int(row.get("Volume")),
                last_trade_days=_trade_age_days(
                    row.get(lt_col) if lt_col else None),
                # Overnight Yahoo zeroes bid/ask on the site well before
                # it drops the chain entirely; keep last-trade-priced
                # rows or after-hours scans collapse to nothing (or to
                # one side, which flips GEX).
                require_quote=False,
            )
            if built is not None:
                rows.append(built)
    return rows


# ── Browser pool ─────────────────────────────────────────────────────────

_tls = threading.local()
_pool_lock = threading.Lock()
_executor = None          # ThreadPoolExecutor, created on first fetch
_all_drivers: list = []   # every driver ever handed out, for atexit


def _create_driver(cfg: dict):
    from selenium import webdriver

    browser = str(cfg.get("browser", "chrome")).lower()
    binary = cfg.get("binary")  # explicit browser executable, e.g. for snap
    if browser == "firefox":
        from selenium.webdriver.firefox.options import Options
        opts = Options()
        if binary:
            opts.binary_location = str(binary)
        if cfg.get("headless", True):
            opts.add_argument("-headless")
        opts.set_preference("general.useragent.override", _UA)
        # Tall viewport: some page variants lazy-render the puts table
        # below the fold, and a table that never enters the viewport
        # never hydrates.
        opts.add_argument("--width=1920")
        opts.add_argument("--height=3000")
        driver = webdriver.Firefox(options=opts)
    else:
        from selenium.webdriver.chrome.options import Options
        opts = Options()
        if binary:
            opts.binary_location = str(binary)
        if cfg.get("headless", True):
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        # Tall viewport — see the Firefox note above.
        opts.add_argument("--window-size=1920,3000")
        opts.add_argument(f"user-agent={_UA}")
        driver = webdriver.Chrome(options=opts)
    with _pool_lock:
        _all_drivers.append(driver)
    return driver


def _get_driver(cfg: dict):
    if getattr(_tls, "driver", None) is None:
        _tls.driver = _create_driver(cfg)
    return _tls.driver


def _discard_driver():
    driver = getattr(_tls, "driver", None)
    _tls.driver = None
    if driver is not None:
        with _pool_lock:
            if driver in _all_drivers:
                _all_drivers.remove(driver)
        try:
            driver.quit()
        except Exception:
            pass


@atexit.register
def _shutdown():
    with _pool_lock:
        drivers, _all_drivers[:] = _all_drivers[:], []
    for d in drivers:
        try:
            d.quit()
        except Exception:
            pass


def _get_executor(cfg: dict):
    from concurrent.futures import ThreadPoolExecutor

    global _executor
    with _pool_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=int(cfg.get("max_workers", 4)),
                thread_name_prefix="yahoo-headless",
            )
        return _executor


def _dismiss_consent(driver) -> None:
    """Click through Yahoo's EU cookie-consent interstitial if shown."""
    from selenium.webdriver.common.by import By

    try:
        url = driver.current_url or ""
        if "consent" not in url and "guce" not in url:
            return
        btn = driver.find_element(
            By.CSS_SELECTOR, "button[name='agree'], button.accept-all")
        btn.click()
        time.sleep(1.5)
    except Exception:
        pass


def _open_expiry_menu(driver) -> None:
    """Click the date-select menu open so its expiry timestamps render.

    The current Yahoo layout keeps the expiration listbox out of the
    DOM until the picker button is clicked; one JS click on the base
    page materializes every expiry as a data-value attribute for
    `extract_expiration_timestamps`. Best-effort — the yfinance
    expirations fallback covers a failed click.
    """
    from selenium.webdriver.common.by import By

    try:
        btns = driver.find_elements(
            By.CSS_SELECTOR, "button[data-type='date']")
        if btns:
            driver.execute_script("arguments[0].click();", btns[0])
            time.sleep(1.5)
    except Exception:
        pass


def _load_page(url: str, cfg: dict, open_expiry_menu: bool = False) -> str:
    """Fetch one options page in a pool thread; return its HTML.

    Waits for a <table> to render, then `settle_seconds` more for the
    fin-streamer price fields to hydrate. A dead/crashed thread-local
    browser is replaced once before giving up.
    """
    from selenium.common.exceptions import (
        TimeoutException, WebDriverException,
    )
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    driver = _get_driver(cfg)
    try:
        driver.get(url)
    except WebDriverException:
        _discard_driver()
        driver = _get_driver(cfg)
        driver.get(url)
    _dismiss_consent(driver)
    try:
        WebDriverWait(driver, float(cfg.get("page_timeout", 25))).until(
            EC.presence_of_element_located((By.TAG_NAME, "table")))
    except TimeoutException:
        log.warning("No chain table rendered for %s", url)
        return driver.page_source or ""
    time.sleep(float(cfg.get("settle_seconds", 2.0)))
    # Sweep the viewport down the page and back so any lazy-rendered
    # content (the puts table sits below the calls table) hydrates even
    # if the tall window wasn't enough.
    try:
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.6)
        driver.execute_script("window.scrollTo(0, 0);")
    except Exception:
        pass
    if open_expiry_menu:
        _open_expiry_menu(driver)
    return driver.page_source or ""


# ── Public fetcher ───────────────────────────────────────────────────────


def fetch_chain_yahoo_headless(
    ticker: str,
    opt_type: str = "both",
    min_dte: int = 30,
    max_dte: int | None = 90,
    headless_config: dict | None = None,
) -> pd.DataFrame:
    """Scrape finance.yahoo.com's option chain pages for `ticker`.

    Same contract as `chain._fetch_chain_yahoo`: rows for every
    expiration with min_dte <= DTE <= max_dte, canonical 17-column
    schema, ValueError when the page yields nothing usable.

    headless_config defaults to `[yahoo_headless]` from config.toml.
    """
    from stocks_shared.yahoo import fetch_live_price, normalize_ticker

    if headless_config is None:
        from options_scanner.config import (
            get_yahoo_headless_config, load_config,
        )
        headless_config = get_yahoo_headless_config(load_config())
    cfg = headless_config

    ticker = normalize_ticker(ticker)
    base_url = f"https://finance.yahoo.com/quote/{ticker}/options"
    ex = _get_executor(cfg)

    # The bare options page carries the expiry picker, the spot price,
    # and the front expiration's tables. Loaded through the pool so
    # browsers only ever live in pool threads.
    base_html = ex.submit(_load_page, base_url, cfg, True).result()

    spot = extract_spot(base_html, ticker)
    if spot is None:
        try:
            spot = fetch_live_price(ticker)
        except Exception:
            spot = None
    if not spot:
        raise ValueError(
            f"Could not extract a price for {ticker} from the Yahoo "
            "options page — the page may be blocked or the layout changed.")

    today = date.today()
    wanted: list[tuple[int, str, int]] = []  # (unix_ts, "YYYY-MM-DD", dte)
    timestamps = extract_expiration_timestamps(base_html)
    if not timestamps:
        # Page rendered without a recognizable expiry picker — the
        # expirations *list* usually still works through the JSON API
        # even when its quotes are zeroed, so fall back to yfinance.
        try:
            import yfinance as yf
            for e in yf.Ticker(ticker).options:
                d = datetime.strptime(e, "%Y-%m-%d").date()
                ts = int(datetime(d.year, d.month, d.day,
                                  tzinfo=timezone.utc).timestamp())
                timestamps.append(ts)
        except Exception as exc:
            log.warning("yfinance expirations fallback failed: %s", exc)
    for ts in timestamps:
        d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        dte = (d - today).days
        if dte >= min_dte and (max_dte is None or dte <= max_dte):
            wanted.append((ts, d.isoformat(), dte))

    if not wanted:
        raise ValueError(
            f"No Yahoo expirations found for {ticker} with DTE "
            f"{min_dte}–{max_dte if max_dte is not None else '∞'}.")
    log.info("  %s: %d expirations with DTE %d–%s (headless)",
             ticker, len(wanted), min_dte,
             max_dte if max_dte is not None else "∞")

    # type=all / straddle=false pin the page to the two-table
    # calls-over-puts view — some layout variants otherwise remember a
    # one-sided or straddle view and render only part of the chain.
    def _exp_url(ts: int) -> str:
        return f"{base_url}?date={ts}&type=all&straddle=false"

    futures = {
        ex.submit(_load_page, _exp_url(ts), cfg): (ts, exp_str, dte)
        for ts, exp_str, dte in wanted
    }

    rows: list[dict] = []
    pages_with_tables = 0
    placeholder_pages = 0
    for fut, (ts, exp_str, dte) in futures.items():
        try:
            html = fut.result()
        except Exception as exc:
            log.warning("  Skipping %s %s: %s", ticker, exp_str, exc)
            continue
        if _NO_DATA_RE.search(html):
            placeholder_pages += 1
        chain = parse_chain_tables(html)
        if opt_type == "both" and (
                (chain["calls"] is None) != (chain["puts"] is None)):
            # Half a chain is a rendering hiccup (lazy table never
            # hydrated), not a market fact — reload the page once.
            missing = "puts" if chain["puts"] is None else "calls"
            log.warning("  %s %s: %s table missing — retrying page",
                        ticker, exp_str, missing)
            try:
                retry = parse_chain_tables(
                    ex.submit(_load_page, _exp_url(ts), cfg).result())
                if (retry["calls"] is not None
                        and retry["puts"] is not None):
                    chain = retry
            except Exception as exc:
                log.warning("  %s %s: retry failed: %s",
                            ticker, exp_str, exc)
        if chain["calls"] is not None or chain["puts"] is not None:
            pages_with_tables += 1
        if not chain_matches_expiration(chain, exp_str):
            log.warning("  %s %s: page served a different expiration "
                        "— skipping", ticker, exp_str)
            continue
        log.info("  %s %s: %d calls / %d puts parsed", ticker, exp_str,
                 0 if chain["calls"] is None else len(chain["calls"]),
                 0 if chain["puts"] is None else len(chain["puts"]))
        rows.extend(rows_from_tables(
            chain, spot=spot, exp_str=exp_str, dte=dte, opt_type=opt_type))

    if pages_with_tables == 0:
        raise ValueError(
            f"Yahoo rendered no option tables for {ticker} — the site "
            "may be blocking automated browsers right now, or the page "
            "layout changed.")
    # A two-sided request that came back entirely one-sided means every
    # page dropped the same table — degraded scrape, and quietly returning
    # it would poison anything sign-sensitive (GEX flips all-negative).
    # No optionable ticker has a genuinely one-sided book at this size.
    if opt_type == "both" and rows:
        n_calls = sum(1 for r in rows if r["type"] == "call")
        n_puts = len(rows) - n_calls
        if min(n_calls, n_puts) == 0 and max(n_calls, n_puts) >= 20:
            got, missing = (("calls", "puts") if n_calls
                            else ("puts", "calls"))
            raise ValueError(
                f"Yahoo returned only {got} for {ticker} ({len(rows)} "
                f"contracts, no {missing}) — Yahoo often drops one side "
                "first when its overnight chain data degrades, and "
                "one-sided data would be misleading (GEX flips sign). "
                "Rescan in a while.")
    if not rows:
        if placeholder_pages:
            raise ValueError(
                f"Yahoo's option chain data is dark for {ticker} right "
                f"now ({placeholder_pages} of {len(wanted)} expiration "
                "pages show 'There are no calls/puts') — its chain "
                "backend goes empty like this in the overnight hours. "
                "Rescan later.")
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(["expiration", "type", "strike"])
            .reset_index(drop=True))
