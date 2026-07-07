"""Fetch and enrich option chain data."""

import logging
from datetime import date, datetime

import pandas as pd

from options_scanner.chain_common import build_option_row, safe_float, safe_int
# Historical aliases — the implementation lives in
# stocks_shared.black_scholes; tests and spreads.py import these names.
from stocks_shared.black_scholes import (
    bs_delta as _bs_delta,
    bs_gamma as _bs_gamma,
    bs_theta as _bs_theta,
    bs_vega as _bs_vega,
    norm_cdf as _norm_cdf,
    norm_pdf as _norm_pdf,
)
from stocks_shared.yahoo import (
    RateLimitError, fetch_live_price, is_rate_limit_error, normalize_ticker,
)

log = logging.getLogger(__name__)

_RISK_FREE_RATE = 0.045


def _trade_age_days(last_trade) -> float:
    """Days since a yfinance lastTradeDate timestamp; NaN if unusable."""
    try:
        if last_trade is None or pd.isna(last_trade):
            return float("nan")
        ts = pd.Timestamp(last_trade)
        now = pd.Timestamp.now(tz=ts.tz) if ts.tz is not None \
            else pd.Timestamp.now()
        return float((now - ts).total_seconds()) / 86400.0
    except (TypeError, ValueError):
        return float("nan")


def _fetch_chain_yahoo(ticker: str, opt_type: str = "both",
                       min_dte: int = 30,
                       max_dte: int | None = 90) -> pd.DataFrame:
    import yfinance as yf

    ticker = normalize_ticker(ticker)
    spot = fetch_live_price(ticker, raise_on_rate_limit=True)
    if not spot:
        raise ValueError(f"Could not fetch live price for {ticker}")

    t = yf.Ticker(ticker)
    today = date.today()

    expirations = []
    for e in t.options:
        dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if dte >= min_dte and (max_dte is None or dte <= max_dte):
            expirations.append(e)
    log.info(
        "  %d expirations with DTE %s",
        len(expirations),
        f"{min_dte}–{max_dte}" if max_dte else f">= {min_dte}",
    )

    rows = []
    raw_seen = 0   # contracts Yahoo sent us, pre-filter
    zeroed = 0     # contracts with no quote at all (bid=ask=0)
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        T = dte / 365.0

        try:
            chain = t.option_chain(exp_str)
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise RateLimitError(
                    f"Yahoo rate limit hit fetching {ticker} {exp_str}"
                ) from exc
            log.warning("  Skipping %s: %s", exp_str, exc)
            continue

        sides = []
        if opt_type in ("both", "calls"):
            sides.append(("call", chain.calls))
        if opt_type in ("both", "puts"):
            sides.append(("put", chain.puts))

        for side, df in sides:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                K = safe_float(row.get("strike"))
                if K <= 0:
                    continue  # math.log(spot / K) below blows up at K=0
                iv = safe_float(row.get("impliedVolatility"))
                raw_seen += 1
                if (safe_float(row.get("bid")) <= 0
                        and safe_float(row.get("ask")) <= 0):
                    zeroed += 1
                delta = _bs_delta(spot, K, T, _RISK_FREE_RATE, iv, side)
                gamma = _bs_gamma(spot, K, T, _RISK_FREE_RATE, iv)
                theta = _bs_theta(spot, K, T, _RISK_FREE_RATE, iv, side)
                vega = _bs_vega(spot, K, T, _RISK_FREE_RATE, iv)
                built = build_option_row(
                    last_trade_days=_trade_age_days(row.get("lastTradeDate")),
                    side=side, strike=K, expiration=exp_str,
                    dte=dte, spot=spot,
                    bid=safe_float(row.get("bid")),
                    ask=safe_float(row.get("ask")),
                    mid=0.0,
                    last=safe_float(row.get("lastPrice")),
                    iv=iv, delta=delta, gamma=gamma,
                    theta=theta, vega=vega,
                    open_interest=safe_int(row.get("openInterest")),
                    volume=safe_int(row.get("volume")),
                )
                if built is not None:
                    rows.append(built)

    # Yahoo's soft throttle doesn't 429 — it serves HTTP-200 chains with
    # every bid/ask zeroed (many IVs reduced to the 0.00001 placeholder)
    # and the strike list truncated. Individual illiquid contracts can
    # have no quote, but an entire chain of them is a throttled response
    # (or a session with no quote data at all), not usable data — either
    # way the scan should wait and retry rather than render nothing.
    if raw_seen >= 10 and zeroed / raw_seen >= 0.85:
        raise RateLimitError(
            f"Yahoo returned a degraded option chain for {ticker} "
            f"({zeroed} of {raw_seen} contracts have no quote — soft "
            "rate limit)")

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_chain(ticker: str, opt_type: str = "both",
                min_dte: int = 30, max_dte: int | None = 90,
                provider: str = "yahoo",
                schwab_config: dict | None = None,
                moomoo_config: dict | None = None) -> pd.DataFrame:
    """Return enriched DataFrame of options with min_dte <= DTE <= max_dte.

    opt_type: "both", "calls", or "puts"
    max_dte:  upper DTE limit; None = no limit
    provider: "yahoo" (default), "yahoo-headless", "schwab", or "moomoo"
    schwab_config: dict with app_key, app_secret, callback_url, token_file
    moomoo_config: dict with host, port (OpenD gateway address)

    "yahoo-headless" scrapes the finance.yahoo.com chain pages with a
    Selenium browser (works after hours when the JSON API serves zeroed
    quotes); it reads its own tuning from [yahoo_headless] in config.toml.
    """
    if provider == "yahoo-headless":
        from options_scanner.yahoo_headless import fetch_chain_yahoo_headless
        return fetch_chain_yahoo_headless(ticker, opt_type, min_dte, max_dte)
    if provider == "schwab":
        from options_scanner.schwab_chain import fetch_chain_schwab
        return fetch_chain_schwab(ticker, opt_type, min_dte, max_dte,
                                  schwab_config)
    if provider == "moomoo":
        from options_scanner.moomoo_chain import fetch_chain_moomoo
        return fetch_chain_moomoo(ticker, opt_type, min_dte, max_dte,
                                  moomoo_config)
    return _fetch_chain_yahoo(ticker, opt_type, min_dte, max_dte)
