"""Cross-ETF correlation awareness for the capital allocator.

Answers "am I about to put a lot of capital into names that all move
together?" — a pairwise correlation matrix of daily returns, plus a
diversification penalty the allocator applies to each candidate's edge
score based on what's already been picked in the same allocation run.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from stocks_shared.yahoo import fetch_history

_MIN_OVERLAP_DAYS = 20  # minimum paired observations before a pair's corr is trusted


def pairwise_log_return_corr(tickers: list[str],
                             lookback_days: int = 90) -> pd.DataFrame:
    """Correlation matrix of daily log returns across `tickers`.

    Square DataFrame indexed/columned by ticker (deduped, order
    preserved). NaN for any pair with fewer than `_MIN_OVERLAP_DAYS`
    overlapping return observations (thin/newly-listed history, or a
    fetch failure for one side) — an unknown correlation should never
    silently read as "uncorrelated" (0.0), since a diversification
    penalty built on that would wrongly clear a genuinely-correlated
    pair.
    """
    unique = list(dict.fromkeys(tickers))
    if not unique:
        return pd.DataFrame()

    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    closes: dict[str, pd.Series] = {}
    for t in unique:
        try:
            hist = fetch_history(t, start=start)
        except Exception:
            continue
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        closes[t] = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

    corr = pd.DataFrame(index=unique, columns=unique, dtype=float)
    for a in unique:
        corr.loc[a, a] = 1.0 if a in closes else np.nan
        for b in unique:
            if a >= b:  # fill upper triangle then mirror, skip diagonal (set above)
                continue
            if a not in closes or b not in closes:
                corr.loc[a, b] = corr.loc[b, a] = np.nan
                continue
            aligned = pd.concat([closes[a], closes[b]], axis=1, join="inner")
            if len(aligned) < _MIN_OVERLAP_DAYS:
                corr.loc[a, b] = corr.loc[b, a] = np.nan
                continue
            c = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            corr.loc[a, b] = corr.loc[b, a] = c
    return corr


def diversification_penalty(ticker: str, already_selected: list[str],
                             corr: pd.DataFrame, threshold: float = 0.7,
                             penalty_per_corr: float = 0.5) -> float:
    """Multiplicative penalty in (0, 1]; 1.0 = no penalty.

    For each already-selected ticker whose |correlation| with `ticker`
    exceeds `threshold`, scales down proportional to how far past the
    threshold the correlation sits (at threshold: no penalty; at |corr|
    = 1.0: scaled down by the full `penalty_per_corr`). Multiple
    correlated picks compound multiplicatively. Unknown correlation
    (NaN — thin history, or `ticker`/`other` missing from `corr`) is
    treated as no penalty for that pair, not as maximally correlated —
    it's a "can't tell," not a red flag.
    """
    penalty = 1.0
    for other in already_selected:
        if other == ticker:
            continue
        if ticker not in corr.index or other not in corr.columns:
            continue
        c = corr.loc[ticker, other]
        if pd.isna(c) or abs(c) <= threshold:
            continue
        over = (abs(c) - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
        penalty *= max(0.0, 1.0 - penalty_per_corr * over)
    return penalty
