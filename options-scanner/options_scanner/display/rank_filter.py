"""Shared "what actually gets displayed" ranking logic.

Single source of truth for the filter -> sort -> head(top_n) chain that
decides which rows show up in a ranked table for one option type. Used
by `display/scan_results.py::show_scan_results()` for the real render,
and by the blocking-scan priority step (tabs/single.py, tabs/
portfolio.py) to know exactly which rows to compute Monte Carlo for
before rendering — both need the identical answer, or the block would
either wait on rows nobody sees or skip rows that do get shown.

This exact duplication mistake already happened once in this codebase:
tabs/single.py's Monte Carlo Trade Analyzer dropdown had its own stale
hand-rolled copy of this logic that unconditionally sorted by
`iv_excess` (wrong — should prefer `signal_score` when present) and
omitted the min_ivpp/min_ann/percentile filters entirely. That call
site now uses this function too.
"""

from __future__ import annotations

import pandas as pd


def filter_sort_top_n(df: pd.DataFrame, opt_type: str, buy: bool,
                      min_oi: int, min_vol: int, top_n: int,
                      min_ivpp: float | None = None,
                      min_ann: float | None = None,
                      min_percentile: float | None = None,
                      min_ann_delta_percentile: float | None = None,
                      ) -> pd.DataFrame:
    """The exact rows shown for one `opt_type` ("call"/"put") slice of
    `df`, in final display order — everything that decides "what's in
    the table" happens here: sort by `signal_score` (falling back to
    `iv_excess`), ascending when buying / descending when selling, OI/
    Vol floors, optional IV+pp/Ann%/percentile floors, then the top-N
    cut. Does not mutate `df` and does not add `_rank` — that's a
    display-only concern the caller adds after, since the priority-key
    extraction callers don't need it.
    """
    iv_asc = buy
    sort_col = "signal_score" if "signal_score" in df.columns else "iv_excess"
    sub = (
        df[df["type"] == opt_type]
        .sort_values([sort_col, "open_interest"], ascending=[iv_asc, False])
    )
    sub = sub[(sub["open_interest"] >= min_oi) & (sub["volume"] >= min_vol)]
    if min_ivpp is not None:
        sub = sub[(sub["iv_excess"] * 100) >= min_ivpp]
    if min_ann is not None:
        sub = sub[sub["ann_yield_pct"] >= min_ann]
    if min_percentile is not None and "iv_percentile" in sub.columns:
        sub = sub[sub["iv_percentile"] >= min_percentile]
    if min_ann_delta_percentile is not None and "ann_delta_percentile" in sub.columns:
        sub = sub[sub["ann_delta_percentile"] >= min_ann_delta_percentile]
    return sub.head(top_n)
