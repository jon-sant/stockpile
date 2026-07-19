"""Monte Carlo metric columns shared by scan_results.py and leaderboard.py.

Both tables already know buy-vs-sell context (a `buy` bool) and build
their `cols`/`col_cfg` dicts independently (no shared table-building
helper — matching the existing duplication between those two files).
This module centralizes just the MC value-formatting/column-spec logic
so the two callers can't drift on formatting or column labels, while
leaving the actual `cols.update(...)`/`col_cfg.update(...)` glue in each
file, same as every other column group there.

Values come from `iv_history`'s `mc_*` columns (see `fetch.py`'s
`_enrich()`), attached to the chain df alongside `mc_status`. A row
whose `mc_status` isn't `'done'` yet (no worker pass has reached it)
renders as the literal string "TBD" — there's no existing pending-cell
convention in this codebase, so these columns render via TextColumn
with pre-formatted strings rather than NumberColumn.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

TBD = "TBD"


def _side_col(base: str, buy: bool) -> str:
    return f"{base}_buy" if buy else f"{base}_sell"


def _fmt(sub: pd.DataFrame, col: str, fmt: str) -> list[str]:
    """Format one mc_* column as display strings, "TBD" wherever
    mc_status isn't 'done' or the value itself is NaN (e.g. a row that
    errored — mc_status='error', value never got set)."""
    status = sub["mc_status"] if "mc_status" in sub.columns else pd.Series(
        [None] * len(sub), index=sub.index)
    values = sub[col] if col in sub.columns else pd.Series(
        [float("nan")] * len(sub), index=sub.index)
    out = []
    for s, v in zip(status, values):
        if s != "done" or pd.isna(v):
            out.append(TBD)
        elif v == float("inf"):
            out.append("+∞")
        elif v == float("-inf"):
            out.append("-∞")
        else:
            out.append(fmt.format(v))
    return out


def exp_pnl_column(sub: pd.DataFrame, buy: bool) -> list[str]:
    """Expected P&L — always-visible, context-appropriate (buy/sell) side."""
    return _fmt(sub, _side_col("mc_expected_pnl", buy), "{:+,.0f}")


def exp_pnl_help() -> str:
    return ("Monte Carlo expected P&L at expiration (mean across simulated "
            "paths), for the side you're currently scanning (buy/sell). "
            "\"TBD\" until the background worker computes it — see the "
            "console log for progress.")


# label, formatted-value fn, side-invariant?, help text
def _prob_profit(sub, buy): return _fmt(sub, _side_col("mc_prob_profit", buy), "{:.0%}")
def _fair_value(sub, buy): return _fmt(sub, "mc_fair_value", "${:,.0f}")
def _edge_vs_market(sub, buy): return _fmt(sub, _side_col("mc_edge_vs_market", buy), "{:+,.0f}")
def _breakeven(sub, buy): return _fmt(sub, "mc_breakeven_move_pct", "{:+.1f}%")
def _cvar(sub, buy): return _fmt(sub, _side_col("mc_cvar5", buy), "{:,.0f}")
def _var(sub, buy): return _fmt(sub, _side_col("mc_var5", buy), "{:,.0f}")
def _sortino(sub, buy): return _fmt(sub, _side_col("mc_sortino", buy), "{:.2f}")

_ColumnFn = Callable[[pd.DataFrame, bool], list]
MC_EXTRA_COLUMNS: tuple[tuple[str, _ColumnFn, str], ...] = (
    ("P(profit)", _prob_profit,
     "Monte Carlo probability of a profitable outcome at expiration, for "
     "the side you're currently scanning."),
    ("MC Fair Value", _fair_value,
     "Monte Carlo model value of this contract (discounted mean payoff) — "
     "same number for either side, compare to Mid."),
    ("Premium vs Model", _edge_vs_market,
     "Market premium vs. the Monte Carlo fair value, for the side you're "
     "currently scanning. Positive = you got a better-than-model price."),
    ("Breakeven Move", _breakeven,
     "% move in the underlying needed to break even at expiration — same "
     "either side (same threshold spot for a single-leg long or short)."),
    ("CVaR", _cvar,
     "Monte Carlo CVaR (worst 5%) — average P&L of the worst 5% of "
     "simulated outcomes, for the side you're currently scanning."),
    ("VaR", _var,
     "Monte Carlo VaR (5%) — the loss threshold exceeded in 5% of "
     "simulated outcomes, for the side you're currently scanning."),
    ("Sortino", _sortino,
     "Monte Carlo Sortino ratio (downside-only risk-adjusted return), for "
     "the side you're currently scanning."),
)


def mc_extra_columns(sub: pd.DataFrame, buy: bool) -> dict[str, list[str]]:
    """The 7 checkbox-gated MC columns (P(profit) through Sortino),
    ready to `cols.update(...)`."""
    return {label: fn(sub, buy) for label, fn, _ in MC_EXTRA_COLUMNS}


def mc_extra_help() -> dict[str, str]:
    return {label: help_text for label, _, help_text in MC_EXTRA_COLUMNS}
