"""Cross-ETF capital allocation for the Watchlist/Portfolio basket.

Ranks sell-side candidates across a scanned basket by a Kelly-inspired
reward-to-tail-risk edge, then greedily waterfills a user-supplied
capital budget across the highest-edge candidates, sizing each position
by how much collateral it needs (cash-secured-put or covered-call
convention) — "which of everything I just scanned deserves my capital,
and how much of it."

Deliberately reuses existing building blocks rather than re-deriving
them: `mc_ui.position_from_chain_row` + `montecarlo.run_simulation` for
the probability/tail-risk edge estimate (no new backtest data needed),
and the same collateral convention `trade_actions.PutSellOrder` already
uses (`strike * 100` per CSP contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from options_scanner.mc_ui import position_from_chain_row
from options_scanner.montecarlo import SimulationConfig, run_simulation

# Basket-wide allocation runs one MC simulation per candidate — lighter
# than the single-position MC Analyze panel's 10k-path default so a
# multi-ticker basket stays responsive. Users who want a precise read on
# one candidate can still open it in the MC Analyze panel directly.
_BASKET_N_PATHS = 2_000


@dataclass(frozen=True)
class AllocationCandidate:
    ticker: str
    strike: float
    expiration: str
    opt_type: Literal["call", "put"]
    premium: float                    # mid, per share
    collateral_per_contract: float    # cash/shares required per contract
    prob_profit: float                # P(profit), from montecarlo.metrics.summarize
    expected_pnl: float                # $ per contract, from MC
    cvar_5pct: float                   # $ per contract (<=0), from MC
    composite_score: float | None = None  # optional tiebreak (e.g. composite_v2)


def edge_score(cand: AllocationCandidate) -> float:
    """Kelly-inspired reward-to-tail-risk ratio: expected P&L per dollar
    of worst-5%-tail risk. Not literal binary Kelly — a cash-secured
    put's payoff isn't a binary win/lose bet — but the same spirit: size
    by reward relative to downside, not reward alone. Floored at 0 (a
    negative-expected-value candidate never gets capital)."""
    downside = abs(cand.cvar_5pct) if cand.cvar_5pct < 0 else 1.0
    return max(0.0, cand.expected_pnl / downside)


def candidates_from_board(
    board: pd.DataFrame,
    n_paths: int = _BASKET_N_PATHS,
    composite_col: str = "signal_score",
) -> list[AllocationCandidate]:
    """Build one `AllocationCandidate` per row of a leaderboard-shaped
    DataFrame (as returned by `display.leaderboard.build_leaderboard`),
    running one lightweight MC simulation per row for the edge estimate.

    Expects `ticker`, `type`, `strike`, `expiration`, `mid`, `iv`, `spot`
    columns (all present on `build_leaderboard`'s output). Rows a
    position can't be built/simulated for (bad data, non-future
    expiration, no positive IV) are silently skipped — this is a
    best-effort ranking aid, not a hard dependency for the rest of the
    scan to function.
    """
    out: list[AllocationCandidate] = []
    for _, r in board.iterrows():
        opt_type: Literal["call", "put"] = (
            "put" if str(r.get("type", "call")).lower().startswith("p") else "call")
        try:
            spot = float(r["spot"])
            position = position_from_chain_row(
                underlying=str(r["ticker"]), spot=spot,
                row={
                    "Strike": r["strike"], "Expiration": r["expiration"],
                    "Mid": r.get("mid", r.get("ask", 0.0)),
                    "IV%": float(r.get("iv", 0.0) or 0.0) * 100.0,
                },
                side="short", opt_type=opt_type, qty=1, risk_free_rate=0.045,
            )
            result = run_simulation(position, SimulationConfig(n_paths=n_paths))
        except Exception:
            continue

        strike = float(r["strike"])
        collateral = strike * 100.0 if opt_type == "put" else spot * 100.0
        out.append(AllocationCandidate(
            ticker=str(r["ticker"]), strike=strike,
            expiration=str(r["expiration"]), opt_type=opt_type,
            premium=float(r.get("mid", 0.0) or 0.0),
            collateral_per_contract=collateral,
            prob_profit=float(result.metrics.get("prob_profit", float("nan"))),
            expected_pnl=float(result.metrics.get("expected_pnl", float("nan"))),
            cvar_5pct=float(result.metrics.get("cvar_5pct", float("nan"))),
            composite_score=(float(r[composite_col])
                             if composite_col in r and pd.notna(r[composite_col])
                             else None),
        ))
    return out


_RESULT_COLS = ["ticker", "strike", "expiration", "opt_type", "contracts",
                "collateral", "edge_score", "expected_pnl", "prob_profit"]


def allocate_capital(
    candidates: list[AllocationCandidate],
    budget: float,
    max_pct_per_position: float = 0.25,
    corr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Greedy waterfill: repeatedly pick the remaining candidate with the
    highest (diversification-adjusted) edge score, size it by how many
    contracts the smaller of (remaining budget, max_pct_per_position *
    budget) affords at its collateral requirement (floor division, same
    convention as `trade_actions.puts_affordable`), subtract that
    dollar amount from the remaining budget, and repeat until the pool
    is exhausted or no remaining candidate can afford even one contract.

    `corr` (added for PR9's diversification penalty — accepted now so
    that PR9 only changes behavior, not this function's signature) is a
    ticker x ticker correlation matrix; when supplied, each candidate's
    edge score is scaled down by how correlated its ticker is with
    already-selected tickers in this same allocation run (recomputed
    every iteration, since it depends on what's already been picked).

    Returns a DataFrame with columns: ticker, strike, expiration,
    opt_type, contracts, collateral, edge_score, expected_pnl,
    prob_profit — empty (right columns) when nothing qualifies.
    """
    if not candidates or budget <= 0:
        return pd.DataFrame(columns=_RESULT_COLS)

    pool = list(candidates)
    selected_tickers: list[str] = []
    remaining = budget
    rows: list[dict] = []

    while pool and remaining > 1e-9:
        scored = []
        for c in pool:
            base = edge_score(c)
            penalty = 1.0
            if corr is not None:
                from options_scanner.compute.correlation import diversification_penalty
                penalty = diversification_penalty(c.ticker, selected_tickers, corr)
            scored.append((base * penalty, c))
        scored.sort(key=lambda t: (t[0], t[1].composite_score or 0.0), reverse=True)
        best_score, best = scored[0]
        pool.remove(best)

        if best_score <= 0.0 or best.collateral_per_contract <= 0:
            continue

        cap = min(remaining, max_pct_per_position * budget)
        qty = int(cap // best.collateral_per_contract)
        if qty <= 0:
            continue

        dollars = qty * best.collateral_per_contract
        remaining -= dollars
        selected_tickers.append(best.ticker)
        rows.append({
            "ticker": best.ticker, "strike": best.strike,
            "expiration": best.expiration, "opt_type": best.opt_type,
            "contracts": qty, "collateral": round(dollars, 2),
            "edge_score": round(best_score, 4),
            "expected_pnl": round(best.expected_pnl * qty, 2),
            "prob_profit": round(best.prob_profit, 4),
        })

    return pd.DataFrame(rows, columns=_RESULT_COLS)
