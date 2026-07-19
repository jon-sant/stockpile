"""Backtest engine: roll a fixed-delta short option every `dte_days`
across a historical window, using Black-Scholes + trailing realized vol
to reconstruct strikes/premiums (see package docstring for the
approximation caveat — no real historical option-chain data exists).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from stocks_shared.black_scholes import bs_price, strike_for_delta
from stocks_shared.yahoo import estimate_option_history, fetch_history


@dataclass(frozen=True)
class BacktestConfig:
    ticker: str
    opt_type: Literal["call", "put"]
    target_delta: float           # magnitude; sign is inferred from opt_type
    dte_days: int
    start_date: date
    end_date: date
    contracts: int = 1
    r: float = 0.045


@dataclass(frozen=True)
class BacktestTrade:
    open_date: date
    close_date: date
    strike: float
    entry_sigma: float
    premium_per_share: float
    terminal_spot: float
    pnl: float                    # $ total for `contracts`, this cycle
    ann_pct_realized: float
    daily_value: pd.DataFrame     # from estimate_option_history — feeds decay curves


@dataclass(frozen=True)
class BacktestResult:
    """`trades` empty (all metrics NaN/0) when there's no price history
    or not one full `dte_days` cycle fits in [start_date, end_date].
    """
    trades: list[BacktestTrade]
    win_rate: float
    avg_ann_pct: float
    max_drawdown: float
    sharpe: float
    sortino: float


_EMPTY_RESULT = BacktestResult(
    trades=[], win_rate=float("nan"), avg_ann_pct=float("nan"),
    max_drawdown=0.0, sharpe=float("nan"), sortino=float("nan"),
)


def _rolling_vol(price_history: pd.DataFrame, window: int = 30) -> pd.Series:
    log_ret = np.log(price_history["Close"] / price_history["Close"].shift(1))
    return log_ret.rolling(window, min_periods=5).std() * np.sqrt(252)


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Roll a `config.target_delta`-delta short `config.opt_type` on
    `config.ticker` from `start_date` to `end_date`, one cycle per
    `dte_days` window, closing each cycle at the real historical spot
    (not modeled) and rolling immediately into the next cycle.

    Each cycle: trailing 30-day realized vol at entry -> invert
    `strike_for_delta` for the target-delta strike -> price via
    `bs_price` -> reconstruct the daily value path via
    `estimate_option_history` (feeds PR11's decay curves) -> close at
    the real terminal spot from `fetch_history`.
    """
    price_history = fetch_history(
        config.ticker,
        start=(config.start_date - timedelta(days=60)).isoformat(),
        end=(config.end_date + timedelta(days=1)).isoformat(),
    )
    if price_history is None or price_history.empty:
        return _EMPTY_RESULT
    price_history = price_history.copy()
    if price_history.index.tz is not None:
        price_history.index = price_history.index.tz_convert(None)
    price_history.index = price_history.index.normalize()

    vol_series = _rolling_vol(price_history)
    # target_delta's sign follows bs_delta's own convention: calls 0..1,
    # puts -1..0 — the caller supplies a magnitude, we apply the sign.
    signed_target_delta = (abs(config.target_delta) if config.opt_type == "call"
                           else -abs(config.target_delta))

    trading_dates = price_history.index
    entry = pd.Timestamp(config.start_date)
    end_ts = pd.Timestamp(config.end_date)
    trades: list[BacktestTrade] = []

    while entry <= end_ts:
        future = trading_dates[trading_dates >= entry]
        if future.empty:
            break
        entry_ts = future[0]

        target_close = entry_ts + pd.Timedelta(days=config.dte_days)
        closeable = trading_dates[trading_dates >= target_close]
        if closeable.empty:
            # Not enough trailing history to close this cycle within the
            # fetched window — stop rather than fabricate a close date.
            break
        close_ts = closeable[0]

        S = float(price_history.loc[entry_ts, "Close"])
        raw_sigma = vol_series.loc[entry_ts] if entry_ts in vol_series.index else np.nan
        sigma = (float(raw_sigma) if pd.notna(raw_sigma) and raw_sigma > 0.01
                 else 0.3)
        T = config.dte_days / 365.0

        strike = strike_for_delta(S, T, config.r, sigma, config.opt_type,
                                  signed_target_delta)
        premium = bs_price(S, strike, T, config.r, sigma, config.opt_type)

        # estimate_option_history's expiration_str regex wants M/D/YYYY —
        # NOT ISO — and its intrinsic branch does an exact `== "Call"`
        # check bypassing bs_price's own case-insensitivity, so this
        # MUST be capitalized here.
        exp_str = close_ts.strftime("%m/%d/%Y")
        daily = estimate_option_history(
            price_history, config.opt_type.capitalize(), strike, exp_str,
            entry_ts, config.contracts, config.r,
        )
        if daily is None or daily.empty:
            entry = close_ts + pd.Timedelta(days=1)
            continue

        terminal_spot = float(price_history.loc[close_ts, "Close"])
        intrinsic = (max(0.0, terminal_spot - strike) if config.opt_type == "call"
                    else max(0.0, strike - terminal_spot))
        # Short option: keep the premium, pay back intrinsic at close.
        # European/cash-settled approximation — no early-assignment
        # modeling, consistent with the rest of this BS-based engine.
        pnl = (premium - intrinsic) * 100.0 * config.contracts

        capital = (S * 100.0 * config.contracts if config.opt_type == "call"
                  else strike * 100.0 * config.contracts)
        cycle_days = max(1, (close_ts - entry_ts).days)
        ann_pct = ((pnl / capital) * (365.0 / cycle_days) * 100.0
                  if capital > 0 else 0.0)

        trades.append(BacktestTrade(
            open_date=entry_ts.date(), close_date=close_ts.date(),
            strike=strike, entry_sigma=sigma, premium_per_share=premium,
            terminal_spot=terminal_spot, pnl=pnl, ann_pct_realized=ann_pct,
            daily_value=daily,
        ))
        entry = close_ts + pd.Timedelta(days=1)

    if not trades:
        return _EMPTY_RESULT

    pnls = np.array([t.pnl for t in trades], dtype=float)
    win_rate = float(np.mean(pnls > 0))
    avg_ann_pct = float(np.mean([t.ann_pct_realized for t in trades]))

    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    max_drawdown = float(np.min(cum - running_max))

    cycles_per_year = 365.0 / max(1, config.dte_days)
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    sharpe = (mean_pnl / std_pnl * np.sqrt(cycles_per_year)
             if std_pnl > 1e-9 else float("nan"))

    # Sortino: same population-downside-deviation-from-zero shape as
    # montecarlo/metrics.py's summarize() (deviation from MAR=0, not
    # from the downside mean), annualized here since this is a
    # multi-cycle strategy series rather than a single-position ratio.
    downside = pnls[pnls < 0]
    downside_dev = (float(np.sqrt(np.mean(downside * downside)))
                    if downside.size > 0 else 0.0)
    if downside_dev > 1e-9:
        sortino = mean_pnl / downside_dev * np.sqrt(cycles_per_year)
    else:
        sortino = float("inf") if mean_pnl > 0 else 0.0

    return BacktestResult(
        trades=trades, win_rate=win_rate, avg_ann_pct=avg_ann_pct,
        max_drawdown=max_drawdown, sharpe=sharpe, sortino=sortino,
    )
