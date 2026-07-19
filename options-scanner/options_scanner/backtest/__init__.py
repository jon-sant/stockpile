"""Historical strategy-return backtest — "always sell an X-delta
call/put on this ticker every N days, for the last Y years."

**Approximation, not a real historical-IV-surface backtest.** No
historical option-chain data source exists anywhere in this repo (no
Polygon integration, nothing paid) — strike and premium are
reconstructed via Black-Scholes + trailing realized volatility from
`shared/stocks_shared/yahoo.py`'s `estimate_option_history`, not real
historical option quotes. Real historical prices reflect skew, term
structure, and demand effects this can't see. Treat results as
directional, not precise — see `BacktestResult`'s docstring and the
disclaimer rendered on the Backtest tab.
"""

from .engine import BacktestConfig, BacktestResult, BacktestTrade, run_backtest

__all__ = ["BacktestConfig", "BacktestResult", "BacktestTrade", "run_backtest"]
