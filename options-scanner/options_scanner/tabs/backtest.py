"""Backtest tab: "always sell an X-delta call/put on this ticker every
N days" — a historical strategy-return simulator.

**Approximation, not a real historical-IV-surface backtest** — see
`options_scanner.backtest`'s package docstring and the on-screen
disclaimer below. No historical option-chain data source exists in this
repo; strikes/premiums are reconstructed via Black-Scholes + trailing
realized volatility, not real historical option quotes.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from options_scanner.backtest import BacktestConfig, run_backtest
from options_scanner.display.decay_curves import render_decay_curves
from options_scanner.format import fmt_strike
from options_scanner.ui_theme import metric_card


def tab_backtest() -> None:
    st.warning(
        "**Modeled, not real historical option prices.** Strikes and "
        "premiums are reconstructed from Black-Scholes + trailing "
        "realized volatility (no historical option-chain data source "
        "exists in this repo) — this can't see skew, term structure, "
        "or demand effects real historical prices would reflect. "
        "Treat results as directional, not precise."
    )

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ticker = st.text_input("Ticker", "SPY", key="bt_ticker").strip().upper()
        with c2:
            opt_type_label = st.radio("Side", ["Puts (sell)", "Calls (sell)"],
                                      key="bt_opt_type", horizontal=True)
            opt_type = "put" if opt_type_label.startswith("Put") else "call"
        with c3:
            target_delta = st.number_input(
                "Target |Delta|", value=0.30, min_value=0.05, max_value=0.50,
                step=0.05, key="bt_delta",
                help="Roll the strike whose Black-Scholes delta is closest "
                     "to this magnitude each cycle.",
            )
        with c4:
            dte_days = st.number_input(
                "DTE per cycle", value=30, min_value=5, max_value=90,
                step=5, key="bt_dte",
            )

        c5, c6, c7 = st.columns(3)
        with c5:
            years_back = st.number_input(
                "Years of history", value=1, min_value=1, max_value=5,
                step=1, key="bt_years",
            )
        with c6:
            contracts = st.number_input(
                "Contracts", value=1, min_value=1, max_value=100,
                key="bt_contracts",
            )
        with c7:
            st.markdown("<div style='height:1.72rem'></div>", unsafe_allow_html=True)
            run_bt = st.button("Run Backtest", type="primary",
                              width="stretch", key="bt_run")

    if run_bt:
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * int(years_back))
        config = BacktestConfig(
            ticker=ticker, opt_type=opt_type, target_delta=float(target_delta),
            dte_days=int(dte_days), start_date=start_date, end_date=end_date,
            contracts=int(contracts),
        )
        with st.spinner(f"Backtesting {ticker}…"):
            result = run_backtest(config)
        st.session_state["bt_result"] = result
        st.session_state["bt_result_label"] = (
            f"{ticker} {opt_type_label} · {target_delta:.2f}Δ · "
            f"{dte_days}DTE · {years_back}y")

    result = st.session_state.get("bt_result")
    if result is None:
        return

    if not result.trades:
        st.info(
            "No completed cycles — the ticker may have too little "
            "history, or the DTE/date-range combination doesn't fit "
            "even one full cycle. Try a shorter DTE or more years."
        )
        return

    st.subheader(st.session_state.get("bt_result_label", "Results"))
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("WIN RATE", f"{result.win_rate * 100:.0f}%",
                    delta_sign="pos" if result.win_rate >= 0.5 else "neg")
    with m2:
        metric_card("AVG ANN%", f"{result.avg_ann_pct:+.1f}%",
                    delta_sign="pos" if result.avg_ann_pct >= 0 else "neg")
    with m3:
        metric_card("MAX DRAWDOWN", f"${result.max_drawdown:,.0f}")
    with m4:
        _sharpe = ("—" if result.sharpe != result.sharpe else f"{result.sharpe:.2f}")
        metric_card("SHARPE", _sharpe)
    with m5:
        _sortino = ("∞" if result.sortino == float("inf")
                    else "—" if result.sortino != result.sortino
                    else f"{result.sortino:.2f}")
        metric_card("SORTINO", _sortino)

    st.markdown("**Trade log**")
    trades_df = pd.DataFrame([{
        "Open": t.open_date, "Close": t.close_date,
        "Strike": fmt_strike(t.strike), "Entry σ": f"{t.entry_sigma * 100:.1f}%",
        "Premium": t.premium_per_share, "Terminal Spot": t.terminal_spot,
        "P&L": t.pnl, "Ann%": t.ann_pct_realized,
    } for t in result.trades])
    st.dataframe(
        trades_df, hide_index=True, width="stretch",
        column_config={
            "Premium": st.column_config.NumberColumn("Premium", format="$%.2f"),
            "Terminal Spot": st.column_config.NumberColumn(
                "Terminal Spot", format="$%.2f"),
            "P&L": st.column_config.NumberColumn("P&L", format="$%+.2f"),
            "Ann%": st.column_config.NumberColumn("Ann%", format="%+.1f%%"),
        },
    )

    st.markdown("**Premium decay by entry IV regime**")
    render_decay_curves(result)
