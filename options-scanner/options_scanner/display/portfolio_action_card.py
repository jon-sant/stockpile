"""Recommended Action card for the Portfolio tab.

For each position scanned, picks the rank-1 IV-rich option and renders it
as an explicit instruction card above the candidates table:

  ROLL                — existing short call/put: BTC + STO steps with net
                        credit and new breakeven.
  SELL TO OPEN        — covered call (≥ 100 shares) or cash-secured put
                        (1-contract reference unit).
  Not actionable      — calls when < 100 shares held; shows top pick for
                        reference only (amber accent).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from options_scanner.format import fmt_strike
from options_scanner.display.outlook_card import OUTLOOK_TONE_HEX


def render_portfolio_action_card(
    ticker: str,
    df_filt: pd.DataFrame,
    spot: float,
    shares: int,
    covered: bool,
    roll_close: float | None,
    open_options: list[dict],
    min_oi: int,
    min_vol: int,
    opt_type: str = "calls",
    buy: bool = False,
    min_ivpp: float | None = None,
    min_ann: float | None = None,
) -> None:
    """Translate the top candidate into an explicit buy/sell action.

    Sell mode (default): picks the rank-1 IV-rich contract.
      covered=True + roll_close supplied → ROLL: BTC existing, STO top pick.
      Otherwise → STO the top pick (covered call for calls; CSP for puts).

    Buy mode: picks the rank-1 IV-cheap contract and frames it as a long
    option (BUY TO OPEN) — debit, max loss, breakeven, directional delta.
    Roll/cover semantics don't apply to a fresh long, so they're skipped.

    opt_type is "calls" or "puts"; never "both" — the caller splits the df
    and calls this function once per side when opt_type is "both".
    """
    type_filter = "put" if opt_type == "puts" else "call"
    eligible = df_filt[
        (df_filt["type"] == type_filter)
        & (df_filt["open_interest"] >= min_oi)
        & (df_filt["volume"] >= min_vol)
    ]
    if min_ivpp is not None:
        eligible = eligible[(eligible["iv_excess"] * 100) >= min_ivpp]
    if min_ann is not None:
        eligible = eligible[eligible["ann_yield_pct"] >= min_ann]
    if eligible.empty:
        return
    sort_col = "signal_score" if "signal_score" in eligible.columns else "iv_excess"
    # Sell wants the richest signal first (descending); buy wants the
    # cheapest (most-negative IV+pp) first (ascending) — same convention as
    # the chain table and leaderboard.
    pick = (
        eligible.sort_values([sort_col, "open_interest"],
                             ascending=[buy, False])
        .iloc[0]
    )
    strike = float(pick["strike"])
    expiry = pd.to_datetime(pick["expiration"]).strftime("%b %d '%y")
    mid = float(pick["mid"])
    iv_excess_pp = float(pick["iv_excess"]) * 100.0
    delta = float(pick.get("delta", 0.0))

    accent = OUTLOOK_TONE_HEX["pos"]   # green — premium income (sell default)
    signal_lbl = "top IV-cheap signal" if buy else "top IV+pp signal"

    if buy:
        # ── BUY TO OPEN (long call or long put) ───────────────────────────
        # A fresh long is a debit position — no roll/cover context applies.
        debit = mid * 100.0
        move_dollars = abs(delta) * 100.0
        if opt_type == "puts":
            accent = OUTLOOK_TONE_HEX["neg"]   # red — bearish / downside
            action_label = "BUY TO OPEN long put"
            action_lines = [
                f"<b>Action:</b> Buy 1×"
                f" <code>{ticker} {fmt_strike(strike)}P exp {expiry}</code>"
                f" at mid ~${mid:.2f}",
                f"<b>Cost (debit):</b> ${debit:,.0f} (1 contract)",
                f"<b>Max loss:</b> ${debit:,.0f} — the debit, if it expires"
                f" worthless",
                f"<b>Delta:</b> ~{delta:.2f} — gains ≈ ${move_dollars:.0f}"
                f" per $1 down-move",
                "<b>Stance:</b> bearish bet / synthetic short as the stock"
                " drops — or a hedge for shares you hold elsewhere.",
                "<b>Hedging instead?</b> Protective puts usually sit further"
                " OTM (|Δ| ~0.15–0.30) — cheaper, with less participation."
                " Lower the delta filter to surface them.",
            ]
            breakeven_line = (
                f"<b>Breakeven (stock):</b> ${strike - mid:.2f}"
                f" (strike − premium) — below this the put is in profit at"
                f" expiry"
            )
        else:
            accent = OUTLOOK_TONE_HEX["pos"]   # green — bullish / upside
            action_label = "BUY TO OPEN long call"
            action_lines = [
                f"<b>Action:</b> Buy 1×"
                f" <code>{ticker} {fmt_strike(strike)}C exp {expiry}</code>"
                f" at mid ~${mid:.2f}",
                f"<b>Cost (debit):</b> ${debit:,.0f} (1 contract)",
                f"<b>Max loss:</b> ${debit:,.0f} — the debit, if it expires"
                f" worthless",
                f"<b>Delta:</b> ~{delta:.2f} — gains ≈ ${move_dollars:.0f}"
                f" per $1 up-move (long exposure)",
            ]
            breakeven_line = (
                f"<b>Breakeven (stock):</b> ${strike + mid:.2f}"
                f" (strike + premium) — above this the call is in profit at"
                f" expiry"
            )

    elif covered and roll_close is not None and open_options:
        # ── ROLL (calls or puts) ──────────────────────────────────────────
        existing = open_options[0]
        net_cr_per_contract = (mid - roll_close) * 100.0
        net_cr_total = net_cr_per_contract * existing["contracts"]
        sign = "+" if net_cr_per_contract >= 0 else "−"

        if opt_type == "puts":
            action_label = "ROLL existing short put"
            opt_code = f"{ticker} {fmt_strike(strike)}P exp {expiry}"
            effective_buy = strike - (mid - roll_close)
            breakeven_line = (
                f"<b>Effective buy price if assigned:</b> ${effective_buy:.2f}"
                f" (strike − net credit)"
            )
        else:
            action_label = "ROLL existing covered call"
            opt_code = f"{ticker} {fmt_strike(strike)}C exp {expiry}"
            new_be = strike + (mid - roll_close)
            breakeven_line = (
                f"<b>New breakeven (stock):</b> ${new_be:.2f}"
                f" — below this the roll costs you net"
            )

        action_lines = [
            f"<b>1) Buy to close</b> {existing['contracts']}×"
            f" <code>{existing['symbol']}</code>"
            f" at mid ~${roll_close:.2f} →"
            f" pay <b>${roll_close * 100 * existing['contracts']:,.0f}</b>",
            f"<b>2) Sell to open</b> {existing['contracts']}×"
            f" <code>{opt_code}</code>"
            f" at mid ~${mid:.2f} →"
            f" collect <b>${mid * 100 * existing['contracts']:,.0f}</b>",
            f"<b>Net result:</b> {sign}${abs(net_cr_total):,.0f}"
            f" ({sign}${abs(net_cr_per_contract):.2f}/contract)",
        ]

    elif opt_type == "puts":
        # ── SELL TO OPEN cash-secured put (1-contract reference unit) ────
        premium = mid * 100.0
        margin_approx = strike * 100.0
        assign_prob = abs(delta) * 100.0
        breakeven_stock = strike - mid
        action_label = "SELL TO OPEN cash-secured put"
        action_lines = [
            f"<b>Action:</b> Sell 1× <code>{ticker} {fmt_strike(strike)}P exp {expiry}</code>"
            f" to open at mid ~${mid:.2f}",
            f"<b>Premium collected:</b> ${premium:,.0f} (1 contract)",
            f"<b>Cash required (approx.):</b> ${margin_approx:,.0f} (strike × 100)",
            f"<b>Assignment probability:</b> ~{assign_prob:.0f}% (|Δ| proxy)",
        ]
        breakeven_line = (
            f"<b>Breakeven (stock):</b> ${breakeven_stock:.2f}"
            f" — below this, net loss on assignment"
        )

    else:
        # ── SELL TO OPEN covered call ─────────────────────────────────────
        if shares < 100:
            pos_desc = ("No stock position" if shares == 0
                        else f"You hold <b>{shares}</b> shares")
            action_label = "Covered call not available"
            action_lines = [
                f"{pos_desc} — need at least 100 shares per contract.",
                f"Top IV-rich call for reference:"
                f" <code>{ticker} {fmt_strike(strike)}C exp {expiry}</code>"
                f" at mid ~${mid:.2f}",
            ]
            breakeven_line = ""
            accent = OUTLOOK_TONE_HEX["neutral"]
        else:
            max_contracts = shares // 100
            premium_per_contract = mid * 100.0
            premium_total = premium_per_contract * max_contracts
            max_profit_per_share = max(0.0, strike - spot) + mid
            max_profit_total = max_profit_per_share * 100 * max_contracts
            assign_prob = abs(delta) * 100.0
            action_label = "SELL TO OPEN covered call"
            action_lines = [
                f"<b>Action:</b> Sell {max_contracts}×"
                f" <code>{ticker} {fmt_strike(strike)}C exp {expiry}</code>"
                f" to open at mid ~${mid:.2f}",
                f"<b>Premium collected:</b> ${premium_total:,.0f}"
                f" ({max_contracts} contract(s) × ${premium_per_contract:,.0f})",
                f"<b>Max profit if assigned at {fmt_strike(strike)}:</b>"
                f" ${max_profit_total:,.0f} (capped — stock gets called away)",
                f"<b>Assignment probability:</b> ~{assign_prob:.0f}% (Δ proxy)",
            ]
            breakeven_line = (
                f"<b>Breakeven (stock):</b> ${spot - mid:.2f}"
                f" — covered down to this price by the premium received"
            )

    lines_html = "".join(f"<li style='margin: 3px 0;'>{l}</li>"
                         for l in action_lines)
    be_html = (
        f"<div style='margin-top: 6px; font-size: 0.78rem;"
        f" color: var(--osc-ink-3);'>{breakeven_line}</div>"
        if breakeven_line else ""
    )
    st.html(
        f"""
        <div style="
            border-left: 4px solid {accent};
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 0.6rem 0;
            font-family: var(--osc-font), -apple-system, sans-serif;
            line-height: 1.5;
        ">
            <div style="font-size: 0.65rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.08em;
                        color: var(--osc-ink-4); margin-bottom: 2px;">
                Recommended action · {signal_lbl} ({iv_excess_pp:+.1f} pp)
            </div>
            <div style="font-size: 1rem; font-weight: 700; color: {accent};
                        margin-bottom: 6px;">
                {action_label}
            </div>
            <ul style="margin: 0; padding-left: 1.1rem; font-size: 0.85rem;
                       color: var(--osc-ink-1);">
                {lines_html}
            </ul>
            {be_html}
        </div>
        """
    )
