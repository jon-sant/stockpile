"""Cross-ticker IV+pp leaderboard for the Portfolio tab.

Aggregates the top-N picks from every scanned ticker into one ranked
table — "across the whole basket, which contracts have the richest
IV+pp right now?" Shown for both the brokerage-CSV and watchlist input
sources, above the per-ticker expanders.

Ranking reuses the same convention as the per-position tables and the
IV chart (`compute.top_ranks`): sort by `signal_score` (descending in
sell mode), `open_interest` as the tiebreaker — so a contract's place
here is consistent with its rank everywhere else.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from options_scanner import iv_scores
from options_scanner.format import EARNINGS_WARN_LEGEND, fmt_strike
from options_scanner.display.scan_stamp import stamp_caption


# Account-info balance fields worth highlighting when selling puts — bolded
# with a hover explanation in the dialog's Account info panel. Keyed by the
# raw Schwab currentBalances field name. Descriptions must stay free of double
# quotes — they're injected into an HTML title="" attribute.
_PUT_BALANCE_NOTES = {
    "availableFundsNonMarginableTrade":
        "Cash-secured put capacity: cash that isn't borrowed, fully "
        "collateralizes a put. Divide by strike x 100 for max contracts. "
        "This is the figure 'Cash for puts' sizes against.",
    "availableFunds":
        "General available-to-trade balance; about the same as your cash-"
        "secured capacity here, slightly less conservative.",
    "buyingPower":
        "Margin buying power (~3x the cash figure). For puts sold on margin "
        "/ naked, NOT cash-secured - the tool ignores it for put sizing.",
    "cashBalance":
        "Actual settled cash. Low when funds sit in securities/SMA; this is "
        "not your put capacity.",
    "liquidationValue":
        "Account total net value - your overall account size.",
    "maintenanceRequirement":
        "Margin already tied up by your current positions.",
    "shortOptionMarketValue":
        "Your existing short-option exposure - gauge how much more to add.",
    "maintenanceCall":
        "Maintenance margin call amount; 0 means no call (good).",
    "regTCall":
        "Reg-T initial-margin call amount; 0 means no call (good).",
    "sma":
        "Special Memorandum Account: margin reserve; matters only if you "
        "trade on margin.",
}


@st.cache_data(ttl=60, show_spinner=False)
def _account_capacity(app_key: str, app_secret: str, callback_url: str,
                      token_file: str) -> dict | None:
    """Cached (60s) read-only Schwab capacity. Returns dict or None.

    Keyed on the credentials so a re-auth (new token file) busts it via the
    same path the rest of the app uses. Read-only — no order entry.
    """
    from stocks_shared.schwab_live import get_client
    from options_scanner import trade_actions
    try:
        client = get_client(app_key, app_secret, callback_url, token_file)
    except Exception:
        return None
    cap = trade_actions.fetch_account_capacity(client)
    if cap is None:
        return None
    return {"cash": cap.cash_available, "bp": cap.buying_power,
            "amount": cap.amount, "type": cap.account_type,
            "mask": cap.account_mask, "balances": cap.balances}


@st.cache_data(ttl=60, show_spinner=False)
def _market_open(app_key: str, app_secret: str, callback_url: str,
                 token_file: str) -> bool | None:
    """Cached (60s) read-only check: is the equity-options market open now?

    True/False, or None when Schwab is unreachable — the dialog keeps Place
    Trade disabled on None (fail safe). Read-only.
    """
    from stocks_shared.schwab_live import get_client
    from options_scanner import trade_actions
    try:
        client = get_client(app_key, app_secret, callback_url, token_file)
    except Exception:
        return None
    return trade_actions.market_is_open(client)


def _submit_put_order(scfg: dict, order, cap: dict | None,
                      paper: bool, fill: dict | None = None) -> dict:
    """Place (or paper-record) a confirmed put order. Returns {ok, msg}.

    LIVE submission happens only when `paper` is False — the existing
    config-level opt-in. Either way the trade is logged to the Trades store.
    `fill` carries the dialog's fill-time {fill_spot, fill_delta, fill_iv} for
    paper trades (a paper trade fills immediately at placement, so it's
    captured here; the live path reconstructs it from the broker fill later).
    """
    from options_scanner import trade_actions, trades_store
    # Store credit PER SHARE (the limit) — that's what the store schema and
    # the Trades tab expect; `order.credit` is the total (limit*100*qty).
    base = {"ticker": order.ticker, "strike": order.strike,
            "expiration": order.expiration, "quantity": order.quantity,
            "credit": order.limit, "status": "open"}
    if paper:
        rec = {**base, "paper": True}
        if fill:
            rec.update({k: v for k, v in fill.items() if v is not None})
        trades_store.add(rec)
        return {"ok": True,
                "msg": (f"📝 Paper trade recorded — {order.describe()}. "
                        "No live order sent. Set `paper = false` in "
                        "config.toml to submit for real.")}
    from stocks_shared.schwab_live import get_client
    try:
        client = get_client(scfg.get("app_key", ""), scfg.get("app_secret", ""),
                            scfg.get("callback_url", ""),
                            scfg.get("token_file", ""))
    except Exception as exc:
        return {"ok": False, "msg": f"Schwab unreachable: {exc}"}
    last4 = ((cap or {}).get("mask") or "")[-4:]
    resolved = trade_actions.resolve_account_hash(client, last4 or None)
    if not resolved:
        return {"ok": False,
                "msg": "Couldn't resolve the target account — order NOT sent."}
    account_hash, mask = resolved
    res = trade_actions.place_put_sell_order(client, order, account_hash)
    if not res["ok"]:
        return {"ok": False, "msg": f"Order rejected: {res['error']}"}
    trades_store.add({**base, "paper": False, "order_id": res["order_id"],
                      "account": mask})
    _oid = f" (id {res['order_id']})" if res["order_id"] else ""
    return {"ok": True,
            "msg": (f"✅ LIVE order sent to {mask}{_oid} — {order.describe()}. "
                    "Verify at your broker.")}


def _investigate_put_dialog(c: dict, ticker_df: "pd.DataFrame | None" = None,
                            min_oi: int = 25, top_n: int = 5,
                            min_vol: int = 0, provider: str = "schwab") -> None:
    """Open the assisted put-selling dialog with the contract identity baked
    into the title bar — right of "Investigate put-sell". The decorator is
    applied at call time (not at import) so the title can be dynamic."""
    _spot_txt = f"${c['spot']:.2f}" if c.get("spot") is not None else "—"
    # Day % change right after the spot (cached 60s, same source as the cards).
    try:
        from options_scanner.display.spot_meta import fetch_spot_meta
        _pct = fetch_spot_meta(str(c["ticker"]), provider).get("pct_change")
    except Exception:
        _pct = None
    if _pct is not None:
        _spot_txt += f" {_pct:+.1f}%"
    _ne = c.get("next_earnings")
    _earn_seg = f", Earnings {_ne.strftime('%b %d')}" if _ne else ""
    # Mode (PAPER vs LIVE) baked into the title bar so it's impossible to miss
    # which one you're in before placing. Comes from config `paper` (set at
    # app start; can't change mid-session without a restart).
    _paper = bool((st.session_state.get("schwab_config") or {}).get("paper",
                                                                    True))
    _mode = "📝 PAPER" if _paper else "🔴 LIVE"
    _title = f"🔍 Sell Put — {c['ticker']} {_spot_txt}{_earn_seg}  ·  {_mode}"

    @st.dialog(_title, width="large")
    def _dlg() -> None:
        _investigate_put_body(c, ticker_df=ticker_df, min_oi=min_oi,
                              top_n=top_n, min_vol=min_vol, provider=provider)

    _dlg()


def _investigate_put_body(c: dict, ticker_df: "pd.DataFrame | None" = None,
                          min_oi: int = 25, top_n: int = 5,
                          min_vol: int = 0, provider: str = "schwab") -> None:
    """Assisted put-selling dialog body (Schwab, watchlist leaderboard).

    Shows the contract snapshot and a 2-D IV chart (where this put sits vs
    the chain), judges whether it looks executable (fill-quality heuristic in
    ``trade_actions``), suggests an editable limit price, then a contracts-to-
    sell input sized against the account's buying capacity, and a validated
    **order preview**. Placement is wired (Confirm Trade → an inline confirm
    panel → submit), gated by market hours + the ``paper`` flag; LIVE only
    when ``paper=false``. See
    ``options-scanner/assisted-put-selling-implementation-plan.md``.
    """
    from options_scanner import trade_actions

    exp = datetime.strptime(c["expiration"], "%Y-%m-%d").strftime("%b %d '%y")

    def _money(v):
        return f"${v:,.2f}" if v is not None else "—"

    iv_txt = f"{c['iv'] * 100:.1f}%" if c.get("iv") is not None else "—"
    delta_txt = f"{c['delta']:.2f}" if c.get("delta") is not None else "—"
    ann_txt = f"{c['ann_pct']:.1f}%" if c.get("ann_pct") is not None else "—"
    # Left snapshot table: contract terms. Right table: prices + liquidity.
    terms = [
        ("Type", "Put"), ("Strike", f"${c['strike']:g}"), ("Expir", exp),
        ("DTE", f"{c['dte']}"), ("IV", iv_txt), ("Delta", delta_txt),
        ("Ann%", ann_txt),
    ]
    prices = [
        ("Bid", _money(c.get("bid"))), ("Ask", _money(c.get("ask"))),
        ("Mid", _money(c.get("mid"))), ("Last", _money(c.get("last"))),
        ("OI", f"{c['open_interest']:,d}"), ("Vol", f"{c['volume']:,d}"),
    ]

    def _kv_html(rows, notes=None):
        # `notes` (field -> description): rows with a note are bolded and get
        # a hover tooltip; the label loses its gray so it stands out.
        notes = notes or {}
        cells = []
        for f, v in rows:
            note = notes.get(f)
            tip = f' title="{note}"' if note else ""
            bold = "font-weight:600;" if note else ""
            label_color = "" if note else "color:#808495;"
            cells.append(
                f"<tr{tip}>"
                f"<td style='padding:5px 14px;{label_color}{bold}'>{f}</td>"
                f"<td style='padding:5px 14px;font-variant-numeric:tabular-nums;"
                f"{bold}'>{v}</td>"
                "</tr>"
            )
        return ("<table style='border-collapse:collapse;font-size:1.05rem'>"
                + "".join(cells) + "</table>")

    assessment = trade_actions.assess_fill(
        bid=c.get("bid"), ask=c.get("ask"), mid=c.get("mid"),
        volume=c.get("volume"), open_interest=c.get("open_interest", 0),
    )

    # Account capacity (read-only) — informs sizing in the trade column.
    scfg = st.session_state.get("schwab_config") or {}
    cap = (_account_capacity(scfg.get("app_key", ""), scfg.get("app_secret", ""),
                             scfg.get("callback_url", ""),
                             scfg.get("token_file", ""))
           if scfg.get("app_key") else None)
    cap_amt = cap.get("amount") if cap else None
    affordable = trade_actions.puts_affordable(cap_amt, c["strike"])
    # Market-hours gate for Place Trade (None = unknown → fail safe, disabled).
    market_open = (_market_open(scfg.get("app_key", ""),
                                scfg.get("app_secret", ""),
                                scfg.get("callback_url", ""),
                                scfg.get("token_file", ""))
                   if scfg.get("app_key") else None)

    # Snapshot (left) beside the trade controls (right), so the bid/ask stays
    # visible while you set the limit. The IV-surface chart goes full-width at
    # the bottom. The contract identity now lives in the dialog title bar, so
    # the left panel just gets a short label that lines up with the right one.
    _ratio = [1, 1.5]
    title_l, title_r = st.columns(_ratio)
    with title_l:
        st.markdown("### Contract")
    with title_r:
        st.markdown("### Sell a cash-secured put")

    # Top row, top-aligned, so "Suggested limit" (top of the right panel) lines
    # up with the top of the tables. The disclaimers + Place Trade button go in
    # a separate footer row below, where they line up with each other.
    snap_col, trade_col = st.columns(_ratio, vertical_alignment="top")
    with snap_col:
        # Two side-by-side HTML tables (terms | prices). Plain HTML rather than
        # st.dataframe, which always draws a header row — here an empty gray bar,
        # since these key/value columns carry no titles.
        terms_col, prices_col = st.columns(2)
        with terms_col:
            st.markdown(_kv_html(terms), unsafe_allow_html=True)
        with prices_col:
            st.markdown(_kv_html(prices), unsafe_allow_html=True)
        for note in assessment.notes:
            st.caption(f"⚠ {note}")

    with trade_col:
        # Default limit: mid when liquid; IV-aligned model price when not (the
        # user can still set their own and place it).
        if assessment.liquid:
            default_limit = assessment.suggested_limit
            st.markdown("**Suggested limit** — mid, rounded to the tick.")
        else:
            st.warning("**Thin/wide market:** " + "; ".join(assessment.reasons)
                       + ". Set your own limit if you still want to place it.")
            model = trade_actions.model_limit(
                spot=c.get("spot"), strike=c["strike"], dte=c["dte"],
                iv=c.get("iv"),
            )
            # Default to the richer of the IV-aligned model and the observed
            # market — never below the higher of mid/last, so a thin contract's
            # default can't undercut the market. The user can still override.
            _base = (model or assessment.suggested_limit or c.get("mid") or 0.05)
            _floor = max((v for v in (c.get("mid"), c.get("last")) if v),
                         default=0.0)
            default_limit = trade_actions.round_to_tick(max(_base, _floor))
            if model is not None:
                st.markdown(f"**Suggested \\${default_limit:.2f}** — IV-aligned "
                            f"(\\${model:.2f}), floored at the higher of "
                            "mid/last. Override as you like.")
            else:
                st.markdown(f"**Suggested \\${default_limit:.2f}** — mid/last "
                            "(no IV model). Override as you like.")

        # Stronger resting border on the two inputs so they clearly read as
        # editable fields. Scoped to the dialog so other number inputs in the
        # app keep their default styling.
        st.markdown(
            "<style>"
            "div[role='dialog'] div[data-testid='stNumberInput'] "
            "div[data-baseweb='input']{border:2px solid #8a8f9c !important;"
            "border-radius:0.5rem;}"
            "</style>",
            unsafe_allow_html=True,
        )
        lim_col, qty_col = st.columns(2)
        with lim_col:
            limit = st.number_input(
                "Limit price",
                min_value=float(trade_actions.tick_for(default_limit)),
                value=float(default_limit),
                step=float(trade_actions.tick_for(default_limit)), format="%.2f",
                key=f"investigate_limit_{c['ticker']}_{c['strike']:g}_{c['expiration']}",
            )
        with qty_col:
            qty = st.number_input(
                "Contracts", min_value=1, value=1, step=1,
                max_value=(affordable if affordable and affordable >= 1 else None),
                key=f"investigate_qty_{c['ticker']}_{c['strike']:g}_{c['expiration']}",
            )
        if cap_amt is not None:
            _aff = f" · up to {affordable}" if affordable is not None else ""
            st.caption(f"Cash for puts \\${cap_amt:,.0f}{_aff} "
                       f"(\\${c['strike'] * 100:,.0f} collateral each). "
                       "See Account info below for full balances.")
        else:
            st.caption("Cash for puts unavailable — connect Schwab "
                       "(Accounts & Trading access required).")

        # Order preview + validation. `order_ok` gates the Place Trade button.
        order_ok = False
        try:
            order = trade_actions.build_put_sell_order(
                ticker=c["ticker"], strike=c["strike"],
                expiration=c["expiration"], limit=float(limit),
                quantity=int(qty), capacity=cap_amt,
            )
            st.success(
                (f"{order.describe()} — credit ${order.credit:,.0f}, "
                 f"collateral ${order.collateral:,.0f}.").replace("$", "\\$"))
            order_ok = True
        except ValueError as exc:
            st.error(f"Can't build this order: {exc}")

    # Session keys scoped to this contract (confirm-pending + last result).
    _ck = f"{c['ticker']}_{c['strike']:g}_{c['expiration']}"
    _confirm_key = f"place_confirm_{_ck}"
    _result_key = f"place_result_{_ck}"
    paper = bool(scfg.get("paper", True))
    # Mode badge on the action buttons so PAPER vs LIVE is unmissable.
    _badge = "📝 PAPER" if paper else "🔴 LIVE"
    _result = st.session_state.get(_result_key)

    # Footer row: disclaimers (left) beside the Place Trade button (right).
    foot_l, foot_r = st.columns(_ratio, vertical_alignment="center")
    with foot_l:
        st.caption("Sells puts only · never fires without your confirm · "
                   "**Schwab only.**")
        if paper:
            st.caption("📝 **Paper mode** (`paper=true`) — Confirm records a "
                       "simulated trade; no live order is sent.")
        else:
            st.caption("🔴 **LIVE** — Confirm sends a real order. Verify at "
                       "your broker.")
    with foot_r:
        # Paper records a simulation any time; a LIVE order is market-gated.
        if not order_ok:
            st.button(f"Confirm Trade · {_badge}", disabled=True, key=f"place_{_ck}",
                      help="Fix the order above first.")
        elif paper or market_open is True:
            if st.button(f"Confirm Trade · {_badge}", key=f"place_{_ck}", type="primary"):
                st.session_state[_confirm_key] = True
                st.session_state.pop(_result_key, None)
        elif market_open is False:
            st.button(f"Confirm Trade · {_badge}", disabled=True, key=f"place_{_ck}",
                      help="Live order — equity options trade 9:30–16:00 ET, "
                           "Mon–Fri.")
            st.caption("⏸ Market closed")
        else:  # None — live, but couldn't confirm the session: stay disabled
            st.button(f"Confirm Trade · {_badge}", disabled=True, key=f"place_{_ck}",
                      help="Live order — can't confirm market hours (Schwab "
                           "unreachable).")
            st.caption("⏸ Market hours unknown")

    # Inline confirm step — Streamlit can't nest a dialog, so this is a review
    # panel within the same dialog: details + Confirm to actually submit.
    if order_ok and st.session_state.get(_confirm_key):
        _acct_lbl = (cap or {}).get("mask") or "your account"
        # Account cash + cash-secured-put capacity, so the order's collateral
        # can be sanity-checked against what's available. Cash falls back to
        # the raw cashBalance when the cash-account field is absent (margin).
        _bal = (cap or {}).get("balances") or {}
        _acct_cash = (cap or {}).get("cash")
        if _acct_cash is None:
            _acct_cash = _bal.get("cashBalance")
        _coll_avail = (cap or {}).get("amount")
        # Collateral leads the bottom line, grouped with the cash figures so the
        # requirement sits right next to what's available.
        _cash_bits = [f"collateral **${order.collateral:,.0f}**"]
        if _acct_cash is not None:
            _cash_bits.append(f"account cash **${_acct_cash:,.2f}**")
        if _coll_avail is not None:
            _cash_bits.append(f"cash for puts **${_coll_avail:,.2f}**")
        # Escape every $ so Streamlit markdown doesn't read $...$ as LaTeX math
        # (which eats the dollar signs and garbles the amounts).
        st.warning((
            f"**Confirm this order** — {order.describe()}  \n"
            f"Account **{_acct_lbl}** · credit **${order.credit:,.0f}** · "
            + ("🔴 **LIVE**" if not paper else "📝 **PAPER**")
            + "  \n" + " · ".join(_cash_bits)).replace("$", "\\$"))
        # Place Trade + a red Cancel, equal-sized and adjacent (a spacer column
        # keeps them compact on the left). Cancel is reddened via CSS scoped to
        # its keyed container — only one confirm panel renders at a time.
        st.markdown(
            "<style>"
            "[class*='st-key-investigate_cancel_box'] button{"
            "background-color:#d9534f !important;border-color:#d9534f "
            "!important;color:#fff !important;}"
            "[class*='st-key-investigate_cancel_box'] button:hover{"
            "background-color:#c9302c !important;border-color:#c9302c "
            "!important;color:#fff !important;}"
            "[class*='st-key-investigate_cancel_box'] button p{"
            "color:#fff !important;}"
            "</style>",
            unsafe_allow_html=True,
        )
        # Cancel collapses the panel via on_click (runs before the rerun body),
        # so it takes effect on the FIRST click. An inline `if button:` would
        # set the flag only after the panel already rendered this run → the
        # collapse wouldn't show until a second click.
        def _cancel_confirm(_k=_confirm_key):
            st.session_state[_k] = False

        _cc1, _cc2, _ = st.columns([1, 1, 3])
        with _cc1:
            _submit = st.button(f"Place Trade · {_badge}", key=f"confirm_{_ck}",
                                type="primary", width="stretch")
        with _cc2:
            _cancel_box = st.container(key="investigate_cancel_box")
            _cancel_box.button("Cancel", key=f"cancel_{_ck}", width="stretch",
                               on_click=_cancel_confirm)
        if _submit:
            _result = _submit_put_order(
                scfg, order, cap, paper,
                fill={"fill_spot": c.get("spot"), "fill_delta": c.get("delta"),
                      "fill_iv": c.get("iv")})
            st.session_state[_result_key] = _result
            st.session_state[_confirm_key] = False
            if _result.get("ok"):
                # Close the dialog and rerun so the Trades tab (an st.tabs panel
                # rendered before this dialog overlay) reflects the new trade
                # without a manual browser refresh. Queue the toast for the NEXT
                # run (emitted in run_app) — a toast created right before
                # st.rerun() is discarded with the current run.
                st.session_state["_osc_toast"] = (
                    _result["msg"] + "  See the Trades tab.")
                st.rerun()

    # Failures stay in the dialog (a success path reruns + toasts above, so the
    # dialog is already closing by the time we'd get here on success).
    if _result and not _result.get("ok"):
        st.error(_result["msg"].replace("$", "\\$"))

    # Account info — collapsed; the full Schwab balance snapshot for the linked
    # account, sitting just above the volatility-surface chart.
    if cap and cap.get("balances"):
        _hdr = " · ".join(x for x in (cap.get("mask"), cap.get("type")) if x)
        with st.expander(f"Account info{(' — ' + _hdr) if _hdr else ''}",
                         expanded=False):
            _bals = cap["balances"]

            def _fmt_bal(k, v):
                return f"{v:,.2f}%" if "percent" in k.lower() else _money(v)

            # Split the balances across two side-by-side tables (sorted, halved).
            _items = [(k, _fmt_bal(k, _bals[k])) for k in sorted(_bals)]
            _half = (len(_items) + 1) // 2
            _ac1, _ac2 = st.columns(2)
            with _ac1:
                st.markdown(_kv_html(_items[:_half], _PUT_BALANCE_NOTES),
                            unsafe_allow_html=True)
            with _ac2:
                st.markdown(_kv_html(_items[_half:], _PUT_BALANCE_NOTES),
                            unsafe_allow_html=True)
            st.caption("Read-only. **Bold** rows are the ones that matter for "
                       "cash-secured puts — hover any for what it means.")

    # IV-surface chart — full width at the bottom (how rich this put is vs the
    # rest of the chain).
    if ticker_df is not None and not ticker_df.empty and c.get("spot"):
        st.markdown("---")
        try:
            from options_scanner.display.iv_chart import show_iv_chart
            show_iv_chart(
                ticker_df, float(c["spot"]), "put", int(min_oi), int(top_n),
                buy=False, ticker=c["ticker"],
                key_prefix=f"inv_{c['ticker']}_{c['strike']:g}_{c['expiration']}",
                min_vol=int(min_vol), provider=provider,
                focus_contract=(float(c["strike"]), str(c["expiration"])),
            )
        except Exception:
            st.caption("IV chart unavailable for this contract.")


def build_leaderboard(results: list[dict], side: str, min_oi: int,
                      top_n: int, min_vol: int = 0,
                      delta_range: tuple[float, float] | None = None,
                      buy: bool = False,
                      min_ivpp: float | None = None,
                      min_ann: float | None = None,
                      ) -> pd.DataFrame:
    """Collect a "best per ticker, then fill" leaderboard for one side.

    `side` is "call" or "put". Selection:

      1. Each ticker's single best contract (its #1) is guaranteed a slot,
         so every scanned ticker that has any qualifying option is
         represented.
      2. Remaining slots are filled with the next-best leftovers globally
         (each ticker contributes at most `top_n`).
      3. Total rows = 2× the number of tickers that have ≥1 qualifying
         option, then everything is sorted by IV+pp so the richest float
         to the top even when several come from the same ticker.

    Returns a DataFrame with a `ticker` column, the chain columns, and a
    boolean `_is_ticker_top` flag (True for each ticker's #1 pick — used
    to shade those rows). Empty frame when nothing qualifies.
    """
    sort_col_for = lambda f: ("signal_score" if "signal_score" in f.columns
                              else "iv_excess")

    per_ticker = []  # each ticker's top-N rows, #1 first
    for res in results:
        if res.get("error"):
            continue
        df = res.get("df")
        if df is None or df.empty:
            continue
        sub = df[(df["type"] == side)
                 & (df["open_interest"] >= min_oi)
                 & (df["volume"] >= min_vol)]
        if min_ivpp is not None:
            sub = sub[(sub["iv_excess"] * 100) >= min_ivpp]
        if min_ann is not None:
            sub = sub[sub["ann_yield_pct"] >= min_ann]
        if delta_range is not None:
            lo, hi = delta_range
            sub = sub[sub["delta"].abs().between(lo, hi)]
        if sub.empty:
            continue
        sub = (sub.sort_values([sort_col_for(sub), "open_interest"],
                               ascending=[buy, False])
               .head(top_n).copy())
        sub["ticker"] = res["position"]["ticker"]
        sub["_is_ticker_top"] = [True] + [False] * (len(sub) - 1)
        per_ticker.append(sub.reset_index(drop=True))

    if not per_ticker:
        return pd.DataFrame()

    n_tickers = len(per_ticker)
    target = 2 * n_tickers

    # 1. Guarantee every ticker's #1 pick.
    guaranteed = pd.concat([t.iloc[[0]] for t in per_ticker], ignore_index=True)

    # 2. Fill remaining slots from the next-best leftovers globally.
    leftovers = [t.iloc[1:] for t in per_ticker if len(t) > 1]
    if leftovers:
        pool = pd.concat(leftovers, ignore_index=True)
        sc = sort_col_for(pool)
        pool = pool.sort_values([sc, "open_interest"], ascending=[buy, False])
        fill = pool.head(max(0, target - len(guaranteed)))
        combined = pd.concat([guaranteed, fill], ignore_index=True)
    else:
        combined = guaranteed

    # 3. Final display sort by signal (richest first when selling, cheapest
    #    first when buying).
    sc = sort_col_for(combined)
    combined = (combined.sort_values([sc, "open_interest"],
                                     ascending=[buy, False])
                .head(target).reset_index(drop=True))
    return combined


def render_leaderboard(results: list[dict], mode: str, min_oi: int,
                       top_n: int, min_vol: int = 0,
                       delta_range: tuple[float, float] | None = None,
                       buy: bool = False,
                       allow_investigate: bool = False,
                       provider: str = "yahoo",
                       min_ivpp: float | None = None,
                       min_ann: float | None = None) -> None:
    """Render the cross-ticker leaderboard table(s).

    `mode` is "call", "put", or "both" (both renders a Calls and a Puts
    leaderboard). `buy` flips the ranking so IV-cheap contracts float to
    the top. Shows an explanatory notice when nothing qualifies at all.

    `allow_investigate` turns each Puts-board row into a selectable control
    that opens the assisted put-selling dialog. The caller gates it to
    watchlist + sell + Schwab; here it only ever attaches to the Puts board
    (you can't sell-to-open a put from the Calls board). `provider` and the
    per-ticker chains (looked up from `results`) feed the dialog's IV chart.
    """
    sides = [mode] if mode in ("call", "put") else ["call", "put"]
    headings = {"call": "Calls", "put": "Puts"}
    ticker_dfs = {
        r["position"]["ticker"]: r["df"]
        for r in results
        if r.get("df") is not None and not r["df"].empty
    }
    # Next earnings date per ticker (the board carries only the 0/1 flag).
    ticker_earnings = {
        r["position"]["ticker"]: (r.get("earnings_dates") or [None])[0]
        for r in results
    }

    rendered_any = False
    for side in sides:
        board = build_leaderboard(results, side, min_oi, top_n, min_vol,
                                  delta_range, buy, min_ivpp, min_ann)
        if board.empty:
            continue
        rendered_any = True
        if len(sides) > 1:
            st.markdown(f"**{headings[side]}**")
        _render_table(board, side, min_vol,
                      investigate=(allow_investigate and side == "put"),
                      min_oi=min_oi, top_n=top_n, ticker_dfs=ticker_dfs,
                      ticker_earnings=ticker_earnings, provider=provider)

    if not rendered_any:
        st.info(
            f"No contracts passed the leaderboard filters "
            f"(Min OI ≥ {min_oi}, Min Vol ≥ {min_vol}"
            + (f", |delta| {delta_range[0]:.2f}–{delta_range[1]:.2f}"
               if delta_range is not None else "")
            + (f", IV+pp ≥ {min_ivpp:+.1f}" if min_ivpp is not None else "")
            + (f", Ann% ≥ {min_ann:.1f}" if min_ann is not None else "")
            + "). Try loosening Min OI / Min Vol — note Vol is *today's* "
              "volume, which is 0 for every contract before the market has "
              "traded."
        )
        return
    st.caption("Shaded rows are each ticker's top pick; other rows fill in "
               "the next-richest contracts across the basket.")
    stamp_caption()


def _prem_pct_em(board: pd.DataFrame) -> pd.Series:
    """Premium as a fraction of the option's own theoretical 1-sigma
    expected move: mid / (spot * iv * sqrt(dte/365)). Informational only.
    dte clamped to >=1 to avoid a 0-DTE divide-by-zero."""
    if not {"mid", "spot", "iv", "dte"} <= set(board.columns):
        return pd.Series([float("nan")] * len(board), index=board.index)
    expected_move = (board["spot"] * board["iv"]
                     * np.sqrt(board["dte"].clip(lower=1) / 365.0))
    return (board["mid"] / expected_move.replace(0, float("nan"))).round(2)


def _render_table(board: pd.DataFrame, side: str, min_vol: int,
                  investigate: bool = False, min_oi: int = 25, top_n: int = 5,
                  ticker_dfs: dict | None = None,
                  ticker_earnings: dict | None = None,
                  provider: str = "yahoo") -> None:
    """Render one leaderboard table, styled like the scan-results table.

    When `investigate` is True the table becomes single-row-selectable and
    selecting a row opens the assisted put-selling dialog. `min_oi`/`top_n`/
    `ticker_dfs`/`provider` feed that dialog's IV chart; `ticker_earnings`
    supplies the next-earnings date shown in its snapshot.
    """
    kind = iv_scores.active_kind(board)

    # ⚠ in the Expiration cell = short-dated (≤60 DTE) and expiring after the
    # next earnings — its IV+pp carries event premium and it's the slice
    # excluded from the surface fit. Cheaper than a whole column.
    _ec = (board["earnings_count"].fillna(0) if "earnings_count" in board.columns
           else pd.Series(0, index=board.index))

    def _exp_cell(e, d, c):
        base = datetime.strptime(e, "%Y-%m-%d").strftime("%b %d '%y")
        return f"{base} ⚠" if (c >= 1 and d <= 60) else base

    _has_warn = any(c >= 1 and d <= 60
                    for c, d in zip(_ec, board["dte"]))

    # Ticker label: symbol + scan spot (currency) + day-change %. pct is
    # per-ticker (cached 60s via fetch_spot_meta); spot is the per-row scan spot.
    from options_scanner.display.spot_meta import fetch_spot_meta
    _pct = {}
    for _tk in board["ticker"].unique():
        try:
            _pct[_tk] = fetch_spot_meta(str(_tk), provider).get("pct_change")
        except Exception:
            _pct[_tk] = None
    _spots = (board["spot"] if "spot" in board.columns
              else pd.Series([None] * len(board), index=board.index))

    def _ticker_label(tk, spot):
        parts = [str(tk)]
        if spot is not None and spot == spot and float(spot) > 0:  # not NaN
            parts.append(f"${float(spot):,.2f}")
        p = _pct.get(tk)
        if p is not None:
            parts.append(f"{p:+.1f}%")
        return " ".join(parts)

    cols = {
        "Ticker": [_ticker_label(tk, sp)
                   for tk, sp in zip(board["ticker"], _spots)],
        "Strike": board["strike"].apply(fmt_strike),
        "Expiration": [_exp_cell(e, d, c) for e, d, c
                       in zip(board["expiration"], board["dte"], _ec)],
        "DTE":   board["dte"].astype(int),
        "Bid":   board["bid"].round(2),
        "Ask":   board["ask"].round(2),
        "Mid":   board["mid"].round(2),
        "Last":  (board["last"].where(board["last"] > 0)
                  if "last" in board.columns
                  else pd.Series([float("nan")] * len(board), index=board.index)),
        "IV+pp": (board["iv_excess"] * 100).round(1),
    }
    if kind != "IV+pp":
        mult, _ = iv_scores.display_for(kind)
        cols[kind] = (board["signal_score"] * mult).round(2)
    cols.update({
        "Delta": board["delta"].round(2),
        "Ann%":  board["ann_yield_pct"].round(1),
        "Ann% / Delta": (board["ann_yield_pct"]
                         / board["delta"].abs().replace(0, float("nan"))).round(1),
        "Prem/EM": _prem_pct_em(board),
        "OI":    board["open_interest"],
        "Vol":   board["volume"],
    })
    disp = pd.DataFrame(cols)

    # Shade each ticker's #1 pick so it stands out from its fill rows.
    top_mask = (board["_is_ticker_top"].tolist()
                if "_is_ticker_top" in board.columns else [False] * len(board))
    _TOP_ROW = "background-color: rgba(53,194,193,0.16)"

    def _shade(_row):
        i = disp.index.get_loc(_row.name)
        return [_TOP_ROW if top_mask[i] else ""] * len(_row)

    styled = disp.style.apply(_shade, axis=1)

    col_cfg = {
        "Ticker":     st.column_config.TextColumn("Ticker", width=160),
        "Strike":     st.column_config.TextColumn("Strike", width=75),
        "Expiration": st.column_config.TextColumn(
            "Expiration", width=125,
            help="⚠ = ≤60 DTE and expiring after the next earnings, so its "
                 "IV+pp includes event premium (and it's excluded from the "
                 "surface fit)."),
        "DTE":   st.column_config.NumberColumn("DTE", format="%d", width=55),
        "Bid":   st.column_config.NumberColumn("Bid", format="$%.2f", width=70),
        "Ask":   st.column_config.NumberColumn("Ask", format="$%.2f", width=70),
        "Mid":   st.column_config.NumberColumn("Mid", format="$%.2f", width=70),
        "Last":  st.column_config.NumberColumn("Last", format="$%.2f", width=70),
        "IV+pp": st.column_config.NumberColumn("IV+pp", format="%+.1f pp",
                                               width=80),
        "Delta": st.column_config.NumberColumn("Delta", format="%.2f",
                                               width=60),
        "Ann%":  st.column_config.NumberColumn("Ann%", format="%.1f%%",
                                               width=65),
        "Ann% / Delta": st.column_config.NumberColumn(
            "Ann% / Delta", format="%.1f", width=95,
            help="Ann% divided by |Delta| — yield per unit of directional "
                 "exposure. Higher = more yield for the assignment risk "
                 "taken on."),
        "Prem/EM": st.column_config.NumberColumn(
            "Prem/EM", format="%.2f", width=75,
            help="Premium ÷ (spot × IV × √(DTE/365)) — how much of the "
                 "option's own theoretical 1-sigma expected move you're "
                 "being paid. >1 means premium exceeds the modeled move."),
        "OI":    st.column_config.NumberColumn("OI", format="%d", width=65),
        "Vol":   st.column_config.NumberColumn("Vol", format="%d", width=65),
    }
    if kind != "IV+pp":
        _, fmt = iv_scores.display_for(kind)
        col_cfg[kind] = st.column_config.NumberColumn(
            kind, format=fmt, width=85,
            help="Active ranking score — the leaderboard is ranked by this "
                 "column. IV+pp shown alongside for context.")

    if not investigate:
        st.dataframe(styled, column_config=col_cfg, hide_index=True,
                     width="stretch")
        if _has_warn:
            st.caption(EARNINGS_WARN_LEGEND)
        return

    # Assisted put-selling (Schwab, watchlist): each row is selectable, and
    # picking one opens the investigate dialog (stub for now).
    st.caption("🔍 **Select a put row** to investigate placing a cash-secured "
               "put sell — Schwab assisted trade (preview).")
    event = st.dataframe(styled, column_config=col_cfg, hide_index=True,
                         width="stretch", on_select="rerun",
                         selection_mode="single-row", key="lb_investigate_put")
    if _has_warn:
        st.caption(EARNINGS_WARN_LEGEND)
    sel = event.selection.rows if hasattr(event, "selection") else []
    if not sel:
        # Deselecting clears the guard so re-selecting the SAME row reopens the
        # dialog. Without this, sel_key still equals _lb_last_investigated and
        # the open below is skipped (the bug where you had to pick another row
        # first).
        st.session_state["_lb_last_investigated"] = None
        return

    def _num(v):
        try:
            f = float(v)
            return f if f == f else None  # NaN → None
        except (TypeError, ValueError):
            return None

    row = board.iloc[sel[0]]
    contract = {
        "ticker": str(row["ticker"]),
        "strike": float(row["strike"]),
        "expiration": str(row["expiration"]),
        "dte": int(row["dte"]),
        "bid": _num(row.get("bid")),
        "ask": _num(row.get("ask")),
        "mid": _num(row.get("mid")),
        "last": _num(row.get("last")) if "last" in board.columns else None,
        "iv": _num(row.get("iv")) if "iv" in board.columns else None,
        "spot": _num(row.get("spot")) if "spot" in board.columns else None,
        "delta": _num(row.get("delta")) if "delta" in board.columns else None,
        "ann_pct": (_num(row.get("ann_yield_pct"))
                    if "ann_yield_pct" in board.columns else None),
        "volume": int(row["volume"]),
        "open_interest": int(row["open_interest"]),
    }
    contract["next_earnings"] = (ticker_earnings or {}).get(contract["ticker"])
    # Only open the modal on a *new* selection, so dismissing it doesn't
    # immediately reopen on the next rerun while the row stays selected.
    sel_key = f"{contract['ticker']}|{contract['strike']}|{contract['expiration']}"
    if st.session_state.get("_lb_last_investigated") != sel_key:
        st.session_state["_lb_last_investigated"] = sel_key
        # Fresh open: drop any stale confirm/result state for this contract so
        # the Place Trade section starts collapsed (it persisted in
        # session_state across a dismiss). Key mirrors _ck in the dialog body.
        _ck = f"{contract['ticker']}_{contract['strike']:g}_{contract['expiration']}"
        st.session_state.pop(f"place_confirm_{_ck}", None)
        st.session_state.pop(f"place_result_{_ck}", None)
        _investigate_put_dialog(
            contract,
            ticker_df=(ticker_dfs or {}).get(contract["ticker"]),
            min_oi=min_oi, top_n=top_n, min_vol=min_vol, provider=provider,
        )
