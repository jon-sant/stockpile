"""Per-expiration volatility-surface chart with top-N pick callouts.

Renders the chain at one chosen expiration as a smile of IV dots with
one reference line, the table's top picks highlighted (large outlined
dots with rank labels), and a spot reference rule.

Reference line (green solid):
  IV surface (IV ≈ a+b·m+c·m²+d·√T+e·m·√T+f·m²·√T) fitted to all dropdown
  expirations using the configured surface filters (default: OTM-only,
  spread ≤ 50%, Δ 0.10–0.95). Dot colors and IV+pp both measure
  distance from this line, so they are fully consistent.

The pick highlighting and ranking come from compute.top_ranks — the same
function the bottom table uses — so chart and table never disagree.
"""

from __future__ import annotations

from datetime import date, datetime

import altair as alt
import pandas as pd
import streamlit as st

from options_scanner.compute.top_ranks import compute_top_ranks
from options_scanner.format import STRIKE_D3_FORMAT, strike_tick_values
from options_scanner.display.scan_stamp import scan_stamp_color, scan_stamp_text
from options_scanner.display.iv_surface_3d import render_iv_surface_3d


_PROVIDER_LINE = {
    "yahoo":  {"color": "#10b981", "strokeDash": [6, 4]},  # green dashed
    "yahoo-headless": {"color": "#0d9488", "strokeDash": [6, 4]},  # teal dashed
    "schwab": {"color": "#3b82f6", "strokeDash": [6, 4]},  # blue dashed
}


def _is_monthly_expiration(exp_str: str) -> bool:
    """Standard monthly options expire the 3rd Friday of the month.

    Inferred from the date since the normalized chain carries no
    weekly/monthly flag (Schwab's API has expirationType but we don't
    plumb it through; Yahoo has none). A heuristic — holiday-shifted or
    AM-settled expirations can deviate — but correct for the vast
    majority of equity options.
    """
    try:
        d = datetime.strptime(exp_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    # 3rd Friday = the Friday in day-of-month range 15–21.
    return d.weekday() == 4 and 15 <= d.day <= 21


def show_iv_chart(df: pd.DataFrame, spot: float, mode: str,
                  min_oi: int, top_n: int, buy: bool,
                  ticker: str = "", key_prefix: str = "s",
                  min_vol: int = 0, provider: str = "yahoo",
                  earnings_dates: list | None = None,
                  surface_filters: tuple | None = None,
                  df_full: pd.DataFrame | None = None,
                  delta_range: tuple[float, float] | None = None,
                  focus_contract: tuple[float, str] | None = None) -> None:
    """Layered chart: per-expiration smile with the table's top-N picks
    highlighted. Background dots are the rest of the chain at the selected
    expiration — filled if they anchored the surface fit, hollow if the
    filters excluded them; bright outlined dots are the top picks.

    `df_full` is the full fetched chain (pre delta-display filter). When
    supplied, a toggle lets the user reveal every contract the surface
    was actually fit to at the chosen expiration, including strikes the
    display delta range hides."""
    if df.empty:
        return

    chart_df = df.copy()
    if mode in ("call", "put"):
        chart_df = chart_df[chart_df["type"] == mode]
    if chart_df.empty:
        return

    top_ranks = compute_top_ranks(chart_df, mode, buy, min_oi, top_n, min_vol)

    from options_scanner import iv_scores as _iv_scores
    _score_kind = _iv_scores.active_kind(chart_df)
    _show_score = _score_kind != "IV+pp" and "signal_score" in chart_df.columns
    _mult = _iv_scores.display_for(_score_kind)[0] if _show_score else 1.0

    def _prep(frame: pd.DataFrame) -> pd.DataFrame:
        """Add the chart's display columns (rank flags, rounded IV fields).

        Ranking comes from `top_ranks`, computed on the displayed subset,
        so extra strikes revealed via the toggle never count as top picks.
        """
        frame = frame.copy()
        frame["is_top"] = frame.apply(
            lambda r: (r["type"], float(r["strike"]), r["expiration"])
            in top_ranks, axis=1)
        frame["rank_label"] = frame.apply(
            lambda r: str(top_ranks.get(
                (r["type"], float(r["strike"]), r["expiration"]), "")), axis=1)
        frame["IV%"]       = (frame["iv"] * 100).round(2)
        frame["FittedIV%"] = (frame["iv_fitted"] * 100).round(2)
        frame["IV+pp"]     = (frame["iv_excess"] * 100).round(2)
        frame["Ann%"]      = frame["ann_yield_pct"].round(2)
        frame["Spread"]    = (frame["ask"] - frame["bid"]).round(2)
        frame["Last"]      = frame["last"].where(frame["last"] > 0)
        if _show_score:
            frame[_score_kind] = (frame["signal_score"] * _mult).round(2)
        return frame

    chart_df = _prep(chart_df)
    exp_dte = chart_df.groupby("expiration")["dte"].first().to_dict()
    chart_df["ExpLabel"] = chart_df["expiration"].apply(
        lambda d: (f"{datetime.strptime(d, '%Y-%m-%d').strftime('%b %d \'%y')}"
                   f" ({exp_dte.get(d, 0)}d)")
    )

    expirations = sorted(chart_df["expiration"].unique())
    exp_labels = {
        e: (f"{datetime.strptime(e, '%Y-%m-%d').strftime('%b %d \'%y')}"
            f" — {exp_dte.get(e, 0)}d")
        for e in expirations
    }
    pick_counts = {
        e: int(chart_df[(chart_df["expiration"] == e) & chart_df["is_top"]].shape[0])
        for e in expirations
    }
    best_exps = {exp for (_, _, exp), r in top_ranks.items() if r == 1}
    picks_df = chart_df[chart_df["is_top"]]
    # A focused contract (e.g. the Sell Put dialog's put) wins the default so
    # the chart opens on its expiration; otherwise default to the most extreme
    # pick's expiration.
    if focus_contract is not None and focus_contract[1] in expirations:
        default_idx = expirations.index(focus_contract[1])
    elif not picks_df.empty:
        extreme_idx = (picks_df["iv_excess"].idxmin() if buy
                       else picks_df["iv_excess"].idxmax())
        default_exp = picks_df.loc[extreme_idx, "expiration"]
        default_idx = expirations.index(default_exp)
    else:
        default_idx = 0

    h1, h2 = st.columns([1, 2], vertical_alignment="bottom")
    with h1:
        st.markdown(
            "<h5 style='margin:0 0 5px 0'>Volatility surface</h5>",
            unsafe_allow_html=True,
        )
    # The 3D view needs the full multi-expiration chain (Single tab only;
    # the Portfolio tab omits df_full).
    _has_full = df_full is not None and not df_full.empty
    _view_opts = (["Single expiration", "All expirations", "3D surface"]
                  if _has_full else ["Single expiration", "All expirations"])
    with h2:
        view = st.radio(
            "View", _view_opts,
            horizontal=True, key=f"{key_prefix}_surface_view",
            label_visibility="collapsed",
            help="Single = one expiration's smile vs. its surface line. "
                 "All expirations = every expiration's fitted surface line "
                 "on one chart, colored by DTE. 3D surface = the whole chain "
                 "as strike × DTE × IV — drag to rotate.",
        )

    # Multi-expiration views ("All expirations" 2D fan and "3D surface")
    # share the same source frame + fit-range setup. Both use the full chain
    # when available (wider strike span), else the displayed frame (e.g. the
    # Portfolio tab, which omits df_full).
    if view in ("All expirations", "3D surface"):
        if df_full is not None and not df_full.empty:
            _src = (df_full[df_full["type"] == mode]
                    if mode in ("call", "put") else df_full)
            overlay_df = _prep(_src)
            _support_src = df_full
        else:
            overlay_df = chart_df
            _support_src = chart_df
        # Strike range that actually anchored the fit (both wings, every
        # expiration). Beyond it the global surface is unsupported
        # extrapolation — the source of the spurious far-OTM/ITM humps — so
        # the views clip to this span.
        fit_range = None
        if "in_fit" in _support_src.columns:
            _anchors = _support_src[_support_src["in_fit"].astype(bool)]
            if not _anchors.empty:
                fit_range = (float(_anchors["strike"].min()),
                             float(_anchors["strike"].max()))
        if view == "3D surface":
            # The fitted mesh needs both wings to span spot: with OTM-only
            # filtering a single option type's in-fit anchors sit on one side of
            # spot, so a single-type surface clips at the spot plane. Pass the
            # full two-sided fit frame (both calls and puts) for the mesh; the
            # dots still use the single-type overlay_df. (For "both", overlay_df
            # is already two-sided, so reuse it.)
            if df_full is not None and not df_full.empty:
                fit_frame = (overlay_df if mode not in ("call", "put")
                             else _prep(df_full))
            else:
                fit_frame = None
            render_iv_surface_3d(overlay_df, spot, ticker, mode, buy,
                                 fit_range, delta_range=delta_range,
                                 min_oi=min_oi, min_vol=min_vol, top_n=top_n,
                                 fit_frame=fit_frame)
        else:
            _render_all_expirations(overlay_df, spot, ticker, mode, fit_range)
        return

    chosen_exp = st.selectbox(
        "Expiration to chart",
        options=expirations,
        index=default_idx,
        format_func=lambda d: (
            f"{'★ ' if d in best_exps else ''}{exp_labels[d]}"
            f"{' Ⓜ' if _is_monthly_expiration(d) else ''}"
            f"  ({pick_counts[d]} pick"
            f"{'s' if pick_counts[d] != 1 else ''})"
        ),
        key=f"{key_prefix}_chart_exp",
        help=("Each expiration has its own volatility smile. The number "
              "in parentheses is how many of the table's top picks live "
              "at that expiration. Ⓜ marks standard monthly expirations "
              "(3rd Friday) — typically the most liquid."),
        label_visibility="collapsed",
    )

    # Optionally reveal the strikes the display delta range hid but the
    # surface was still fit on — only offered when there are extra ones.
    show_all_fit = False
    full_mode = None
    if df_full is not None and not df_full.empty:
        full_mode = df_full
        if mode in ("call", "put"):
            full_mode = full_mode[full_mode["type"] == mode]
        _n_full = int((full_mode["expiration"] == chosen_exp).sum())
        _n_shown = int((chart_df["expiration"] == chosen_exp).sum())
        if _n_full > _n_shown:
            show_all_fit = st.checkbox(
                f"Show all fit points at this expiration "
                f"({_n_full - _n_shown} more outside the display Δ range)",
                value=False, key=f"{key_prefix}_show_all_fit",
                help="The surface is fit on a wider delta range than the "
                     "table shows. Turn this on to see every contract the "
                     "line was actually fit to at this expiration.",
            )

    if show_all_fit and full_mode is not None:
        sub = _prep(full_mode[full_mode["expiration"] == chosen_exp]).sort_values(
            ["type", "strike"]
        ).copy()
    else:
        sub = chart_df[chart_df["expiration"] == chosen_exp].sort_values(
            ["type", "strike"]
        ).copy()
    if sub.empty:
        return

    # Warn when this expiration's line isn't a fit to its own contracts.
    _fit_method = (str(sub["fit_method"].iloc[0])
                   if "fit_method" in sub.columns else "")
    if _fit_method == "fallback":
        _exp_date = datetime.strptime(chosen_exp, "%Y-%m-%d").date()
        _excl_earn = surface_filters and any(
            n == "exclude_earnings" for n, _ in surface_filters
        )
        _earn_spans = earnings_dates and any(
            date.today() < ed <= _exp_date for ed in earnings_dates
        )
        if _excl_earn and _earn_spans:
            st.warning(
                f"**{exp_labels[chosen_exp]}** has an earnings event before "
                "expiration — the earnings-exclusion filter removed those "
                "contracts from the fit, leaving too few to fit this slice "
                "locally. The line shown is the cross-expiration surface. "
                "Earnings IV premium still shows up as positive IV+pp.",
                icon="⚠️",
            )
        else:
            st.warning(
                f"The per-expiry fit couldn't fit **{exp_labels[chosen_exp]}** "
                "from its own contracts (too few passed the surface-fit "
                "filters), so this line is the cross-expiration surface — it "
                "may not reflect this expiry's own smile. Switch the **Fit:** "
                "toggle to *Global*, or relax the surface-fit filters / widen "
                "the DTE range.",
                icon="⚠️",
            )
    elif _fit_method == "none":
        st.warning(
            "Not enough clean contracts to fit a surface for this scan, so "
            "the line just traces the quotes (IV+pp ≈ 0). Widen the DTE "
            "range or relax the surface-fit filters.",
            icon="⚠️",
        )

    iv_cols = ["IV%", "FittedIV%"]
    y_min = max(0.0, float(sub[iv_cols].values.min()) * 0.92)
    y_max = float(sub[iv_cols].values.max()) * 1.05

    excess_max = max(abs(sub["IV+pp"].min()), abs(sub["IV+pp"].max()), 1.0)
    if buy:
        color_range = ["#22c55e", "#cbd5e1", "#ef4444"]
    else:
        color_range = ["#ef4444", "#cbd5e1", "#22c55e"]
    color_scale = alt.Scale(
        domain=[-excess_max, 0, excess_max], range=color_range)
    shape_scale = alt.Scale(
        domain=["call", "put"], range=["circle", "square"])

    x_min = min(float(sub["strike"].min()), spot) * 0.97
    x_max = max(float(sub["strike"].max()), spot) * 1.03

    base_x = alt.X(
        "strike:Q", title="Strike",
        scale=alt.Scale(domain=[x_min, x_max]),
        axis=alt.Axis(
            format=STRIKE_D3_FORMAT,
            values=strike_tick_values(sub["strike"], x_min, x_max) or alt.Undefined,
        ),
    )
    y_scale = alt.Scale(domain=[y_min, y_max])
    base_y = alt.Y("IV%:Q", title="Implied Volatility (%)", scale=y_scale)

    # D3 number formats for each score kind (Altair tooltips use D3, not printf).
    _SCORE_D3_FMT = {"IV z": "+.2f", "IV rel": "+.1%", "Score": "+.2f",
                     "VRP": ".2f", "IV %ile": ".0f"}
    _score_d3_fmt = _SCORE_D3_FMT.get(_score_kind, "+.2f")
    tooltip_fields = [
        alt.Tooltip("strike:Q",       title="Strike",          format=STRIKE_D3_FORMAT),
        alt.Tooltip("type:N",         title="Type"),
        alt.Tooltip("IV%:Q",                                   format=".1f"),
        alt.Tooltip("FittedIV%:Q",    title="Surface IV%",     format=".1f"),
        alt.Tooltip("IV+pp:Q",        title="IV excess (pp)",  format="+.1f"),
        *([alt.Tooltip(f"{_score_kind}:Q", title=_score_kind,
                       format=_score_d3_fmt)]
          if _show_score else []),
        alt.Tooltip("delta:Q",                                 format=".2f"),
        alt.Tooltip("Ann%:Q",         title="Ann%",            format=".1f"),
        alt.Tooltip("volume:Q",       title="Volume",          format=",.0f"),
        alt.Tooltip("open_interest:Q", title="OI"),
        alt.Tooltip("bid:Q",          title="Bid",             format="$.2f"),
        alt.Tooltip("ask:Q",          title="Ask",             format="$.2f"),
        alt.Tooltip("Spread:Q",       title="Spread",          format="$.2f"),
        alt.Tooltip("Last:Q",         title="Last",            format="$.2f"),
        *([alt.Tooltip("in_fit:N", title="In surface fit")]
          if "in_fit" in sub.columns else []),
    ]

    # Dashed line — color encodes data source (blue=Yahoo, green=Schwab)
    _line_style = _PROVIDER_LINE.get(provider, _PROVIDER_LINE["yahoo"])
    line_surface = alt.Chart(sub).mark_line(
        size=2, **_line_style,
    ).encode(
        x=base_x,
        y=alt.Y("FittedIV%:Q", scale=y_scale),
        detail="type:N",
    )

    # Background dots, split by whether they anchored the surface fit:
    # filled = used in the regression, hollow = excluded by the filters.
    bg_rest = sub[~sub["is_top"]]
    if "in_fit" in bg_rest.columns:
        bg_fit  = bg_rest[bg_rest["in_fit"].astype(bool)]
        bg_excl = bg_rest[~bg_rest["in_fit"].astype(bool)]
    else:
        bg_fit, bg_excl = bg_rest, bg_rest.iloc[0:0]

    background = alt.Chart(bg_fit).mark_point(
        size=60, opacity=1.0, filled=True,
    ).encode(
        x=base_x,
        y=base_y,
        color=alt.Color("IV+pp:Q", scale=color_scale,
                        legend=alt.Legend(title="IV excess (pp)")),
        shape=alt.Shape("type:N", scale=shape_scale,
                        legend=alt.Legend(title="Type")),
        tooltip=tooltip_fields,
    )

    excluded = alt.Chart(bg_excl).mark_point(
        size=70, opacity=1.0, filled=False, strokeWidth=2.6,
    ).encode(
        x=base_x,
        y=base_y,
        color=alt.Color("IV+pp:Q", scale=color_scale, legend=None),
        shape=alt.Shape("type:N", scale=shape_scale, legend=None),
        tooltip=tooltip_fields,
    )

    picks = alt.Chart(sub[sub["is_top"]]).mark_point(
        size=260, opacity=1.0, filled=True,
        stroke="#0f172a", strokeWidth=2,
    ).encode(
        x=base_x,
        y=base_y,
        color=alt.Color("IV+pp:Q", scale=color_scale, legend=None),
        shape=alt.Shape("type:N", scale=shape_scale, legend=None),
        tooltip=tooltip_fields,
    )

    ranks = alt.Chart(sub[sub["is_top"]]).mark_text(
        fontSize=14, dy=-20, fontWeight="bold", color="#0f172a",
    ).encode(
        x=base_x,
        y=base_y,
        text="rank_label:N",
    )

    spot_df = pd.DataFrame({
        "x": [spot], "y": [y_max], "label": [f"Spot ${spot:.2f}"],
    })
    spot_rule = alt.Chart(spot_df).mark_rule(
        color="#0f172a", strokeDash=[3, 3], size=2,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[x_min, x_max])),
        tooltip=[alt.Tooltip("x:Q", title="Spot", format="$,.2f")],
    )
    spot_label = alt.Chart(spot_df).mark_text(
        align="left", baseline="top", dx=5, dy=2,
        color="#0f172a", fontWeight="bold", fontSize=11,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[x_min, x_max])),
        y="y:Q",
        text="label:N",
    )

    # When the "show all fit points" toggle reveals strikes outside the
    # display delta range, shade the strike bands beyond the normally
    # displayed span so the extra points read as out-of-range context.
    shade_layers = []
    if show_all_fit:
        _shown = chart_df[chart_df["expiration"] == chosen_exp]
        if not _shown.empty:
            shown_lo = float(_shown["strike"].min())
            shown_hi = float(_shown["strike"].max())
            bands = []
            if x_min < shown_lo:
                bands.append({"x": x_min, "x2": shown_lo})
            if shown_hi < x_max:
                bands.append({"x": shown_hi, "x2": x_max})
            if bands:
                shade = alt.Chart(pd.DataFrame(bands)).mark_rect(
                    color="#64748b", opacity=0.09,
                ).encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[x_min, x_max])),
                    x2="x2:Q",
                )
                shade_layers.append(shade)

    # Focus marker: a gold ring + label around the specific contract in focus
    # (e.g. the dialog's put), drawn only when it lives at the charted
    # expiration and survived the display filters.
    focus_layers = []
    if focus_contract is not None and focus_contract[1] == chosen_exp:
        _frow = sub[(sub["strike"] - float(focus_contract[0])).abs() < 1e-6]
        if not _frow.empty:
            _fword = {"call": "this call", "put": "this put"}.get(
                mode, "this option")
            _fdf = _frow.assign(_flabel=f"◆ {_fword}")
            focus_layers = [
                alt.Chart(_fdf).mark_point(
                    size=640, filled=False, shape="circle",
                    stroke="#f59e0b", strokeWidth=3.5,
                ).encode(x=base_x, y=base_y),
                alt.Chart(_fdf).mark_text(
                    fontSize=12, fontWeight="bold", color="#b45309", dy=24,
                ).encode(x=base_x, y=base_y, text="_flabel:N"),
            ]

    type_word = {"call": "calls", "put": "puts", "both": "options"}[mode]
    title_text = (f"{ticker} {type_word} — {exp_labels[chosen_exp]}"
                  if ticker else f"{type_word} — {exp_labels[chosen_exp]}")
    chart = alt.layer(
        *shade_layers,
        line_surface, background, excluded, picks, ranks,
        *focus_layers,
        spot_rule, spot_label,
    ).properties(
        height=380,
        title=alt.TitleParams(
            text=title_text,
            subtitle=scan_stamp_text() or None,
            subtitleColor=scan_stamp_color(),
            subtitleFontSize=11,
            fontSize=16, fontWeight="bold", anchor="start",
            color="#0f172a",
        ),
    )
    st.altair_chart(chart, width='stretch')

    shade_note = (
        "<br>"
        "<span style='background:rgba(100,116,139,0.16);padding:0 0.35em'>"
        "&nbsp;&nbsp;&nbsp;</span>"
        " <b>Shaded band</b> = strikes outside the display &Delta; range,"
        " revealed by the toggle above."
        if show_all_fit and shade_layers else ""
    )
    focus_note = (
        "<br><span style='color:#f59e0b;font-size:1.1em'>&#9711;</span>"
        " <b>Gold ring</b> = the contract you're investigating."
        if focus_layers else ""
    )
    st.markdown(
        "<div style='font-size:0.8rem;line-height:1.9;color:var(--osc-ink-3)'>"
        "<span style='color:#10b981'>&#9632;&#9632; &mdash; &mdash;</span>"
        "&nbsp;<b>Green dashed</b> (Yahoo Finance)&nbsp;&nbsp;"
        "<span style='color:#3b82f6'>&#9632;&#9632; &mdash; &mdash;</span>"
        "&nbsp;<b>Blue dashed</b> (Schwab) &mdash;"
        " IV surface fit across all fetched expirations (within your DTE range),"
        " using only clean data (configurable under <i>Advanced surface fit</i>)."
        " <b>Dot color and IV+pp both measure distance above/below this line</b>"
        " &mdash; green dot = IV-rich, red = IV-cheap."
        "<br>"
        "<b>Filled dot</b> = anchored the surface fit;"
        " <b>hollow dot</b> = a filter excluded it from the fit"
        " (it still gets an IV+pp read)."
        "<br>"
        "<b>Large outlined dot + number</b> = top pick;"
        " number matches rank in table below (1&nbsp;=&nbsp;strongest signal)."
        " Vertical dashed line = current spot price."
        + shade_note + focus_note +
        "</div>",
        unsafe_allow_html=True,
    )


def _render_all_expirations(frame: pd.DataFrame, spot: float,
                            ticker: str, mode: str,
                            fit_range: tuple[float, float] | None = None
                            ) -> None:
    """Overlay every expiration's fitted surface line on one chart.

    Each line is one expiration's `iv_fitted` vs. strike, colored by DTE —
    the term-structure fan. It plots the already-computed surface, so it
    reflects whatever algorithm produced it: in Global mode the lines
    share one surface (curvature in common, fanned by √T); in Per-expiry
    mode each line is that slice's own smile. `frame` must already carry
    the `_prep` display columns (FittedIV%, IV%).

    `fit_range` (lo, hi strike) clips the lines to the strikes that
    anchored the fit. Past that span the global surface is pure
    extrapolation — the cause of the spurious far-OTM/ITM humps — so we
    simply don't draw it."""
    frame = frame.dropna(subset=["FittedIV%", "strike"]).copy()
    if frame.empty:
        st.info("No fitted surface to display for this scan.")
        return
    clip_note = ""
    if fit_range is not None:
        lo, hi = fit_range
        in_range = frame[(frame["strike"] >= lo) & (frame["strike"] <= hi)]
        if not in_range.empty:
            frame = in_range
            clip_note = (f" Lines are drawn only across the strike range that "
                         f"anchored the fit (${lo:,.0f}–${hi:,.0f}); past it "
                         f"the surface is unsupported extrapolation.")
    frame = frame.sort_values(["expiration", "strike"])
    frame["ExpDate"] = frame["expiration"].apply(
        lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%b %d '%y"))

    x_min = min(float(frame["strike"].min()), spot) * 0.97
    x_max = max(float(frame["strike"].max()), spot) * 1.03
    y_min = max(0.0, float(frame["FittedIV%"].min()) * 0.92)
    y_max = float(frame["FittedIV%"].max()) * 1.05

    base_x = alt.X(
        "strike:Q", title="Strike",
        scale=alt.Scale(domain=[x_min, x_max]),
        axis=alt.Axis(
            format=STRIKE_D3_FORMAT,
            values=strike_tick_values(frame["strike"], x_min, x_max) or alt.Undefined,
        ),
    )
    y_enc = alt.Y("FittedIV%:Q", title="Fitted IV (%)",
                  scale=alt.Scale(domain=[y_min, y_max]))
    dte_color = alt.Color("dte:Q", scale=alt.Scale(scheme="viridis"),
                          legend=alt.Legend(title="DTE"))
    tooltip = [
        alt.Tooltip("ExpDate:N",   title="Expiration"),
        alt.Tooltip("dte:Q",       title="DTE",         format="d"),
        alt.Tooltip("strike:Q",    title="Strike",      format=STRIKE_D3_FORMAT),
        alt.Tooltip("FittedIV%:Q", title="Surface IV%", format=".1f"),
        alt.Tooltip("IV%:Q",       title="IV%",         format=".1f"),
    ]

    lines = alt.Chart(frame).mark_line(size=2).encode(
        x=base_x, y=y_enc, color=dte_color, detail="expiration:N",
    )
    # Nodes carry the tooltip — Altair line tooltips alone are unreliable.
    nodes = alt.Chart(frame).mark_circle(size=28, opacity=0.55).encode(
        x=base_x, y=y_enc, color=dte_color, detail="expiration:N",
        tooltip=tooltip,
    )

    spot_df = pd.DataFrame({
        "x": [spot], "y": [y_max], "label": [f"Spot ${spot:.2f}"],
    })
    spot_rule = alt.Chart(spot_df).mark_rule(
        color="#0f172a", strokeDash=[3, 3], size=2,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[x_min, x_max])),
        tooltip=[alt.Tooltip("x:Q", title="Spot", format="$,.2f")],
    )
    spot_label = alt.Chart(spot_df).mark_text(
        align="left", baseline="top", dx=5, dy=2,
        color="#0f172a", fontWeight="bold", fontSize=11,
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[x_min, x_max])),
        y="y:Q", text="label:N",
    )

    type_word = {"call": "calls", "put": "puts", "both": "options"}[mode]
    subj = f"{ticker} {type_word}" if ticker else type_word
    _stamp = scan_stamp_text()
    chart = (lines + nodes + spot_rule + spot_label).properties(
        height=380,
        title=alt.TitleParams(
            text=f"{subj} — fitted IV surface · all expirations",
            fontSize=16, fontWeight="bold", anchor="start",
            color="#0f172a",
            # Subtitle keys only when there's a stamp — Altair rejects a
            # None subtitle (the slice chart relies on scan_ts always set).
            **({"subtitle": _stamp, "subtitleColor": scan_stamp_color(),
                "subtitleFontSize": 11} if _stamp else {}),
        ),
    )
    st.altair_chart(chart, width='stretch')

    st.markdown(
        "<div style='font-size:0.8rem;line-height:1.9;color:var(--osc-ink-3)'>"
        "Each line is one expiration's <b>fitted surface</b> (the IV the model"
        " expects at each strike), colored by <b>days to expiration</b>."
        " In <b>Global</b> fit every line comes from one surface fit across"
        " all expirations, so they share curvature and fan out by term"
        " structure; in <b>Per-expiry</b> fit each line is that expiration's"
        " own smile." + clip_note +
        " Vertical dashed line = current spot price."
        "</div>",
        unsafe_allow_html=True,
    )
