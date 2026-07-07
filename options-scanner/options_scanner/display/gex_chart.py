"""Gamma Exposure (GEX) bar chart by strike.

Aggregates per-strike dealer gamma across all expirations in the
current scan and renders it as a colored bar chart with the spot
overlay, summary metrics row (Total GEX / Regime / Zero-gamma level),
and a DTE-scope footnote so screenshots stay self-documenting.

Convention matches `compute.gex_summary`: calls contribute positive
gamma (pinning), puts negative (amplifying). Net positive total =
mean-reverting regime; net negative = trending regime.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from options_scanner.compute.gex_summary import per_strike_gex, gamma_flip_strike
from options_scanner.format import STRIKE_D3_FORMAT, strike_tick_values
from options_scanner.display.scan_stamp import scan_stamp_color, scan_stamp_text


def show_gex_chart(df: pd.DataFrame, spot: float,
                   provider: str = "yahoo",
                   ticker: str = "") -> None:
    """Render the GEX bar chart + summary metrics for the supplied chain.

    Args:
        df: Chain DataFrame with `type`, `strike`, `open_interest`,
            `gamma`, and (optionally) `dte` / `expiration` columns.
        spot: Current underlying spot — drawn as a dashed vertical
            reference line and used to zoom the x-axis to the
            meaningful strike range.
        provider: "yahoo" or "schwab" — surfaces the right caveat
            footnote about IV freshness.
        ticker: Optional ticker symbol prepended to the chart title
            for screenshot-friendly context. Falls back to the bare
            chart name when omitted.
    """
    if df.empty or "gamma" not in df.columns:
        st.info("No GEX to display — this chain has no gamma data.")
        return

    gex = per_strike_gex(df, spot)
    if gex.empty or gex["gex"].abs().sum() == 0:
        st.info("No GEX to display — no open interest across this chain's "
                "strikes in the selected DTE range.")
        return

    total_gex = gex["gex"].sum()
    gex["color"] = gex["gex"].apply(lambda v: "Pinning" if v >= 0 else "Amplifying")

    # Gamma flip (zero-gamma level): strike where cumulative GEX flips
    # sign, nearest spot. See gamma_flip_strike for why "nearest the actual
    # sign change" beats "lowest strike with cumulative >= 0".
    zero_cross = gamma_flip_strike(gex, spot)

    g1, g2, g3 = st.columns(3)
    regime = "Pinning (mean-reverting)" if total_gex >= 0 else "Amplifying (trending)"
    g1.metric("Total GEX", f"{total_gex:,.0f}", help=(
        "Positive = dealers net long gamma across this chain — price "
        "tends to mean-revert. Negative = dealers net short gamma — "
        "moves tend to be amplified."
    ))
    g2.metric("Regime", regime)
    if not pd.isna(zero_cross):
        g3.metric("Gamma flip", f"${zero_cross:,.2f}", help=(
            "Gamma flip (a.k.a. zero-gamma level): strike where cumulative "
            "dealer gamma flips sign. Price above this level tends to be "
            "more volatile."
        ))

    # Zoom the x-axis to the strikes that actually carry GEX. Chains
    # often include far-OTM strikes with near-zero gamma — leaving them
    # in shrinks the meaningful bars to a sliver in the middle. Take
    # the strikes that hold ~99% of total |GEX| as the core.
    gex_sorted_abs = gex.assign(abs_gex=gex["gex"].abs()) \
                        .sort_values("abs_gex", ascending=False)
    total_abs = float(gex_sorted_abs["abs_gex"].sum())
    if total_abs > 0:
        cum = gex_sorted_abs["abs_gex"].cumsum() / total_abs
        core = gex_sorted_abs[cum <= 0.99]
        if core.empty:
            core = gex_sorted_abs.head(1)
        core_lo = float(core["strike"].min())
        core_hi = float(core["strike"].max())
    else:
        core_lo = float(gex["strike"].min())
        core_hi = float(gex["strike"].max())

    # Center the window on spot so a lopsided GEX distribution (or one far
    # strike in the core) can't shove the bars to one side — spot reads at
    # the middle every scan. The half-width reaches the farthest core edge
    # (so the 99%-|GEX| core is always included), floored at ±2% of spot.
    # Also reach the gamma flip when that keeps the window within ~1.6x the
    # core's own reach, so the flip line stays visible; a far flip would
    # shrink the real bars to a sliver, so it's left to the metric card
    # only (the in-range guard below drops the line). Padded 4%.
    core_half = max(spot - min(core_lo, spot), max(core_hi, spot) - spot,
                    spot * 0.02)
    half = core_half
    if not pd.isna(zero_cross):
        half = max(half, min(abs(spot - float(zero_cross)), core_half * 1.6))
    half *= 1.04
    x_min, x_max = spot - half, spot + half

    # Trim the dataframe to the zoom range so bars rescale to fill the
    # chart width (Altair's `scale=domain=` alone just clips off-screen
    # bars without expanding the in-range ones).
    gex_zoomed = gex[(gex["strike"] >= x_min) & (gex["strike"] <= x_max)]
    if gex_zoomed.empty:
        gex_zoomed = gex
    y_max_gex = float(gex_zoomed["gex"].max())

    # Interactive x-axis zoom: scroll/drag to zoom the strike axis so more
    # strikes (incl. .5s) label as you zoom in. Bound to scales, x only, so
    # the bars keep their vertical scale. No explicit scale `domain` on the
    # layers (an explicit domain would override and disable the zoom), and no
    # `value` init on the selection (a scales-bound init leaves the binding
    # inert). The initial view is pinned to the core window [x_min, x_max] by
    # a transparent anchor layer (xspan) that just widens the auto-domain. Any
    # Streamlit rerun redraws to this default view.
    x_zoom = alt.selection_interval(bind="scales", encodings=["x"])
    xspan = alt.Chart(pd.DataFrame({"x": [x_min, x_max]})).mark_rule(
        opacity=0).encode(x="x:Q")

    bars = alt.Chart(gex_zoomed).mark_bar(opacity=0.85, clip=True).encode(
        x=alt.X("strike:Q", title="Strike",
                scale=alt.Scale(nice=False),
                axis=alt.Axis(
                    format=STRIKE_D3_FORMAT,
                    # All core strikes as candidate ticks; labelOverlap thins
                    # them at the default view and zooming in reveals more.
                    values=strike_tick_values(gex_zoomed["strike"], max_ticks=1000)
                    or alt.Undefined,
                    labelOverlap="greedy")),
        y=alt.Y("gex:Q", title="Net GEX ($)"),
        color=alt.Color("color:N",
                        scale=alt.Scale(
                            domain=["Pinning", "Amplifying"],
                            range=["#22c55e", "#ef4444"],
                        ),
                        legend=alt.Legend(title=None)),
        tooltip=[
            alt.Tooltip("strike:Q",  title="Strike",  format=STRIKE_D3_FORMAT),
            alt.Tooltip("gex:Q",     title="Net GEX", format=",.0f"),
            alt.Tooltip("color:N",   title="Effect"),
        ],
    )

    spot_df = pd.DataFrame({"x": [spot], "y": [y_max_gex],
                            "label": [f"Spot ${spot:.2f}"]})
    spot_rule = alt.Chart(spot_df).mark_rule(
        color="#0f172a", strokeDash=[3, 3], strokeWidth=1.5,
    ).encode(
        x=alt.X("x:Q"),
    )
    spot_label = alt.Chart(spot_df).mark_text(
        align="left", baseline="top", dx=5, dy=2,
        color="#0f172a", fontWeight="bold", fontSize=11,
    ).encode(
        x=alt.X("x:Q"),
        y="y:Q",
        text="label:N",
    )

    # Gamma-flip vertical line (the zero-gamma level), drawn like the spot
    # line but in violet. Only drawn when the flip sits inside the zoomed
    # strike window [x_min, x_max]. A flip far outside it — common on chains
    # whose cumulative GEX first crosses zero way out in a low-OI tail (seen
    # on Schwab SPY: flip ~245 with spot ~753) — would otherwise blow out the
    # layered chart's shared x-axis and collapse the real bars to an unseeable
    # sliver. The flip value is still shown in the metric card above.
    flip_layers = []
    if not pd.isna(zero_cross) and x_min <= float(zero_cross) <= x_max:
        flip_df = pd.DataFrame({"x": [float(zero_cross)], "y": [y_max_gex],
                                "label": [f"Gamma flip ${zero_cross:,.2f}"]})
        flip_rule = alt.Chart(flip_df).mark_rule(
            color="#7c3aed", strokeDash=[6, 3], strokeWidth=1.5, clip=True,
        ).encode(
            x=alt.X("x:Q"),
        )
        flip_label = alt.Chart(flip_df).mark_text(
            align="left", baseline="top", dx=5, dy=16,
            color="#7c3aed", fontWeight="bold", fontSize=11, clip=True,
        ).encode(
            x=alt.X("x:Q"),
            y="y:Q",
            text="label:N",
        )
        flip_layers = [flip_rule, flip_label]

    # Build a screenshot-friendly title: ticker first, then chart type.
    # Falls back to just the chart name if no ticker is passed. The dealer-
    # positioning warning rides in the title so it's impossible to miss on a
    # screenshot; the full caveat sits in the caption below the chart.
    _warn = ("  (⚠️ Dealer positioning is assumed, not measured, "
             "more details below chart)")
    title_text = (f"{ticker} — Gamma Exposure (GEX) by strike{_warn}"
                  if ticker else f"Gamma Exposure (GEX) by strike{_warn}")

    # DTE scope footnote so screenshots taken days later still convey
    # which slice of the chain the bars are summed over.
    if "dte" in df.columns and not df["dte"].empty:
        dte_lo = int(df["dte"].min())
        dte_hi = int(df["dte"].max())
        n_exp = int(df["expiration"].nunique())
        dte_note = (f"Aggregated across {n_exp} expiration"
                    f"{'s' if n_exp != 1 else ''} "
                    f"({dte_lo}–{dte_hi} DTE).")
    else:
        dte_note = "Aggregated across all expirations in the current scan."

    chart = alt.layer(
        bars, spot_rule, spot_label, *flip_layers, xspan
    ).add_params(x_zoom)
    st.altair_chart(
        chart.properties(
            height=240,
            title=alt.TitleParams(
                text=title_text,
                subtitle=scan_stamp_text() or None,
                subtitleColor=scan_stamp_color(),
                subtitleFontSize=11,
                fontSize=14, fontWeight="bold", anchor="start",
                color="#0f172a",
            ),
        ).configure_view(strokeWidth=0),
        width='stretch',
    )

    st.caption("Scroll or drag on the chart to zoom the strike axis — more "
               "strikes label as you zoom in. Rescan to reset the view.")

    provider_caveat = (
        "GEX estimated from Black-Scholes gamma (Yahoo IV may be stale on LEAPS)."
        if provider in ("yahoo", "yahoo-headless")
        else "GEX computed from Schwab's native gamma values."
    )
    st.caption(f"{dte_note} {provider_caveat}")
    st.caption(
        "⚠️ **Dealer positioning is assumed, not measured.** GEX treats "
        "dealers as long calls / short puts — a market-wide tendency, not "
        "verifiable without trade-level data. Concentrated flows (e.g. funds "
        "writing covered calls) can flip the real sign. Use as directional "
        "context, not fact."
    )
