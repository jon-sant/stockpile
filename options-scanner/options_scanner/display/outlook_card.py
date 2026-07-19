"""Market View card for the Single Ticker tab.

Renders the (Direction × Option Type) -> stance mapping from
`options_scanner.market_view` as a small accent-bordered callout.
`OUTLOOK_TABLE`/`OUTLOOK_TONE_HEX` are re-exported here for existing
importers (e.g. `display/portfolio_action_card.py`) — the actual data
lives in `market_view.py` now, streamlit-free, since `mc_batch.py`'s
compute path also needs it (to derive Monte Carlo drift) and must not
import anything Streamlit-flavored.
"""

from __future__ import annotations

import streamlit as st

from options_scanner.market_view import OUTLOOK_TABLE, OUTLOOK_TONE_HEX


def render_outlook_card(buy: bool, opt_type: str) -> None:
    """Render the directional-outlook callout for the Single Ticker tab.

    Maps the user's (Direction × Option Type) selection to a structured
    market-view summary so users know what the scan is actually screening
    for. Renders as a small card in the third column of Group 2.
    """
    cfg = OUTLOOK_TABLE.get((buy, opt_type))
    if not cfg:
        return
    accent = OUTLOOK_TONE_HEX[cfg["tone"]]
    # st.markdown (not st.html) so the card lives in the main document
    # and picks up html[data-osc-theme] dark-mode rules from inject_theme().
    st.markdown(
        f"""
        <div class="mv-card" style="border-left-color:{accent};">
            <div class="mv-eyebrow">Market view</div>
            <details>
                <summary class="mv-stance" style="color:{accent};">
                    {cfg['stance']}
                    <span class="mv-hint">▾</span>
                </summary>
                <div class="mv-body">{cfg['summary']}</div>
                <div class="mv-eg">e.g. {cfg['examples']}</div>
            </details>
        </div>
        """,
        unsafe_allow_html=True,
    )
