import datetime
import math
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional: yfinance for Gold/FX if available
try:
    import yfinance as yf
except Exception:
    yf = None

COLORS = {
    "DARK_BLUE": "#0B3D91",
    "LIGHT_GREY": "#e0e0e0",
    "MID_GREY": "#9E9E9E",
    "DARK_RED": "#8B0000",
    "BLACK": "#000000",
    "ECB_OFFICIAL": "#374151",
    "GREEN": "#10B981",
    "RED": "#EF4444",
}

ECB_DATA_API = "https://data-api.ecb.europa.eu/service/data"
START_DATE_GLOBAL = "2015-01-01"
START_DATE_POLICY = "2020-01-01"

# =============================================================================
# NETWORK SESSION
# =============================================================================
def make_session() -> requests.Session:
    """Create requests session with retries (same as EU_Dashboard.py)."""
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; ECB data fetcher)",
        "Accept": "*/*",
    })
    return s

SESSION = make_session()

# =============================================================================
# ECB DATA FETCHING (CACHED)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def ecb_get_panel(
    flow: str,
    series_key: str,
    start_period: str | None = None,
    end_period: str | None = None,
    timeout: int = 60*12,
) -> pd.DataFrame:
    """Returns SDMX-CSV as a DataFrame. If 404, returns empty DataFrame."""
    url = f"{ECB_DATA_API}/{flow}/{series_key}"
    params = {"format": "csvdata"}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    try:
        r = SESSION.get(url, params=params, headers={"Accept": "text/csv"}, timeout=timeout)
        if r.status_code == 404:
            return pd.DataFrame()
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame()


def ecb_panel_to_series(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Converts panel SDMX-CSV DataFrame into standard 2-col time series."""
    if df_panel is None or df_panel.empty:
        return pd.DataFrame({"date": pd.to_datetime([]), "value": []})

    tcol = "TIME_PERIOD" if "TIME_PERIOD" in df_panel.columns else None
    vcol = "OBS_VALUE" if "OBS_VALUE" in df_panel.columns else None
    if tcol is None or vcol is None:
        return pd.DataFrame({"date": pd.to_datetime([]), "value": []})

    out = df_panel[[tcol, vcol]].rename(columns={tcol: "date", vcol: "value"}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date", "value"]).sort_values("date")


@st.cache_data(ttl=3600, show_spinner=False)
def get_ecb_series(flow: str, key: str, start_period: str) -> pd.Series:
    """Fetch a single ECB time series and return as pd.Series."""
    panel = ecb_get_panel(flow, key, start_period)
    df = ecb_panel_to_series(panel)
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["value"]


# =============================================================================
# ECB WATCH (market-implied meeting distribution) (CACHED)
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ecb_watch_probabilities(timeout: int = 12) -> dict | None:
    """
    Fetch market-implied ECB meeting rate probabilities from ecb-watch.eu.
    (Same call + headers as EU_Dashboard.py)
    """
    url = "https://ecb-watch.eu/probabilities"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; EU dashboard; +https://ecb-watch.eu/)",
        "Accept": "application/json,*/*",
    }
    try:
        r = SESSION.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _ecb_watch_percentile(probabilities: dict, percentile: float) -> float:
    """Compute a percentile from a discrete distribution { '3.75%': p, ... }."""
    if not probabilities:
        return float("nan")

    items = []
    for k, p in probabilities.items():
        try:
            rate = float(str(k).rstrip("%"))
            prob = float(p)
            if np.isfinite(rate) and np.isfinite(prob):
                items.append((rate, prob))
        except Exception:
            continue

    if not items:
        return float("nan")

    items.sort(key=lambda x: x[0])
    rates = [r for r, _ in items]
    probs = [p for _, p in items]
    total = sum(probs)
    if total <= 0:
        return float("nan")

    probs = [p / total for p in probs]

    cum = 0.0
    for i, (rate, p) in enumerate(zip(rates, probs)):
        prev_cum = cum
        cum += p
        if cum >= percentile:
            if i == 0 or cum == prev_cum:
                return rate
            w = (percentile - prev_cum) / (cum - prev_cum)
            return rates[i - 1] + w * (rates[i] - rates[i - 1])
    return rates[-1]


def _ecb_watch_mode(probabilities: dict) -> float:
    """Return the modal rate (highest probability mass point) from ECB Watch distribution."""
    if not probabilities:
        return float("nan")

    best_rate = float("nan")
    best_prob = -1.0

    for k, p in probabilities.items():
        try:
            rate = float(str(k).rstrip("%"))
            prob = float(p)
            if not (np.isfinite(rate) and np.isfinite(prob)):
                continue
            if prob > best_prob or (prob == best_prob and np.isfinite(best_rate) and rate < best_rate):
                best_prob = prob
                best_rate = rate
        except Exception:
            continue

    return best_rate


def process_ecb_watch_data(data: dict | None) -> pd.DataFrame:
    """Convert ECB Watch JSON into a DataFrame indexed by meeting date."""
    if not data or not isinstance(data, dict):
        return pd.DataFrame()

    abs_data = data.get("abs_data", {})
    if not isinstance(abs_data, dict) or not abs_data:
        return pd.DataFrame()

    rows = []
    for date_str, dist in abs_data.items():
        try:
            dt = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(dt) or not isinstance(dist, dict):
                continue

            p25 = _ecb_watch_percentile(dist, 0.25)
            p50 = _ecb_watch_percentile(dist, 0.50)
            p75 = _ecb_watch_percentile(dist, 0.75)
            mode = _ecb_watch_mode(dist)

            if not (np.isfinite(p25) and np.isfinite(p50) and np.isfinite(p75) and np.isfinite(mode)):
                continue

            rows.append(
                {
                    "date": dt,
                    "p25": float(p25),
                    "median": float(p50),
                    "p75": float(p75),
                    "mode": float(mode),
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("date").set_index("date")


# =============================================================================
# HELPERS
# =============================================================================
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fig_from_spec(data: list[dict], layout: dict, height: int) -> go.Figure:
    """Build a Plotly figure from dict specs (keeps your existing trace/layout logic).

    Legend handling:
    - Your layouts place legends below the chart using negative `legend.y`.
    - Streamlit can become scrollable when that legend falls outside the Plotly canvas.
    - We keep the legend *below the plot* (not over the data) by allocating a larger
      bottom margin and increasing the figure height accordingly, so the legend remains
      inside the Plotly canvas and inside the rounded card.
    """
    fig = go.Figure(data=data, layout=layout)

    # ---- Defensive Plotly hygiene ----
    # Plotly/Streamlit + unified hover can show a bold "undefined" when:
    #   - x-axis auto-type inference misfires for datetime-like x values, or
    #   - a helper trace has name=None.
    # We harden both cases without changing the visual output.
    try:
        if getattr(fig.layout.xaxis, "type", None) in (None, "") and fig.data:
            xs = getattr(fig.data[0], "x", None)
            if xs is not None and len(xs) > 0:
                idxs = np.linspace(0, len(xs) - 1, num=min(5, len(xs)), dtype=int)
                sample = [xs[i] for i in idxs]
                parsed = pd.to_datetime(sample, errors="coerce")
                if hasattr(parsed, "notna") and float(parsed.notna().mean()) >= 0.6:
                    fig.update_xaxes(type="date")
    except Exception:
        pass

    for tr in fig.data:
        try:
            nm = getattr(tr, "name", None)
            if nm is None or str(nm).strip().lower() == "undefined":
                tr.name = ""
                tr.showlegend = False
                tr.hoverinfo = "skip"
        except Exception:
            pass

    # Base layout
    base_margin = layout.get("margin", dict(l=60, r=60, t=30, b=70))
    fig.update_layout(
        hovermode=layout.get("hovermode", "x unified"),
        height=height,
        margin=base_margin,
    )

    # Force black font for all elements using safest update methods
    fig.update_layout(
        font=dict(color="black"),
        hoverlabel=dict(namelength=-1),
        legend_font_color="black",
    )

    # Avoid Plotly rendering a bold "undefined" title when only title styling is applied.
    # This can happen if a figure has no explicit title text but we set title font/color.
    try:
        t_text = None
        if getattr(fig.layout, "title", None) is not None:
            t_text = getattr(fig.layout.title, "text", None)

        if t_text is None or str(t_text).strip().lower() == "undefined":
            fig.update_layout(title_text="")

        fig.update_layout(title=dict(font=dict(color="black")))
    except Exception:
        pass

    # Force axis colors to black
    fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"), linecolor="black", gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"), linecolor="black", gridcolor="rgba(0,0,0,0.05)")

    # --- Keep bottom legends inside the figure canvas (no scrollbars) ---
    try:
        leg = fig.layout.legend
        if leg and getattr(leg, "orientation", None) == "h":
            ly = getattr(leg, "y", None)
            if ly is not None and float(ly) < 0:
                # Estimate required legend space (px) from number of visible legend items.
                n_items = 0
                for tr in fig.data:
                    show = getattr(tr, "showlegend", True)
                    if show is False:
                        continue
                    n_items += 1

                # Rough row estimate for a horizontal legend (responsive -> conservative).
                per_row = 4
                rows = max(1, int(math.ceil(n_items / per_row)))

                # Reserve space for: legend rows + padding + x-axis labels.
                legend_px = 26 * rows + 28
                needed_b = max(int(base_margin.get("b", 0) or 0), 70 + legend_px)

                # Increase bottom margin, and increase figure height by the same delta
                # so the *plot area* doesn't shrink (bigger white "frame" at the bottom).
                current_b = int(fig.layout.margin.b or 0)
                if needed_b > current_b:
                    delta = needed_b - current_b
                    fig.update_layout(
                        margin=dict(
                            l=int(fig.layout.margin.l or 0),
                            r=int(fig.layout.margin.r or 0),
                            t=int(fig.layout.margin.t or 0),
                            b=needed_b,
                        ),
                        height=int((fig.layout.height or height) + delta),
                    )

                # Keep legend centered; keep the negative y (below plot) as in your specs.
                # If it's *too* low, cap it so it stays within the expanded canvas.
                if float(ly) < -0.40:
                    fig.update_layout(legend=dict(y=-0.32))
    except Exception:
        pass

    return fig


def _show_chart(
    title: str,
    fig: go.Figure | None,
    observations: list[str] | None,
    sources: list[str] | None,
    label_guide: list[str] | None = None,
    *,
    show_details: bool = True,
    key: str | None = None,
):
    """Render a chart; details panel is optional (on by default)."""
    if title:
        st.subheader(title)
    if fig is None:
        st.warning("Chart unavailable (missing data).")
        return

    # Streamlit auto-generates element IDs based on the call signature.
    # If the same Plotly figure is rendered twice in a single run, Streamlit can raise
    # StreamlitDuplicateElementId. Providing an explicit unique key prevents it.
    if key is None:
        import hashlib
        key = "plotly_" + hashlib.md5(title.encode("utf-8")).hexdigest()[:10]

    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={
            "scrollZoom": False,
            "displaylogo": False,
            "responsive": True,
        },
    )

    if not show_details:
        return

    st.markdown('<div class="details-box">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Key observations")
        if observations:
            st.markdown("\n".join([f"- {o}" for o in observations]))
        else:
            st.write("—")

    with c2:
        st.markdown("#### Data sources")
        if sources:
            st.markdown("\n".join([f"- {s}" for s in sources]))
        else:
            st.write("—")

    if label_guide:
        st.markdown("---")
        st.markdown("#### Label guide")
        st.markdown("\n".join([f"- {lg}" for lg in label_guide]))

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# CHART BUILDERS (adapted from your original http.server dashboard)
# =============================================================================
def build_policy_path_fig():
    """ECB policy + yields + ECB Watch path."""
    dfr = get_ecb_series("FM", "D.U2.EUR.4F.KR.DFR.LEV", START_DATE_POLICY).dropna()
    estr = get_ecb_series("EST", "B.EU000A2X2A25.WT", START_DATE_POLICY).dropna()
    aaa_10y = get_ecb_series("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", START_DATE_POLICY).dropna()
    all_10y = get_ecb_series("YC", "B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y", START_DATE_POLICY).dropna()

    df_watch = process_ecb_watch_data(fetch_ecb_watch_probabilities())

    data = []

    if not dfr.empty:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"ECB Deposit Facility (DFR) (latest: {float(dfr.iloc[-1]):.2f}%)",
                "x": dfr.index,
                "y": dfr.values.astype(float),
                "line": {"width": 2.2, "color": COLORS["ECB_OFFICIAL"], "shape": "hv"},
            }
        )

    if not estr.empty:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"€STR (latest: {float(estr.iloc[-1]):.2f}%)",
                "x": estr.index,
                "y": estr.values.astype(float),
                "line": {"width": 1.6, "color": COLORS["MID_GREY"]},
                "opacity": 0.85,
            }
        )

    if not aaa_10y.empty:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"10Y AAA yield (latest: {float(aaa_10y.iloc[-1]):.2f}%)",
                "x": aaa_10y.index,
                "y": aaa_10y.values.astype(float),
                "line": {"width": 2.0, "color": COLORS["BLACK"]},
            }
        )

    if not all_10y.empty:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"10Y Total Euro Area yield (latest: {float(all_10y.iloc[-1]):.2f}%)",
                "x": all_10y.index,
                "y": all_10y.values.astype(float),
                "line": {"width": 1.9, "color": COLORS["DARK_RED"]},
            }
        )

    # ECB Watch band + median + mode, anchored from last €STR
    if (not df_watch.empty) and (not estr.empty):
        anchor_date = pd.to_datetime(estr.index[-1]).normalize()
        anchor_val = float(estr.iloc[-1])

        dfw = df_watch[df_watch.index >= anchor_date].copy()
        if not dfw.empty:
            path_dates = [anchor_date] + dfw.index.tolist()
            med = [anchor_val] + [float(x) for x in dfw["median"].tolist()]
            p25 = [anchor_val] + [float(x) for x in dfw["p25"].tolist()]
            p75 = [anchor_val] + [float(x) for x in dfw["p75"].tolist()]
            mode_vals = [anchor_val] + [float(x) for x in dfw["mode"].tolist()]

            # band
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "ECB Watch: 25–75 percentile band",
                    "x": path_dates,
                    "y": p25,
                    "line": {"width": 0},
                    "showlegend": False,
                    "hoverinfo": "skip",
                }
            )
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "ECB Watch: 25–75 percentile band",
                    "x": path_dates,
                    "y": p75,
                    "line": {"width": 0},
                    "fill": "tonexty",
                    "fillcolor": "rgba(11,61,145,0.18)",
                }
            )

            # median
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "ECB Watch: market-implied median",
                    "x": path_dates,
                    "y": med,
                    "line": {"width": 2.6, "color": COLORS["DARK_BLUE"]},
                }
            )
            # mode (stars)
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": "ECB Watch: market-implied mode",
                    "x": path_dates,
                    "y": mode_vals,
                    "line": {"width": 2.2, "color": COLORS["DARK_BLUE"]},
                    "marker": {"size": 10, "symbol": "star", "color": COLORS["DARK_BLUE"]},
                }
            )

    # X-axis range: show last ~6 years + some forward
    max_hist = None
    candidates = []
    for s in [dfr, estr, aaa_10y, all_10y]:
        if s is not None and not s.empty:
            candidates.append(pd.to_datetime(s.index[-1]))
    if candidates:
        max_hist = max(candidates)

    max_watch = pd.to_datetime(df_watch.index[-1]) if (df_watch is not None and not df_watch.empty) else None
    max_x = max([d for d in [max_hist, max_watch] if d is not None], default=pd.Timestamp.now())
    max_x = pd.to_datetime(max_x) + pd.Timedelta(days=60)

    start_x = pd.Timestamp(year=pd.Timestamp.now().year - 6, month=1, day=1)

    layout = {
        "xaxis": {"title": "", "range": [start_x, max_x], "tickformat": "%Y", "dtick": "M12"},
        "yaxis": {"title": "Rate / Yield (%)", "ticksuffix": "%"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 60, "t": 45, "b": 85},
    }

    fig = _fig_from_spec(data, layout, height=650)

    obs = []
    if not df_watch.empty:
        obs.append("ECB Watch meetings: median + 25–75th percentile band (implied distribution).")
    else:
        obs.append("ECB Watch path unavailable (fetch failed or empty).")

    src = [
        "ECB Data API: FM (policy rates), EST (€STR), YC (10Y yields)",
        "ECB Watch: meeting-rate probability distributions (ecb-watch.eu/probabilities)",
    ]

    return fig, obs, src



def _show_details_block(
    block_title: str,
    observations: list[str] | None,
    sources: list[str] | None,
    label_guide: list[str] | None = None,
):
    """Render a details block (observations + sources) without plotting."""
    if block_title:
        st.markdown(f"### {block_title}")

    st.markdown('<div class="details-box">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Key observations")
        if observations:
            st.markdown("\n".join([f"- {o}" for o in observations]))
        else:
            st.write("—")
    with c2:
        st.markdown("#### Data sources")
        if sources:
            st.markdown("\n".join([f"- {s}" for s in sources]))
        else:
            st.write("—")

    if label_guide:
        st.markdown("---")
        st.markdown("#### Label guide")
        st.markdown("\n".join([f"- {lg}" for lg in label_guide]))

    st.markdown("</div>", unsafe_allow_html=True)



def build_yield_curve_figs(lookback_months: int = 24, spread_years: int = 5):
    """Yield curve snapshots + sovereign spreads vs AAA and vs Total EA."""
    end_dt = pd.Timestamp.today().normalize()
    start_curve = (end_dt - pd.DateOffset(months=lookback_months + 2)).strftime("%Y-%m-%d")
    start_spread = (end_dt - pd.DateOffset(years=spread_years, months=2)).strftime("%Y-%m-%d")

    order = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    curve_total = "G_N_C"
    curve_aaa = "G_N_A"

    series_total = {}
    series_aaa = {}

    for tenor in order:
        s_total = get_ecb_series("YC", f"B.U2.EUR.4F.{curve_total}.SV_C_YM.SR_{tenor}", start_curve)
        if s_total is not None and not s_total.dropna().empty:
            series_total[tenor] = s_total.rename(tenor)

        s_aaa = get_ecb_series("YC", f"B.U2.EUR.4F.{curve_aaa}.SV_C_YM.SR_{tenor}", start_curve)
        if s_aaa is not None and not s_aaa.dropna().empty:
            series_aaa[tenor] = s_aaa.rename(tenor)

    fig1 = fig2 = fig3 = None
    obs1 = src1 = obs2 = src2 = obs3 = src3 = None

    if series_total:
        df_total = pd.concat(series_total.values(), axis=1).sort_index().dropna(how="all").ffill()
        df_total_m = df_total.resample("ME").last().dropna(how="all")
        df_total_m = df_total_m.tail(lookback_months + 1)

        if not df_total_m.empty:
            latest_total_dt = df_total_m.index[-1]
            xcats = [t for t in order if t in df_total_m.columns]

            data1 = []
            # older snapshots (fading)
            for dt in df_total_m.index[:-1]:
                row = df_total_m.loc[dt]
                data1.append(
                    {
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": dt.strftime("%Y-%m"),
                        "x": xcats,
                        "y": [float(row[t]) for t in xcats],
                        "line": {"width": 1, "color": "rgba(160,160,160,0.55)"},
                        "marker": {"size": 4, "color": "rgba(160,160,160,0.55)"},
                        "showlegend": False,
                        "hoverinfo": "skip",
                    }
                )

            # latest Total EA
            row_total_latest = df_total_m.loc[latest_total_dt]
            data1.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": f"Latest Total EA ({latest_total_dt.strftime('%Y-%m')})",
                    "x": xcats,
                    "y": [float(row_total_latest[t]) for t in xcats],
                    "line": {"width": 3.2, "color": COLORS["DARK_BLUE"]},
                    "marker": {"size": 6, "color": COLORS["DARK_BLUE"]},
                }
            )

            # latest AAA
            obs1 = []
            if series_aaa:
                df_aaa = pd.concat(series_aaa.values(), axis=1).sort_index().dropna(how="all").ffill()
                df_aaa_m = df_aaa.resample("ME").last().dropna(how="all")
                if not df_aaa_m.empty:
                    latest_aaa_dt = df_aaa_m.index[-1]
                    xcats_aaa = [t for t in xcats if t in df_aaa_m.columns]
                    if xcats_aaa:
                        row_aaa_latest = df_aaa_m.loc[latest_aaa_dt]
                        data1.append(
                            {
                                "type": "scatter",
                                "mode": "lines+markers",
                                "name": f"Latest AAA ({latest_aaa_dt.strftime('%Y-%m')})",
                                "x": xcats_aaa,
                                "y": [float(row_aaa_latest[t]) for t in xcats_aaa],
                                "line": {"width": 3.2, "color": COLORS["BLACK"]},
                                "marker": {"size": 6, "color": COLORS["BLACK"]},
                            }
                        )

                        aaa10 = _safe_float(row_aaa_latest.get("10Y"))
                        tot10 = _safe_float(row_total_latest.get("10Y"))
                        if np.isfinite(aaa10) and np.isfinite(tot10):
                            obs1.append(f"EA−AAA 10Y Spread: {((tot10 - aaa10) * 100):+.0f} bps")

            y2 = _safe_float(row_total_latest.get("2Y"))
            y10 = _safe_float(row_total_latest.get("10Y"))
            y30 = _safe_float(row_total_latest.get("30Y"))
            if np.isfinite(y2) and np.isfinite(y10):
                obs1.append(f"EA 2s10 slope: {(y10 - y2):+.2f} pp")
            if np.isfinite(y10) and np.isfinite(y30):
                obs1.append(f"EA 10s30 slope: {(y30 - y10):+.2f} pp")
            obs1.append(f"Grey lines: Total EA monthly snapshots (last {lookback_months} months).")

            src1 = [
                "ECB YC dataset: Euro area yield curves (spot rates, Svensson model)",
                "Curves shown: Total Euro Area (G_N_C) and AAA (G_N_A)",
            ]

            layout1 = {
                "xaxis": {"title": "Maturity", "type": "category"},
                "yaxis": {"title": "Yield (%)"},
                "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
                "margin": {"l": 60, "r": 60, "t": 45, "b": 85},
            }
            fig1 = _fig_from_spec(data1, layout1, height=650)

    # Spreads (monthly): country 10Y IRS - baseline 10Y YC
    countries = [
        ("Germany", "DE"),
        ("Netherlands", "NL"),
        ("Austria", "AT"),
        ("France", "FR"),
        ("Belgium", "BE"),
        ("Spain", "ES"),
        ("Portugal", "PT"),
        ("Italy", "IT"),
        ("Greece", "GR"),
        ("Ireland", "IE"),
    ]

    yields = {}
    for name, code in countries:
        s = get_ecb_series("IRS", f"M.{code}.L.L40.CI.0000.EUR.N.Z", start_spread)
        if s is None or s.dropna().empty:
            continue
        try:
            idx = pd.to_datetime(s.index).to_period("M").to_timestamp()
            s_m = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=idx).groupby(level=0).mean().sort_index()
            yields[name] = s_m.rename(name)
        except Exception:
            yields[name] = s.dropna().rename(name)

    base_aaa = get_ecb_series("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", start_spread)
    base_all = get_ecb_series("YC", "B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y", start_spread)

    def _to_monthly(s: pd.Series) -> pd.Series:
        if s is None or s.dropna().empty:
            return pd.Series(dtype=float)
        try:
            idx = pd.to_datetime(s.index).to_period("M").to_timestamp()
            out = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=idx).groupby(level=0).mean().sort_index()
            return out.dropna()
        except Exception:
            return s.dropna()

    base_aaa_m = _to_monthly(base_aaa)
    base_all_m = _to_monthly(base_all)

    def _build_spread_fig(baseline: pd.Series, baseline_name: str) -> tuple[go.Figure | None, list[str] | None, list[str] | None]:
        if not yields or baseline is None or baseline.dropna().empty:
            return None, None, None

        df_y = pd.concat(yields.values(), axis=1).sort_index().dropna(how="all")
        if df_y.empty:
            return None, None, None

        baseline = baseline.reindex(df_y.index, method="ffill")
        common = df_y.index.intersection(baseline.index)
        if common.empty:
            return None, None, None

        df_y = df_y.loc[common]
        baseline = baseline.loc[common]

        min_dt = end_dt - pd.DateOffset(years=spread_years)
        df_y = df_y[df_y.index >= min_dt].dropna(how="all")
        baseline = baseline[baseline.index >= min_dt].dropna()

        common = df_y.index.intersection(baseline.index)
        if common.empty:
            return None, None, None

        spread_bps = (df_y.loc[common].sub(baseline.loc[common], axis=0) * 100.0).dropna(how="all")
        if spread_bps.empty:
            return None, None, None

        data = []
        for col in spread_bps.columns:
            s = spread_bps[col].dropna()
            if s.empty:
                continue
            data.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": col,
                    "x": s.index,
                    "y": s.values.astype(float),
                    "line": {"width": 1.8},
                }
            )

        shapes = []
        if len(spread_bps.index) > 1:
            shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y",
                    "x0": spread_bps.index.min(),
                    "x1": spread_bps.index.max(),
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "black", "width": 1},
                }
            )

        layout = {
            "xaxis": {"title": "", "tickformat": "%Y-%m"},
            "yaxis": {"title": f"10Y spread vs {baseline_name} (bps)"},
            "shapes": shapes,
            "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.30},
            "margin": {"l": 60, "r": 60, "t": 45, "b": 95},
        }

        obs = [
            f"Spread = country 10Y (ECB IRS) − {baseline_name} 10Y (ECB YC), in bps.",
            f"History shown: last {spread_years} years (monthly).",
        ]
        src = [
            "ECB IRS dataset: Long-term interest rates (10Y) by country",
            f"ECB YC dataset: {baseline_name} 10Y (spot rate, Svensson model)",
        ]

        return _fig_from_spec(data, layout, height=600), obs, src

    fig2, obs2, src2 = _build_spread_fig(base_aaa_m, "Euro Area AAA")
    fig3, obs3, src3 = _build_spread_fig(base_all_m, "Total Euro Area")

    return (fig1, obs1, src1), (fig2, obs2, src2), (fig3, obs3, src3)

def build_money_market_fig(start_period: str):
    """Repo rates/volumes + ECB corridor + €STR + excess liquidity."""
    start_dt = pd.to_datetime(start_period)

    # Corridor + €STR
    dfr = get_ecb_series("FM", "D.U2.EUR.4F.KR.DFR.LEV", start_period)
    mro = get_ecb_series("FM", "D.U2.EUR.4F.KR.MRR_FR.LEV", start_period)
    mlf = get_ecb_series("FM", "D.U2.EUR.4F.KR.MLFR.LEV", start_period)
    estr = get_ecb_series("EST", "B.EU000A2X2A25.WT", start_period)

    # Excess liquidity: millions EUR -> EUR tn
    exliq = get_ecb_series("ILM", "D.U2.C.EXLIQ.U2.EUR", start_period)
    if exliq is not None and not exliq.dropna().empty:
        exliq = pd.to_numeric(exliq, errors="coerce").dropna().sort_index() / 1_000_000.0
    else:
        exliq = pd.Series(dtype=float)

    # MMSR secured repo (overnight), borrowing side, by collateral issuer
    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch_mmsr_single(coll_code: str, data_type: str, start_period_: str) -> pd.Series:
        key = f"B.U2._X.{coll_code}.._Z.T.BO.{data_type}..MA._Z._Z.EUR._Z"
        url = f"{ECB_DATA_API}/MMSR/{key}"
        params = {"startPeriod": start_period_, "format": "csvdata"}
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return pd.Series(dtype=float)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
                return pd.Series(dtype=float)

            out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
            out["TIME_PERIOD"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce")
            out["OBS_VALUE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
            out = out.dropna(subset=["TIME_PERIOD", "OBS_VALUE"]).sort_values("TIME_PERIOD")

            s = pd.Series(out["OBS_VALUE"].values, index=pd.DatetimeIndex(out["TIME_PERIOD"]))
            return s[~s.index.duplicated(keep="last")].sort_index()
        except Exception:
            return pd.Series(dtype=float)

    collaterals = [("Germany", "DE"), ("France", "FR"), ("Italy", "IT")]
    repo_rates = {nm: _fetch_mmsr_single(code, "WR", start_period) for nm, code in collaterals}
    repo_vols = {nm: _fetch_mmsr_single(code, "TT", start_period) for nm, code in collaterals}

    def _cut(s: pd.Series) -> pd.Series:
        if s is None or s.dropna().empty:
            return pd.Series(dtype=float)
        ss = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        ss = ss[ss.index >= start_dt]
        return ss

    data = []

    dfr_f = _cut(dfr)
    mlf_f = _cut(mlf)

    # Corridor shading (DFR–MLF)
    if not dfr_f.empty and not mlf_f.empty:
        common = dfr_f.index.intersection(mlf_f.index)
        if len(common) > 1:
            lo = dfr_f.loc[common].astype(float)
            hi = mlf_f.loc[common].astype(float)
            data.append(
                {"type": "scatter", "mode": "lines", "name": "ECB corridor (DFR–MLF)", "x": common, "y": lo.values, "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"}
            )
            data.append(
                {"type": "scatter", "mode": "lines", "name": "ECB corridor (DFR–MLF)", "x": common, "y": hi.values, "line": {"width": 0}, "fill": "tonexty", "fillcolor": "rgba(158,158,158,0.18)"}
            )

    def _add_step(name: str, s: pd.Series, color: str, width: float, dash: str | None = None):
        ss = _cut(s)
        if ss.empty:
            return
        line = {"width": float(width), "color": color, "shape": "hv"}
        if dash:
            line["dash"] = dash
        data.append({"type": "scatter", "mode": "lines", "name": name, "x": ss.index, "y": ss.values.astype(float), "line": line})

    _add_step("DFR", dfr, COLORS["ECB_OFFICIAL"], 2.0)
    _add_step("MRO", mro, COLORS["MID_GREY"], 1.6, dash="dot")
    _add_step("MLF", mlf, COLORS["BLACK"], 1.8)

    es = _cut(estr)
    if not es.empty:
        data.append({"type": "scatter", "mode": "lines", "name": "€STR", "x": es.index, "y": es.values.astype(float), "line": {"width": 2.0, "color": "#6B7280"}})

    for nm, s in repo_rates.items():
        ss = _cut(s)
        if ss.empty:
            continue
        data.append({"type": "scatter", "mode": "lines", "name": f"{nm} Repo rate", "x": ss.index, "y": ss.values.astype(float), "line": {"width": 1.7}, "opacity": 0.95})

    for nm, s in repo_vols.items():
        ss = _cut(s)
        if ss.empty:
            continue
        data.append({"type": "bar", "name": f"{nm} Repo Volume", "x": ss.index, "y": ss.values.astype(float), "yaxis": "y2", "opacity": 0.35})

    ex = _cut(exliq)
    if not ex.empty:
        data.append(
            {"type": "scatter", "mode": "lines", "name": "Excess liquidity (EUR tn)", "x": ex.index, "y": ex.values.astype(float), "yaxis": "y3", "line": {"width": 1.6, "dash": "dash", "color": COLORS["DARK_BLUE"]}, "opacity": 0.95}
        )

    # X-range
    max_dt = start_dt
    for tr in data:
        xs = tr.get("x")
        if isinstance(xs, (pd.DatetimeIndex, list)) and len(xs) > 0:
            try:
                max_dt = max(max_dt, pd.to_datetime(xs[-1]))
            except Exception:
                pass

    layout = {
        "xaxis": {"title": "", "tickformat": "%Y-%m", "range": [start_dt, max_dt], "domain": [0.0, 0.86]},
        "yaxis": {"title": "Rate (%)"},
        "yaxis2": {"title": "Volume (EUR mn)", "overlaying": "y", "side": "right", "showgrid": False, "anchor": "free", "position": 0.89},
        "yaxis3": {"title": "Excess liquidity (EUR tn)", "overlaying": "y", "side": "right", "showgrid": False, "anchor": "free", "position": 0.98},
        "barmode": "stack",
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.32},
        "margin": {"l": 60, "r": 110, "t": 45, "b": 100},
    }

    fig = _fig_from_spec(data, layout, height=700)

    obs = [
        "DFR - UNSEC: [Deposit Facility Rate] Interest Rate Banks receive on overnight deposits with Eurosystem - Primary anchor for short-term money market rates.",
        "€STR - UNSEC: [Euro Short-Term Rate] Benchmark for average interest rate for Unsec overnight borrowing of Banks in EA.",
        "MLF SEC: [Marginal Lending Facility] Penalty Interest Rate Banks pay to borrow overnight from Eurosystem - Aboslute ceiling of the policy corridor.",
        "MRO SEC: [Main Refinancing Operations] Interest Rate for one-week liquidity - Liquidity backstop for Banking Sector.",
        "Excess Liquidity: Volume of Central Bank Reserves held by the banking system (in excess of minimum requirements and autonomous factors) - High levels drive market rates toward the floor of the corridor.",
        "ECB Corridor: Spread between the MLF (ceiling) and the DFR (floor).",
        "Repo volumes are stacked (Germany + France + Italy) to show total secured borrowing volume across these collateral buckets.",
        "Repo rates are secured borrowing repo rates by collateral issuer (DE/FR/IT), alongside ECB corridor and €STR.",
        "Excess liquidity is converted to EUR tn (from ECB ILM series).",
    ]

    src = [
        "ECB MMSR dataset (secured segment): repo rates (WR) and total volumes (TT) by collateral issuer (DE/FR/IT), borrowing side",
        "ECB FM dataset: ECB policy corridor (DFR, MRO, MLF)",
        "ECB EST dataset: €STR overnight rate",
        "ECB ILM dataset: Excess liquidity (Eurosystem)",
    ]

    return fig, obs, src

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yf(symbols: list[str], start: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbols, start=start, progress=False)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_eurostat_net_flow(start_date: str) -> pd.Series:
    """Eurostat JSON-stat: ext_st_easitc (EA20 trade, SITC) -> monthly net flow (exports − imports)."""
    BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    DATASET = "ext_st_easitc"

    base_params = {
        "format": "JSON",
        "lang": "EN",
        "freq": "M",
        "geo": "EA20",
        "partner": "EXT_EA20",
    }

    def _get(params: dict) -> dict:
        r = SESSION.get(f"{BASE}/{DATASET}", params=params, timeout=45)
        if r.status_code != 200:
            raise RuntimeError(f"Eurostat HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def _ordered_codes(cat: dict) -> list[str]:
        idx = cat.get("index")
        if isinstance(idx, dict):
            return sorted(idx.keys(), key=lambda k: idx[k])
        if isinstance(idx, list):
            return idx
        return list(cat.get("label", {}).keys())

    def _pick_code_by_label(dim: dict, must_contain: list[str]) -> str | None:
        labels = dim.get("category", {}).get("label", {})
        for code, lab in labels.items():
            low = str(lab).lower()
            if all(k in low for k in must_contain):
                return code
        return None

    def _jsonstat_single_series_to_pd(j: dict) -> pd.Series:
        ids = j["id"]
        sizes = j["size"]
        dims = j["dimension"]

        time_id = None
        for cand in ids:
            if cand.lower() in ("time", "time_period", "timeperiod"):
                time_id = cand
                break
        if time_id is None:
            raise ValueError("Could not find time dimension in JSON-stat response.")

        time_cat = dims[time_id]["category"]
        time_codes = _ordered_codes(time_cat)

        values = j["value"]
        n = int(np.prod(sizes))

        if isinstance(values, list):
            arr = values
        elif isinstance(values, dict):
            arr = [np.nan] * n
            for k, v in values.items():
                arr[int(k)] = v
        else:
            raise ValueError("Unexpected JSON-stat 'value' format.")

        arr = arr[: len(time_codes)]

        def _parse_time(code: str) -> pd.Timestamp:
            code = str(code).strip()
            if len(code) == 7 and code[4] == "M":
                code = f"{code[:4]}-{code[5:]}"
            return pd.to_datetime(code, errors="coerce") + pd.offsets.MonthEnd(0)

        idx = pd.to_datetime([_parse_time(c) for c in time_codes])
        return pd.Series(pd.to_numeric(arr, errors="coerce"), index=idx).dropna().sort_index()

    # Discovery to identify codes robustly
    try:
        meta = _get({**base_params, "sitc06": "SITC9", "lastTimePeriod": 1})
    except Exception:
        try:
            meta = _get({**base_params, "lastTimePeriod": 1})
        except Exception:
            return pd.Series(dtype=float)

    dims = meta.get("dimension", {})

    sitc_code = "SITC9"
    if "sitc06" in dims:
        labels = dims["sitc06"]["category"]["label"]
        if "SITC9" not in labels:
            found = _pick_code_by_label(dims["sitc06"], ["commodities", "transactions"])
            if found:
                sitc_code = found

    indic_code = None
    if "indic_et" in dims:
        indic_code = (
            _pick_code_by_label(dims["indic_et"], ["trade value", "million"])
            or _pick_code_by_label(dims["indic_et"], ["value", "million"])
            or _pick_code_by_label(dims["indic_et"], ["trade value"])
        )
    if not indic_code:
        return pd.Series(dtype=float)

    bal_code = imp_code = exp_code = None
    if "stk_flow" in dims:
        bal_code = _pick_code_by_label(dims["stk_flow"], ["balance"])
        imp_code = _pick_code_by_label(dims["stk_flow"], ["import"])
        exp_code = _pick_code_by_label(dims["stk_flow"], ["export"])

    final_params = {**base_params, "sitc06": sitc_code, "indic_et": indic_code, "sinceTimePeriod": start_date[:7]}

    try:
        if bal_code:
            j_bal = _get({**final_params, "stk_flow": bal_code})
            return _jsonstat_single_series_to_pd(j_bal)

        if imp_code and exp_code:
            j_exp = _get({**final_params, "stk_flow": exp_code})
            j_imp = _get({**final_params, "stk_flow": imp_code})
            s_exp = _jsonstat_single_series_to_pd(j_exp)
            s_imp = _jsonstat_single_series_to_pd(j_imp)
            return (s_exp - s_imp).dropna()

        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def build_gold_figs():
    """Gold in EUR + trend band + M2 + M2 YoY (top) and net trade flow (bottom)."""
    if yf is None:
        return None, None, ["Install `yfinance` to load gold & FX data."], ["Yahoo Finance (via yfinance)"]

    # 1) Gold in EUR (daily)
    data = _fetch_yf(["GC=F", "EURUSD=X"], start=START_DATE_GLOBAL)
    if data is None or data.empty:
        return None, None, ["Gold/FX data unavailable."], ["Yahoo Finance (via yfinance)"]

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        try:
            closes = data.xs("Close", level=0, axis=1)
        except Exception:
            closes = data["Close"] if "Close" in data else data
    else:
        closes = data["Close"] if "Close" in data.columns else data

    gold_usd = closes.get("GC=F")
    eurusd = closes.get("EURUSD=X")
    if gold_usd is None or eurusd is None:
        return None, None, ["Gold/FX data unavailable."], ["Yahoo Finance (via yfinance)"]

    gold_eur = (gold_usd / eurusd).ffill().dropna()
    if gold_eur.empty:
        return None, None, ["Gold/FX data unavailable."], ["Yahoo Finance (via yfinance)"]

    # Display window: last ~3 years
    end_display = gold_eur.index.max()
    display_start = end_display - pd.DateOffset(years=3)
    gold = gold_eur[gold_eur.index >= display_start].dropna()
    if gold.empty:
        return None, None, ["Gold/FX data unavailable."], ["Yahoo Finance (via yfinance)"]

    # Exponential trend band (US method)
    days = (gold.index - gold.index[0]).days.values.astype(float)
    b1, b0 = np.polyfit(days, np.log(gold.values.astype(float)), 1)
    trend = np.exp(b0 + b1 * days)
    lower = trend * 0.90
    upper = trend * 1.10

    # 2) M2 stock + YoY
    m2_growth_key = "M.U2.Y.V.M20.X.I.U2.2300.Z01.A"
    m2_stock_key = "M.U2.Y.V.M20.X.1.U2.2300.Z01.E"
    m2_growth = get_ecb_series("BSI", m2_growth_key, START_DATE_GLOBAL)
    m2_stock = get_ecb_series("BSI", m2_stock_key, START_DATE_GLOBAL)

    if not m2_stock.empty:
        m2_stock.index = pd.to_datetime(m2_stock.index) + pd.offsets.MonthEnd(0)
        m2_stock = pd.to_numeric(m2_stock, errors="coerce").dropna().sort_index()
        m2_tn = (m2_stock / 1_000_000.0)
        m2_tn = m2_tn[m2_tn.index >= display_start]
    else:
        m2_tn = pd.Series(dtype=float)

    if not m2_growth.empty:
        m2_growth.index = pd.to_datetime(m2_growth.index) + pd.offsets.MonthEnd(0)
        m2_yoy = pd.to_numeric(m2_growth, errors="coerce").dropna().sort_index()
        m2_yoy = m2_yoy[m2_yoy.index >= display_start]
    else:
        m2_yoy = pd.Series(dtype=float)

    # 3) Net trade flow (EUR bn)
    net_flow_m = _fetch_eurostat_net_flow(START_DATE_GLOBAL)
    if net_flow_m is not None and not net_flow_m.dropna().empty:
        net_flow_bn = (net_flow_m / 1000.0).dropna()
        net_flow_bn = net_flow_bn[net_flow_bn.index >= display_start]
    else:
        net_flow_bn = pd.Series(dtype=float)

    # --- TOP FIG ---
    data_top = [
        {"type": "scatter", "mode": "lines", "name": "Trend band (±10%)", "x": gold.index, "y": lower, "line": {"width": 0}, "showlegend": False, "hoverinfo": "skip"},
        {"type": "scatter", "mode": "lines", "name": "Trend band (±10%)", "x": gold.index, "y": upper, "line": {"width": 0}, "fill": "tonexty", "fillcolor": "rgba(158,158,158,0.18)"},
        {"type": "scatter", "mode": "lines", "name": "Gold Exp. trend", "x": gold.index, "y": trend, "line": {"dash": "dot", "width": 1.4, "color": COLORS["MID_GREY"]}},
        {"type": "scatter", "mode": "lines", "name": "Gold (EUR/oz)", "x": gold.index, "y": gold.values.astype(float), "line": {"width": 1.9, "color": COLORS["BLACK"]}},
    ]

    if not m2_tn.dropna().empty:
        data_top.append({"type": "scatter", "mode": "lines", "name": "M2 stock (EUR tn)", "x": m2_tn.dropna().index, "y": m2_tn.dropna().values.astype(float), "yaxis": "y2", "line": {"dash": "dash", "width": 1.6, "color": COLORS["DARK_BLUE"]}})

    if not m2_yoy.dropna().empty:
        data_top.append({"type": "bar", "name": "M2 YoY (%)", "x": m2_yoy.dropna().index, "y": m2_yoy.dropna().values.astype(float), "yaxis": "y3", "opacity": 0.25})

    layout_top = {
        "xaxis": {"title": "", "tickformat": "%b %Y", "domain": [0.0, 0.86]},
        "yaxis": {"title": "Gold (EUR/oz)"},
        "yaxis2": {"title": "M2 (EUR tn)", "overlaying": "y", "side": "right", "showgrid": False, "anchor": "free", "position": 0.89},
        "yaxis3": {"title": "M2 YoY (%)", "overlaying": "y", "side": "right", "showgrid": False, "anchor": "free", "position": 0.98},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22},
        "margin": {"l": 60, "r": 110, "t": 45, "b": 85},
    }
    fig_top = _fig_from_spec(data_top, layout_top, height=600)

    # --- BOTTOM FIG ---
    fig_bot = None
    if not net_flow_bn.dropna().empty:
        sflow = net_flow_bn.dropna()
        data_bot = [{"type": "bar", "name": "Net trade flow", "x": sflow.index, "y": sflow.values.astype(float), "opacity": 0.75}]

        shapes = []
        if len(sflow.index) > 1:
            shapes.append({"type": "line", "xref": "x", "yref": "y", "x0": sflow.index.min(), "x1": sflow.index.max(), "y0": 0, "y1": 0, "line": {"color": "black", "width": 1}})

        layout_bot = {
            "xaxis": {"title": "", "tickformat": "%b %Y", "domain": [0.0, 0.86]},
            "yaxis": {"title": "Net trade flow (EUR bn)"},
            "shapes": shapes,
            "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
            "margin": {"l": 60, "r": 110, "t": 45, "b": 85},
        }
        fig_bot = _fig_from_spec(data_bot, layout_bot, height=420)

    obs = [
        "Net trade flow is computed as exports − imports (EA20 vs rest of world); positive values indicate net exports.",
    ]

    src = [
        "Yahoo Finance (via yfinance): GC=F (gold futures), EURUSD=X (FX rate)",
        "ECB BSI dataset: M2 stock and annual growth rate",
        "Eurostat dataset: ext_st_easitc (EA20 trade, SITC); net flow = exports − imports",
    ]

    return fig_top, fig_bot, obs, src


# =============================================================================
# STREAMLIT APP
# =============================================================================
def _inject_light_ui():
    # --- PAGE CONFIG & STYLING ---
    st.set_page_config(page_title="EU Dashboard", page_icon="🇪🇺", layout="wide")

    st.markdown(
        """
        <style>
            .stApp { background: #F7F8FB; }
.block-container { padding-top: 1.25rem; padding-bottom: 2.25rem; max-width: 1200px; }
          h1, h2, h3 { letter-spacing: -0.01em; }
          
          /* Show header so the sidebar toggle is visible on mobile */
          header { visibility: visible !important; height: auto !important; }
          footer { visibility: hidden; height: 0; }
          
          /* Fixed sidebar width for readability and stability */
          [data-testid="stSidebar"] {
              min-width: 260px;
              max-width: 300px;
          }


          /* Sticky Navigation Bar */
          div[data-testid="stRadio"] {
            position: sticky;
            top: 2.85rem;
            z-index: 1000;
            background: #F7F8FB;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
          }

          /* Adjust block container for sticky header */
          .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Removed duplicate title


def _series_last_and_prev(s: pd.Series) -> tuple[pd.Timestamp | None, float | None, pd.Timestamp | None, float | None]:
    """Return (last_date, last_val, prev_date, prev_val) for a series."""
    if s is None:
        return None, None, None, None
    ss = pd.to_numeric(s, errors="coerce").dropna()
    if ss.empty:
        return None, None, None, None
    last_dt = pd.to_datetime(ss.index[-1])
    last_val = float(ss.iloc[-1])
    if len(ss) >= 2:
        prev_dt = pd.to_datetime(ss.index[-2])
        prev_val = float(ss.iloc[-2])
        return last_dt, last_val, prev_dt, prev_val
    return last_dt, last_val, None, None


def _metric_from_series(
    name: str,
    s: pd.Series,
    unit: str = "%",
    decimals: int = 2,
    *,
    delta_mode: str = "diff",  # "diff" (default) or "pct"
) -> dict | None:
    """Build a st.metric payload with explicit as-of and comparison date."""
    last_dt, last_val, prev_dt, prev_val = _series_last_and_prev(s)
    if last_dt is None or last_val is None:
        return None

    label = f"{name} (as of {last_dt.strftime('%Y-%m-%d')})"
    value = f"{last_val:.{decimals}f}{unit}"

    delta = None
    if prev_dt is not None and prev_val is not None and np.isfinite(prev_val) and prev_val != 0:
        if delta_mode == "pct":
            d = (last_val - prev_val) / prev_val * 100.0
            delta = f"{d:+.{max(1, min(2, decimals))}f}% vs {prev_dt.strftime('%Y-%m-%d')}"
        else:
            d = last_val - prev_val
            suffix = " pp" if unit == "%" else ""
            delta = f"{d:+.{decimals}f}{suffix} vs {prev_dt.strftime('%Y-%m-%d')}"

    return {"label": label, "value": value, "delta": delta}


@st.cache_data(ttl=1800, show_spinner=False)
def load_dashboard_data() -> dict:
    """Prefetch all datasets so navigation doesn't trigger downloads (data calls match EU_Dashboard.py)."""
    # KPI series
    dfr = get_ecb_series("FM", "D.U2.EUR.4F.KR.DFR.LEV", START_DATE_POLICY).dropna()
    estr = get_ecb_series("EST", "B.EU000A2X2A25.WT", START_DATE_POLICY).dropna()
    aaa_10y = get_ecb_series("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", START_DATE_POLICY).dropna()

    # Optional FX and gold spot (fast window, for KPIs only)
    fx_eurusd = pd.Series(dtype=float)
    gold_eur = pd.Series(dtype=float)
    if yf is not None:
        start = (pd.Timestamp.today().normalize() - pd.DateOffset(days=40)).strftime("%Y-%m-%d")
        yfdata = _fetch_yf(["EURUSD=X", "GC=F"], start=start)
        if yfdata is not None and not yfdata.empty:
            if isinstance(yfdata.columns, pd.MultiIndex):
                try:
                    closes = yfdata.xs("Close", level=0, axis=1)
                except Exception:
                    closes = yfdata["Close"] if "Close" in yfdata else yfdata
            else:
                closes = yfdata["Close"] if "Close" in yfdata.columns else yfdata

            fx = closes.get("EURUSD=X")
            gc = closes.get("GC=F")
            if fx is not None:
                fx_eurusd = pd.to_numeric(fx, errors="coerce").dropna()
            if fx is not None and gc is not None:
                g = pd.to_numeric(gc, errors="coerce")
                fxv = pd.to_numeric(fx, errors="coerce")
                gold_eur = (g / fxv).dropna()

    # Figures (fixed windows; no user parameters)
    policy_fig, policy_obs, policy_src = build_policy_path_fig()

    # Yield curve + spreads (defaults match EU_Dashboard.py: last 24 months / last 5 years)
    yc_pack, sp_aaa_pack, sp_all_pack = build_yield_curve_figs()

    # Money market (same two windows as EU_Dashboard.py)
    mm_start_long = "2020-01-01"
    mm_start_short = "2025-01-01"
    mm_fig_long, mm_obs_long, mm_src_long = build_money_market_fig(mm_start_long)
    mm_fig_short, mm_obs_short, mm_src_short = build_money_market_fig(mm_start_short)

    # Gold & liquidity
    gold_top, gold_bot, gold_obs, gold_src = build_gold_figs()

    # KPI payloads
    kpis = []
    for payload in [
        _metric_from_series("DFR", dfr, unit="%", decimals=2),
        _metric_from_series("€STR", estr, unit="%", decimals=2),
        _metric_from_series("10Y AAA", aaa_10y, unit="%", decimals=2),
        _metric_from_series("EUR/USD", fx_eurusd, unit="", decimals=4, delta_mode="pct") if not fx_eurusd.empty else None,
        _metric_from_series("Gold (EUR/oz)", gold_eur, unit="", decimals=0, delta_mode="pct") if not gold_eur.empty else None,
    ]:
        if payload:
            kpis.append(payload)

    return {
        "kpis": kpis,
        "policy": (policy_fig, policy_obs, policy_src),
        "yield_curve": yc_pack,
        "spreads_aaa": sp_aaa_pack,
        "spreads_all": sp_all_pack,
        "money_market_long": (mm_fig_long, mm_obs_long, mm_src_long, mm_start_long),
        "money_market_short": (mm_fig_short, mm_obs_short, mm_src_short, mm_start_short),
        "gold": (gold_top, gold_bot, gold_obs, gold_src),
    }

def main():
    _inject_light_ui()

    with st.spinner("Loading data (cached)…"):
        data = load_dashboard_data()

    # Sidebar
    st.sidebar.title("Controls")
    if st.sidebar.button("Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: data is prefetched so switching sections is instant.")

    # Navigation removed from sidebar (duplicates main tabs)

    # Header row
    h1, h2, h3 = st.columns([0.72, 0.18, 0.10], vertical_alignment="bottom")
    with h1:
        st.title("🇪🇺 European Union Macro Dashboard")
    with h2:
        st.caption(f"Local time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with h3:
        if st.button("↻ Refresh"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading data (cached)…"):
        data = load_dashboard_data()

    # Removed KPI row

    st.markdown("---")

    # Navigation (radio as pill tabs)
    page = st.radio(
        "Navigation",
        ["Monetary Policy", "Yield Curve", "Money Market", "Gold & Liquidity"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Removed Overview section logic

    # ===== MONETARY POLICY =====
    if page == "Monetary Policy":
        fig, obs, src = data["policy"]
        _show_chart("Monetary Policy Path: €STR vs Market-implied path", fig, obs, src)

    # ===== YIELD CURVE =====
    elif page == "Yield Curve":
        fig, obs, src = data["yield_curve"]
        _show_chart("Euro Area Yield Curve (Monthly Snapshots, last 24 months)", fig, obs, src)

        st.markdown("---")
        fig, obs, src = data["spreads_aaa"]
        _show_chart("Euro Area 10Y Sovereign Spreads vs Euro Area AAA", fig, obs, src)

        st.markdown("---")
        fig, obs, src = data["spreads_all"]
        _show_chart("Euro Area 10Y Sovereign Spreads vs Euro Area", fig, obs, src)

    # ===== MONEY MARKET =====
    elif page == "Money Market":
        fig_l, obs_l, src_l, start_l = data["money_market_long"]
        fig_s, obs_s, src_s, start_s = data["money_market_short"]

        st.subheader("Euro Money Market: Repo Rates, ECB Corridor & Excess Liquidity")
        # Plot long window
        _show_chart("", fig_l, None, None, show_details=False, key=f"mm_long_{start_l}")

        st.markdown("---")
        # Plot short window
        _show_chart("", fig_s, None, None, show_details=False, key=f"mm_short_{start_s}")

        st.markdown('<div class="details-box">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Key observations")
            # Merged unique observations to avoid repeating the repo/liquidity boilerplate
            all_obs = []
            if obs_l: all_obs.extend(obs_l)
            if obs_s: all_obs.extend(obs_s)
            
            # Remove the boilerplate from individual observations if it snuck in
            boilerplate = [
                "Repo volumes are stacked (Germany + France + Italy) to show total secured borrowing volume across these collateral buckets.",
                "Repo rates are secured borrowing repo rates by collateral issuer (DE/FR/IT), alongside ECB corridor and €STR.",
                "Excess liquidity is converted to EUR tn (from ECB ILM series)."
            ]
            clean_obs = [o for o in all_obs if o not in boilerplate]
            
            # Show boilerplate once at the top of observations or as a caption
            st.markdown("\n".join([f"- {o}" for o in boilerplate]))
            st.markdown("---")
            
            if clean_obs:
                # Remove duplicates while preserving order
                seen = set()
                uniq_obs = []
                for o in clean_obs:
                    if o not in seen:
                        seen.add(o)
                        uniq_obs.append(o)
                st.markdown("\n".join([f"- {o}" for o in uniq_obs]))
            elif not boilerplate:
                st.write("—")

        with c2:
            st.markdown("#### Data sources")
            src_all = sorted(list(set((src_l or []) + (src_s or []))))
            if src_all:
                st.markdown("\n".join([f"- {s}" for s in src_all]))
            else:
                st.write("—")

        st.markdown("</div>", unsafe_allow_html=True)


    # ===== GOLD & LIQUIDITY =====
    elif page == "Gold & Liquidity":
        st.header("Gold & Liquidity")

        gold_top, gold_bot, obs, src = data["gold"]
        if gold_top is None:
            st.info("Install `yfinance` to enable Gold/FX panels.")
            return

        st.subheader("Gold vs Liquidity (M2)")
        _show_chart("", gold_top, None, None, show_details=False, key="gold_top")

        if gold_bot is None:
            st.info("Trade flow chart unavailable (Eurostat fetch failed).")
        else:
            st.markdown("---")
            st.subheader("Non-Monetary Gold Net Trade Flow")
            _show_chart("", gold_bot, None, None, show_details=False, key="gold_bot")

        st.markdown("---")

        st.markdown('<div class="details-box">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Key observations")
            if obs:
                st.markdown("\n".join([f"- {o}" for o in obs]))
            else:
                st.write("—")

        with c2:
            st.markdown("#### Data sources")
            if src:
                st.markdown("\n".join([f"- {s}" for s in src]))
            else:
                st.write("—")

        st.markdown("</div>", unsafe_allow_html=True)


    else:
        st.error(f"Unknown page selection: {page}")

if __name__ == "__main__":
    main()