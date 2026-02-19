import datetime
import math
import gzip
import io
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
START_DATE_GLOBAL = "2015-01-01"
START_DATE_POLICY = "2020-01-01"

# =============================================================================
# FINANCIAL CONDITIONS & LIQUIDITY (ECB)
# Stress proxy: CISS (ECB)
# Net Liquidity proxy (Millions EUR) = Assets - GovDeposits - DebtCert - FixedTermDeposits
# Smoothing: 5-day average for Net Liquidity proxy
# Start: 2020-01-01
# =============================================================================
FINCOND_START_DATE = "2020-01-01"
FINCOND_LIQ_SMOOTH_DAYS = 5

# --- Series keys ---
FINCOND_CISS_KEY = "D.U2.Z0Z.4F.EC.SS_CIN.IDX"              # CISS / daily
FINCOND_ASSETS_KEY = "W.U2.C.T000000.Z5.Z01"               # Total assets/liabilities / weekly / mn EUR
FINCOND_GOVDEP_EOP_KEY = "M.U2.C.L050100.U2.EUR"           # Central gov deposits (EOP) / monthly / mn EUR
FINCOND_GOVDEP_MP_KEY  = "M.U2.C.L050100MP.U2.EUR"         # Central gov deposits (maint. period avg) / monthly / mn EUR
FINCOND_DEBTCERT_KEY = "W.U2.C.L040000.Z5.EUR"             # Debt certificates issued / weekly / mn EUR
FINCOND_FTD_KEY      = "M.U2.C.L020300.U2.EUR"             # Fixed-term deposits / monthly / mn EUR


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


def _tenor_to_years(tenor: str) -> float:
    """Convert tenor label like 3M/1Y/30Y to maturity in years."""
    t = str(tenor).strip().upper()
    if t.endswith("M"):
        return float(t[:-1]) / 12.0
    if t.endswith("Y"):
        return float(t[:-1])
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
                # Guard: numeric x values (e.g., horizontal bars) must stay linear, not date.
                is_numeric_sample = all(
                    isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                    for v in sample
                )
                if not is_numeric_sample:
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
    fig.update_xaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        linecolor="black",
        gridcolor="rgba(0,0,0,0.05)",
        hoverformat="%Y-%m-%d",
    )
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
            xvals = [_tenor_to_years(t) for t in xcats]
            tenor_map = {float(x): t for x, t in zip(xvals, xcats)}
            tenor_labels = [tenor_map.get(float(x), str(x)) for x in xvals]
            curve_hover = "%{y:.2f}%<extra>%{fullData.name}</extra>"

            data1 = []
            # older snapshots (fading)
            for dt in df_total_m.index[:-1]:
                row = df_total_m.loc[dt]
                data1.append(
                    {
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "",
                        "x": xvals,
                        "y": [float(row[t]) for t in xcats],
                        "line": {"width": 1, "color": "rgba(160,160,160,0.55)"},
                        "marker": {"size": 4, "color": "rgba(160,160,160,0.55)"},
                        "showlegend": False,
                        "hoverinfo": "skip",
                    }
                )

            # latest Total EA
            row_total_latest = df_total_m.loc[latest_total_dt]
            # Invisible helper trace: prints mapped tenor labels (3M/6M/10Y) in unified hover.
            data1.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": "",
                    "x": xvals,
                    "y": [float(row_total_latest[t]) for t in xcats],
                    "text": tenor_labels,
                    "hovertemplate": "Tenor %{text}<extra></extra>",
                    "line": {"width": 0, "color": "rgba(0,0,0,0)"},
                    "showlegend": False,
                }
            )
            data1.append(
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": f"Latest Total EA ({latest_total_dt.strftime('%Y-%m')})",
                    "x": xvals,
                    "y": [float(row_total_latest[t]) for t in xcats],
                    "text": xcats,
                    "customdata": xcats,
                    "hovertemplate": curve_hover,
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
                        xvals_aaa = [_tenor_to_years(t) for t in xcats_aaa]
                        row_aaa_latest = df_aaa_m.loc[latest_aaa_dt]
                        data1.append(
                            {
                                "type": "scatter",
                                "mode": "lines+markers",
                                "name": f"Latest AAA ({latest_aaa_dt.strftime('%Y-%m')})",
                                "x": xvals_aaa,
                                "y": [float(row_aaa_latest[t]) for t in xcats_aaa],
                                "text": xcats_aaa,
                                "customdata": xcats_aaa,
                                "hovertemplate": curve_hover,
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
                "hovermode": "x unified",
                "xaxis": {
                    "title": "Maturity (years)",
                    "type": "linear",
                    "range": [0.0, (max(xvals) * 1.03) if xvals else 30.0],
                    "unifiedhovertitle": {"text": ""},
                    "tickangle": 0,
                    "tickfont": {"size": 11},
                    "automargin": True,
                },
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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fred_series_public(series_id: str, start_date: str = "1900-01-01") -> pd.Series:
    """Public FRED fetch via fredgraph CSV endpoint (no API key required)."""
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    params = {"id": str(series_id).strip(), "cosd": start_date, "coed": end_date}

    try:
        r = SESSION.get(FRED_GRAPH_URL, params=params, timeout=50)
        r.raise_for_status()
        text = (r.text or "").strip()
        if not text or text.lower().startswith("<!doctype html") or text.lower().startswith("<html"):
            return pd.Series(dtype=float)

        df = pd.read_csv(StringIO(text))
        if df.empty:
            return pd.Series(dtype=float)

        date_col = "DATE" if "DATE" in df.columns else df.columns[0]
        val_col = series_id if series_id in df.columns else df.columns[-1]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[date_col, val_col])
        if df.empty:
            return pd.Series(dtype=float)
        return df.set_index(date_col)[val_col].sort_index()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def build_cross_country_sectoral_debt_latest_fig():
    """Cross-country stacked debt to GDP chart (latest available quarter)."""
    countries = {
        "US": "USA",
        "XM": "Euro Area",
        "GB": "UK",
        "JP": "Japan",
        "DE": "Germany",
        "FR": "France",
        "IT": "Italy",
    }
    sectors = {
        "G": "Government",
        "H": "Households",
        "N": "Non-Fin Corps",
    }
    colors = {
        "Government": "#1f77b4",
        "Households": "#2ca02c",
        "Non-Fin Corps": "#d62728",
    }

    # Pull only a 3-year window and take the latest non-null point per series.
    end_date = pd.Timestamp.today().normalize()
    start_date = (end_date - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    tickers = {}
    for c_code, c_name in countries.items():
        for s_code, s_name in sectors.items():
            sid = f"Q{c_code}{s_code}AM770A"
            tickers[sid] = {"Country": c_name, "Sector": s_name}

    results = []
    latest_dates = []
    for sid, meta in tickers.items():
        s = fred_series_public(sid, start_date=start_date)
        valid = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        if valid.empty:
            continue
        results.append(
            {
                "Country": meta["Country"],
                "Sector": meta["Sector"],
                "Debt_to_GDP": float(valid.iloc[-1]),
            }
        )
        latest_dates.append(pd.to_datetime(valid.index[-1]))

    src = [
        "FRED BIS credit database series: Q[Country][Sector]AM770A (Government, Households, Non-Financial Corporates)",
    ]
    if not results:
        return None, ["Cross-country sectoral debt data unavailable."], src

    df_results = pd.DataFrame(results)
    df_pivot = df_results.pivot(index="Country", columns="Sector", values="Debt_to_GDP")
    if df_pivot.empty:
        return None, ["Cross-country sectoral debt data unavailable after pivoting."], src

    df_pivot["Total"] = df_pivot.sum(axis=1, min_count=1)
    df_pivot = df_pivot.sort_values(by="Total", ascending=True).drop(columns=["Total"])

    data = []
    for sector in ["Government", "Households", "Non-Fin Corps"]:
        if sector not in df_pivot.columns:
            continue
        s = pd.to_numeric(df_pivot[sector], errors="coerce").fillna(0.0)
        data.append(
            {
                "type": "bar",
                "name": sector,
                "x": s.values.astype(float),
                "y": list(df_pivot.index),
                "orientation": "h",
                "marker": {"color": colors.get(sector, COLORS["MID_GREY"])},
                "opacity": 0.9,
                "width": 0.7,
            }
        )

    if not data:
        return None, ["Cross-country sectoral debt data unavailable after cleaning."], src

    reported_period = max(latest_dates).strftime("%Y-%m") if latest_dates else "n/a"
    layout = {
        "title": {"text": f"{reported_period}"},
        "barmode": "stack",
        "xaxis": {
            "title": "Total Debt (% of GDP)",
            "type": "linear",
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.12)",
            "griddash": "dash",
        },
        "yaxis": {"title": "", "categoryorder": "array", "categoryarray": list(df_pivot.index)},
        "legend": {
            "title": {"text": "Sector (Excl. Financials)"},
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.25,
        },
        "margin": {"l": 120, "r": 40, "t": 45, "b": 95},
    }
    fig = _fig_from_spec(data, layout, height=560)

    obs = [
        f"Latest available reporting period across included series: {reported_period}.",
        "Stacked components: government, households, and non-financial corporates debt.",
        "Values are BIS debt to GDP ratios distributed through FRED.",
    ]
    return fig, obs, src


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def build_historical_niip_gdp_fig():
    """Historical NIIP (% of GDP) for major EU economies from Eurostat."""
    url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/TIPSII40/?format=SDMX-CSV"
    src = ["Eurostat SDMX API dataset TIPSII40 (NIIP, % of GDP)"]

    try:
        r = SESSION.get(url, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return None, ["EU historical NIIP data unavailable (Eurostat fetch failed)."], src

    if df.empty:
        return None, ["EU historical NIIP data unavailable (empty Eurostat response)."], src

    df.columns = [str(c).upper() for c in df.columns]
    required = {"GEO", "TIME_PERIOD", "OBS_VALUE"}
    if not required.issubset(set(df.columns)):
        return None, [f"EU historical NIIP schema mismatch. Columns found: {', '.join(df.columns)}"], src

    target_countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "DK", "PT", "PL"]
    df_f = df[df["GEO"].isin(target_countries)][["GEO", "TIME_PERIOD", "OBS_VALUE"]].copy()
    if df_f.empty:
        return None, ["EU historical NIIP data unavailable after country filtering."], src

    df_f["OBS_VALUE"] = pd.to_numeric(df_f["OBS_VALUE"], errors="coerce")
    df_f = df_f.dropna(subset=["OBS_VALUE"])
    if df_f.empty:
        return None, ["EU historical NIIP data unavailable after value parsing."], src

    df_pivot = df_f.pivot_table(index="TIME_PERIOD", columns="GEO", values="OBS_VALUE", aggfunc="last")
    if df_pivot.empty:
        return None, ["EU historical NIIP data unavailable after pivoting."], src

    try:
        df_pivot.index = pd.PeriodIndex(df_pivot.index.astype(str), freq="Q").to_timestamp("Q")
    except Exception:
        df_pivot.index = pd.to_datetime(df_pivot.index, errors="coerce")
    df_pivot = df_pivot[~df_pivot.index.isna()].sort_index()
    df_pivot = df_pivot[df_pivot.index >= pd.Timestamp("2010-01-01")]
    if df_pivot.empty:
        return None, ["EU historical NIIP data unavailable after date alignment."], src

    data = []
    for code in target_countries:
        if code not in df_pivot.columns:
            continue
        s = pd.to_numeric(df_pivot[code], errors="coerce").dropna()
        if s.empty:
            continue
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": code,
                "x": s.index,
                "y": s.values.astype(float),
                "line": {"width": 1.9},
                "opacity": 0.9,
            }
        )

    if not data:
        return None, ["EU historical NIIP data unavailable after cleaning."], src

    shapes = []
    x0 = df_pivot.index.min()
    x1 = df_pivot.index.max()
    if x0 is not None and x1 is not None:
        shapes.append(
            {"type": "line", "xref": "x", "yref": "y", "x0": x0, "x1": x1, "y0": 0, "y1": 0, "line": {"color": "black", "width": 1}}
        )
        shapes.append(
            {"type": "line", "xref": "x", "yref": "y", "x0": x0, "x1": x1, "y0": -35, "y1": -35, "line": {"color": COLORS["RED"], "width": 2, "dash": "dash"}}
        )

    layout = {
        "xaxis": {"title": "", "tickformat": "%Y", "dtick": "M24", "range": [pd.Timestamp("2010-01-01"), df_pivot.index.max()]},
        "yaxis": {"title": "NIIP (% of GDP)"},
        "shapes": shapes,
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.30},
        "margin": {"l": 60, "r": 40, "t": 45, "b": 95},
    }
    fig = _fig_from_spec(data, layout, height=620)

    obs = [
        "NIIP above zero indicates a net external creditor position; below zero indicates net debtor position.",
        "Dashed red line marks the MIP warning threshold (-35% of GDP).",
        "Countries shown: DE, FR, IT, ES, NL, BE, AT, DK, PT, PL.",
    ]
    return fig, obs, src


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



# =============================================================================
# EUROSTAT HICP (Headline/Core) (CACHED)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eurostat_hicp_midx(timeout: int = 60*12) -> pd.DataFrame:
    """Eurostat SDMX TSV (compressed): prc_hicp_midx -> EA20 headline & core index levels.

    Returns a DataFrame indexed by month-end with columns:
      - Headline (CP00)
      - Core (TOT_X_NRG_FOOD)
    Values are index levels (unit=I15).
    """
    url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/prc_hicp_midx?format=TSV&compressed=true"

    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        with gzip.open(io.BytesIO(r.content), "rt") as f:
            df = pd.read_csv(f, sep="\t")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # The first column contains the dimension headers (comma-separated), with TIME_PERIOD embedded.
    first_col = df.columns[0]
    headers = [h.strip() for h in str(first_col).split(",")]
    if headers:
        headers[-1] = headers[-1].replace(r"\\TIME_PERIOD", "").replace(r"\TIME_PERIOD", "").strip()

    try:
        df[headers] = df[first_col].astype(str).str.split(",", expand=True)
    except Exception:
        return pd.DataFrame()

    df = df.drop(columns=[first_col])

    for col in headers:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Filter: EA20 + unit=I15 (index level) + headline/core codes
    geo_col = "geo" if "geo" in df.columns else (headers[-1] if headers else "geo")
    if geo_col in df.columns:
        df = df[df[geo_col] == "EA20"]

    if "unit" in df.columns:
        df = df[df["unit"] == "I15"]

    if "coicop" not in df.columns:
        return pd.DataFrame()

    target_codes = ["CP00", "TOT_X_NRG_FOOD"]
    df = df[df["coicop"].isin(target_codes)]
    if df.empty:
        return pd.DataFrame()

    # Pivot wide-by-date: melt monthly columns
    id_vars = [c for c in headers if c in df.columns]
    value_vars = [c for c in df.columns if c not in id_vars]

    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="period", value_name="value")

    df_long["value"] = pd.to_numeric(
        df_long["value"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce",
    )

    dt = pd.to_datetime(
        df_long["period"].astype(str).str.strip().str.replace("M", "-", regex=False),
        format="%Y-%m",
        errors="coerce",
    )

    # Use month-end timestamps to align with other monthly series in this dashboard
    df_long["date"] = dt + pd.offsets.MonthEnd(0)

    df_pivot = (
        df_long.pivot_table(index="date", columns="coicop", values="value")
        .rename(columns={"CP00": "Headline", "TOT_X_NRG_FOOD": "Core"})
        .sort_index()
    )

    return df_pivot.dropna()


def calculate_hicp_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY inflation, 3m/3m annualized momentum, and core index + SMAs."""
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy().sort_index()

    # Year-over-year inflation (YoY)
    if "Headline" in data.columns:
        data["Headline YoY"] = data["Headline"].pct_change(12) * 100.0
    if "Core" in data.columns:
        data["Core YoY"] = data["Core"].pct_change(12) * 100.0

    # 3m/3m annualized momentum (based on 3m moving average)
    for col in ["Headline", "Core"]:
        if col not in data.columns:
            continue
        avg_3m = data[col].rolling(window=3).mean()
        data[f"{col} 3m/3m Annualized"] = ((avg_3m / avg_3m.shift(3)) ** 4 - 1.0) * 100.0

    # Core index (initially rebased; will be rebased again at chart start)
    if "Core" in data.columns and not data["Core"].dropna().empty:
        data["Core Index"] = (data["Core"] / float(data["Core"].iloc[0])) * 100.0
        data["Core SMA 6"] = data["Core Index"].rolling(window=6).mean()
        data["Core SMA 12"] = data["Core Index"].rolling(window=12).mean()

    return data


def build_inflation_figs(start_date: str = "2005-01-01"):
    """Build the inflation panel figures (Plotly, dashboard style)."""
    df_raw = fetch_eurostat_hicp_midx()
    if df_raw is None or df_raw.empty:
        src = ["Eurostat SDMX API: prc_hicp_midx (HICP monthly index)"]
        return (None, ["Eurostat HICP data unavailable (fetch failed or empty)."], src), (None, None, src), (None, None, src)

    dfm = calculate_hicp_metrics(df_raw)

    start_dt = pd.to_datetime(start_date)
    dfp = dfm[dfm.index >= start_dt].copy()

    # Rebase core index to 100 at chart start (e.g., Jan 2005)
    if "Core" in dfp.columns and not dfp["Core"].dropna().empty:
        dfp["Core Index"] = (dfp["Core"] / float(dfp["Core"].iloc[0])) * 100.0
        dfp["Core SMA 6"] = dfp["Core Index"].rolling(window=6).mean()
        dfp["Core SMA 12"] = dfp["Core Index"].rolling(window=12).mean()

    dfp = dfp.dropna()

    src = ["Eurostat SDMX API: prc_hicp_midx (EA20, unit=I15; CP00 headline; TOT_X_NRG_FOOD core)"]

    if dfp.empty:
        return (None, ["Eurostat HICP data unavailable after filtering."], src), (None, None, src), (None, None, src)

    x0 = dfp.index.min()
    x1 = dfp.index.max()
    last_dt = pd.to_datetime(dfp.index[-1])

    # -----------------------------------------------------------------------------
    # FIG 1: Headline vs Core YoY + target band (1–3) and 2% line
    # -----------------------------------------------------------------------------
    shapes1 = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": x0,
            "x1": x1,
            "y0": 1,
            "y1": 3,
            "fillcolor": "rgba(239,68,68,0.12)",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": x0,
            "x1": x1,
            "y0": 2,
            "y1": 2,
            "line": {"color": COLORS["RED"], "width": 2, "dash": "dash"},
        },
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": x0,
            "x1": x1,
            "y0": 0,
            "y1": 0,
            "line": {"color": "black", "width": 1},
        },
    ]

    data1 = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Headline HICP (YoY)",
            "x": dfp.index,
            "y": dfp["Headline YoY"].values.astype(float),
            "line": {"width": 2.2, "color": COLORS["BLACK"]},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Core HICP (YoY)",
            "x": dfp.index,
            "y": dfp["Core YoY"].values.astype(float),
            "line": {"width": 2.2, "color": COLORS["DARK_BLUE"]},
        },
    ]

    layout1 = {
        "xaxis": {"title": "", "tickformat": "%Y", "dtick": "M24"},
        "yaxis": {"title": "YoY Change %"},
        "shapes": shapes1,
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 60, "t": 45, "b": 95},
    }

    fig1 = _fig_from_spec(data1, layout1, height=520)

    obs1 = [
        f"Latest (as of {last_dt.strftime('%Y-%m')}): Headline {float(dfp['Headline YoY'].iloc[-1]):.1f}%, Core {float(dfp['Core YoY'].iloc[-1]):.1f}%.",
        "Red band shows 1–3% range; dashed line marks 2%.",
    ]

    # -----------------------------------------------------------------------------
    # FIG 2: Momentum (3m/3m annualized) in stacked panels
    # -----------------------------------------------------------------------------
    fig_sub = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)

    fig_sub.add_trace(
        go.Bar(
            x=dfp.index,
            y=dfp["Headline 3m/3m Annualized"].values.astype(float),
            name="Headline 3m/3m",
            marker_color=COLORS["BLACK"],
            opacity=1.0,
        ),
        row=1,
        col=1,
    )
    fig_sub.add_trace(
        go.Bar(
            x=dfp.index,
            y=dfp["Core 3m/3m Annualized"].values.astype(float),
            name="Core 3m/3m",
            marker_color=COLORS["DARK_BLUE"],
            opacity=1.0,
        ),
        row=2,
        col=1,
    )

    fig_sub.update_yaxes(title_text="Headline (%)", row=1, col=1)
    fig_sub.update_yaxes(title_text="Core (%)", row=2, col=1)
    fig_sub.update_xaxes(tickformat="%Y", dtick="M24")

    shapes2 = [
        {"type": "line", "xref": "x", "yref": "y", "x0": x0, "x1": x1, "y0": 0, "y1": 0, "line": {"color": "black", "width": 1}},
        {"type": "line", "xref": "x", "yref": "y2", "x0": x0, "x1": x1, "y0": 0, "y1": 0, "line": {"color": "black", "width": 1}},
    ]

    layout2 = fig_sub.layout.to_plotly_json()
    layout2.update(
        {
            "shapes": shapes2,
            "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.28},
            "margin": {"l": 60, "r": 60, "t": 45, "b": 105},
        }
    )

    fig2 = _fig_from_spec(list(fig_sub.data), layout2, height=740)

    obs2 = [
        f"Latest momentum (as of {last_dt.strftime('%Y-%m')}): Headline {float(dfp['Headline 3m/3m Annualized'].iloc[-1]):+.1f}%, Core {float(dfp['Core 3m/3m Annualized'].iloc[-1]):+.1f}%.",
        "Momentum is 3m/3m annualized, based on a 3-month moving average of index levels.",
    ]

    # -----------------------------------------------------------------------------
    # FIG 3: Core index + SMAs (rebased to 100 at chart start)
    # -----------------------------------------------------------------------------
    data3 = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Core Index",
            "x": dfp.index,
            "y": dfp["Core Index"].values.astype(float),
            "line": {"width": 2.2, "color": COLORS["BLACK"]},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "6-Month SMA",
            "x": dfp.index,
            "y": dfp["Core SMA 6"].values.astype(float),
            "line": {"width": 2.0, "color": COLORS["DARK_BLUE"], "dash": "dash"},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "12-Month SMA",
            "x": dfp.index,
            "y": dfp["Core SMA 12"].values.astype(float),
            "line": {"width": 2.0, "color": COLORS["DARK_RED"], "dash": "dash"},
        },
    ]

    layout3 = {
        "xaxis": {"title": "", "tickformat": "%Y", "dtick": "M24"},
        "yaxis": {"title": "Index"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 60, "t": 45, "b": 95},
    }

    fig3 = _fig_from_spec(data3, layout3, height=620)

    cum = float(dfp["Core Index"].iloc[-1]) - 100.0
    obs3 = [
        f"Cumulative core price-level change since {start_dt.strftime('%b %Y')}: {cum:+.1f} index points.",
        "Dashed lines show 6- and 12-month moving averages of the rebased core index.",
    ]

    return (fig1, obs1, src), (fig2, obs2, src), (fig3, obs3, src)



# =============================================================================
# FINANCIAL CONDITIONS & LIQUIDITY (CISS vs Net Liquidity proxy) (CACHED)
# =============================================================================
def _parse_ecb_time_period_robust(tp: pd.Series) -> pd.DatetimeIndex:
    """Robust ECB TIME_PERIOD parsing (daily / monthly / ISO-weekly)."""
    s = tp.astype(str).str.strip()

    # Weekly ISO format: YYYY-Www
    if s.str.contains("W").any():
        dt = pd.to_datetime(s + "-1", format="%G-W%V-%u", errors="coerce")  # Monday
        return pd.DatetimeIndex(dt)

    # Monthly: YYYY-MM -> month-end
    if s.str.fullmatch(r"\d{4}-\d{2}").all():
        dt = pd.PeriodIndex(s, freq="M").to_timestamp("M")
        return pd.DatetimeIndex(dt)

    # Daily: YYYY-MM-DD
    if s.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        # If it looks like monthly delivered as YYYY-MM-01, shift to month-end (heuristic)
        if s.str.endswith("-01").all() and len(s) < 2000:
            dt = dt.to_period("M").to_timestamp("M")
        return pd.DatetimeIndex(dt)

    # Fallback
    return pd.DatetimeIndex(pd.to_datetime(s, errors="coerce"))


def _ecb_panel_to_series_robust(df_panel: pd.DataFrame) -> pd.Series:
    """Convert ECB SDMX-CSV panel to a pd.Series with robust time parsing."""
    if df_panel is None or df_panel.empty:
        return pd.Series(dtype=float)

    if "TIME_PERIOD" not in df_panel.columns or "OBS_VALUE" not in df_panel.columns:
        return pd.Series(dtype=float)

    df = df_panel[["TIME_PERIOD", "OBS_VALUE"]].dropna()
    if df.empty:
        return pd.Series(dtype=float)

    idx = _parse_ecb_time_period_robust(df["TIME_PERIOD"])
    vals = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    s = pd.Series(vals.values, index=idx).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def get_ecb_series_robust(flow: str, key: str, start_period: str) -> pd.Series:
    """Fetch a single ECB time series (robust TIME_PERIOD parsing)."""
    panel = ecb_get_panel(flow, key, start_period)
    return _ecb_panel_to_series_robust(panel)


@st.cache_data(ttl=3600, show_spinner=False)
def get_eu_liquidity_dashboard(
    start: str = FINCOND_START_DATE,
    liq_smooth_days: int = FINCOND_LIQ_SMOOTH_DAYS,
) -> tuple[pd.DataFrame, str]:
    """EU Liquidity & Financial Stress dashboard (daily grid, forward-filled)."""
    # Stress (daily)
    ciss_daily = get_ecb_series_robust("CISS", FINCOND_CISS_KEY, start)

    # Liquidity components
    assets_w = get_ecb_series_robust("ILM", FINCOND_ASSETS_KEY, start)       # weekly
    govdep_mp = get_ecb_series_robust("ILM", FINCOND_GOVDEP_MP_KEY, start)   # monthly (often shorter/patchy)

    if govdep_mp.dropna().shape[0] < 24:
        govdep = get_ecb_series_robust("ILM", FINCOND_GOVDEP_EOP_KEY, start) # monthly (longer)
        govdep_source = "EOP (L050100)"
    else:
        govdep = govdep_mp
        govdep_source = "MP (L050100MP)"

    debtcert_w = get_ecb_series_robust("ILM", FINCOND_DEBTCERT_KEY, start)   # weekly
    ftd_m = get_ecb_series_robust("ILM", FINCOND_FTD_KEY, start)             # monthly

    df = pd.concat(
        [ciss_daily, assets_w, govdep, debtcert_w, ftd_m],
        axis=1,
        keys=["CISS", "Assets", "GovDeposits", "DebtCert", "FixedTermDeposits"],
    ).sort_index()

    if df.empty:
        return pd.DataFrame(), govdep_source

    # Combine on a daily grid and forward-fill (connect points)
    df = df.resample("D").ffill().bfill()

    # If any of the drains are still missing, fill with 0 to avoid breaks in the proxy
    for col in ["GovDeposits", "DebtCert", "FixedTermDeposits"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Net Liquidity proxy (Millions EUR)
    df["NetLiq_mn"] = df["Assets"] - df["GovDeposits"] - df["DebtCert"] - df["FixedTermDeposits"]

    # Rolling averages (calendar days) for both series
    if liq_smooth_days and liq_smooth_days > 1:
        df["NetLiq_5d_mn"] = df["NetLiq_mn"].rolling(liq_smooth_days, min_periods=1).mean()
        df["CISS_s"] = df["CISS"].rolling(liq_smooth_days, min_periods=1).mean()
    else:
        df["NetLiq_5d_mn"] = df["NetLiq_mn"]
        df["CISS_s"] = df["CISS"]

    # Ensure no line breaks after smoothing
    df["CISS_plot"] = df["CISS_s"].interpolate(method="time").ffill().bfill()

    df = df[df.index >= pd.to_datetime(start)]
    return df, govdep_source


def build_financial_conditions_liquidity_fig(
    start_date: str = FINCOND_START_DATE,
    liq_smooth_days: int = FINCOND_LIQ_SMOOTH_DAYS,
):
    """Single chart: CISS (left) vs Net Liquidity proxy (right), both solid lines."""
    df, govdep_source = get_eu_liquidity_dashboard(start=start_date, liq_smooth_days=liq_smooth_days)

    src = [
        "ECB Data API: CISS dataset (CISS), ILM dataset (Assets, Gov. deposits, Debt certificates, Fixed-term deposits)",
    ]

    if df is None or df.empty:
        obs = ["ECB liquidity / CISS data unavailable (fetch failed or empty)."]
        return None, obs, src

    ciss = df["CISS_plot"].dropna()
    netliq_tn = (df["NetLiq_5d_mn"] / 1_000_000.0).dropna()  # mn -> tn

    if ciss.empty or netliq_tn.empty:
        obs = ["ECB liquidity / CISS data unavailable after alignment."]
        return None, obs, src

    start_dt = pd.to_datetime(start_date)
    max_dt = pd.to_datetime(df.index.max())

    data = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "CISS",
            "x": ciss.index,
            "y": ciss.values.astype(float),
            "line": {"width": 2.2, "color": COLORS["DARK_BLUE"]},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Net Liquidity Proxy",
            "x": netliq_tn.index,
            "y": netliq_tn.values.astype(float),
            "yaxis": "y2",
            "line": {"width": 2.2, "color": COLORS["BLACK"]},
        },
    ]

    layout = {
        "xaxis": {"title": "", "tickformat": "%Y", "dtick": "M12", "range": [start_dt, max_dt], "domain": [0.0, 0.86]},
        "yaxis": {"title": "CISS (index)"},
        "yaxis2": {"title": "Net Liquidity Proxy (Trillion €)", "overlaying": "y", "side": "right", "showgrid": False, "anchor": "free", "position": 0.90},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 110, "t": 45, "b": 95},
    }

    fig = _fig_from_spec(data, layout, height=650)

    last_dt = pd.to_datetime(df.index.max())
    obs = [
        f"CISS: [Composite Indicator of Systemic Stress for EA] - Aggregates stress across Banks, Money, Equity, Bonds and FX Markets. 1 indicates high systemic risk, values below 0.3 generally stable financial conditions.",
        f"Net liquidity proxy = Assets - GovDeposits - DebtCert - FixedTermDeposits.",
        f"Both CISS and net liquidity proxy are shown as {liq_smooth_days}-day rolling averages.",
        f"GovDeposits source chosen: {govdep_source}.",
    ]

    return fig, obs, src

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
    cross_debt_pack = build_cross_country_sectoral_debt_latest_fig()
    niip_pack = build_historical_niip_gdp_fig()

    # Money market (same two windows as EU_Dashboard.py)
    mm_start_long = "2020-01-01"
    mm_start_short = "2025-01-01"
    mm_fig_long, mm_obs_long, mm_src_long = build_money_market_fig(mm_start_long)
    mm_fig_short, mm_obs_short, mm_src_short = build_money_market_fig(mm_start_short)

    # Gold & liquidity
    gold_top, gold_bot, gold_obs, gold_src = build_gold_figs()


    # Inflation (Eurostat HICP)
    infl_pack = build_inflation_figs(start_date="2005-01-01")

    # Financial Conditions & Liquidity (ECB CISS + net liquidity proxy)
    fincond_pack = build_financial_conditions_liquidity_fig(
        start_date=FINCOND_START_DATE,
        liq_smooth_days=FINCOND_LIQ_SMOOTH_DAYS,
    )
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
        "cross_country_sectoral_debt": cross_debt_pack,
        "historical_niip": niip_pack,
        "money_market_long": (mm_fig_long, mm_obs_long, mm_src_long, mm_start_long),
        "money_market_short": (mm_fig_short, mm_obs_short, mm_src_short, mm_start_short),
        "gold": (gold_top, gold_bot, gold_obs, gold_src),
        "inflation": infl_pack,
        "financial_conditions_liquidity": fincond_pack,
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
        ["Monetary Policy", "Inflation", "Yield Curves/Spreads & Macro", "Money Market", "Financial Conditions & Liquidity", "Gold & Liquidity"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Removed Overview section logic

    # ===== MONETARY POLICY =====
    if page == "Monetary Policy":
        fig, obs, src = data["policy"]
        _show_chart("Monetary Policy Path: €STR vs Market-implied path", fig, obs, src)

    # ===== INFLATION =====
    elif page == "Inflation":
        (fig1, obs1, src1), (fig2, obs2, src2), (fig3, obs3, src3) = data["inflation"]

        _show_chart("Headline vs Core Inflation", fig1, obs1, src1)

        st.markdown("---")
        _show_chart("Momentum - 3m/3m Annualized", fig2, obs2, src2)

        st.markdown("---")
        _show_chart("Core Inflation Index & Moving Averages", fig3, obs3, src3)

    # ===== YIELD CURVE =====
    elif page == "Yield Curves/Spreads & Macro":
        fig, obs, src = data["yield_curve"]
        _show_chart("Euro Area Yield Curve (Monthly Snapshots, last 24 months)", fig, obs, src)

        st.markdown("---")
        fig, obs, src = data["spreads_aaa"]
        _show_chart("Euro Area 10Y Sovereign Spreads vs Euro Area AAA", fig, obs, src)

        st.markdown("---")
        fig, obs, src = data["spreads_all"]
        _show_chart("Euro Area 10Y Sovereign Spreads vs Euro Area", fig, obs, src)

        st.markdown("---")
        fig, obs, src = data["cross_country_sectoral_debt"]
        _show_chart("Country Sectoral Debt as % of GDP (@ latest avail)", fig, obs, src)

        st.markdown("---")
        fig, obs, src = data["historical_niip"]
        _show_chart("NIIP as % of GDP", fig, obs, src)

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



    # ===== FINANCIAL CONDITIONS & LIQUIDITY =====
    elif page == "Financial Conditions & Liquidity":
        fig, obs, src = data["financial_conditions_liquidity"]
        _show_chart("Euro Area Stress vs Net Liquidity Proxy", fig, obs, src)

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
