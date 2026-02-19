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

st.set_page_config(page_title="Macro Leverage & Credit", page_icon="📊", layout="wide")

COLORS = {
    "BLUE": "#1f77b4",
    "GREEN": "#2ca02c",
    "RED": "#d62728",
    "DARK_BLUE": "#0B3D91",
    "DARK_RED": "#8B0000",
    "BLACK": "#000000",
    "BROWN": "#8c564b",
}

FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ECB_DATA_API = "https://data-api.ecb.europa.eu/service/data"

START_DATE_LEVERAGE = "2010-01-01"
START_DATE_DELINQ = "2010-01-01"
START_PERIOD_NPL = "2010-Q1"


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; Macro Leverage & Credit Dashboard)",
            "Accept": "*/*",
        }
    )
    return s


SESSION = make_session()


def _inject_light_ui():
    st.markdown(
        """
        <style>
          .stApp { background: #F7F8FB; }
          .block-container { padding-top: 1.25rem; padding-bottom: 2.25rem; max-width: 1200px; }
          h1, h2, h3 { letter-spacing: -0.01em; }

          header { visibility: visible !important; height: auto !important; }
          footer { visibility: hidden; height: 0; }

          [data-testid="stSidebar"] {
              min-width: 260px;
              max-width: 300px;
          }

          div[data-testid="stRadio"] {
            position: sticky;
            top: 2.85rem;
            z-index: 1000;
            background: #F7F8FB;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
          }

          .details-box {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15,23,42,0.06);
            border-radius: 14px;
            padding: 0.9rem 1.0rem;
            margin-top: 0.35rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fig_from_spec(data: list[dict], layout: dict, height: int) -> go.Figure:
    fig = go.Figure(data=data, layout=layout)

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

    base_margin = layout.get("margin", dict(l=60, r=60, t=30, b=80))
    fig.update_layout(
        hovermode=layout.get("hovermode", "x unified"),
        height=height,
        margin=base_margin,
        font=dict(color="black"),
        hoverlabel=dict(namelength=-1),
        legend_font_color="black",
    )

    try:
        t_text = None
        if getattr(fig.layout, "title", None) is not None:
            t_text = getattr(fig.layout.title, "text", None)
        if t_text is None or str(t_text).strip().lower() == "undefined":
            fig.update_layout(title_text="")
        fig.update_layout(title=dict(font=dict(color="black")))
    except Exception:
        pass

    fig.update_xaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        linecolor="black",
        gridcolor="rgba(0,0,0,0.05)",
        hoverformat="%Y-%m-%d",
    )
    fig.update_yaxes(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        linecolor="black",
        gridcolor="rgba(0,0,0,0.05)",
    )

    try:
        leg = fig.layout.legend
        if leg and getattr(leg, "orientation", None) == "h":
            ly = getattr(leg, "y", None)
            if ly is not None and float(ly) < 0:
                n_items = 0
                for tr in fig.data:
                    show = getattr(tr, "showlegend", True)
                    if show is False:
                        continue
                    n_items += 1
                per_row = 4
                rows = max(1, int(math.ceil(n_items / per_row)))
                legend_px = 26 * rows + 28
                needed_b = max(int(base_margin.get("b", 0) or 0), 70 + legend_px)
                fig.update_layout(
                    margin=dict(**base_margin, b=needed_b),
                    height=height + max(0, needed_b - base_margin.get("b", 0)),
                )
    except Exception:
        pass

    return fig


def _show_chart(title: str, fig: go.Figure | None, observations: list[str] | None, sources: list[str] | None):
    if title:
        st.subheader(title)
    if fig is None:
        st.warning("Chart unavailable (missing data).")
        return

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": False,
            "displaylogo": False,
            "responsive": True,
            "displayModeBar": True,
        },
    )

    st.markdown('<div class="details-box">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Key observations")
        if observations:
            st.markdown("\n".join([f"- {o}" for o in observations]))
        else:
            st.write("-")
    with c2:
        st.markdown("#### Data sources")
        if sources:
            st.markdown("\n".join([f"- {s}" for s in sources]))
        else:
            st.write("-")
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fred_series_public(series_id: str, start_date: str = "1900-01-01") -> pd.Series:
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_us_delinquency_panel(start_date: str = START_DATE_DELINQ) -> pd.DataFrame:
    tickers = {
        "DRALACBS": "Total Economy (All Loans)",
        "DRBLACBS": "Firms (Business Loans)",
        "DRCLACBS": "Households (Consumer Loans)",
    }
    panel = {}
    for sid, label in tickers.items():
        s = fred_series_public(sid, start_date=start_date)
        if s is not None and not s.dropna().empty:
            panel[label] = s
    if not panel:
        return pd.DataFrame()
    return pd.concat(panel, axis=1).sort_index().dropna(how="all")


def build_us_delinquency_fig() -> tuple[go.Figure | None, list[str], list[str]]:
    df = fetch_us_delinquency_panel()
    src = ["FRED series: DRALACBS, DRBLACBS, DRCLACBS"]
    if df.empty:
        return None, ["US commercial bank loan delinquency data unavailable."], src

    colors = {
        "Total Economy (All Loans)": COLORS["BLACK"],
        "Firms (Business Loans)": COLORS["DARK_RED"],
        "Households (Consumer Loans)": COLORS["DARK_BLUE"],
    }
    data = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": col,
                "x": s.index,
                "y": s.values.astype(float),
                "line": {"width": 2.2, "color": colors.get(col, COLORS["BLACK"])},
                "marker": {"size": 5, "color": colors.get(col, COLORS["BLACK"])},
            }
        )

    if not data:
        return None, ["US commercial bank loan delinquency data unavailable after cleaning."], src

    layout = {
        "xaxis": {"title": "", "tickformat": "%Y"},
        "yaxis": {"title": "Delinquency rate (%)"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 40, "t": 45, "b": 95},
    }
    fig = _fig_from_spec(data, layout, height=560)

    obs = [
        "Total Economy (All Loans): broad delinquency rate across the commercial bank loan book.",
        "Firms (Business Loans): delinquency rate on loans to businesses/non-financial corporates.",
        "Households (Consumer Loans): delinquency rate on household/consumer credit exposures.",
    ]
    return fig, obs, src


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ecb_sup_series(series_key: str, start_period: str = START_PERIOD_NPL) -> pd.Series:
    url = f"{ECB_DATA_API}/SUP/{series_key}"
    params = {"format": "csvdata", "startPeriod": start_period}
    try:
        r = SESSION.get(url, params=params, timeout=60)
        if r.status_code == 404:
            return pd.Series(dtype=float)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            return pd.Series(dtype=float)

        out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        out["TIME_PERIOD"] = out["TIME_PERIOD"].astype(str).str.strip()
        out["OBS_VALUE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
        out = out.dropna(subset=["TIME_PERIOD", "OBS_VALUE"])
        if out.empty:
            return pd.Series(dtype=float)

        idx = pd.PeriodIndex(out["TIME_PERIOD"], freq="Q").to_timestamp("Q")
        s = pd.Series(out["OBS_VALUE"].values, index=idx).dropna().sort_index()
        return s[~s.index.duplicated(keep="last")]
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eu_npl_panel(start_period: str = START_PERIOD_NPL) -> pd.DataFrame:
    series_keys = {
        "Total Economy": "Q.B01.W0._Z.I7000._T.SII._Z._Z._Z.PCT.C",
        "Firms (NFCs)": "Q.B01.W0.S11.I7005._T.SII._Z._Z._Z.PCT.C",
        "Households": "Q.B01.W0.S14.I7005._T.SII._Z._Z._Z.PCT.C",
    }
    panel = {}
    for label, key in series_keys.items():
        s = fetch_ecb_sup_series(key, start_period=start_period)
        if s is not None and not s.dropna().empty:
            panel[label] = s
    if not panel:
        return pd.DataFrame()
    return pd.concat(panel, axis=1).sort_index().dropna(how="all")


def build_eu_npl_fig() -> tuple[go.Figure | None, list[str], list[str]]:
    df = fetch_eu_npl_panel()
    src = ["ECB Data API (SUP): NPL ratio series for total economy, NFCs, and households"]
    if df.empty:
        return None, ["Euro area NPL ratio data unavailable."], src

    colors = {
        "Total Economy": COLORS["BLACK"],
        "Firms (NFCs)": COLORS["DARK_RED"],
        "Households": COLORS["DARK_BLUE"],
    }
    data = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": col,
                "x": s.index,
                "y": s.values.astype(float),
                "line": {"width": 2.1, "color": colors.get(col, COLORS["BLACK"])},
                "marker": {"size": 5},
            }
        )

    if not data:
        return None, ["Euro area NPL ratio data unavailable after cleaning."], src

    layout = {
        "xaxis": {"title": "", "tickformat": "%Y"},
        "yaxis": {"title": "NPL ratio (%)"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 40, "t": 45, "b": 95},
    }
    fig = _fig_from_spec(data, layout, height=560)

    obs = [
        "Total Economy: NPL ratio across the broad supervised banking loan portfolio.",
        "Firms (NFCs): non-performing loan ratio for non-financial corporations.",
        "Households: non-performing loan ratio for household credit.",
    ]
    return fig, obs, src


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_bis_debt_panel(start_date: str = START_DATE_LEVERAGE) -> pd.DataFrame:
    countries = ["US", "XM", "DE", "FR", "IT", "GB", "JP"]
    sectors = ["G", "H", "N"]
    panel = {}
    for c_code in countries:
        for s_code in sectors:
            sid = f"Q{c_code}{s_code}AM770A"
            s = fred_series_public(sid, start_date=start_date)
            if s is not None and not s.dropna().empty:
                panel[sid] = s
    if not panel:
        return pd.DataFrame()
    return pd.concat(panel, axis=1).sort_index().dropna(how="all")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_us_financials_pct(start_date: str = START_DATE_LEVERAGE) -> pd.Series:
    dodfs = fred_series_public("DODFS", start_date=start_date)
    gdp = fred_series_public("GDP", start_date=start_date)
    if dodfs.empty or gdp.empty:
        return pd.Series(dtype=float)

    common = dodfs.index.intersection(gdp.index)
    if common.empty:
        return pd.Series(dtype=float)

    dodfs = pd.to_numeric(dodfs.reindex(common), errors="coerce")
    gdp = pd.to_numeric(gdp.reindex(common), errors="coerce")
    ratio = (dodfs / (gdp * 1000.0)) * 100.0
    return ratio.dropna().rename("US_Financials_Pct")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def prepare_country_debt_frames(start_date: str = START_DATE_LEVERAGE) -> dict[str, pd.DataFrame]:
    countries = {
        "US": "United States",
        "XM": "Euro Area",
        "DE": "Germany",
        "FR": "France",
        "IT": "Italy",
        "GB": "United Kingdom",
        "JP": "Japan",
    }
    sector_map = {
        "G": "Government",
        "H": "Households",
        "N": "Non-Fin Corps",
    }

    bis_df = fetch_bis_debt_panel(start_date=start_date)
    us_fin = fetch_us_financials_pct(start_date=start_date)
    if bis_df.empty:
        return {}

    out = {}
    for c_code, c_name in countries.items():
        df_c = pd.DataFrame(index=bis_df.index.copy())
        core_cols = []
        for s_code, s_name in sector_map.items():
            sid = f"Q{c_code}{s_code}AM770A"
            if sid in bis_df.columns:
                df_c[s_name] = pd.to_numeric(bis_df[sid], errors="coerce")
                core_cols.append(s_name)

        if not core_cols:
            continue
        df_c = df_c.dropna(subset=core_cols, how="any")
        if df_c.empty:
            continue

        df_c["Total Non-Financial"] = df_c[core_cols].sum(axis=1)
        if c_code == "US" and not us_fin.empty:
            df_c["Total Incl Financials"] = df_c["Total Non-Financial"] + us_fin.reindex(df_c.index, method="ffill")
        else:
            df_c["Total Incl Financials"] = np.nan

        out[c_name] = df_c.sort_index()

    return out


def build_historical_macro_leverage_fig(
    country_frames: dict[str, pd.DataFrame],
    country_name: str,
) -> tuple[go.Figure | None, list[str], list[str]]:
    src = [
        "FRED BIS credit database: Q[Country][Sector]AM770A series (Government, Households, Non-Financial Corporates)",
        "FRED: DODFS and GDP (US total including financials overlay)",
    ]

    if not country_frames or country_name not in country_frames:
        return None, ["Historical macro leverage data unavailable."], src

    df_c = country_frames[country_name].copy()
    if df_c.empty:
        return None, ["Historical macro leverage data unavailable after alignment."], src

    sector_colors = {
        "Government": COLORS["BLUE"],
        "Households": COLORS["GREEN"],
        "Non-Fin Corps": COLORS["RED"],
    }
    sector_fill = {
        "Government": "rgba(31,119,180,0.42)",
        "Households": "rgba(44,160,44,0.42)",
        "Non-Fin Corps": "rgba(214,39,40,0.42)",
    }
    data = []

    for sector in ["Government", "Households", "Non-Fin Corps"]:
        if sector not in df_c.columns:
            continue
        s = pd.to_numeric(df_c[sector], errors="coerce").dropna()
        if s.empty:
            continue
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": sector,
                "x": s.index,
                "y": s.values.astype(float),
                "stackgroup": "one",
                "line": {"width": 1.1, "color": sector_colors[sector]},
                "fillcolor": sector_fill[sector],
            }
        )

    tn = pd.to_numeric(df_c.get("Total Non-Financial"), errors="coerce").dropna()
    if not tn.empty:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Sum (Gov+HH+Corp)",
                "x": tn.index,
                "y": tn.values.astype(float),
                "line": {"color": COLORS["BLACK"], "width": 2.4, "dash": "dash"},
            }
        )

    tif = pd.to_numeric(df_c.get("Total Incl Financials"), errors="coerce").dropna()
    if not tif.empty:
        data.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Total Systemic (Incl. Financials)",
                "x": tif.index,
                "y": tif.values.astype(float),
                "line": {"color": COLORS["BLACK"], "width": 2.8},
            }
        )

    if not data:
        return None, ["Historical macro leverage data unavailable after cleaning."], src

    layout = {
        "xaxis": {"title": "", "tickformat": "%Y"},
        "yaxis": {"title": "Debt to GDP ratio (%)"},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.25},
        "margin": {"l": 60, "r": 40, "t": 45, "b": 95},
    }
    fig = _fig_from_spec(data, layout, height=620)

    obs = [
        "Government: public-sector debt as a share of GDP.",
        "Households: household debt as a share of GDP (mainly mortgages and consumer credit).",
        "Non-Fin Corps: non-financial corporate debt as a share of GDP.",
        "Sum (Gov+HH+Corp): aggregate non-financial leverage.",
        "Total Systemic (Incl. Financials): only for US.",
    ]

    return fig, obs, src


def main():
    _inject_light_ui()

    st.sidebar.title("Controls")
    if st.sidebar.button("Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("Tip: data is cached, so switching tabs should be fast after first load.")

    h1, h2, h3 = st.columns([0.72, 0.18, 0.10], vertical_alignment="bottom")
    with h1:
        st.title("Macro Leverage & Credit")
    with h2:
        st.caption(f"Local time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with h3:
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Leverage: Debt to GDP Ratio",
            "US Loan Deliquency (Comm Banks)",
            "EA NPL Ratios",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Leverage: Debt to GDP Ratio":
        with st.spinner("Loading leverage data (cached)..."):
            country_frames = prepare_country_debt_frames()
        if not country_frames:
            st.warning("Historical macro leverage section unavailable (FRED fetch failed).")
            return

        countries = sorted(country_frames.keys())
        default_idx = countries.index("United States") if "United States" in countries else 0
        selected_country = st.selectbox("Country", countries, index=default_idx)
        fig, obs, src = build_historical_macro_leverage_fig(country_frames, selected_country)
        _show_chart("", fig, obs, src)

    elif page == "US Loan Deliquency (Comm Banks)":
        with st.spinner("Loading US delinquency data (cached)..."):
            fig, obs, src = build_us_delinquency_fig()
        _show_chart("", fig, obs, src)

    elif page == "EA NPL Ratios":
        with st.spinner("Loading euro area NPL data (cached)..."):
            fig, obs, src = build_eu_npl_fig()
        _show_chart("", fig, obs, src)


if __name__ == "__main__":
    main()
