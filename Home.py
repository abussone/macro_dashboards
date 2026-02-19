import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Macro Dashboards",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# UI STYLING
# =============================================================================
st.markdown(
    """
    <style>
      .stApp { background: #F7F8FB; }
      .block-container { padding-top: 2rem; padding-bottom: 2.25rem; max-width: 1200px; }
      h1, h2, h3 { letter-spacing: -0.01em; }
      
      /* Show header so the sidebar toggle is visible on mobile */
      header { visibility: visible !important; height: auto !important; }
      footer { visibility: hidden; height: 0; }
      
      /* Fixed sidebar width for readability and stability */
      [data-testid="stSidebar"] {
          min-width: 260px;
          max-width: 300px;
      }
      
      .dashboard-card {
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      }
      
      .dashboard-card h3 {
        margin-top: 0;
        color: #0B3D91;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }
      
      .flag-icon {
        font-size: 1.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# HOME PAGE CONTENT
# =============================================================================
st.title("📊 Macro Economic Dashboards")

st.markdown(
    """
    Institutional macro and market analytics for the US, Euro Area, and cross-cycle pattern analysis.
    Explore policy, inflation, yield curves/spreads & macro balance-sheet risk, liquidity, positioning, gold flows, and historical analog setups
    from integrated public data sources in one interface.
    """
)

st.markdown("---")

# Macro Leverage & Credit Card
st.markdown(
    """
    <div class="dashboard-card">
        <h3><span class="flag-icon">ML</span> Macro Leverage & Credit</h3>
        <p>Cross-market credit risk and balance-sheet stress monitoring including:</p>
        <ul>
            <li>Historical macro leverage (debt-to-GDP structure by sector and country)</li>
            <li>US commercial bank loan delinquency dynamics</li>
            <li>Euro area non-performing loan (NPL) ratios</li>
        </ul>
        <p><strong>Data Sources:</strong> FRED, ECB Data API</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# EU Dashboard Card
st.markdown(
    """
    <div class="dashboard-card">
        <h3><span class="flag-icon">🇪🇺</span> European Union Dashboard</h3>
        <p>Comprehensive analysis of Euro Area macro and market conditions including:</p>
        <ul>
            <li>ECB monetary policy path and market-implied pricing</li>
            <li>Headline/Core inflation, momentum, and index trend tracking</li>
            <li>Yield Curves/Spreads & Macro (sovereign spreads, NIIP, sectoral debt)</li>
            <li>Money market rates, corridor, and excess liquidity dynamics</li>
            <li>Financial conditions vs net liquidity proxy (CISS framework)</li>
            <li>Gold vs M2 and non-monetary net gold trade flow</li>
        </ul>
        <p><strong>Data Sources:</strong> ECB Data API, ECB Watch, Eurostat, Yahoo Finance</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# US Dashboard Card
st.markdown(
    """
    <div class="dashboard-card">
        <h3><span class="flag-icon">🇺🇸</span> United States Dashboard</h3>
        <p>In-depth view of US macroeconomic and financial market data including:</p>
        <ul>
            <li>FOMC policy paths and SEP projections</li>
            <li>Headline/Core inflation, impulse, and momentum monitoring</li>
            <li>Treasury yield curves and breakeven inflation expectations</li>
            <li>Money market rates, funding spreads, and Fed liquidity signals</li>
            <li>Hedge fund UST futures positioning (gross and net)</li>
            <li>Financial conditions vs net liquidity proxy (NFCI framework)</li>
            <li>Gold trajectory, M2 liquidity, and physical gold flow tracking</li>
        </ul>
        <p><strong>Data Sources:</strong> FRED, Atlanta Fed MPT, Federal Reserve Board, OFR, CFTC, BEA, Yahoo Finance</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Market Memory Explorer Card
st.markdown(
    """
    <div class="dashboard-card">
        <h3><span class="flag-icon">MM</span> Market Memory Explorer</h3>
        <p>Pattern-matching tool for historical market analogs and forward-return context:</p>
        <ul>
            <li>Ticker-based analog search on daily adjusted closes</li>
            <li>Correlation matching on cumulative return paths</li>
            <li>Strict no-overlap historical windowing with distinctness filtering</li>
            <li>Top analog table with forward return horizons (L->2L and L->3L)</li>
            <li>Rebased overlay chart of current regime vs matched episodes</li>
        </ul>
        <p><strong>Data Sources:</strong> Yahoo Finance (via yfinance)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; color: #9E9E9E; font-size: 0.9rem; margin-top: 2rem;">
        Use the sidebar to navigate between dashboards and tools
    </div>
    """,
    unsafe_allow_html=True,
)
