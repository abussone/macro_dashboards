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
    Welcome to the Macro Economic Dashboards. Select a dashboard from the sidebar to explore detailed economic data and visualizations.
    """
)

st.markdown("---")

# EU Dashboard Card
st.markdown(
    """
    <div class="dashboard-card">
        <h3><span class="flag-icon">🇪🇺</span> European Union Dashboard</h3>
        <p>Comprehensive analysis of European macroeconomic indicators including:</p>
        <ul>
            <li>ECB monetary policy rates and paths</li>
            <li>Euro area yield curves and sovereign spreads</li>
            <li>Money market rates and liquidity conditions</li>
            <li>Trade flows and economic indicators</li>
        </ul>
        <p><strong>Data Sources:</strong> ECB Data API, ECB Watch, Eurostat</p>
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
            <li>Treasury yield curves and breakeven inflation</li>
            <li>Money market rates and Fed operations</li>
            <li>Gold flows, liquidity, and hedge fund positioning</li>
        </ul>
        <p><strong>Data Sources:</strong> FRED, Atlanta Fed MPT, Federal Reserve Board, CFTC, BEA</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; color: #9E9E9E; font-size: 0.9rem; margin-top: 2rem;">
        Use the sidebar to navigate between dashboards
    </div>
    """,
    unsafe_allow_html=True,
)
