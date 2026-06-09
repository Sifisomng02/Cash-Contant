# =============================================================================
# CashContant v2.0 — Main Application
# Author: Sifiso Mnguni
# Description: A financial behaviour tracking dashboard analysing cash demand
#              and security risks in South Africa using real SARB & World Bank data.
# =============================================================================

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    WORLD_BANK_FILE, ATM_FILE, SMARTCASH_FILE, EXCEL_FILE,
    BRANCH_LOCATIONS_FILE, CASH_CIRCULATION_FILE, LOGO_FILE,
    APP_TITLE, APP_ICON, APP_AUTHOR, APP_VERSION,
    FORECAST_DEFAULT_DAYS, FORECAST_MIN_DAYS, FORECAST_MAX_DAYS,
    SARB_STATS,
)
from models.model_demand import forecast_cash_demand
from models.model_security import security_analysis

# =============================================================================
# Page Configuration & Custom Styling
# =============================================================================

st.set_page_config(
    page_title=f"{APP_TITLE} Dashboard",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — SA-inspired dark green/gold theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2b1d 0%, #1a4731 100%);
    }

    [data-testid="stSidebar"] * {
        color: #e8f5e9 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f0faf4;
        border: 1px solid #c8e6c9;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricValue"] {
        color: #1a4731 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 800;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        color: #1a4731;
    }

    /* Plotly chart backgrounds */
    .js-plotly-plot {
        border-radius: 12px;
    }

    /* Section dividers */
    hr {
        border-color: #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading (with Streamlit caching)
# =============================================================================

@st.cache_data
def load_world_bank_data():
    try:
        df = pd.read_csv(WORLD_BANK_FILE, skiprows=4)
        df_sa = df[df["Country Name"] == "South Africa"].copy()
        if df_sa.empty:
            return None
        years = [str(y) for y in range(2004, 2025)]
        df_long = df_sa.melt(id_vars=["Country Name"], value_vars=years,
                             var_name="Year", value_name="ATM_Density")
        df_long["Year"] = pd.to_datetime(df_long["Year"], format="%Y")
        df_long["ATM_Density"] = pd.to_numeric(df_long["ATM_Density"], errors="coerce")
        df_long["YoY_Change"] = df_long["ATM_Density"].pct_change() * 100
        return df_long.dropna(subset=["ATM_Density"])
    except FileNotFoundError:
        st.warning(f"World Bank data not found at: `{WORLD_BANK_FILE}`")
        return None


@st.cache_data
def load_atm_withdrawals():
    try:
        df = pd.read_csv(ATM_FILE, delimiter=";")
        # Drop unnamed columns from messy exports
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.columns = df.columns.str.strip()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        return df
    except FileNotFoundError:
        st.warning(f"ATM withdrawals file not found at: `{ATM_FILE}`")
        return None


@st.cache_data
def load_smartcash_data():
    try:
        df = pd.read_csv(SMARTCASH_FILE, delimiter=";")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"])
    except FileNotFoundError:
        st.warning(f"SmartCash dataset not found at: `{SMARTCASH_FILE}`")
        return None


@st.cache_data
def load_branch_locations():
    try:
        df = pd.read_csv(BRANCH_LOCATIONS_FILE)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        return df.dropna(subset=["lat", "lon"])
    except FileNotFoundError:
        st.info("Branch locations file not found. Add `branch_locations.csv` to your `data/` folder.")
        return None


@st.cache_data
def load_m0_data():
    try:
        df = pd.read_csv(CASH_CIRCULATION_FILE, skiprows=1)
        df = df[["Date", "M0"]].copy()
        df["M0"] = pd.to_numeric(df["M0"].astype(str).str.replace(",", ""), errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"], format="%b, %Y", errors="coerce")
        df = df.dropna().sort_values("Date")
        # Compute YoY % change
        df["YoY_Change"] = df["M0"].pct_change(12) * 100
        return df
    except FileNotFoundError:
        st.warning(f"Cash circulation file not found at: `{CASH_CIRCULATION_FILE}`")
        return None


# =============================================================================
# Plotting Helpers (Plotly)
# =============================================================================

COLORS = {
    "green_dark": "#1a4731",
    "green_mid": "#2e7d52",
    "green_light": "#4caf50",
    "gold": "#c9a84c",
    "gold_light": "#f0d080",
    "red": "#e53935",
    "orange": "#fb8c00",
    "bg": "#f9fbf9",
}

PLOTLY_LAYOUT = dict(
    font_family="DM Sans",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
)


def plot_atm_density(df):
    fig = px.line(
        df, x="Year", y="ATM_Density",
        title="📍 ATM Density — South Africa (2004–2023)",
        labels={"ATM_Density": "ATMs per 100,000 Adults", "Year": ""},
        color_discrete_sequence=[COLORS["green_mid"]],
    )
    fig.add_scatter(x=df["Year"], y=df["ATM_Density"],
                    mode="markers", marker=dict(size=6, color=COLORS["gold"]),
                    showlegend=False)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e0ece4")
    return fig


def plot_m0_trend(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["M0"] / 1e3,
        name="M0 (R Billions)", line=dict(color=COLORS["green_mid"], width=2.5),
        fill="tozeroy", fillcolor="rgba(46,125,82,0.08)"
    ), secondary_y=False)
    if "YoY_Change" in df.columns:
        fig.add_trace(go.Bar(
            x=df["Date"], y=df["YoY_Change"],
            name="YoY % Change", marker_color=df["YoY_Change"].apply(
                lambda x: COLORS["green_light"] if x >= 0 else COLORS["red"]
            ), opacity=0.5
        ), secondary_y=True)
    fig.update_layout(
        title="💵 Total Cash in Circulation (M0) with YoY Growth",
        **PLOTLY_LAYOUT
    )
    fig.update_yaxes(title_text="R Billions", secondary_y=False, gridcolor="#e0ece4")
    fig.update_yaxes(title_text="YoY % Change", secondary_y=True, showgrid=False)
    return fig


def plot_monthly_habits(df):
    monthly = df.set_index("timestamp")["withdrawal_amount"].resample("M").sum().reset_index()
    monthly.columns = ["Month", "Amount"]
    monthly["Month_Label"] = monthly["Month"].dt.strftime("%b %Y")
    avg = monthly["Amount"].mean()
    fig = px.bar(
        monthly, x="Month_Label", y="Amount",
        title="📅 Your Monthly Cash Withdrawals",
        labels={"Amount": "Total Amount (R)", "Month_Label": ""},
        color="Amount",
        color_continuous_scale=[[0, COLORS["green_light"]], [1, COLORS["green_dark"]]],
    )
    fig.add_hline(y=avg, line_dash="dash", line_color=COLORS["gold"],
                  annotation_text=f"Monthly Avg: R{avg:,.0f}",
                  annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_regional_trends(df):
    fig = px.line(
        df, x="Date", y="Cash_Withdrawn", color="ATM_ID",
        title="📊 Local ATM Cash Demand by Location",
        labels={"Cash_Withdrawn": "Cash Withdrawn (R)", "Date": "", "ATM_ID": "ATM"},
        color_discrete_sequence=px.colors.sequential.Greens_r,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def plot_lr_forecast(lr_result, lr_metrics):
    historical = lr_result[lr_result["y"].notna()]
    future = lr_result[lr_result["y"].isna()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical["ds"], y=historical["y"],
        mode="lines", name="Actual", line=dict(color=COLORS["green_mid"], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=historical["ds"], y=historical["lr_forecast"],
        mode="lines", name="LR Fit (Historical)",
        line=dict(color=COLORS["gold"], width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=future["ds"], y=future["lr_forecast"],
        mode="lines", name="LR Forecast",
        line=dict(color=COLORS["gold"], width=2, dash="dash")
    ))
    fig.update_layout(
        title=f"📈 Linear Regression Forecast — RMSE: R{lr_metrics['RMSE']:,} | MAE: R{lr_metrics['MAE']:,}",
        xaxis_title="", yaxis_title="Daily Withdrawals (R)",
        **PLOTLY_LAYOUT
    )
    return fig


def plot_anomaly_distribution(scored_df):
    risk_counts = scored_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    color_map = {
        "✅ Normal": COLORS["green_light"],
        "🟡 Low Risk": COLORS["gold_light"],
        "🟠 Medium Risk": COLORS["orange"],
        "🔴 High Risk": COLORS["red"],
    }
    fig = px.pie(
        risk_counts, names="Risk Level", values="Count",
        title="🛡️ Transaction Risk Distribution",
        color="Risk Level", color_discrete_map=color_map,
        hole=0.45,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# =============================================================================
# Main App
# =============================================================================

def main():

    # --- Sidebar ---
    st.sidebar.markdown(f"""
    <div style="text-align:center; padding: 8px 0 16px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#c9a84c;">
            💡 CashContant
        </div>
        <div style="font-size:0.75rem; color:#a5d6a7; margin-top:2px;">
            v{APP_VERSION} · by {APP_AUTHOR}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(LOGO_FILE):
        st.sidebar.image(LOGO_FILE, use_container_width=True)

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", [
        "📈 My Dashboard",
        "🗺️ Branch Hotspots",
        "🔮 Withdrawal Forecast",
        "🛡️ Security & Risk",
        "🗂️ Raw Data Explorer",
    ])
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"""
        **Key SARB 2023 Stats**
        - **{SARB_STATS['cash_users_pct']}%** of South Africans use cash
        - **{SARB_STATS['small_txn_cash_pct']}%** of R1–R100 purchases are cash
        - **{SARB_STATS['digital_fee_barrier_pct']}%** cite high fees as barrier to digital
        - **R{SARB_STATS['atm_withdrawals_2022_trillion']}T** withdrawn at ATMs in 2022
        """
    )

    # --- Load Data ---
    df_sa_trend     = load_world_bank_data()
    df_atm_habits   = load_atm_withdrawals()
    df_smartcash    = load_smartcash_data()
    df_m0           = load_m0_data()

    # =========================================================================
    # PAGE 1: My Dashboard
    # =========================================================================
    if page == "📈 My Dashboard":
        st.title("📈 My Dashboard")
        st.caption("Personal, regional, and national cash trends — all in one place.")
        st.markdown("---")

        # --- KPI Cards ---
        if df_atm_habits is not None:
            avg_monthly = df_atm_habits.set_index("timestamp")["withdrawal_amount"].resample("M").sum().mean()
            total_withdrawals = df_atm_habits["withdrawal_amount"].sum()
            visit_count = len(df_atm_habits)
            most_used_branch = df_atm_habits["branch"].mode()[0] if "branch" in df_atm_habits.columns else "N/A"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg. Monthly Withdrawal", f"R {avg_monthly:,.0f}")
            c2.metric("Total Withdrawn (All Time)", f"R {total_withdrawals:,.0f}")
            c3.metric("Total ATM Visits", f"{visit_count:,}")
            c4.metric("Most Used Branch", most_used_branch)
        else:
            st.warning("Personal ATM data not loaded. Showing national trends only.")

        # --- Key Insights Expander ---
        with st.expander("🔍 Key Findings from SARB & World Bank Data"):
            st.markdown("""
            | Insight | Detail |
            |---|---|
            | Cash dominates small payments | 58% of all R1–R100 transactions are made in cash |
            | Top cash use cases | Taxis, informal traders, peer-to-peer payments |
            | Digital fee barrier | 34% of consumers prefer cash due to high digital transaction fees |
            | ATM withdrawals 2022 | R1.2 Trillion withdrawn nationally |
            | Cash in circulation | M0 has grown consistently, even as digital payments expand |
            """)

        st.markdown("---")

        # --- Personal Habits Chart ---
        st.subheader("Your Withdrawal Habits")
        if df_atm_habits is not None:
            st.plotly_chart(plot_monthly_habits(df_atm_habits), use_container_width=True)

        # --- National Trends ---
        st.subheader("National Trends")
        col1, col2 = st.columns(2)
        with col1:
            if df_sa_trend is not None:
                st.plotly_chart(plot_atm_density(df_sa_trend), use_container_width=True)
            else:
                st.info("World Bank ATM density data not loaded.")
        with col2:
            if df_m0 is not None:
                st.plotly_chart(plot_m0_trend(df_m0), use_container_width=True)
            else:
                st.info("Cash in circulation (M0) data not loaded.")

        # --- Regional Trends ---
        st.subheader("Local ATM Demand")
        if df_smartcash is not None:
            st.plotly_chart(plot_regional_trends(df_smartcash), use_container_width=True)

    # =========================================================================
    # PAGE 2: Branch Hotspots & Map
    # =========================================================================
    elif page == "🗺️ Branch Hotspots":
        st.title("🗺️ Branch Hotspots & Map")
        st.caption("Your most-visited branches, their locations, and withdrawal activity.")
        st.markdown("---")

        df_locations = load_branch_locations()

        if df_atm_habits is None or df_locations is None:
            st.warning("Ensure both `atm_withdrawals.csv` and `branch_locations.csv` exist in `data/`.")
        else:
            branch_stats = df_atm_habits.groupby("branch").agg(
                total_withdrawn=("withdrawal_amount", "sum"),
                visits=("withdrawal_amount", "count"),
                avg_withdrawal=("withdrawal_amount", "mean"),
            ).reset_index()

            df_map = df_locations.merge(branch_stats, on="branch", how="left")
            df_map[["total_withdrawn", "visits", "avg_withdrawal"]] = \
                df_map[["total_withdrawn", "visits", "avg_withdrawal"]].fillna(0)

            st.subheader("Branch Statistics")
            st.dataframe(
                df_map.style.format({
                    "total_withdrawn": "R {:,.0f}",
                    "avg_withdrawal": "R {:,.0f}",
                    "visits": "{:,.0f}",
                }),
                use_container_width=True,
            )

            st.subheader("Branch Map")
            fig_map = px.scatter_mapbox(
                df_map, lat="lat", lon="lon", hover_name="branch",
                hover_data={"total_withdrawn": True, "visits": True},
                size="total_withdrawn", color="total_withdrawn",
                color_continuous_scale="Greens",
                zoom=5, mapbox_style="carto-positron",
                title="Branch Activity Heatmap",
            )
            fig_map.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_map, use_container_width=True)
            st.info("Larger circles = higher total withdrawals at that branch.")

    # =========================================================================
    # PAGE 3: Withdrawal Forecast
    # =========================================================================
    elif page == "🔮 Withdrawal Forecast":
        st.title("🔮 Withdrawal Forecast")
        st.caption("Forecasting your future cash withdrawals using time-series modelling.")
        st.markdown("---")

        if df_atm_habits is None:
            st.warning("Cannot generate forecast — ATM withdrawal data not loaded.")
        else:
            days = st.slider(
                "Forecast horizon (days):",
                min_value=FORECAST_MIN_DAYS,
                max_value=FORECAST_MAX_DAYS,
                value=FORECAST_DEFAULT_DAYS,
                step=1,
            )

            col1, col2 = st.columns([2, 1])
            with col2:
                st.markdown("### Model Selection")
                use_prophet = st.toggle("Use Prophet (recommended)", value=True)
                st.caption("Prophet captures seasonality automatically. Linear Regression is the baseline.")

            with st.spinner("Running forecast models..."):
                # --- Linear Regression (always runs) ---
                demand_results = forecast_cash_demand(df_atm_habits, periods=days)
                lr_fig = plot_lr_forecast(demand_results["lr_result"], demand_results["lr_metrics"])

                if use_prophet:
                    try:
                        from prophet import Prophet  # type: ignore
                        daily_df = demand_results["daily_df"]
                        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                        m.fit(daily_df)
                        future = m.make_future_dataframe(periods=days)
                        forecast = m.predict(future)

                        # Prophet plot via Plotly
                        from prophet.plot import plot_plotly, plot_components_plotly
                        prophet_fig = plot_plotly(m, forecast)
                        prophet_fig.update_layout(
                            title=f"🔮 Prophet Forecast — {days}-Day Outlook",
                            xaxis_title="", yaxis_title="Daily Withdrawals (R)",
                            **PLOTLY_LAYOUT,
                        )

                        with col1:
                            st.plotly_chart(prophet_fig, use_container_width=True)

                        st.subheader("Forecast Components (Trend + Seasonality)")
                        components_fig = plot_components_plotly(m, forecast)
                        st.plotly_chart(components_fig, use_container_width=True)

                    except ImportError:
                        st.warning("Prophet not installed (`pip install prophet`). Showing Linear Regression only.")
                        with col1:
                            st.plotly_chart(lr_fig, use_container_width=True)
                else:
                    with col1:
                        st.plotly_chart(lr_fig, use_container_width=True)

            # --- Model Comparison ---
            st.subheader("📏 Model Evaluation (Linear Regression Baseline)")
            mc1, mc2 = st.columns(2)
            mc1.metric("RMSE", f"R {demand_results['lr_metrics']['RMSE']:,}",
                       help="Root Mean Squared Error — lower is better")
            mc2.metric("MAE", f"R {demand_results['lr_metrics']['MAE']:,}",
                       help="Mean Absolute Error — average prediction error")

            with st.expander("ℹ️ What do these metrics mean?"):
                st.markdown("""
                - **RMSE (Root Mean Squared Error):** The average magnitude of prediction errors, 
                  with larger errors penalised more heavily. A lower RMSE means better accuracy.
                - **MAE (Mean Absolute Error):** The average absolute difference between predicted 
                  and actual values. Easier to interpret — it's simply the average error in Rands.
                - **Why two models?** Linear Regression is a transparent baseline. 
                  Prophet adds seasonal intelligence (weekly, monthly, yearly patterns).
                  Comparing them shows how much value the more complex model adds.
                """)

    # =========================================================================
    # PAGE 4: Security & Risk
    # =========================================================================
    elif page == "🛡️ Security & Risk":
        st.title("🛡️ Security & Risk")
        st.markdown("---")

        st.warning("""
        **The National Reality:** Cash-in-transit heists, ATM bombings, and personal 
        robberies remain serious risks in South Africa. Every ATM visit carries potential 
        personal safety exposure. This module helps you detect fraud and reduce that exposure.
        """)

        st.subheader("How CashContant Protects You")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **1. Reduce Exposure**
            - Plan withdrawals using the Forecast page → fewer ATM trips
            - Identify high-frequency branches on the Map page
            - Replace routine cash trips with digital alternatives
            """)
        with col2:
            st.markdown("""
            **2. Detect Fraud**
            - Scans for rapid repeated withdrawals (< 1 min apart)
            - Flags impossible travel (different ATMs < 10 min apart)
            - Highlights after-hours transactions (00h–04h)
            - Identifies high-value outliers using IQR method
            - Assigns an **anomaly score (0–100)** per transaction
            """)

        st.markdown("---")
        st.header("Personal Security Scan")

        if df_atm_habits is None:
            st.warning("ATM withdrawal data not loaded. Cannot run security scan.")
        else:
            if st.button("🔍 Scan My Withdrawal History for Anomalies", type="primary"):
                with st.spinner("Analysing your transaction patterns..."):
                    # Rename column to match security model's expected 'amount' column
                    scan_df = df_atm_habits.copy()
                    if "withdrawal_amount" in scan_df.columns and "amount" not in scan_df.columns:
                        scan_df = scan_df.rename(columns={"withdrawal_amount": "amount"})

                    summary, scored_df = security_analysis(scan_df)

                st.markdown(summary)
                st.markdown("---")

                # Visualise risk distribution
                st.subheader("Transaction Risk Breakdown")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.plotly_chart(plot_anomaly_distribution(scored_df), use_container_width=True)
                with col2:
                    risk_counts = scored_df["risk_level"].value_counts()
                    for level, count in risk_counts.items():
                        st.metric(str(level), f"{count:,} transactions")

                # Show flagged transactions
                st.subheader("Flagged Transactions (Score > 0)")
                flagged = scored_df[scored_df["anomaly_score"] > 0].sort_values("anomaly_score", ascending=False)
                if not flagged.empty:
                    display_cols = [c for c in ["timestamp", "amount", "atm_id", "card_id",
                                                "anomaly_score", "risk_level"] if c in flagged.columns]
                    st.dataframe(flagged[display_cols].head(50), use_container_width=True)
                else:
                    st.success("✅ No anomalies detected in your transaction history.")

    # =========================================================================
    # PAGE 5: Raw Data Explorer
    # =========================================================================
    elif page == "🗂️ Raw Data Explorer":
        st.title("🗂️ Raw Data Explorer")
        st.caption("Inspect the underlying datasets powering this dashboard.")
        st.markdown("---")

        datasets = {
            "Your ATM Withdrawals": df_atm_habits,
            "Cash in Circulation (M0)": df_m0,
            "World Bank ATM Density": df_sa_trend,
            "SmartCash Local ATM Data": df_smartcash,
        }

        for name, df in datasets.items():
            if df is not None:
                with st.expander(f"📋 {name} — {len(df):,} rows"):
                    st.dataframe(df.head(200), use_container_width=True)
                    st.caption(f"Showing first 200 of {len(df):,} rows. Columns: {list(df.columns)}")

        df_locations = load_branch_locations()
        if df_locations is not None:
            with st.expander(f"📋 Branch Locations — {len(df_locations):,} rows"):
                st.dataframe(df_locations, use_container_width=True)


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    main()
