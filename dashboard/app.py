import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import urllib.parse
import os
from dotenv import load_dotenv

# -------------------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# DATABASE CONNECTION & DATA LOADING
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads processed feature data from the PostgreSQL warehouse."""
    load_dotenv()
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "Preet@3753")
    DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")
    DB_NAME = os.getenv("POSTGRES_DB", "insurance_db")

    password = urllib.parse.quote_plus(DB_PASS)
    db_uri = f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_uri)
    
    query = "SELECT * FROM processed_feature_data"
    df = pd.read_sql(query, engine)
    
    # Optional mapping for readability on target
    df['claim_outcome_label'] = df['claim_outcome'].map({1: "Claim", 0: "No Claim"})
    return df

try:
    data = load_data()
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()

# -------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to Page:", ["Data Quality Monitor", "Exploratory Data Analysis", "Business Personas"])

st.sidebar.markdown("---")
st.sidebar.info("AI-Driven Vehicle Insurance Risk Intelligence Platform")

# -------------------------------------------------------------------
# PAGE 1: DATA QUALITY MONITOR
# -------------------------------------------------------------------
if page == "Data Quality Monitor":
    st.title("🛡️ Data Quality Monitor")
    st.markdown("Assess the integrity, completeness, and cleanliness of your data warehouse tables.")
    
    # 1. KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(data):,}")
    
    duplicate_count = data.duplicated().sum()
    col2.metric("Duplicate Records", duplicate_count)
    
    total_missing = data.isna().sum().sum()
    col3.metric("Total Missing Values", total_missing)

    st.markdown("---")
    
    # 2. Missing Value & Data Type Analysis
    st.subheader("Data Types & Missing Values Consistency")
    colA, colB = st.columns(2)
    
    with colA:
        missing_df = pd.DataFrame({
            'Data Type': data.dtypes.astype(str),
            'Missing Values': data.isna().sum(),
            'Missing %': (data.isna().sum() / len(data) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)

    with colB:
        # Missing value heatmap trick for Plotly
        # taking a sample of up to 1000 rows to keep UI performant
        sample_size = min(1000, len(data))
        missing_matrix = data.sample(sample_size).isna().astype(int).T
        fig_missing = px.imshow(
            missing_matrix,
            color_continuous_scale=["#f9f9f9", "#ff4b4b"],
            aspect="auto",
            title=f"Missing Value Heatmap (Sample of {sample_size} records)",
            labels=dict(x="Record Index", y="Features", color="Is Missing")
        )
        fig_missing.update_xaxes(showticklabels=False)
        st.plotly_chart(fig_missing, use_container_width=True)

    # 3. Outlier Detection (using IQR)
    st.markdown("---")
    st.subheader("Outlier Detection (IQR Method)")
    
    numeric_cols = data.select_dtypes(include=np.number).drop(columns=['claim_outcome'], errors='ignore').columns
    
    # Calculate IQR Outliers
    Q1 = data[numeric_cols].quantile(0.25)
    Q3 = data[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    outlier_counts = ((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
    outlier_df = pd.DataFrame({'Outlier Count': outlier_counts}).reset_index().rename(columns={'index': 'Feature'})
    
    colC, colD = st.columns([1, 2])
    with colC:
        st.dataframe(outlier_df.sort_values(by="Outlier Count", ascending=False), hide_index=True, use_container_width=True)
        
    with colD:
        # Let user choose feature to see outlier boxplot
        outlier_feature = st.selectbox("Select Feature to view Boxplot:", numeric_cols)
        fig_box = px.box(data, y=outlier_feature, points="outliers", color="claim_outcome_label",
                         title=f"Boxplot Distribution: {outlier_feature}")
        st.plotly_chart(fig_box, use_container_width=True)

# -------------------------------------------------------------------
# PAGE 2: EXPLORATORY DATA ANALYSIS
# -------------------------------------------------------------------
elif page == "Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Explore variable distributions and their relationship with claim occurrences.")
    
    # Top Level Overview
    st.subheader("Target Distribution")
    fig_target = px.pie(data, names="claim_outcome_label", title="Claim Outcome Breakdown", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Feature Relationships against Claims")
    
    # Row 1 of Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Claims by Age Group
        if 'driver_age' in data.columns:
            fig_age = px.histogram(data, x="driver_age", color="claim_outcome_label", barmode="group",
                                   title="Claims by Driver Age Group")
            st.plotly_chart(fig_age, use_container_width=True)
            
    with col2:
        # 2. Claims by Vehicle Age
        if 'vehicle_age' in data.columns:
            fig_v_age = px.histogram(data, x="vehicle_age", color="claim_outcome_label", barmode="group",
                                     title="Claims by Vehicle Age Parameter")
            st.plotly_chart(fig_v_age, use_container_width=True)

    # Row 2 of Charts
    col3, col4 = st.columns(2)
    
    with col3:
        # 3. Claims by Driving Experience
        if 'driving_experience' in data.columns:
            fig_exp = px.histogram(data, x="driving_experience", color="claim_outcome_label", barmode="group",
                                   title="Claims by Driving Experience")
            st.plotly_chart(fig_exp, use_container_width=True)
            
    with col4:
        # 4. Premium vs Claim Probability (Scatter or Binning)
        fig_scatter = px.scatter(data.sample(min(2000, len(data))), x="annual_premium", y="annual_mileage", 
                                 color="claim_outcome_label", opacity=0.6,
                                 title="Premium vs Annual Mileage (Sampled)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    
    # 5. Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    num_df = data.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                         color_continuous_scale="RdBu_r", 
                         title="Pearson Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)



# -------------------------------------------------------------------
# PAGE 3: BUSINESS PERSONAS
# -------------------------------------------------------------------
elif page == "Business Personas":
    st.title("Role-Based Business Dashboards")
    st.markdown("Tailored decision insights for key stakeholders in the insurance ecosystem.")
    
    # Create the three Persona tabs
    tab_risk, tab_underwriter, tab_exec = st.tabs([
        "Insurance Risk Analyst", 
        "Underwriting Manager", 
        "Executive Business Leader"
    ])
    
    # =======================================================
    # PERSONA 1: RISK ANALYST
    # =======================================================
    with tab_risk:
        st.header("Risk Analyst View")
        st.write("Focus: Granular driver risk metrics and immediate claim likelihood factors.")
        
        total_claims = data['claim_outcome'].sum()
        high_risk_threshold = data['driver_risk_score'].quantile(0.85)
        high_risk_drivers = len(data[data['driver_risk_score'] >= high_risk_threshold])
        avg_risk_score = data['driver_risk_score'].mean()
        
        # Risk Analyst KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Claims Flagged", f"{total_claims:,}")
        c2.metric("High Risk Drivers Identified", f"{high_risk_drivers:,}")
        c3.metric("Avg Driver Risk Score", f"{avg_risk_score:.2f}")
        
        st.markdown("---")
        
        # Risk Analyst Charts
        r1_col1, r1_col2 = st.columns(2)
        
        with r1_col1:
            fig_risk_dist = px.histogram(data, x="driver_risk_score", color="claim_outcome_label", 
                                         nbins=30, title="Driver Risk Score Distribution")
            st.plotly_chart(fig_risk_dist, use_container_width=True)
            
        with r1_col2:
            age_claim_rate = data.groupby("driver_age")["claim_outcome"].mean().reset_index()
            age_claim_rate["Claim Rate %"] = age_claim_rate["claim_outcome"] * 100
            fig_age_claim = px.bar(age_claim_rate, x="driver_age", y="Claim Rate %", 
                                   title="Driver Age vs Claim Rate (%)", color="Claim Rate %", color_continuous_scale="Reds")
            st.plotly_chart(fig_age_claim, use_container_width=True)
            
        st.subheader("Claims by Driving Violations (Proxy)")
        # Calculate a proxy for driving violations using risk_score / driving_experience length logic mapping
        # Or just using driving_experience directly
        fig_viol = px.histogram(data, x="driving_experience", color="claim_outcome_label", 
                                barmode="group",
                                title="Claims Breakdown by Driving Experience Tiers")
        st.plotly_chart(fig_viol, use_container_width=True)


    # =======================================================
    # PERSONA 2: UNDERWRITING MANAGER
    # =======================================================
    with tab_underwriter:
        st.header("Underwriting Manager View")
        st.write("Focus: Pricing optimization, premium relationships, and vehicle-level risk.")
        
        avg_premium = data['annual_premium'].mean()
        high_prem_threshold = data['annual_premium'].quantile(0.85)
        high_prem_policies = len(data[data['annual_premium'] >= high_prem_threshold])
        
        # Finding the top vehicle age claim probability
        v_age_prob = data.groupby('vehicle_age')['claim_outcome'].mean()
        highest_v_age_prob = v_age_prob.idxmax()
        highest_v_age_val = v_age_prob.max() * 100
        
        # Underwriter KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Annual Premium", f"${avg_premium:,.2f}")
        c2.metric("High Premium Policies", f"{high_prem_policies:,}")
        c3.metric(f"Highest Risk Vehicle Age (Class {highest_v_age_prob})", f"{highest_v_age_val:.1f}% Claim Rate")
        
        st.markdown("---")
        
        # Underwriter Charts
        u1_col1, u1_col2 = st.columns(2)
        
        with u1_col1:
            fig_prem_risk = px.scatter(data.sample(min(2000, len(data))), x="driver_risk_score", y="annual_premium", 
                                       color="claim_outcome_label", opacity=0.5,
                                       title="Premium vs Claim Risk Factor")
            st.plotly_chart(fig_prem_risk, use_container_width=True)
            
        with u1_col2:
            fig_v_risk = px.histogram(data, x="vehicle_risk_score", color="claim_outcome_label", barmode='group',
                                      title="Vehicle Risk Score Distribution")
            fig_v_risk.update_layout(xaxis_type='category')
            st.plotly_chart(fig_v_risk, use_container_width=True)
            
        st.subheader("Annual Mileage vs Claim Probability")
        fig_mileage = px.box(data, x="claim_outcome_label", y="annual_mileage", color="claim_outcome_label",
                              title="Mileage Spread by Claim Outcome")
        st.plotly_chart(fig_mileage, use_container_width=True)


    # =======================================================
    # PERSONA 3: EXECUTIVE BUSINESS LEADER
    # =======================================================
    with tab_exec:
        st.header("Executive Business Leader View")
        st.write("Focus: High-level portfolio health, revenue generation, and overarching loss metrics.")
        
        total_policies = len(data)
        claim_rate = (data['claim_outcome'].sum() / total_policies) * 100
        total_premium = data['annual_premium'].sum()
        
        # Assuming an average standard payout of $5,200 per claim for Executive metrics logic if actual isn't explicitly held
        assumed_payout_per_claim = 5200.0
        total_claim_payout = total_claims * assumed_payout_per_claim
        
        # Executive KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Active Policies", f"{total_policies:,}")
        c2.metric("Portfolio Claim Rate", f"{claim_rate:.1f}%")
        c3.metric("Total Premium Collected", f"${total_premium:,.0f}")
        c4.metric("Estimated Total Payout", f"${total_claim_payout:,.0f}", delta=f"{total_premium - total_claim_payout:,.0f} Margin")
        
        st.markdown("---")
        
        e1_col1, e1_col2 = st.columns(2)
        
        with e1_col1:
            # Segment the portfolio into risk tiers
            # Because driver_risk_score can have many 0s (safe drivers), qcut bins overlap. Adding duplicates="drop" handles this.
            # Using cut instead to guarantee safety
            
            # Using custom bins to guarantee segments based on the explicit spread of scores
            risk_bins = [-1, data['driver_risk_score'].mean(), data['driver_risk_score'].quantile(0.85), data['driver_risk_score'].max() + 1]
            risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
            # If the max isn't high enough to distinct the 85th percentile, fallback to basic cut limit
            if risk_bins[1] >= risk_bins[2]:
                data['Risk_Segment'] = pd.cut(data['driver_risk_score'], bins=3, labels=risk_labels)
            else:
                try:
                    data['Risk_Segment'] = pd.cut(data['driver_risk_score'], bins=risk_bins, labels=risk_labels)
                except ValueError:
                    data['Risk_Segment'] = pd.qcut(data['driver_risk_score'], q=3, labels=risk_labels, duplicates='drop')

            fig_segment = px.pie(data, names='Risk_Segment', title="Policy Risk Segmentation", 
                                 hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_segment, use_container_width=True)
            
        with e1_col2:
            # Financial Comparison
            fin_df = pd.DataFrame({
                "Category": ["Total Premium Revenue", "Estimated Claim Payout"],
                "Amount": [total_premium, total_claim_payout]
            })
            fig_fin = px.bar(fin_df, x="Category", y="Amount", color="Category", text="Amount",
                             title="Premium Revenue vs Estimated Claims")
            fig_fin.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_fin.update_layout(showlegend=False)
            st.plotly_chart(fig_fin, use_container_width=True)

        st.subheader("Monthly Claim Trends (Simulated YTD)")
        
        # Generating synthetic time series for executive view simulating year-to-date monthly data
        np.random.seed(total_policies)
        # Create a random distribution of months reflecting slight seasonality
        months = pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%Y-%m")
        sys_months = np.random.choice(months, size=len(data))
        data['policy_month'] = sys_months
        
        monthly_trend = data.groupby('policy_month')['claim_outcome'].sum().reset_index()
        fig_trend = px.line(monthly_trend, x='policy_month', y='claim_outcome', markers=True,
                            title="Total Claims Flagged Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)
        
