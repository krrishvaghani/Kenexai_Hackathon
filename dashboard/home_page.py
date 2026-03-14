import streamlit as st

def show_home():
    st.title("🛡️ AI-Driven Vehicle Insurance Risk Intelligence Platform")
    st.markdown("### *An intelligent data platform for insurance risk analytics, predictive modeling, and AI-powered decision support.*")
    st.markdown("---")

    st.header("Platform Overview")
    st.markdown('''
    This platform integrates data engineering, machine learning, and generative AI to provide a comprehensive insurance analytics solution. 
    It ingests raw policy data, processes it through automated ETL pipelines, trains predictive models to estimate claim risks, and delivers actionable insights through role-based dashboards and an AI assistant.
    ''')
    st.markdown("---")

    st.header("System Architecture Overview")
    st.markdown('''
    The system follows **modern data architecture principles such as Medallion Architecture** to guarantee clean, validated information for all AI pipelines.
    
    1. **Data Sources**
    2. ⬇️ **Synthetic Data Generator**
    3. ⬇️ **Raw Data Layer (PostgreSQL - Bronze)**
    4. ⬇️ **ETL Pipeline (Data Cleaning + Feature Engineering)**
    5. ⬇️ **Processed Feature Store (Silver)**
    6. ⬇️ **Machine Learning Models & Copilot KB (Gold)**
    7. ⬇️ **Business Intelligence Dashboards & AI Insurance Copilot (RAG Chatbot)**
    ''')
    st.markdown("---")

    st.header("Platform Capabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.container(border=True).markdown('''
        #### ⚙️ Feature 1 - Data Engineering Pipeline
        Automated ETL pipelines clean and transform raw insurance data into analytics-ready datasets with feature engineering and data quality checks.
        ''')
        st.container(border=True).markdown('''
        #### 🤖 Feature 3 - Predictive Machine Learning
        ML models predict claim probability, estimate potential claim costs, and identify high-risk drivers.
        ''')
        st.container(border=True).markdown('''
        #### 💬 Feature 5 - AI Insurance Copilot
        A RAG-powered chatbot uses company data to answer analytics questions and assist decision makers.
        ''')
    with col2:
        st.container(border=True).markdown('''
        #### 🎲 Feature 2 - Synthetic Data Simulation
        A synthetic data engine continuously generates realistic insurance records to simulate production data growth.
        ''')
        st.container(border=True).markdown('''
        #### 📊 Feature 4 - Role-Based Dashboards
        Interactive dashboards provide tailored insights for risk analysts, underwriting managers, and executives.
        ''')
        
    st.markdown("---")

    st.header("Who Uses This Platform")
    col3, col4, col5 = st.columns(3)
    col3.info("**Insurance Risk Analyst**\n\nAnalyzes driver risk patterns, claim probabilities, and behavioral clusters.")
    col4.warning("**Underwriting Manager**\n\nOptimizes insurance pricing strategies and evaluates policy risk before approval.")
    col5.success("**Executive Business Leader**\n\nMonitors portfolio performance, revenue metrics, and overall insurance risk exposure.")

    st.markdown("---")

    st.header("Platform Impact")
    st.markdown('''
    This platform transforms traditional insurance analytics into an AI-driven decision intelligence system. 
    By combining data engineering, predictive modeling, and generative AI, the platform enables faster risk assessment, smarter underwriting decisions, and improved portfolio management.
    ''')