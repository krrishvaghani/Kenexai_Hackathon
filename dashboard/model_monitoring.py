import streamlit as st
import pandas as pd
import json
import os

def show_model_monitoring():
    st.title("📈 Model Performance Monitoring")
    st.markdown("Track and compare machine learning model performance over time.")
    
    registry_file = "model_registry.json"
    
    # Try looking in root path if not in dashboard
    if not os.path.exists(registry_file):
        registry_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_registry.json")

    if not os.path.exists(registry_file):
        st.warning("No model registry found. Have you trained a model yet?")
        return

    with open(registry_file, "r") as f:
        try:
            registry = json.load(f)
        except:
            registry = []
            
    if not registry:
        st.warning("Model registry is empty.")
        return
        
    df_models = pd.DataFrame(registry)
    
    # Latest vs Previous metrics
    latest_model = df_models.iloc[-1]
    prev_accuracy = 0.0
    if len(df_models) > 1:
        prev_accuracy = df_models.iloc[-2]["accuracy"]
        
    latest_acc = latest_model["accuracy"]
    acc_change = latest_acc - prev_accuracy
    
    # Layout top KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Model Version", latest_model["model_version"])
    col2.metric("Latest Accuracy", f"{latest_acc:.4f}", f"{acc_change:+.4f}" if len(df_models) > 1 else None)
    col3.metric("Previous Accuracy", f"{prev_accuracy:.4f}" if len(df_models) > 1 else "N/A")
    col4.metric("Current Dataset Size", f"{latest_model['dataset_size']:,}")
    
    st.markdown("---")
    
    st.header("Performance Trends")
    
    # Transform data for plotting
    df_models["Version_Index"] = df_models["model_version"]
    df_models.set_index("Version_Index", inplace=True)
    
    # Model Accuracy Over Time
    st.subheader("Model Accuracy Over Time")
    st.line_chart(df_models[["accuracy"]], use_container_width=True)
    
    # Columns for Precision, Recall, F1
    st.subheader("Model Metrics Breakdown")
    col_p, col_r, col_f = st.columns(3)
    
    with col_p:
        st.markdown("**Precision Trend**")
        st.line_chart(df_models[["precision"]], color="#2ca02c")
        
    with col_r:
        st.markdown("**Recall Trend**")
        st.line_chart(df_models[["recall"]], color="#d62728")
        
    with col_f:
        st.markdown("**F1-score Trend**")
        st.line_chart(df_models[["f1_score"]], color="#9467bd")

    st.markdown("---")
    st.subheader("Registry History")
    st.dataframe(df_models.reset_index(drop=True), use_container_width=True)