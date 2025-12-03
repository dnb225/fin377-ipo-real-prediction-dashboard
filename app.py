"""
IPO Risk Prediction Dashboard
Real SDC Platinum Data (1980-2017)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="IPO Risk Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        with open('models/best_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        with open('models/best_regressor.pkl', 'rb') as f:
            regressor = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        return classifier, regressor, scaler, feature_columns, metadata
    except FileNotFoundError:
        st.error("Model files not found. Please run all Jupyter notebooks (Parts 1-5) first to train models.")
        return None, None, None, None, None

@st.cache_data
def load_data():
    """Load test predictions and results"""
    try:
        test_preds = pd.read_csv('data/final/test_predictions.csv')
        clf_results = pd.read_csv('data/final/classification_results.csv')
        reg_results = pd.read_csv('data/final/regression_results.csv')
        strategy_results = pd.read_csv('data/final/strategy_summary.csv')
        feature_importance = pd.read_csv('data/final/feature_importance.csv')

        return test_preds, clf_results, reg_results, strategy_results, feature_importance
    except FileNotFoundError:
        st.error("Data files not found. Please run all Jupyter notebooks (Parts 1-5) first.")
        return None, None, None, None, None

# Load everything
classifier, regressor, scaler, feature_columns, metadata = load_models()
test_preds, clf_results, reg_results, strategy_results, feature_importance = load_data()

# Check if data loaded successfully
if classifier is None or test_preds is None:
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Introduction", "Home & IPO Search", "Model Performance", "Investment Strategies", "Feature Analysis", "IPO Sandbox"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **IPO Risk Prediction Dashboard**
    
    This dashboard uses machine learning to predict IPO first-day returns and identify high-risk offerings.
    
    **Data:** SDC Platinum IPO Database (1980-2017)
    
    **Models:** Classification & Regression
    
    **Team:** JLD Inc. LLC. Partners
    - Logan Wesselt
    - Julian Tashjian  
    - Dylan Bollinger
    
    **Course:** FIN 377 - Lehigh University
    """
)

# ============================================================================
# PAGE 0: INTRODUCTION
# ============================================================================
if page == "Introduction":
    st.title("📈 IPO Risk Prediction Dashboard")
    st.markdown("### Machine Learning for Intelligent IPO Investment Decisions")

    st.markdown("---")

    # Project overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total IPOs", f"{metadata['n_ipos']:,}" if metadata else "N/A")
    with col2:
        st.metric("Time Period", metadata.get('date_range', 'N/A') if metadata else "N/A")
    with col3:
        st.metric("Features", metadata.get('n_features', 'N/A') if metadata else "N/A")

    st.markdown("---")

    # Research question
    st.markdown("## 🎯 Research Question")
    st.markdown("""
    **Can machine learning models, using only information available before an IPO's first trading day, 
    predict first-day returns and identify high-risk IPOs prone to large negative price moves?**
    
    This question is highly relevant because all investors—retail or institutional—enter an IPO allocation 
    decision with limited information. If machine learning can extract predictive patterns from pre-IPO data, 
    it offers practical value for investment decisions.
    """)

    st.markdown("---")

    # Key findings
    st.markdown("## 📊 Key Findings")

    if clf_results is not None and len(clf_results) > 0:
        best_auc = clf_results['Validation AUC'].max()

        if best_auc >= 0.75:
            perf_desc = "strong discriminatory power"
            perf_icon = "✅"
        elif best_auc >= 0.65:
            perf_desc = "moderate predictive ability"
            perf_icon = "✓"
        elif best_auc >= 0.55:
            perf_desc = "weak but above-random predictive ability"
            perf_icon = "⚠️"
        else:
            perf_desc = "limited predictive power (near-random performance)"
            perf_icon = "❌"

        st.markdown(f"""
        **{perf_icon} Classification Performance:** The model shows {perf_desc} for identifying 
        high-risk IPOs (AUC: {best_auc:.3f}).
        
        **📈 Data Source:** Analysis based on {len(test_preds):,} IPOs from the SDC Platinum database 
        spanning {metadata.get('date_range', '1980-2017')}.
        
        **💡 Investment Strategies:** ML-driven investment strategies demonstrate improved risk-adjusted 
        returns compared to naive baseline approaches.
        """)

    st.markdown("---")

    # Methodology
    st.markdown("## 🔬 Methodology")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Classification Models")
        st.markdown("""
        **Objective:** Predict high-risk IPOs (bottom 25% returns)
        
        **Models Tested:**
        - Logistic Regression
        - Random Forest
        - XGBoost
        - LightGBM
        
        **Metric:** AUC-ROC Score
        
        **Best Model:** {0}
        """.format(clf_results.iloc[0]['Model'] if clf_results is not None else "N/A"))

    with col2:
        st.markdown("### Regression Models")
        st.markdown("""
        **Objective:** Predict numerical first-day returns
        
        **Models Tested:**
        - Linear Regression
        - Ridge Regression
        - Decision Tree
        - Random Forest
        - XGBoost
        - LightGBM
        
        **Metrics:** MAE, RMSE, R²
        
        **Best Model:** {0}
        """.format(reg_results.iloc[0]['Model'] if reg_results is not None else "N/A"))

    st.markdown("---")

    # Features
    st.markdown("## 📋 Features Used")

    st.markdown("""
    Our models use only information available **before** the IPO begins trading:
    
    **Deal Structure:**
    - Offer price, gross proceeds, shares offered
    - Price revision from filing range
    - Primary vs secondary shares
    
    **Firm Characteristics:**
    - Firm age at IPO
    - Revenue and assets
    - Industry classification
    
    **Quality Indicators:**
    - VC-backed status
    - Underwriter quality (amendment count)
    
    **Market Conditions:**
    - Hot market periods (1995-2000, 2013+)
    - Crisis periods (2001-2002, 2008-2009)
    - IPO timing (year, quarter, month)
    
    **Total Features:** {0}
    """.format(len(feature_columns) if feature_columns else "N/A"))

    st.markdown("---")

    # How to use
    st.markdown("## 🧭 How to Use This Dashboard")

    st.markdown("""
    1. **Home & IPO Search:** Explore individual IPO predictions and search by company
    2. **Model Performance:** View detailed model evaluation metrics and comparisons
    3. **Investment Strategies:** Compare ML-driven investment strategies against baseline
    4. **Feature Analysis:** Understand which features drive predictions
    5. **IPO Sandbox:** Create hypothetical IPO scenarios and see predictions
    """)

# ============================================================================
# PAGE 1: HOME & IPO SEARCH
# ============================================================================
elif page == "Home & IPO Search":
    st.title("🏠 Home & IPO Search")
    st.markdown("### Explore Individual IPO Predictions")

    st.markdown("---")

    # Summary statistics
    st.markdown("## 📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total IPOs", f"{len(test_preds):,}")
    with col2:
        st.metric("Avg First-Day Return", f"{test_preds['first_day_return'].mean()*100:.2f}%")
    with col3:
        st.metric("High-Risk Rate", f"{test_preds['high_risk'].mean()*100:.1f}%")
    with col4:
        st.metric("VC-Backed Rate", f"{test_preds['vc_backed'].mean()*100:.0f}%")

    st.markdown("---")

    # Filters
    st.markdown("## 🔍 Search & Filter")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Industry filter
        industries = ['All'] + sorted(test_preds['industry'].unique().tolist()) if 'industry' in test_preds.columns else ['All']
        industry_filter = st.selectbox("Industry", industries)

    with col2:
        # Year filter
        years = ['All'] + sorted(test_preds['ipo_year'].unique().tolist(), reverse=True) if 'ipo_year' in test_preds.columns else ['All']
        year_filter = st.selectbox("Year", years)

    with col3:
        # Search by company
        search_term = st.text_input("Search Company", placeholder="Enter company name...")

    # Apply filters
    filtered_df = test_preds.copy()

    if industry_filter != 'All' and 'industry' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['industry'] == industry_filter]

    if year_filter != 'All' and 'ipo_year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ipo_year'] == int(year_filter)]

    if search_term:
        filtered_df = filtered_df[filtered_df['Issuer'].str.contains(search_term, case=False, na=False)]

    st.markdown(f"**Showing {len(filtered_df)} IPOs**")

    # Display results
    if len(filtered_df) > 0:
        # Sort options
        sort_by = st.selectbox(
            "Sort by:",
            ["Predicted Return (High to Low)", "Predicted Return (Low to High)",
             "Actual Return (High to Low)", "Risk Probability (High to Low)"]
        )

        if "High to Low" in sort_by:
            ascending = False
        else:
            ascending = True

        if "Predicted Return" in sort_by:
            sort_col = 'predicted_return'
        elif "Actual Return" in sort_by:
            sort_col = 'first_day_return'
        else:
            sort_col = 'risk_probability'

        filtered_df = filtered_df.sort_values(sort_col, ascending=ascending)

        # Display table
        display_cols = ['Issuer', 'ipo_year', 'industry', 'OfferPrice', 'first_day_return',
                       'predicted_return', 'risk_probability', 'vc_backed']
        display_cols = [col for col in display_cols if col in filtered_df.columns]

        st.dataframe(
            filtered_df[display_cols].head(20),
            column_config={
                "first_day_return": st.column_config.NumberColumn("Actual Return", format="%.2f%%"),
                "predicted_return": st.column_config.NumberColumn("Predicted Return", format="%.2f%%"),
                "risk_probability": st.column_config.NumberColumn("Risk Prob", format="%.1f%%"),
                "OfferPrice": st.column_config.NumberColumn("Offer Price", format="$%.2f")
            },
            hide_index=True,
            use_container_width=True
        )

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="ipo_predictions.csv",
            mime="text/csv"
        )

        # Sample detailed view
        st.markdown("---")
        st.markdown("## 🔬 Detailed View")

        selected_ipo = st.selectbox(
            "Select an IPO for detailed view:",
            filtered_df['Issuer'].tolist()
        )

        if selected_ipo:
            row = filtered_df[filtered_df['Issuer'] == selected_ipo].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Company Info**")
                st.write(f"Company: {row['Issuer']}")
                if 'TickerSymbol' in row:
                    st.write(f"Ticker: {row['TickerSymbol']}")
                if 'ipo_year' in row:
                    st.write(f"IPO Year: {row['ipo_year']:.0f}")

            with col2:
                st.markdown("**Predictions**")
                if 'predicted_return' in row:
                    st.write(f"Predicted Return: {row['predicted_return']*100:.2f}%")
                if 'risk_probability' in row:
                    st.write(f"Risk Probability: {row['risk_probability']*100:.1f}%")
                if 'predicted_high_risk' in row:
                    st.write(f"Risk Classification: {'High Risk' if row['predicted_high_risk'] else 'Low Risk'}")

            with col3:
                st.markdown("**Actual Outcome**")
                if 'first_day_return' in row:
                    st.write(f"Actual Return: {row['first_day_return']*100:.2f}%")
                if 'high_risk' in row:
                    st.write(f"Actual Risk: {'High Risk' if row['high_risk'] else 'Low Risk'}")
                if 'correct_classification' in row:
                    st.write(f"Correct Prediction: {'✓ Yes' if row['correct_classification'] else '✗ No'}")
    else:
        st.warning("No IPOs match your search criteria.")

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.title("🎯 Model Performance Analysis")
    st.markdown("### Detailed Evaluation of ML Models")

    st.markdown("---")

    # Classification Results
    st.markdown("## Classification Models (High-Risk Detection)")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot model comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=clf_results['Validation AUC'],
            y=clf_results['Model'],
            orientation='h',
            marker=dict(
                color=clf_results['Validation AUC'],
                colorscale='Viridis',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=[f"{x:.3f}" for x in clf_results['Validation AUC']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Classification Model Performance (AUC-ROC)",
            xaxis_title="AUC Score (Higher is Better)",
            yaxis_title="",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Random")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Best Model")
        best_clf = clf_results.iloc[0]
        st.metric("Model", best_clf['Model'])
        st.metric("Validation AUC", f"{best_clf['Validation AUC']:.3f}")

        # Interpretation
        auc = best_clf['Validation AUC']
        if auc >= 0.75:
            st.success("✅ Excellent discriminatory power")
        elif auc >= 0.65:
            st.info("✓ Good predictive ability")
        elif auc >= 0.55:
            st.warning("⚠️ Weak but above-random")
        else:
            st.error("❌ Near-random performance")

    # Full results table
    with st.expander("View All Classification Results"):
        st.dataframe(clf_results, use_container_width=True)

    st.markdown("---")

    # Regression Results
    st.markdown("## Regression Models (Return Prediction)")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot model comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=reg_results['Validation MAE'],
            y=reg_results['Model'],
            orientation='h',
            marker=dict(
                color=reg_results['Validation MAE'],
                colorscale='Reds',
                reversescale=True,
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=[f"{x:.4f}" for x in reg_results['Validation MAE']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Regression Model Performance (MAE)",
            xaxis_title="Mean Absolute Error (Lower is Better)",
            yaxis_title="",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Best Model")
        best_reg = reg_results.iloc[0]
        st.metric("Model", best_reg['Model'])
        st.metric("Validation MAE", f"{best_reg['Validation MAE']:.4f}")
        if 'Validation R²' in best_reg:
            st.metric("R² Score", f"{best_reg['Validation R²']:.4f}")

    # Full results table
    with st.expander("View All Regression Results"):
        st.dataframe(reg_results, use_container_width=True)

    st.markdown("---")

    # Sample predictions
    st.markdown("## 🔬 Sample Predictions")

    sample_preds = test_preds.head(10).copy()

    # Create comparison chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sample_preds.index,
        y=sample_preds['first_day_return'] * 100,
        mode='markers',
        name='Actual',
        marker=dict(size=10, color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=sample_preds.index,
        y=sample_preds['predicted_return'] * 100,
        mode='markers',
        name='Predicted',
        marker=dict(size=10, color='red', symbol='x')
    ))

    fig.update_layout(
        title="Actual vs Predicted Returns (Sample)",
        xaxis_title="IPO Index",
        yaxis_title="First-Day Return (%)",
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(240,240,240,0.5)'
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: INVESTMENT STRATEGIES (ENHANCED)
# ============================================================================
elif page == "Investment Strategies":
    st.title("💼 Investment Strategy Analysis")
    st.markdown("### Compare ML-Driven Investment Approaches Against Baseline")

    st.markdown("---")

    # Calculate true best strategy based on composite score
    strategy_results_copy = strategy_results.copy()
    strategy_results_copy['composite_score'] = (
        strategy_results_copy['Avg Return (%)'] * 0.6 +
        strategy_results_copy['Sharpe Ratio'] * 20 * 0.4
    )
    best_strategy_idx = strategy_results_copy['composite_score'].idxmax()
    best_strategy = strategy_results_copy.loc[best_strategy_idx]

    # Get baseline
    baseline_mask = strategy_results['Strategy'] == 'Buy All'
    if baseline_mask.any():
        baseline_strategy = strategy_results[baseline_mask].iloc[0]
        baseline_return = baseline_strategy['Avg Return (%)']
        baseline_sharpe = baseline_strategy['Sharpe Ratio']
    else:
        baseline_return = strategy_results['Avg Return (%)'].mean()
        baseline_sharpe = strategy_results['Sharpe Ratio'].mean()

    # Key Metrics
    st.markdown("## 🎯 Performance Highlights")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🏆 Best Strategy",
            best_strategy['Strategy'],
            help="Based on composite score of return and Sharpe ratio"
        )
    with col2:
        st.metric(
            "📈 Avg Return",
            f"{best_strategy['Avg Return (%)']:.2f}%",
            delta=f"{best_strategy['Avg Return (%)'] - baseline_return:.2f}pp vs baseline"
        )
    with col3:
        st.metric(
            "⚖️ Sharpe Ratio",
            f"{best_strategy['Sharpe Ratio']:.3f}",
            delta=f"{best_strategy['Sharpe Ratio'] - baseline_sharpe:.3f} vs baseline"
        )
    with col4:
        if 'High-Risk Rate (%)' in best_strategy:
            baseline_risk = baseline_strategy.get('High-Risk Rate (%)', 0) if baseline_mask.any() else 0
            st.metric(
                "🛡️ High Risk Rate",
                f"{best_strategy['High-Risk Rate (%)']:.1f}%",
                delta=f"{baseline_risk - best_strategy['High-Risk Rate (%)']:.1f}% lower",
                delta_color="inverse"
            )

    st.markdown("---")

    # Interactive filters
    st.markdown("## 🔍 Interactive Strategy Explorer")

    col_filter1, col_filter2 = st.columns([2, 1])

    with col_filter1:
        selected_strategies = st.multiselect(
            "Select strategies to compare:",
            options=strategy_results['Strategy'].tolist(),
            default=strategy_results['Strategy'].tolist(),
            help="Choose which strategies to display"
        )

    with col_filter2:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Avg Return (%)', 'Sharpe Ratio', 'High-Risk Rate (%)'] if 'High-Risk Rate (%)' in strategy_results.columns else ['Avg Return (%)', 'Sharpe Ratio'],
            index=0
        )

    # Filter data
    if selected_strategies:
        filtered_strategies = strategy_results[strategy_results['Strategy'].isin(selected_strategies)].copy()
        ascending = (sort_by == 'High-Risk Rate (%)')
        filtered_strategies = filtered_strategies.sort_values(by=sort_by, ascending=ascending)
    else:
        filtered_strategies = strategy_results.copy()
        st.warning("⚠️ Please select at least one strategy")

    st.markdown("---")

    # Risk vs Return Scatter
    if len(filtered_strategies) > 0 and 'High-Risk Rate (%)' in filtered_strategies.columns:
        st.markdown("## 📊 Risk-Return Profile")
        st.markdown("*Higher returns with lower risk = better performance*")

        fig_scatter = go.Figure()

        colors = filtered_strategies['Sharpe Ratio'].values
        sizes = filtered_strategies.get('IPOs Invested', [20]*len(filtered_strategies)) / 2

        fig_scatter.add_trace(go.Scatter(
            x=filtered_strategies['High-Risk Rate (%)'],
            y=filtered_strategies['Avg Return (%)'],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sharpe<br>Ratio"),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=filtered_strategies['Strategy'],
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><br>' +
                          'Return: %{y:.2f}%<br>' +
                          'Risk Rate: %{x:.1f}%<br>' +
                          '<extra></extra>'
        ))

        fig_scatter.update_layout(
            title="Risk vs. Return: Strategy Performance Map",
            xaxis_title="High-Risk Rate (%) - Lower is Better",
            yaxis_title="Average Return (%) - Higher is Better",
            height=500,
            plot_bgcolor='rgba(240,240,240,0.5)',
            showlegend=False
        )

        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("---")

    # Performance Charts
    st.markdown("## 📈 Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Average Return
        fig_return = go.Figure()
        fig_return.add_trace(go.Bar(
            y=filtered_strategies['Strategy'],
            x=filtered_strategies['Avg Return (%)'],
            orientation='h',
            marker=dict(
                color=filtered_strategies['Avg Return (%)'],
                colorscale='Viridis',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=[f"{x:.2f}%" for x in filtered_strategies['Avg Return (%)']],
            textposition='outside'
        ))

        fig_return.add_vline(x=baseline_return, line_dash="dash", line_color="red",
                            annotation_text=f"Baseline: {baseline_return:.2f}%")

        fig_return.update_layout(
            title="Average First-Day Return",
            xaxis_title="Return (%)",
            yaxis_title="",
            height=400,
            plot_bgcolor='rgba(240,240,240,0.5)'
        )

        st.plotly_chart(fig_return, use_container_width=True)

        # High-Risk Rate
        if 'High-Risk Rate (%)' in filtered_strategies.columns:
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                y=filtered_strategies['Strategy'],
                x=filtered_strategies['High-Risk Rate (%)'],
                orientation='h',
                marker=dict(
                    color=filtered_strategies['High-Risk Rate (%)'],
                    colorscale='Reds',
                    reversescale=True,
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=[f"{x:.1f}%" for x in filtered_strategies['High-Risk Rate (%)']],
                textposition='outside'
            ))

            fig_risk.update_layout(
                title="High-Risk IPO Rate (Lower is Better)",
                xaxis_title="High-Risk Rate (%)",
                yaxis_title="",
                height=400,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

            st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        # Sharpe Ratio
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Bar(
            y=filtered_strategies['Strategy'],
            x=filtered_strategies['Sharpe Ratio'],
            orientation='h',
            marker=dict(
                color=filtered_strategies['Sharpe Ratio'],
                colorscale='Bluered',
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=[f"{x:.3f}" for x in filtered_strategies['Sharpe Ratio']],
            textposition='outside'
        ))

        fig_sharpe.add_vline(x=baseline_sharpe, line_dash="dash", line_color="red",
                            annotation_text=f"Baseline: {baseline_sharpe:.3f}")

        fig_sharpe.update_layout(
            title="Risk-Adjusted Performance (Sharpe Ratio)",
            xaxis_title="Sharpe Ratio",
            yaxis_title="",
            height=400,
            plot_bgcolor='rgba(240,240,240,0.5)'
        )

        st.plotly_chart(fig_sharpe, use_container_width=True)

        # IPOs Invested
        if 'IPOs Invested' in filtered_strategies.columns:
            fig_count = go.Figure()
            fig_count.add_trace(go.Bar(
                y=filtered_strategies['Strategy'],
                x=filtered_strategies['IPOs Invested'],
                orientation='h',
                marker=dict(
                    color=filtered_strategies['IPOs Invested'],
                    colorscale='Blues',
                    showscale=False,
                    line=dict(color='white', width=1)
                ),
                text=filtered_strategies['IPOs Invested'].astype(int),
                textposition='outside'
            ))

            fig_count.update_layout(
                title="Number of IPOs Invested",
                xaxis_title="IPO Count",
                yaxis_title="",
                height=400,
                plot_bgcolor='rgba(240,240,240,0.5)'
            )

            st.plotly_chart(fig_count, use_container_width=True)

    st.markdown("---")

    # Detailed table
    st.markdown("## 📋 Detailed Strategy Metrics")
    st.dataframe(filtered_strategies, use_container_width=True, height=300)

    # Download
    csv = filtered_strategies.to_csv(index=False)
    st.download_button(
        label="📥 Download Strategy Results",
        data=csv,
        file_name="strategy_comparison.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Strategy Descriptions
    st.markdown("## 📚 Strategy Descriptions")

    strategies_info = {
        "Buy All": {
            "description": "**Baseline strategy** - invest in all IPOs without selection criteria",
            "pros": "Maximum diversification",
            "cons": "Includes high-risk IPOs"
        },
        "Avoid High Risk": {
            "description": "**Risk-averse** - invest only in IPOs with low predicted risk",
            "pros": "Reduces downside exposure",
            "cons": "May miss high-return opportunities"
        },
        "Top 25% Returns": {
            "description": "**Growth-focused** - invest in top quartile predicted returns",
            "pros": "Maximizes potential returns",
            "cons": "Higher risk concentration"
        },
        "Combined Strategy": {
            "description": "**Balanced** - low risk AND above-median returns",
            "pros": "Best risk-adjusted returns",
            "cons": "Fewer opportunities"
        },
        "High Confidence": {
            "description": "**Selective** - invest only with high model confidence",
            "pros": "High conviction trades",
            "cons": "Very selective"
        }
    }

    for strategy in filtered_strategies['Strategy']:
        if strategy in strategies_info:
            info = strategies_info[strategy]
            with st.expander(f"**{strategy}** {'🏆' if strategy == best_strategy['Strategy'] else ''}"):
                st.markdown(info["description"])

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**✅ Pros:** {info['pros']}")
                with col2:
                    st.markdown(f"**❌ Cons:** {info['cons']}")

                if strategy in strategy_results['Strategy'].values:
                    row = strategy_results[strategy_results['Strategy'] == strategy].iloc[0]
                    st.markdown("**Performance:**")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        st.metric("Avg Return", f"{row['Avg Return (%)']:.2f}%")
                    with perf_col2:
                        st.metric("Sharpe Ratio", f"{row['Sharpe Ratio']:.3f}")
                    with perf_col3:
                        if 'IPOs Invested' in row:
                            st.metric("IPOs", f"{int(row['IPOs Invested'])}")

# ============================================================================
# PAGE 4: FEATURE ANALYSIS
# ============================================================================
elif page == "Feature Analysis":
    st.title("🔬 Feature Importance Analysis")
    st.markdown("### Understanding What Drives IPO Performance Predictions")

    st.markdown("---")

    # Top features
    st.markdown("## 🏆 Most Important Features")

    top_10_features = feature_importance.head(10)['feature'].tolist()

    col1, col2 = st.columns(2)

    with col1:
        # Classification importance
        fig = go.Figure()
        top_clf = feature_importance.nlargest(15, 'classification_importance')

        fig.add_trace(go.Bar(
            y=top_clf['feature'],
            x=top_clf['classification_importance'],
            orientation='h',
            marker=dict(
                color=top_clf['classification_importance'],
                colorscale='Blues',
                showscale=False
            ),
            name='Classification'
        ))

        fig.update_layout(
            title="Top Features for Risk Classification",
            xaxis_title="Importance",
            yaxis_title="",
            height=500,
            showlegend=False
        )
        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Regression importance
        fig = go.Figure()
        top_reg = feature_importance.nlargest(15, 'regression_importance')

        fig.add_trace(go.Bar(
            y=top_reg['feature'],
            x=top_reg['regression_importance'],
            orientation='h',
            marker=dict(
                color=top_reg['regression_importance'],
                colorscale='Oranges',
                showscale=False
            ),
            name='Regression'
        ))

        fig.update_layout(
            title="Top Features for Return Prediction",
            xaxis_title="Importance",
            yaxis_title="",
            height=500,
            showlegend=False
        )
        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Combined importance
    st.markdown("## 📊 Combined Feature Importance")

    feature_importance['avg_importance'] = (
        feature_importance['classification_importance'] +
        feature_importance['regression_importance']
    ) / 2

    top_combined = feature_importance.nlargest(20, 'avg_importance')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Classification',
        y=top_combined['feature'],
        x=top_combined['classification_importance'],
        orientation='h',
        marker=dict(color='steelblue')
    ))

    fig.add_trace(go.Bar(
        name='Regression',
        y=top_combined['feature'],
        x=top_combined['regression_importance'],
        orientation='h',
        marker=dict(color='coral')
    ))

    fig.update_layout(
        title="Top 20 Features: Classification vs Regression",
        xaxis_title="Importance",
        yaxis_title="",
        barmode='group',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature insights
    st.markdown("## 💡 Key Insights")

    st.markdown("""
    The most important features for predicting IPO outcomes include:
    
    **Deal Structure:**
    - Price revision from filing range (strong signal of demand)
    - Offer price and gross proceeds
    - Share structure (primary vs secondary)
    
    **Firm Characteristics:**
    - Firm age (younger firms = higher volatility)
    - Financial metrics (revenue, assets)
    - VC-backing status
    
    **Market Timing:**
    - Hot market periods (1995-2000, 2013+)
    - Crisis periods (2001-2002, 2008-2009)
    - Seasonal effects (quarter, month)
    
    **Interaction Effects:**
    - Technology + VC-backed
    - Young firm + VC-backed
    - Large offer + established firm
    """)

    # Full table
    with st.expander("View All Feature Importance"):
        st.dataframe(
            feature_importance[['feature', 'classification_importance', 'regression_importance', 'avg_importance']].sort_values('avg_importance', ascending=False),
            use_container_width=True
        )

# ============================================================================
# PAGE 5: IPO SANDBOX
# ============================================================================
elif page == "IPO Sandbox":
    st.title("🧪 IPO Sandbox: Create Your Scenario")
    st.markdown("### Build a Custom IPO and See Model Predictions")

    st.markdown("---")

    st.markdown("""
    Use the controls below to create a hypothetical IPO scenario. The models will predict 
    the expected first-day return and risk classification based on your inputs.
    """)

    st.markdown("---")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💼 Deal Structure")

        offer_price = st.slider("Offer Price ($)", 5.0, 50.0, 15.0, 0.5)
        gross_proceeds = st.slider("Gross Proceeds ($M)", 10.0, 500.0, 50.0, 10.0)
        price_revision = st.slider("Price Revision (%)", -30.0, 30.0, 0.0, 1.0) / 100
        primary_pct = st.slider("Primary Shares (%)", 50.0, 100.0, 100.0, 5.0) / 100

        st.markdown("### 🏢 Firm Characteristics")

        firm_age = st.slider("Firm Age (years)", 0.0, 30.0, 7.0, 0.5)
        revenue = st.slider("Revenue ($M)", 0.0, 500.0, 50.0, 10.0)
        assets = st.slider("Assets ($M)", 0.0, 500.0, 50.0, 10.0)

    with col2:
        st.markdown("### 🏆 Quality Indicators")

        vc_backed = st.selectbox("VC-Backed", ["Yes", "No"])
        vc_backed_val = 1 if vc_backed == "Yes" else 0

        industry = st.selectbox("Industry", ["Technology", "Healthcare", "Financial", "Other"])

        underwriter_rank = st.slider("Underwriter Quality (0-10)", 0, 10, 7)

        st.markdown("### 📅 Market Conditions")

        ipo_year = st.slider("IPO Year", 2015, 2025, 2024)
        ipo_month = st.slider("IPO Month", 1, 12, 6)

        hot_market = 1 if ipo_year >= 2013 else 0
        crisis_period = 0

    st.markdown("---")

    # Create input vector
    if st.button("🔮 Generate Prediction", type="primary"):
        # Create feature vector matching training features
        input_data = {}

        # Basic features
        input_data['OfferPrice'] = offer_price
        input_data['firm_age'] = firm_age
        input_data['vc_backed'] = vc_backed_val
        input_data['price_revision'] = price_revision
        input_data['gross_proceeds'] = gross_proceeds
        input_data['log_proceeds'] = np.log1p(gross_proceeds)
        input_data['revenue'] = revenue
        input_data['log_revenue'] = np.log1p(revenue)
        input_data['assets'] = assets
        input_data['log_assets'] = np.log1p(assets)
        input_data['primary_pct'] = primary_pct
        input_data['underwriter_rank'] = underwriter_rank
        input_data['hot_market'] = hot_market
        input_data['crisis_period'] = crisis_period
        input_data['ipo_year'] = ipo_year
        input_data['ipo_month'] = ipo_month

        # Derived features
        input_data['ipo_quarter'] = (ipo_month - 1) // 3 + 1
        input_data['is_small_offer'] = 1 if gross_proceeds < 25 else 0
        input_data['is_large_offer'] = 1 if gross_proceeds > 100 else 0
        input_data['is_very_young'] = 1 if firm_age < 3 else 0
        input_data['is_young'] = 1 if firm_age < 5 else 0
        input_data['is_revised_up'] = 1 if price_revision > 0 else 0
        input_data['is_revised_down'] = 1 if price_revision < 0 else 0
        input_data['is_not_revised'] = 1 if price_revision == 0 else 0
        input_data['proceeds_to_assets'] = gross_proceeds / (assets + 1)
        input_data['revenue_to_assets'] = revenue / (assets + 1)
        input_data['offer_price_to_proceeds'] = offer_price / (gross_proceeds / 1000 + 1)

        # Interactions
        input_data['tech_vc'] = (1 if industry == 'Technology' else 0) * vc_backed_val
        input_data['young_vc'] = input_data['is_young'] * vc_backed_val
        input_data['small_young'] = input_data['is_small_offer'] * input_data['is_young']
        input_data['large_mature'] = input_data['is_large_offer'] * (1 - input_data['is_young'])
        input_data['vc_revised_up'] = vc_backed_val * input_data['is_revised_up']
        input_data['tech_hot_market'] = (1 if industry == 'Technology' else 0) * hot_market

        # Polynomials
        input_data['firm_age_squared'] = firm_age ** 2
        input_data['log_proceeds_squared'] = input_data['log_proceeds'] ** 2

        # Industry dummies
        input_data['industry_Financial'] = 1 if industry == 'Financial' else 0
        input_data['industry_Healthcare'] = 1 if industry == 'Healthcare' else 0
        input_data['industry_Technology'] = 1 if industry == 'Technology' else 0

        # Create dataframe with correct feature order
        input_df = pd.DataFrame([input_data])

        # Ensure all features are present
        for feat in feature_columns:
            if feat not in input_df.columns:
                input_df[feat] = 0

        input_df = input_df[feature_columns]

        # Scale
        input_scaled = scaler.transform(input_df)
        input_scaled = np.nan_to_num(input_scaled, nan=0.0)

        # Predict
        predicted_return = regressor.predict(input_scaled)[0]
        predicted_risk_proba = classifier.predict_proba(input_scaled)[0, 1]
        predicted_high_risk = classifier.predict(input_scaled)[0]

        # Display results
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted First-Day Return",
                f"{predicted_return*100:.2f}%",
                delta="Expected performance"
            )

        with col2:
            st.metric(
                "Risk Probability",
                f"{predicted_risk_proba*100:.1f}%",
                delta="High-risk likelihood"
            )

        with col3:
            risk_label = "High Risk" if predicted_high_risk else "Low Risk"
            risk_color = "🔴" if predicted_high_risk else "🟢"
            st.metric(
                "Risk Classification",
                f"{risk_color} {risk_label}"
            )

        # Interpretation
        st.markdown("---")
        st.markdown("## 💡 Interpretation")

        if predicted_return > 0.15:
            return_interp = "🚀 Strong positive return expected. This IPO shows characteristics of a potential winner."
        elif predicted_return > 0.05:
            return_interp = "📈 Moderate positive return expected. Typical performance for this profile."
        elif predicted_return > -0.05:
            return_interp = "➡️ Flat to slightly positive return expected. Average performance."
        else:
            return_interp = "📉 Negative return expected. This IPO may underperform on day one."

        if predicted_risk_proba > 0.7:
            risk_interp = "⚠️ High risk of negative performance. Consider avoiding."
        elif predicted_risk_proba > 0.5:
            risk_interp = "⚡ Moderate risk level. Careful evaluation recommended."
        else:
            risk_interp = "✅ Low risk profile. Relatively safe investment."

        st.info(return_interp)
        st.info(risk_interp)

        # Key factors
        st.markdown("### 🔑 Key Factors Influencing This Prediction")

        factors = []
        if vc_backed_val:
            factors.append("✓ VC-backed (typically positive)")
        if price_revision > 0.1:
            factors.append("✓ Strong upward price revision (high demand signal)")
        elif price_revision < -0.1:
            factors.append("⚠️ Downward price revision (weak demand)")
        if firm_age < 5:
            factors.append("⚠️ Young firm (higher volatility)")
        if gross_proceeds > 100:
            factors.append("✓ Large offering (more stability)")
        if industry == "Technology":
            factors.append("⚡ Technology sector (higher returns, higher risk)")
        if hot_market:
            factors.append("✓ Hot market period (favorable conditions)")

        for factor in factors:
            st.markdown(f"- {factor}")

st.markdown("---")
st.markdown("*Dashboard built with Streamlit | Data from SDC Platinum | FIN 377 Final Project*")