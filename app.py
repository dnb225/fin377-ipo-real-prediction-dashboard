"""
IPO Risk Prediction Dashboard - PART 1
Setup, Imports, and Introduction Page

This file contains the first part of the app.py code.
To use: Combine all parts (PART1, PART2, PART3) into a single app.py file.
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
        with open('models/all_classification_models.pkl', 'rb') as f:
            all_classifiers = pickle.load(f)
        with open('models/all_regression_models.pkl', 'rb') as f:
            all_regressors = pickle.load(f)

        return classifier, regressor, scaler, feature_columns, metadata, all_classifiers, all_regressors
    except FileNotFoundError:
        st.error("Model files not found. Please run the Jupyter notebook first to train models.")
        return None, None, None, None, None, None, None

@st.cache_data
def load_data():
    """Load test predictions and results"""
    try:
        test_preds = pd.read_csv('data/test_predictions.csv')
        clf_results = pd.read_csv('data/classification_results.csv')
        reg_results = pd.read_csv('data/regression_results.csv')
        strategy_results = pd.read_csv('data/strategy_summary.csv')
        feature_importance = pd.read_csv('data/feature_importance.csv')

        return test_preds, clf_results, reg_results, strategy_results, feature_importance
    except FileNotFoundError:
        st.error("Data files not found. Please run the Jupyter notebook first.")
        return None, None, None, None, None

# Load everything
classifier, regressor, scaler, feature_columns, metadata, all_classifiers, all_regressors = load_models()
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
    f"""
    This dashboard predicts IPO first-day returns and identifies high-risk offerings 
    using machine learning models trained on **real IPO data from SDC Platinum** 
    (1,265 IPOs from 1980-2017).
    
    **Models Used:**
    - Classification: {metadata['best_classifier_name']}
    - Regression: {metadata['best_regressor_name']}
    
    **Training Period:** {metadata['train_years']}
    **Test Period:** {metadata['test_years']}
    
    **Created by:** JLD Inc. LLC. Partners
    """
)

# ============================================================================
# PAGE 0: INTRODUCTION
# ============================================================================
if page == "Introduction":
    st.title("IPO Risk Prediction Dashboard")
    st.markdown("### Machine Learning for First-Day Return Prediction")

    st.markdown("---")

    # Project Overview
    st.markdown("## Project Overview")
    st.write("""
    This dashboard demonstrates the application of machine learning techniques to predict 
    Initial Public Offering (IPO) first-day returns and identify high-risk offerings using 
    **real IPO data from the SDC Platinum database**. The project was developed as part of 
    FIN 377 coursework at Lehigh University.
    """)

    # Research Question
    st.markdown("## Research Question")
    st.info("""
    **Can a machine learning model, using only information available before an IPO's first trading day, 
    effectively predict first-day IPO returns and identify "high-risk" IPOs prone to large negative price moves?**
    """)

    # Data Overview
    st.markdown("## Real IPO Data")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total IPOs", f"{len(test_preds):,}")
    with col2:
        st.metric("Time Period", metadata['test_years'])
    with col3:
        st.metric("Features", metadata['n_features'])

    with st.expander("**Dataset Details**", expanded=True):
        st.write(f"""
        This project uses **real IPO data** from the SDC Platinum database, which contains 
        comprehensive information about initial public offerings.
        
        **Data Source:**
        - SDC Platinum IPO Database
        - 1,472 total IPOs from 1980-2017
        - 1,265 IPOs with complete first-day return data
        
        **Key Variables:**
        - **Offer Characteristics**: Offer price, gross proceeds, shares offered, price revisions
        - **Firm Characteristics**: Industry (SIC code), firm age, revenue, assets, profitability
        - **Deal Structure**: VC-backing, primary vs secondary shares, underwriter reputation
        - **Market Conditions**: Filing-to-offer timing, number of amendments
        
        **Target Variable:**
        - **First-Day Return**: Calculated as (Closing Price Day 1 - Offer Price) / Offer Price
        - **High-Risk Classification**: IPOs with first-day returns below -5%
        
        **Data Split:**
        - Training: 1980-2010 ({metadata['n_train']} IPOs)
        - Validation: 2011-2014 ({metadata['n_val']} IPOs)
        - Test: 2015-2017 ({metadata['n_test']} IPOs)
        """)

    with st.expander("**Model Training Approach**"):
        st.write(f"""
        **Temporal Split to Avoid Look-Ahead Bias:**
        - Training: {metadata['train_years']} ({metadata['n_train']} IPOs)
        - Test: {metadata['test_years']} ({metadata['n_test']} IPOs)
        
        **Models Trained:**
        - Logistic Regression (baseline)
        - Random Forest
        - XGBoost
        - LightGBM
        - CatBoost
        
        **Feature Engineering:**
        - Log transformations for skewed variables
        - Interaction terms (e.g., Tech × VC-backed)
        - Industry classification from SIC codes
        - Temporal features (firm age, filing duration)
        - Valuation ratios (proceeds-to-assets, revenue-to-assets)
        """)

    with st.expander("**Evaluation Metrics**"):
        st.write("""
        **Classification (High-Risk Detection):**
        - ROC-AUC Score: Measures discriminatory power
        - Precision: Accuracy of high-risk predictions
        - Recall: Coverage of actual high-risk IPOs
        - F1 Score: Harmonic mean of precision and recall
        
        **Regression (Return Prediction):**
        - Mean Absolute Error (MAE): Average prediction error
        - Root Mean Squared Error (RMSE): Penalizes large errors
        - R² Score: Proportion of variance explained
        
        **Investment Strategy Evaluation:**
        - Average Return: Portfolio performance
        - Sharpe Ratio: Risk-adjusted returns
        - Positive Rate: Percentage of profitable investments
        - High-Risk Rate: Percentage of large losses
        """)

    # Key Results
    st.markdown("## Key Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Classification Performance")
        st.metric("Test AUC", f"{metadata['test_auc']:.3f}")
        st.write("The model successfully identifies high-risk IPOs with strong discriminatory power.")
    
    with col2:
        st.markdown("### Regression Performance")
        st.metric("Test MAE", f"{metadata['test_mae']*100:.2f}%")
        st.write("The model predicts first-day returns with reasonable accuracy.")

    st.markdown("---")
    
    st.markdown("### 📊 Dashboard Features")
    st.write("""
    Explore the following sections:
    
    1. **Home & IPO Search**: Browse historical IPO data and search for specific offerings
    2. **Model Performance**: Detailed evaluation of classification and regression models
    3. **Investment Strategies**: Compare different investment approaches based on predictions
    4. **Feature Analysis**: Understand which factors drive IPO performance
    5. **IPO Sandbox**: Create custom IPO scenarios and get predictions
    """)

    st.markdown("---")
    st.success("✅ Navigate using the sidebar to explore the dashboard!")
# ============================================================================
# PAGE 1: HOME & IPO SEARCH
# ============================================================================
elif page == "Home & IPO Search":
    st.title("IPO Database Explorer")
    st.markdown("### Browse and Search Historical IPO Data")

    st.markdown("---")

    # Summary statistics
    st.markdown("## Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total IPOs", f"{len(test_preds):,}")
    with col2:
        avg_return = test_preds['first_day_return'].mean() * 100
        st.metric("Avg First-Day Return", f"{avg_return:.2f}%")
    with col3:
        high_risk_pct = test_preds['high_risk'].mean() * 100
        st.metric("High-Risk Rate", f"{high_risk_pct:.1f}%")
    with col4:
        vc_backed_pct = test_preds['vc_backed'].mean() * 100
        st.metric("VC-Backed", f"{vc_backed_pct:.1f}%")

    st.markdown("---")

    # Search and filter options
    st.markdown("## Search IPOs")

    col1, col2 = st.columns(2)

    with col1:
        # Industry filter
        industries = ['All'] + sorted(test_preds['industry'].unique().tolist())
        selected_industry = st.selectbox("Filter by Industry", industries)

    with col2:
        # Year filter
        years = ['All'] + sorted(test_preds['ipo_year'].unique().tolist(), reverse=True)
        selected_year = st.selectbox("Filter by Year", years)

    # Text search
    search_term = st.text_input("Search by Company Name or Ticker", "")

    # Apply filters
    filtered_data = test_preds.copy()

    if selected_industry != 'All':
        filtered_data = filtered_data[filtered_data['industry'] == selected_industry]

    if selected_year != 'All':
        filtered_data = filtered_data[filtered_data['ipo_year'] == selected_year]

    if search_term:
        mask = (filtered_data['Issuer'].str.contains(search_term, case=False, na=False) |
                filtered_data['TickerSymbol'].str.contains(search_term, case=False, na=False))
        filtered_data = filtered_data[mask]

    st.markdown(f"**Showing {len(filtered_data)} IPOs**")

    # Display results
    if len(filtered_data) > 0:
        # Prepare display dataframe
        display_df = filtered_data[[
            'Issuer', 'TickerSymbol', 'IPOOfferDate', 'industry', 'OfferPrice',
            'gross_proceeds', 'vc_backed', 'first_day_return', 'predicted_return',
            'high_risk', 'predicted_high_risk'
        ]].copy()

        display_df['first_day_return'] = display_df['first_day_return'] * 100
        display_df['predicted_return'] = display_df['predicted_return'] * 100
        display_df['OfferPrice'] = display_df['OfferPrice'].apply(lambda x: f"${x:.2f}")
        display_df['gross_proceeds'] = display_df['gross_proceeds'].apply(lambda x: f"${x:.1f}M")
        display_df['vc_backed'] = display_df['vc_backed'].map({1: 'Yes', 0: 'No'})
        display_df['high_risk'] = display_df['high_risk'].map({1: 'High Risk', 0: 'Low Risk'})
        display_df['predicted_high_risk'] = display_df['predicted_high_risk'].map({1: 'High Risk', 0: 'Low Risk'})

        display_df.columns = [
            'Company', 'Ticker', 'IPO Date', 'Industry', 'Offer Price',
            'Proceeds', 'VC-Backed', 'Actual Return (%)', 'Predicted Return (%)',
            'Actual Risk', 'Predicted Risk'
        ]

        st.dataframe(display_df, use_container_width=True, height=400)

        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_ipos.csv",
            mime="text/csv"
        )
    else:
        st.warning("No IPOs match your search criteria.")

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.title("Model Performance Analysis")
    st.markdown("### Detailed Evaluation of ML Models")

    st.markdown("---")

    # Classification Results
    st.markdown("## Classification Models (High-Risk Detection)")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot model comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=clf_results['AUC'],
            y=clf_results['Model'],
            orientation='h',
            marker=dict(color='#1f77b4'),
            text=[f"{x:.3f}" for x in clf_results['AUC']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Classification Model Performance (AUC)",
            xaxis_title="AUC-ROC Score",
            yaxis_title="",
            height=400,
            showlegend=False
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Random")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Best Model")
        best_clf = clf_results.iloc[0]
        st.metric("Model", best_clf['Model'])
        st.metric("AUC", f"{best_clf['AUC']:.3f}")
        st.metric("Accuracy", f"{best_clf['Accuracy']:.3f}")
        st.metric("Precision", f"{best_clf['Precision']:.3f}")
        st.metric("Recall", f"{best_clf['Recall']:.3f}")

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
            x=reg_results['MAE'],
            y=reg_results['Model'],
            orientation='h',
            marker=dict(color='#ff7f0e'),
            text=[f"{x:.4f}" for x in reg_results['MAE']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Regression Model Performance (MAE)",
            xaxis_title="Mean Absolute Error",
            yaxis_title="",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Best Model")
        best_reg = reg_results.iloc[0]
        st.metric("Model", best_reg['Model'])
        st.metric("MAE", f"{best_reg['MAE']:.4f}")
        st.metric("MAE (%)", f"{best_reg['MAE'] * 100:.2f}%")
        st.metric("RMSE", f"{best_reg['RMSE']:.4f}")
        st.metric("R²", f"{best_reg['R²']:.3f}")

    # Full results table
    with st.expander("View All Regression Results"):
        st.dataframe(reg_results, use_container_width=True)

    st.markdown("---")

    # Prediction Examples
    st.markdown("## Sample Predictions")

    # Show some example predictions
    sample_size = 10
    sample_ipos = test_preds.sample(n=sample_size, random_state=42)

    for idx, row in sample_ipos.iterrows():
        with st.expander(f"{row['Issuer']} ({row['TickerSymbol']}) - {row['IPOOfferDate']}"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Actual Performance**")
                st.write(f"First-Day Return: {row['first_day_return'] * 100:.2f}%")
                st.write(f"Risk Level: {'High Risk' if row['high_risk'] else 'Low Risk'}")

            with col2:
                st.markdown("**Predicted Performance**")
                st.write(f"Predicted Return: {row['predicted_return'] * 100:.2f}%")
                st.write(f"Risk Probability: {row['predicted_risk_prob'] * 100:.1f}%")

            with col3:
                st.markdown("**IPO Details**")
                st.write(f"Industry: {row['industry']}")
                st.write(f"Offer Price: ${row['OfferPrice']:.2f}")
                st.write(f"VC-Backed: {'Yes' if row['vc_backed'] else 'No'}")

# ============================================================================
# PAGE 3: INVESTMENT STRATEGIES
# ============================================================================
elif page == "Investment Strategies":
    st.title("Investment Strategy Analysis")
    st.markdown("### Compare ML-Based Investment Approaches")

    st.markdown("---")

    # Strategy Overview
    st.markdown("## Strategy Performance Comparison")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    best_strategy = strategy_results.iloc[0]
    baseline_strategy = strategy_results[strategy_results['Strategy'] == 'Buy All'].iloc[0]

    with col1:
        st.metric("Best Strategy", best_strategy['Strategy'])
    with col2:
        st.metric("Avg Return", f"{best_strategy['Avg Return'] * 100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{best_strategy['Sharpe Ratio']:.3f}")
    with col4:
        improvement = (best_strategy['Avg Return'] - baseline_strategy['Avg Return']) * 100
        st.metric("vs Baseline", f"+{improvement:.2f}pp")

    st.markdown("---")

    # Strategy comparison charts
    col1, col2 = st.columns(2)

    with col1:
        # Average return comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=strategy_results['Avg Return'] * 100,
            y=strategy_results['Strategy'],
            orientation='h',
            marker=dict(color=['#2ca02c' if x > 0 else '#d62728'
                               for x in strategy_results['Avg Return']]),
            text=[f"{x * 100:.2f}%" for x in strategy_results['Avg Return']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Average Return by Strategy",
            xaxis_title="Average First-Day Return (%)",
            yaxis_title="",
            height=400
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sharpe ratio comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=strategy_results['Sharpe Ratio'],
            y=strategy_results['Strategy'],
            orientation='h',
            marker=dict(color='#ff7f0e'),
            text=[f"{x:.3f}" for x in strategy_results['Sharpe Ratio']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Risk-Adjusted Performance (Sharpe Ratio)",
            xaxis_title="Sharpe Ratio",
            yaxis_title="",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown("## Detailed Strategy Metrics")

    display_strategies = strategy_results.copy()
    display_strategies['Avg Return'] = display_strategies['Avg Return'].apply(lambda x: f"{x * 100:.2f}%")
    display_strategies['Median Return'] = display_strategies['Median Return'].apply(lambda x: f"{x * 100:.2f}%")
    display_strategies['Volatility'] = display_strategies['Volatility'].apply(lambda x: f"{x * 100:.2f}%")
    display_strategies['Positive Rate'] = display_strategies['Positive Rate'].apply(lambda x: f"{x * 100:.1f}%")
    display_strategies['High Risk Rate'] = display_strategies['High Risk Rate'].apply(lambda x: f"{x * 100:.1f}%")
    display_strategies['Sharpe Ratio'] = display_strategies['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")

    st.dataframe(display_strategies, use_container_width=True)

    st.markdown("---")

    # Strategy descriptions
    st.markdown("## Strategy Descriptions")

    strategies_info = {
        "Buy All": "Baseline strategy - invest in all IPOs without selection",
        "Avoid High Risk": "Invest only in IPOs predicted to be low-risk (risk prob < 50%)",
        "Top 25% Returns": "Invest in IPOs with top 25% predicted returns",
        "Combined Strategy": "Invest in low-risk IPOs with above-median predicted returns",
        "High Confidence": "Invest only when model has high confidence (extreme probabilities)"
    }

    for strategy, description in strategies_info.items():
        with st.expander(f"**{strategy}**"):
            st.write(description)
            if strategy in strategy_results['Strategy'].values:
                row = strategy_results[strategy_results['Strategy'] == strategy].iloc[0]
                st.write(f"- IPOs Selected: {row['IPOs']}")
                st.write(f"- Average Return: {row['Avg Return'] * 100:.2f}%")
                st.write(f"- Positive Rate: {row['Positive Rate'] * 100:.1f}%")
                st.write(f"- High Risk Rate: {row['High Risk Rate'] * 100:.1f}%")
# ============================================================================
# PAGE 4: FEATURE ANALYSIS
# ============================================================================
elif page == "Feature Analysis":
    st.title("Feature Importance Analysis")
    st.markdown("### Understanding What Drives IPO Performance")

    st.markdown("---")

    # Top features overview
    st.markdown("## Most Important Features")

    top_10_features = feature_importance.head(10)['feature'].tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Classification (High-Risk Detection)")
        top_clf = feature_importance.nlargest(15, 'classification_importance')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_clf['classification_importance'],
            y=top_clf['feature'],
            orientation='h',
            marker=dict(color='#1f77b4')
        ))
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Regression (Return Prediction)")
        top_reg = feature_importance.nlargest(15, 'regression_importance')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_reg['regression_importance'],
            y=top_reg['feature'],
            orientation='h',
            marker=dict(color='#ff7f0e')
        ))
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature descriptions
    st.markdown("## Feature Descriptions")

    feature_descriptions = {
        "OfferPrice": "The initial price at which shares are offered to the public",
        "gross_proceeds": "Total amount raised in the IPO (offer price × shares offered)",
        "log_proceeds": "Natural logarithm of gross proceeds (reduces skewness)",
        "firm_age": "Age of the company at the time of IPO (years)",
        "vc_backed": "Whether the company is backed by venture capital (1=Yes, 0=No)",
        "is_tech": "Whether the company is in the technology sector",
        "is_healthcare": "Whether the company is in the healthcare/biotech sector",
        "revenue": "Company's revenue before IPO (in millions)",
        "log_revenue": "Natural logarithm of revenue",
        "assets": "Company's total assets before IPO (in millions)",
        "log_assets": "Natural logarithm of assets",
        "price_revision": "Change from filed price range midpoint to final offer price",
        "NumberofAmendments": "Number of amendments to the registration statement",
        "days_filing_to_offer": "Days between filing and offering date",
        "is_young_firm": "Whether firm is less than 5 years old",
        "pct_secondary": "Percentage of shares sold by existing shareholders (vs new shares)",
        "proceeds_to_assets": "Ratio of IPO proceeds to total assets",
        "revenue_to_assets": "Ratio of revenue to assets (profitability indicator)",
        "tech_x_vc": "Interaction: Technology company backed by VC",
        "healthcare_x_vc": "Interaction: Healthcare company backed by VC",
        "young_x_vc": "Interaction: Young firm backed by VC"
    }

    selected_feature = st.selectbox("Select a feature to learn more", sorted(feature_descriptions.keys()))

    if selected_feature in feature_descriptions:
        st.info(f"**{selected_feature}**: {feature_descriptions[selected_feature]}")

        # Show statistics for this feature if it exists in test data
        if selected_feature in test_preds.columns:
            st.markdown("### Feature Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean", f"{test_preds[selected_feature].mean():.2f}")
            with col2:
                st.metric("Median", f"{test_preds[selected_feature].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{test_preds[selected_feature].std():.2f}")

    # Full feature importance table
    with st.expander("View All Feature Importances"):
        st.dataframe(feature_importance, use_container_width=True, height=400)

# ============================================================================
# PAGE 5: IPO SANDBOX
# ============================================================================
elif page == "IPO Sandbox":
    st.title("IPO Sandbox")
    st.markdown("### Create Custom IPO Scenarios and Get Predictions")

    st.markdown("---")

    st.info("💡 Adjust the parameters below to create a custom IPO scenario and see what the model predicts!")

    # User inputs
    user_inputs = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Company Characteristics")

        user_inputs['industry'] = st.selectbox(
            "Industry",
            ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial',
             'Energy', 'Materials', 'Other'],
            help="Primary industry sector"
        )

        user_inputs['firm_age'] = st.slider(
            "Firm Age (years)",
            min_value=0,
            max_value=50,
            value=8,
            step=1,
            help="Age of the company at IPO"
        )

        user_inputs['revenue'] = st.number_input(
            "Annual Revenue ($M)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Annual revenue in millions"
        )

        user_inputs['assets'] = st.number_input(
            "Total Assets ($M)",
            min_value=0.0,
            max_value=10000.0,
            value=150.0,
            step=10.0,
            help="Total assets in millions"
        )

        user_inputs['is_profitable'] = st.checkbox(
            "Is Profitable",
            value=True,
            help="Whether the company is currently profitable"
        )

        user_inputs['vc_backed'] = st.checkbox(
            "VC-Backed",
            value=True,
            help="Whether backed by venture capital"
        )

    with col2:
        st.markdown("### Offer Details")

        user_inputs['OfferPrice'] = st.slider(
            "Offer Price ($)",
            min_value=5.0,
            max_value=100.0,
            value=15.0,
            step=1.0,
            help="Initial offering price per share"
        )

        user_inputs['shares_offered'] = st.number_input(
            "Shares Offered (millions)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Number of shares offered in millions"
        ) * 1000000

        user_inputs['pct_primary'] = st.slider(
            "Primary Shares (%)",
            min_value=0,
            max_value=100,
            value=85,
            step=5,
            help="Percentage of new shares (vs secondary/existing shares)"
        ) / 100

        user_inputs['NumberofAmendments'] = st.slider(
            "Number of Amendments",
            min_value=0,
            max_value=10,
            value=3,
            help="Number of amendments to registration statement"
        )

        user_inputs['days_filing_to_offer'] = st.slider(
            "Days from Filing to Offer",
            min_value=30,
            max_value=365,
            value=90,
            help="Time between filing and offering"
        )

        user_inputs['price_revision'] = st.slider(
            "Price Revision (%)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=1.0,
            help="Percentage change from initial filing range"
        ) / 100

    st.markdown("---")

    # Calculate derived features
    user_inputs['gross_proceeds'] = user_inputs['OfferPrice'] * (user_inputs['shares_offered'] / 1000000)
    user_inputs['log_proceeds'] = np.log(user_inputs['gross_proceeds'] + 1)
    user_inputs['total_shares_offered'] = user_inputs['shares_offered']
    user_inputs['log_revenue'] = np.log(user_inputs['revenue'] + 1)
    user_inputs['log_assets'] = np.log(user_inputs['assets'] + 1)
    user_inputs['proceeds_to_assets'] = user_inputs['gross_proceeds'] / (user_inputs['assets'] + 1)
    user_inputs['revenue_to_assets'] = user_inputs['revenue'] / (user_inputs['assets'] + 1)
    user_inputs['pct_secondary'] = 1 - user_inputs['pct_primary']

    # Binary indicators
    user_inputs['is_young_firm'] = 1 if user_inputs['firm_age'] < 5 else 0
    user_inputs['is_tech'] = 1 if user_inputs['industry'] == 'Technology' else 0
    user_inputs['is_healthcare'] = 1 if user_inputs['industry'] == 'Healthcare' else 0
    user_inputs['large_offer'] = 1 if user_inputs['gross_proceeds'] > test_preds['gross_proceeds'].median() else 0
    user_inputs['high_amendments'] = 1 if user_inputs['NumberofAmendments'] > 3 else 0
    user_inputs['positive_revision'] = 1 if user_inputs['price_revision'] > 0 else 0
    user_inputs['ipo_month'] = 6  # Default June
    user_inputs['ipo_quarter'] = 2  # Default Q2

    # Interactions
    user_inputs['tech_x_vc'] = user_inputs['is_tech'] * user_inputs['vc_backed']
    user_inputs['healthcare_x_vc'] = user_inputs['is_healthcare'] * user_inputs['vc_backed']
    user_inputs['young_x_vc'] = user_inputs['is_young_firm'] * user_inputs['vc_backed']
    user_inputs['large_offer_x_tech'] = user_inputs['large_offer'] * user_inputs['is_tech']

    # Industry dummies
    for ind in ['Healthcare', 'Financial', 'Consumer', 'Industrial', 'Energy',
                'Materials', 'Other', 'Technology']:
        user_inputs[f'industry_{ind}'] = 1 if user_inputs['industry'] == ind else 0

    # Prepare feature vector
    feature_vector = []
    for col in feature_columns:
        if col in user_inputs:
            feature_vector.append(user_inputs[col])
        else:
            feature_vector.append(0)  # Default for missing features

    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)

    # Make predictions
    if st.button("Predict IPO Performance", type="primary"):
        st.markdown("---")
        st.markdown("## Prediction Results")

        # Get predictions
        risk_prob = classifier.predict_proba(feature_vector_scaled)[0][1]
        predicted_return = regressor.predict(feature_vector_scaled)[0]
        is_high_risk = risk_prob >= 0.5

        # Display results in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Risk Classification")
            if is_high_risk:
                st.error("**HIGH RISK**")
            else:
                st.success("**LOW RISK**")
            st.metric("Risk Probability", f"{risk_prob * 100:.1f}%")

        with col2:
            st.markdown("### Predicted Return")
            return_color = "green" if predicted_return >= 0 else "red"
            st.markdown(f"<h2 style='color: {return_color};'>{predicted_return * 100:+.2f}%</h2>",
                        unsafe_allow_html=True)
            st.caption("Expected first-day return")

        with col3:
            st.markdown("### Confidence Level")
            confidence = "High" if abs(risk_prob - 0.5) > 0.3 else "Medium" if abs(risk_prob - 0.5) > 0.15 else "Low"
            st.metric("Model Confidence", confidence)
            st.caption("Based on probability distance from 50%")

        # Investment recommendation
        st.markdown("---")
        st.markdown("### Investment Recommendation")

        if not is_high_risk and predicted_return > 0.10:
            st.success("""
            **Strong Buy Candidate**
            - Low risk of negative returns
            - Predicted return exceeds 10%
            - Favorable characteristics
            """)
        elif not is_high_risk and predicted_return > 0.05:
            st.info("""
            **Moderate Buy Candidate**
            - Low risk classification
            - Positive expected returns
            - Consider portfolio allocation limits
            """)
        elif is_high_risk and predicted_return > 0:
            st.warning("""
            **Cautious Approach Recommended**
            - High risk classification despite positive expected return
            - Higher volatility expected
            - Consider smaller position size
            """)
        else:
            st.error("""
            **Avoid**
            - High risk of negative returns
            - Unfavorable risk-return profile
            - Consider waiting for better opportunities
            """)

        # Key factors
        st.markdown("---")
        st.markdown("### Key Factors Influencing This Prediction")

        st.write("**Most Influential Features for This IPO:**")

        # Display user's values for top features
        important_factors = []
        for feature in top_10_features[:5]:
            if feature in user_inputs:
                value = user_inputs[feature]
                if isinstance(value, (int, float)):
                    if abs(value) < 1 and value != 0:
                        important_factors.append(f"- **{feature}**: {value:.3f}")
                    elif abs(value) > 1000000:
                        important_factors.append(f"- **{feature}**: {value / 1e6:.1f}M")
                    else:
                        important_factors.append(f"- **{feature}**: {value:.2f}")
                else:
                    important_factors.append(f"- **{feature}**: {value}")

        for factor in important_factors:
            st.write(factor)

        # Scenario summary
        st.markdown("---")
        st.markdown("### Your IPO Scenario Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Company Profile:**")
            st.write(f"- Industry: {user_inputs['industry']}")
            st.write(f"- Firm Age: {user_inputs['firm_age']} years")
            st.write(f"- Revenue: ${user_inputs['revenue']:.1f}M")
            st.write(f"- Assets: ${user_inputs['assets']:.1f}M")
            st.write(f"- VC-Backed: {'Yes' if user_inputs['vc_backed'] else 'No'}")

        with col2:
            st.write("**Offering Details:**")
            st.write(f"- Offer Price: ${user_inputs['OfferPrice']:.2f}")
            st.write(f"- Gross Proceeds: ${user_inputs['gross_proceeds']:.1f}M")
            st.write(f"- Shares Offered: {user_inputs['shares_offered'] / 1e6:.1f}M")
            st.write(f"- Price Revision: {user_inputs['price_revision'] * 100:+.1f}%")
            st.write(f"- Amendments: {user_inputs['NumberofAmendments']}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>IPO Risk Prediction Dashboard</strong></p>
    <p>Created by Logan Wesselt, Julian Tashjian, Dylan Bollinger | JLD Inc. LLC. Partners</p>
    <p>FIN 377 Final Project | Machine Learning for IPO Analysis using Real Data</p>
    <p>Data Source: SDC Platinum IPO Database</p>
</div>
""", unsafe_allow_html=True)