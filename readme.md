# IPO Risk Prediction Dashboard

Machine learning models for predicting first-day IPO returns and identifying high-risk offerings using pre-IPO information.

## Project Overview

This project uses machine learning to analyze Initial Public Offerings (IPOs) and predict first-day performance. Using only information available before an IPO's first trading day, our models classify high-risk offerings and predict expected returns.

**Live Dashboard:** https://fin377-ipo-real-prediction-dashboard-jld.streamlit.app/

## Research Question

Can machine learning models, using only pre-IPO data, effectively predict first-day returns and identify high-risk IPOs prone to poor performance?

## Data

- **Source:** SDC Platinum IPO Database
- **Period:** 1980-2017
- **Sample Size:** 1,265 IPOs with complete first-day return data
- **Split:** 80% training, 20% test (stratified)

## Methodology

**Target Variables:**
- First-day return (continuous)
- High-risk classification (bottom 25% of returns or negative)

**Models Evaluated:**
- Classification: Logistic Regression, Random Forest, XGBoost, LightGBM
- Regression: Linear, Ridge, Decision Tree, Random Forest, XGBoost, LightGBM

**Techniques:**
- 5-fold cross-validation for hyperparameter tuning
- SMOTE for class imbalance
- Feature engineering (40+ features)
- StandardScaler preprocessing

## Results

**Best Classification Model:** Random Forest (Test AUC: 0.680)
**Best Regression Model:** Random Forest (Test MAE: 0.154)

Investment strategies based on model predictions demonstrate improved risk-adjusted returns compared to baseline approaches.

## Repository Structure
```
├── data/
│   └── final/              # Model outputs and predictions
├── models/                 # Trained models (.pkl files)
├── notebooks/              # Jupyter notebooks (Parts 1-5)
├── app.py                  # Streamlit dashboard
└── requirements.txt        # Python dependencies
```

## Running the Dashboard Locally
```bash
git clone https://github.com/YOUR-USERNAME/fin377-ipo-real-prediction-dashboard.git
cd fin377-ipo-real-prediction-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard Features

1. **Introduction:** Project overview and key findings
2. **Methodology:** Detailed technical approach
3. **Home & IPO Search:** Explore predictions for individual IPOs
4. **Model Performance:** Evaluation metrics and comparisons
5. **Investment Strategies:** Backtest ML-driven investment approaches
6. **Feature Analysis:** Feature importance visualization
7. **IPO Sandbox:** Create hypothetical scenarios

## Team

- Dylan Bollinger
- Logan Wesselt
- Julian Tashjian

**Course:** FIN 377 - Data Science for Finance, Lehigh University

## Technical Stack

- Python 3.8+
- scikit-learn, XGBoost, LightGBM
- pandas, numpy
- Streamlit, Plotly
- imbalanced-learn (SMOTE)

## Limitations

- Data ends in 2017; may not capture recent market dynamics
- Model performance suggests IPO first-day returns remain largely unpredictable
- Does not account for transaction costs or allocation mechanisms
- Missing data excluded (14% of original dataset)

## References

**Data Source:**
- Securities Data Corporation (SDC) Platinum IPO Database

**Academic Literature:**
- Ritter, J. R. (1991). The long-run performance of initial public offerings. *The Journal of Finance*, 46(1), 3-27.
- Loughran, T., & Ritter, J. R. (2004). Why has IPO underpricing changed over time? *Financial Management*, 33(3), 5-37.

## License

This project was completed for academic purposes at Lehigh University.
```