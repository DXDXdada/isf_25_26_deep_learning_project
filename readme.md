to do list 
- [] commit/push on github 
- [] make the repo public for the professor
- [] add the names of the group
- [] readme : rework the the jupyter notebooks description
- [] format each notebook accordingly + add good description for each cell output
- [] add references + reference to my internship @calgary
- [] add a part for the use of chatgpt in the readme

# Deep Learning for Asset Price Movement Prediction

This project is part of the course "Deep Learning" for the ISF master's degree at Dauphine University for the 2025/2026 academic year.

## Problem Statement

This project's goal is to answer the following questions:
- Can deep learning predict an asset's price movement (up or down)? 
- Can it outperform classical machine learning algorithms? 

Multiple sub-questions can be derived from these:
- **Question 1**: Which deep learning architectures perform best on market data?
- **Question 2**: Can the models generalize across different market conditions, time horizons, and asset volatilities? 
- **Question 3**: How do the models perform compared to classical machine learning algorithms?

## Implementation

This project systematically investigates how deep learning architectures perform under these conditions, with a focus on understanding generalization capabilities across different volatility regimes and time horizons. 

Initially, the goal was to predict exact closing prices. However, after investigation, we found this to be extremely difficult with very high error rates. We ultimately chose a **binary classification approach** to predict stock price movement direction (up or down) across multiple time horizons and financial instruments, as this problem was more tractable and had substantial literature available.

We will systematically address each question outlined in the **Problem Statement** section.

**Remark:**
This project is not an exhaustive survey of all deep learning models available. Due to time constraints, we selected a few architectures that showed promise for financial time series forecasting based on recent literature.

## Project Structure

The code is organized into Jupyter notebooks, each covering a specific task:

### Notebooks
1. **Data Exploration & EDA** (`1_data_exploration_eda.ipynb`)
    - Load and validate raw data
    - Exploratory data analysis
    - Data quality assessment
    - Volatility and correlation analysis

2. **Feature Engineering** (`2_feature_engineering_preprocessing.ipynb`)
    - Feature engineering: 
        - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
        - Derived features (returns, momentum, volatility)
        - Time-based features
    - Preprocessing:
        - Data normalization/scaling
        - Sequence generation for deep learning
        - Train/validation/test splits

4. **Model Architectures** (`4_model_architectures.ipynb`)
   - Define neural network architectures
   - LSTM, GRU, CNN, Transformer models

5. **Baseline Models** (`5_baseline_models.ipynb`)
   - Simple baselines (random, buy-and-hold, moving average)
   - Classical machine learning (Random Forest, XGBoost, SVM)

6. **LSTM Training** (`6_lstm_training.ipynb`)
   - Train LSTM models
   - Hyperparameter tuning
   - Performance evaluation

7. **GRU Training** (`7_gru_training.ipynb`)
   - Train GRU models
   - Compare with LSTM

8. **CNN Training** (`8_cnn_training.ipynb`)
   - Train 1D Convolutional Neural Networks
   - Analyze temporal pattern recognition

9. **Transformer Training** (`9_transformer_training.ipynb`)
   - Train Transformer models
   - Attention mechanism analysis

10. **Hybrid Models** (`10_hybrid_training.ipynb`)
    - CNN-LSTM combinations
    - Ensemble methods

11. **Hyperparameter Tuning** (`11_hyperparameter_tuning.ipynb`)
    - Grid search and random search
    - Optimal configuration selection

12. **Cross-Asset Experiments** (`12_cross_asset_experiments.ipynb`)
    - Train on one asset, test on others
    - Generalization analysis

13. **Backtesting** (`13_backtesting.ipynb`)
    - Simulate trading strategies
    - Risk-adjusted returns analysis

14. **Model Interpretation** (`14_model_interpretation.ipynb`)
    - Feature importance
    - Attention visualization
    - Error analysis

15. **Final Report** (`15_final_report.ipynb`)
    - Consolidated results
    - Conclusions and recommendations

## Data

### Data Source
Data was obtained using the **Yahoo Finance API** (yfinance). We chose to work with **daily data** for several reasons:
- **Signal-to-noise ratio**: Daily data avoids excessive noise that can lead to poor performance
- **Historical depth**: yfinance provides extensive historical daily data (up to 25 years) for free, whereas hourly data is limited to ~730 days
- **Predictability**: Daily movements are more influenced by fundamental factors and show clearer patterns than intraday fluctuations
- **Literature support**: Most academic research on financial prediction uses daily data

### Asset Selection
To study model generalization across different volatility regimes, we selected 5 financial instruments with distinct characteristics:

| Ticker | Asset Name | Type | Volatility Profile | Data Period |
|--------|-----------|------|-------------------|-------------|
| AAPL | Apple Inc. | Tech Stock | Low-Medium | ~25 years |
| AMZN | Amazon.com Inc. | Tech Stock | Medium | ~25 years |
| NVDA | NVIDIA Corporation | Tech Stock | Medium-High | ~25 years |
| SPY | S&P 500 ETF | Index ETF | Low | ~25 years |
| BTC-USD | Bitcoin | Cryptocurrency | Very High | ~11 years |

This diverse asset selection enables us to:
- Test generalization across different volatility levels
- Compare traditional stocks vs. cryptocurrency
- Analyze sector-specific patterns (tech stocks vs. broad market)

### Data Collection
The data collection script can be found in:
- **Script**: `get_data_yfinance.py`
- **Usage**: Run with `python get_data_yfinance.py`
- **Output directory**: `/data_new/`

### Data Format
Each CSV file contains the following columns:
- **Date**: Trading date (index)
- **Open**: Opening price for the day
- **High**: Highest price during the day
- **Low**: Lowest price during the day
- **Close**: Closing price for the day
- **Volume**: Trading volume for the day

### Data Statistics
- **Start date**: 2000-01-01 (or earliest available)
- **End date**: 2024-12-08 (current date)
- **Frequency**: Daily (1d interval)
- **Total observations**: ~6,000-6,500 days per stock (varies by IPO date and availability)

## Contributors

ISF Master's Students - Dauphine University 2025/2026

## License

This project is for academic purposes only.