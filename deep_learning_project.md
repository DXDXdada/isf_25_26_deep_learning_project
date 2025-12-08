### a) Multi-Layer LSTM (Long Short-Term Memory)
**Justification**: LSTMs were specifically designed to capture long-term dependencies in sequential data and have shown success in financial forecasting. The gating mechanisms help prevent vanishing gradients and allow the model to learn which historical information is relevant.

**Architecture Details**:
- Input layer: (sequence_length, n_features)
- LSTM Layer 1: 128 units, return_sequences=True
- Dropout: 0.2
- LSTM Layer 2: 64 units, return_sequences=False
- Dropout: 0.2
- Dense Layer 1: 32 units, ReLU activation
- Dense Layer 2: 16 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation

**Hyperparameters to Tune**:
- Number of LSTM layers: [1, 2, 3]
- Hidden units: [64, 128, 256]
- Dropout rate: [0.1, 0.2, 0.3]
- Learning rate: [0.001, 0.0001]

#### b) GRU (Gated Recurrent Unit)
**Justification**: GRUs are computationally more efficient than LSTMs (fewer parameters) while maintaining similar performance. This is important for financial applications where training speed matters.

**Architecture Details**:
- Input layer: (sequence_length, n_features)
- GRU Layer 1: 128 units, return_sequences=True
- Dropout: 0.2
- GRU Layer 2: 64 units, return_sequences=False
- Dropout: 0.2
- Dense Layer 1: 32 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation

**Hyperparameters to Tune**:
- Number of GRU layers: [1, 2, 3]
- Hidden units: [64, 128, 256]
- Dropout rate: [0.1, 0.2, 0.3]

#### c) 1D Convolutional Neural Network (CNN)
**Justification**: CNNs can capture local patterns and temporal features in time series data. They are translation-invariant, meaning they can detect patterns regardless of when they occur in the sequence. CNNs are also computationally efficient.

**Architecture Details**:
- Input layer: (sequence_length, n_features)
- Conv1D Layer 1: 64 filters, kernel_size=3, ReLU activation
- MaxPooling1D: pool_size=2
- Conv1D Layer 2: 128 filters, kernel_size=3, ReLU activation
- MaxPooling1D: pool_size=2
- Conv1D Layer 3: 64 filters, kernel_size=3, ReLU activation
- GlobalMaxPooling1D
- Dense Layer 1: 64 units, ReLU activation
- Dropout: 0.3
- Dense Layer 2: 32 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation

**Hyperparameters to Tune**:
- Number of conv layers: [2, 3, 4]
- Filter sizes: [32, 64, 128]
- Kernel sizes: [3, 5, 7]
- Pooling: [MaxPooling, AveragePooling]

#### d) Transformer-based Architecture
**Justification**: Transformers with self-attention mechanisms have revolutionized sequence modeling. They can capture long-range dependencies more effectively than RNNs and can process sequences in parallel. The attention mechanism allows the model to focus on the most relevant time steps.

**Architecture Details**:
- Input layer: (sequence_length, n_features)
- Positional Encoding layer
- Multi-Head Attention: 8 heads, key_dim=64
- Layer Normalization
- Feed-Forward Network: Dense(128, ReLU) -> Dense(n_features)
- Layer Normalization
- Global Average Pooling
- Dense Layer 1: 64 units, ReLU activation
- Dropout: 0.2
- Dense Layer 2: 32 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation

**Hyperparameters to Tune**:
- Number of attention heads: [4, 8, 16]
- Number of transformer blocks: [1, 2, 3]
- Feed-forward dimension: [64, 128, 256]
- Attention dropout: [0.1, 0.2]

#### e) Hybrid CNN-LSTM Architecture
**Justification**: Combines the feature extraction capabilities of CNNs with the sequence modeling power of LSTMs. CNN layers extract local patterns, and LSTM layers model temporal dependencies between these patterns.

**Architecture Details**:
- Input layer: (sequence_length, n_features)
- Conv1D Layer: 64 filters, kernel_size=3, ReLU activation
- MaxPooling1D: pool_size=2
- LSTM Layer 1: 64 units, return_sequences=True
- Dropout: 0.2
- LSTM Layer 2: 32 units, return_sequences=False
- Dropout: 0.2
- Dense Layer: 32 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation

**Hyperparameters to Tune**:
- CNN filters: [32, 64, 128]
- LSTM units: [32, 64, 128]
- Kernel size: [3, 5]


**Experimental Scenarios**:

#### Scenario A: Within-Asset Performance (Baseline)
**Setup**: Train and test on the same asset with 80-20 temporal split
**Purpose**: Establish baseline performance for each asset

| Experiment | Training Set | Test Set | Expected Outcome |
|------------|--------------|----------|------------------|
| A1 | AAPL (80%) | AAPL (20%) | Best performance - model sees same dynamics |
| A2 | AMZN (80%) | AMZN (20%) | Good performance - same asset |
| A3 | NVDA (80%) | NVDA (20%) | Good performance - same asset |
| A4 | BTC (80%) | BTC (20%) | Challenging due to high volatility |

#### Scenario B: Cross-Asset Transfer (Low to High Volatility)
**Setup**: Train on low/medium volatility, test on higher volatility
**Purpose**: Test if models trained on stable markets generalize to volatile ones

| Experiment | Training Set | Test Set | Hypothesis |
|------------|--------------|----------|------------|
| B1 | AAPL | NVDA | Moderate degradation - both equities |
| B2 | AAPL | BTC | Significant degradation - different dynamics |
| B3 | AMZN | NVDA | Small degradation - similar volatility |
| B4 | AMZN | BTC | Significant degradation - different asset class |
| B5 | NVDA | BTC | Moderate degradation - both high vol |

#### Scenario C: Cross-Asset Transfer (High to Low Volatility)
**Setup**: Train on high volatility, test on lower volatility
**Purpose**: Test if models trained on noisy data work on cleaner signals

| Experiment | Training Set | Test Set | Hypothesis |
|------------|--------------|----------|------------|
| C1 | BTC | AAPL | May overfit to noise, poor generalization |
| C2 | BTC | AMZN | Similar to C1 |
| C3 | BTC | NVDA | Better than C1/C2, still degradation |
| C4 | NVDA | AAPL | Slight degradation |
| C5 | NVDA | AMZN | Minimal degradation |

#### Scenario D: Multi-Asset Training
**Setup**: Train on combined data from multiple assets
**Purpose**: Test if diverse training improves generalization

| Experiment | Training Set | Test Set | Hypothesis |
|------------|--------------|----------|------------|
| D1 | AAPL + AMZN + NVDA | BTC | Better generalization to unseen dynamics |
| D2 | AAPL + AMZN + NVDA | AAPL | May underperform vs A1 (diluted signal) |
| D3 | All 4 assets (mixed) | Each asset (hold-out) | Robust model across regimes |
| D4 | BTC + NVDA (high vol) | AAPL + AMZN | Test high-vol -> low-vol transfer |

**Cross-Validation Strategy for Multi-Asset**:
- Use stratified splitting to ensure balanced representation of each asset
- Implement time-series cross-validation (expanding window) to prevent look-ahead bias
- Track per-asset performance metrics separately

---

## Feature Engineering

### Raw Features (from OHLCV data)
1. **Price Features**:
   - `open`, `high`, `low`, `close` (normalized)
   - `close_to_open_ratio`: close/open
   - `high_to_low_ratio`: high/low
   - `typical_price`: (high + low + close) / 3

2. **Volume Features**:
   - `volume` (normalized)
   - `volume_change`: (volume[t] - volume[t-1]) / volume[t-1]

### Technical Indicators
**Justification**: Technical indicators are widely used in quantitative finance and encode market dynamics that may not be obvious from raw prices.

3. **Momentum Indicators**:
   - **RSI (Relative Strength Index)** [14, 21 periods]: Measures overbought/oversold conditions
   - **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
   - **ROC (Rate of Change)** [12, 24 periods]: Momentum indicator

4. **Trend Indicators**:
   - **SMA (Simple Moving Average)** [20, 50, 200 periods]: Basic trend indicator
   - **EMA (Exponential Moving Average)** [12, 26 periods]: Weighted trend indicator
   - **Bollinger Bands**: Volatility bands (middle, upper, lower)
   - **ADX (Average Directional Index)**: Trend strength

5. **Volatility Indicators**:
   - **ATR (Average True Range)** [14 periods]: Measures volatility
   - **Bollinger Band Width**: (upper_band - lower_band) / middle_band
   - **Historical Volatility** [20, 50 periods]: Standard deviation of returns

6. **Volume Indicators**:
   - **OBV (On-Balance Volume)**: Cumulative volume indicator
   - **Volume SMA** [20 periods]: Average volume
   - **VWAP (Volume Weighted Average Price)**: Average price weighted by volume

### Derived Features
7. **Price Changes & Returns**:
   - `returns`: log(close[t] / close[t-1])
   - `cumulative_returns`: Sum of returns over window
   - `lagged_returns`: [1, 3, 6, 12, 24 hour lags]

8. **Time-based Features**:
   - `hour_of_day`: Cyclical encoding (sin/cos) - market open/close patterns
   - `day_of_week`: Cyclical encoding - weekend effects
   - `is_market_hours`: For US stocks (9:30 AM - 4:00 PM EST)

9. **Volatility Regime Features**:
   - `realized_volatility_ratio`: recent_vol / long_term_vol
   - `volatility_regime`: Categorical [low, medium, high] based on percentiles

### Feature Normalization
**Methods to compare**:
- **MinMax Scaling**: Scale features to [0, 1] range - preserves distribution
- **Standard Scaling**: Zero mean, unit variance - assumes normal distribution
- **Robust Scaling**: Uses median and IQR - robust to outliers
- **Rolling Window Normalization**: Normalize using statistics from rolling window (prevents look-ahead bias)

**Recommendation**: Use **Rolling Window Normalization** with a 720-hour (30-day) window to respect temporal ordering and adapt to non-stationary data.

---

## Data Preprocessing Pipeline

### Step 1: Data Loading and Validation
```
For each asset CSV file:
1. Load data using pandas
2. Parse datetime column and set as index
3. Check for missing values
4. Check for data quality issues (negative prices, zero volume, etc.)
5. Ensure consistent time intervals (hourly)
6. Handle gaps in data (forward fill, interpolation, or removal)
```

### Step 2: Exploratory Data Analysis (EDA)
```
For each asset:
1. Plot price time series (close price)
2. Calculate and plot returns distribution
3. Compute descriptive statistics (mean, std, skewness, kurtosis)
4. Calculate volatility metrics (realized vol, Parkinson estimator)
5. Check for stationarity (ADF test, KPSS test)
6. Analyze autocorrelation (ACF, PACF plots)
7. Compare volatility across assets
8. Identify regime changes or structural breaks
9. Create correlation matrix across assets
```

### Step 3: Feature Engineering
```
For each asset:
1. Calculate all raw features (ratios, typical price)
2. Compute technical indicators using TA-Lib or custom functions
3. Generate derived features (returns, lagged values)
4. Add time-based features with cyclical encoding
5. Calculate volatility regime indicators
6. Handle NaN values created by rolling calculations (drop initial rows)
7. Save engineered features to new CSV file
```

### Step 4: Target Variable Creation
```
For each time horizon (24h, 168h, 720h):
1. Shift close prices by horizon: future_close = close.shift(-horizon)
2. Create binary labels: label = (future_close > close).astype(int)
3. Remove last 'horizon' rows (no future data available)
4. Check class balance (UP vs DOWN)
5. Save labels separately for each horizon
```

### Step 5: Data Splitting Strategy
**Important**: Use **time-series split** (NOT random split) to prevent look-ahead bias.

```
For Within-Asset Experiments (Scenario A):
1. Sort data by timestamp
2. Split: First 80% for training, Last 20% for testing
3. From training set, take last 20% as validation set
4. Final split: 64% train, 16% validation, 20% test

For Cross-Asset Experiments (Scenarios B, C):
1. Use 80% of source asset for training, 20% for validation
2. Use 100% of target asset for testing (or last 20% if same as source)

For Multi-Asset Experiments (Scenario D):
1. Concatenate asset data
2. Implement time-series cross-validation with 5 folds
3. Use expanding window approach: train on [0:t], validate on [t:t+δ]
```

### Step 6: Feature Scaling
```
1. Fit scaler on TRAINING set only
2. Transform training, validation, and test sets
3. Save scaler object for later use
4. Apply rolling window normalization if selected
```

### Step 7: Sequence Creation
```
For each sample at time t:
1. Extract sequence of length = time_horizon
2. X[i] = features[t-horizon:t]  # Past 'horizon' hours
3. y[i] = label[t]  # Future movement label
4. Create 3D arrays: (n_samples, sequence_length, n_features)
5. Ensure no overlap between train/val/test in sequence creation
```

### Step 8: Class Balancing
**Problem**: Financial data is often imbalanced (e.g., more UP than DOWN days)

**Solutions to implement**:
1. **Class Weights**: Assign higher weights to minority class in loss function
2. **SMOTE**: Synthetic Minority Over-sampling Technique (use with caution for time series)
3. **Undersampling**: Randomly undersample majority class
4. **Threshold Adjustment**: Adjust classification threshold based on validation set

**Recommendation**: Start with class weights, compare with other methods.

---

## Model Training Procedure

### Training Configuration

**Loss Function**: Binary Cross-Entropy
```
Justification: Standard loss for binary classification. 
With class weights: loss = -w_0 * y * log(ŷ) - w_1 * (1-y) * log(1-ŷ)
```

**Optimizer**: Adam
```
Justification: Adaptive learning rate, works well with RNNs and deep networks.
Parameters: learning_rate=0.001, beta_1=0.9, beta_2=0.999
```

**Regularization Techniques**:
1. **Dropout**: 0.2-0.3 after RNN/Conv layers to prevent overfitting
2. **L2 Regularization**: Weight decay in dense layers (lambda=0.001)
3. **Early Stopping**: Monitor validation loss, patience=10 epochs
4. **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)

**Training Hyperparameters**:
- **Batch Size**: 32, 64, 128 (larger for stability, smaller for better generalization)
- **Epochs**: Maximum 100, with early stopping
- **Validation Frequency**: Every epoch
- **Checkpoint Saving**: Save best model based on validation accuracy

### Training Loop Pseudocode
```python
for each architecture:
    for each time_horizon:
        for each training_scenario:
            # Load data
            X_train, y_train = load_sequences(train_data, horizon)
            X_val, y_val = load_sequences(val_data, horizon)
            X_test, y_test = load_sequences(test_data, horizon)
            
            # Calculate class weights
            class_weights = compute_class_weight(y_train)
            
            # Build model
            model = build_model(architecture, input_shape, hyperparams)
            
            # Compile model
            model.compile(
                optimizer=Adam(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC', 'Precision', 'Recall']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                ModelCheckpoint(filepath='best_model.h5', save_best_only=True),
                TensorBoard(log_dir='logs/')
            ]
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=64,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # Load best model
            model.load_weights('best_model.h5')
            
            # Evaluate on test set
            test_metrics = evaluate_model(model, X_test, y_test)
            
            # Save results
            save_results(architecture, horizon, scenario, history, test_metrics)
```

### Hyperparameter Tuning Strategy

**Method**: Random Search with Cross-Validation (more efficient than Grid Search)

**Hyperparameters to Tune**:
```python
hyperparameter_space = {
    'lstm_units': [64, 128, 256],
    'lstm_layers': [1, 2, 3],
    'dense_units': [16, 32, 64],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
    'batch_size': [32, 64, 128],
    'optimizer': ['Adam', 'RMSprop'],
}
```

**Procedure**:
1. Randomly sample 30-50 configurations from hyperparameter space
2. Train each configuration for reduced epochs (e.g., 20 epochs)
3. Evaluate on validation set
4. Select top 5 configurations
5. Train top configurations fully with early stopping
6. Select best based on validation performance
7. Evaluate final model on test set

---

## Evaluation Metrics

### Primary Metrics

**1. Accuracy**
```
Formula: (TP + TN) / (TP + TN + FP + FN)
Interpretation: Overall correctness, but can be misleading with imbalanced classes
Baseline: 50% (random guessing for balanced classes)
```

**2. Precision**
```
Formula: TP / (TP + FP)
Interpretation: Of predicted UPs, how many were actually UP
Financial Meaning: Reduces false buy signals
```

**3. Recall (Sensitivity)**
```
Formula: TP / (TP + FN)
Interpretation: Of actual UPs, how many did we predict
Financial Meaning: Captures more true opportunities
```

**4. F1 Score**
```
Formula: 2 * (Precision * Recall) / (Precision + Recall)
Interpretation: Harmonic mean of precision and recall
Usage: Good for imbalanced datasets
```

**5. ROC-AUC (Area Under ROC Curve)**
```
Interpretation: Probability that model ranks random UP higher than random DOWN
Range: 0.5 (random) to 1.0 (perfect)
Advantage: Threshold-independent metric
```

### Financial-Specific Metrics

**6. Sharpe Ratio (Backtesting)**
```
Formula: (Mean Return - Risk-Free Rate) / Std(Returns)
Purpose: Risk-adjusted returns from model predictions
Strategy: Buy when predict UP, Sell/Hold when predict DOWN
```

**7. Maximum Drawdown**
```
Formula: Max(Peak - Trough) / Peak
Purpose: Worst-case loss scenario
Target: Minimize drawdown while maintaining returns
```

**8. Win Rate**
```
Formula: Number of Profitable Trades / Total Trades
Benchmark: > 50% for profitable strategy
```

**9. Profit Factor**
```
Formula: Gross Profit / Gross Loss
Interpretation: Total winning $ / Total losing $
Target: > 1.0 for profitability
```

### Comparison Metrics

**10. Statistical Significance Testing**
```
Use: McNemar's test for comparing paired predictions
Purpose: Determine if one model is significantly better than another
Threshold: p-value < 0.05
```

**11. Confusion Matrix Analysis**
```
Detailed breakdown:
- True Positives: Correctly predicted UP
- True Negatives: Correctly predicted DOWN
- False Positives: Predicted UP, actually DOWN (Type I error - buy when shouldn't)
- False Negatives: Predicted DOWN, actually UP (Type II error - miss opportunity)
```

### Evaluation Procedure
```
For each trained model:
1. Generate predictions on test set: y_pred_prob = model.predict(X_test)
2. Apply threshold (default 0.5): y_pred = (y_pred_prob > 0.5).astype(int)
3. Calculate classification metrics (accuracy, precision, recall, F1, AUC)
4. Plot confusion matrix
5. Plot ROC curve and calculate AUC
6. Plot precision-recall curve
7. Analyze errors: which time periods had most errors?
8. Feature importance analysis (SHAP values, attention weights)
9. Backtest strategy: simulate trading based on predictions
10. Calculate financial metrics (Sharpe, drawdown, win rate)
11. Compare with baseline models (random, buy-and-hold, moving average crossover)
```

---

## Baseline Models for Comparison

To validate that deep learning models provide value, compare against:

**1. Random Classifier**
- Randomly predicts UP or DOWN with 50% probability
- Expected accuracy: ~50%
- Purpose: Absolute lower bound

**2. Persistence Model (Naive)**
- Predicts: "Future = Current State"
- If currently UP, predict next period is UP
- Purpose: Test if there's predictable momentum

**3. Moving Average Crossover**
- Buy signal: Short MA > Long MA
- Sell signal: Short MA < Long MA
- Purpose: Classic technical analysis benchmark

**4. Logistic Regression**
- Linear model on engineered features
- Purpose: Test if non-linear deep learning is necessary

**5. Random Forest Classifier**
- Tree-based ensemble method
- Purpose: Strong non-linear baseline without sequential modeling

---

## Project Structure and Code Organization

### Recommended Directory Structure
```
isf_25_26_deep_learning_project/
|
+-- data/
|   +-- AAPL_hourly_yfinance.csv      # Raw data files
|   +-- AMZN_hourly_yfinance.csv
|   +-- NVDA_hourly_yfinance.csv
|   +-- BTC_USD_hourly_binance.csv
|   |
|   +-- processed/                     # Saved processed datasets
|   |   +-- AAPL_features.pkl
|   |   +-- AMZN_features.pkl
|   |   +-- NVDA_features.pkl
|   |   +-- BTC_features.pkl
|   |
|   +-- sequences/                     # Saved sequence data
|       +-- AAPL_24h_sequences.npz
|       +-- AAPL_168h_sequences.npz
|       +-- AAPL_720h_sequences.npz
|       +-- ... (for other assets)
|
+-- notebooks/                         # Main project notebooks
|   |
|   +-- 01_data_exploration_and_eda.ipynb
|   |   # Purpose: Load data, validate quality, perform EDA
|   |   # Outputs: Visualizations, data quality report, volatility analysis
|   |   # Cells: ~20-30 cells
|   |
|   +-- 02_feature_engineering.ipynb
|   |   # Purpose: Create technical indicators and derived features
|   |   # Outputs: Processed feature datasets saved to data/processed/
|   |   # Cells: ~25-35 cells
|   |
|   +-- 03_data_preprocessing_and_sequences.ipynb
|   |   # Purpose: Scaling, target creation, sequence generation
|   |   # Outputs: Ready-to-train sequences saved to data/sequences/
|   |   # Cells: ~20-30 cells
|   |
|   +-- 04_baseline_models.ipynb
|   |   # Purpose: Implement and evaluate baseline models
|   |   # Models: Random, Persistence, MA Crossover, Logistic Regression, Random Forest
|   |   # Outputs: Baseline performance metrics
|   |   # Cells: ~30-40 cells
|   |
|   +-- 05_lstm_models.ipynb
|   |   # Purpose: Build, train, and evaluate LSTM architectures
|   |   # Experiments: Different horizons, hyperparameters
|   |   # Outputs: Trained models, performance metrics, visualizations
|   |   # Cells: ~40-60 cells
|   |
|   +-- 06_gru_models.ipynb
|   |   # Purpose: Build, train, and evaluate GRU architectures
|   |   # Experiments: Different horizons, hyperparameters
|   |   # Outputs: Trained models, performance metrics
|   |   # Cells: ~35-50 cells
|   |
|   +-- 07_cnn_models.ipynb
|   |   # Purpose: Build, train, and evaluate 1D CNN architectures
|   |   # Experiments: Different horizons, filter sizes, kernel sizes
|   |   # Outputs: Trained models, performance metrics
|   |   # Cells: ~35-50 cells
|   |
|   +-- 08_transformer_models.ipynb
|   |   # Purpose: Build, train, and evaluate Transformer architectures
|   |   # Experiments: Attention heads, transformer blocks
|   |   # Outputs: Trained models, attention visualizations
|   |   # Cells: ~40-60 cells
|   |
|   +-- 09_hybrid_cnn_lstm_models.ipynb
|   |   # Purpose: Build, train, and evaluate Hybrid architectures
|   |   # Experiments: CNN-LSTM combinations
|   |   # Outputs: Trained models, performance metrics
|   |   # Cells: ~35-50 cells
|   |
|   +-- 10_hyperparameter_tuning.ipynb
|   |   # Purpose: Systematic hyperparameter optimization
|   |   # Method: Random search or Bayesian optimization
|   |   # Outputs: Optimal configurations, tuning results
|   |   # Cells: ~25-40 cells
|   |
|   +-- 11_cross_asset_experiments.ipynb
|   |   # Purpose: Test generalization across assets
|   |   # Scenarios: A, B, C, D (within-asset, transfers, multi-asset)
|   |   # Outputs: Transfer learning analysis, generalization metrics
|   |   # Cells: ~50-70 cells
|   |
|   +-- 12_financial_backtesting.ipynb
|   |   # Purpose: Backtest trading strategies based on predictions
|   |   # Metrics: Sharpe ratio, drawdown, win rate, profit factor
|   |   # Outputs: Trading performance, equity curves
|   |   # Cells: ~30-45 cells
|   |
|   +-- 13_model_interpretation.ipynb
|   |   # Purpose: Feature importance, error analysis, model insights
|   |   # Methods: SHAP values, attention weights, error patterns
|   |   # Outputs: Interpretation visualizations, insights
|   |   # Cells: ~25-40 cells
|   |
|   +-- 14_final_results_and_conclusions.ipynb
|       # Purpose: Comprehensive results summary and analysis
|       # Contents: All comparisons, statistical tests, final conclusions
|       # Outputs: Final report, recommendations
|       # Cells: ~40-60 cells
|
+-- models/                            # Saved trained models
|   +-- lstm_AAPL_24h_best.h5
|   +-- lstm_AAPL_168h_best.h5
|   +-- transformer_BTC_24h_best.h5
|   +-- ... (all trained models)
|
+-- results/                           # Generated outputs
|   +-- figures/                       # All plots and visualizations
|   |   +-- eda/
|   |   +-- training_curves/
|   |   +-- confusion_matrices/
|   |   +-- roc_curves/
|   |   +-- backtesting/
|   |
|   +-- metrics/                       # Performance metrics
|   |   +-- baseline_results.csv
|   |   +-- lstm_results.csv
|   |   +-- all_models_comparison.csv
|   |   +-- cross_asset_results.csv
|   |
|   +-- logs/                          # Training logs (optional)
|       +-- tensorboard_logs/
|
+-- utils/                             # Helper functions (if needed)
|   +-- helper_functions.py           # Reusable functions across notebooks
|
+-- requirements.txt                   # Python dependencies
+-- environment.yml                    # Conda environment specification
+-- README.md                          # Project overview and setup instructions
+-- deep_learning_project.md          # This file - detailed specification
+-- .gitignore                        # Git ignore file
```

### Notebook Organization Philosophy

**All code lives in notebooks** with the following structure for each notebook:

#### Standard Notebook Structure:
1. **Title and Introduction** (Markdown)
   - Notebook purpose and objectives
   - Expected inputs and outputs
   
2. **Setup Section** (Code)
   - Import libraries
   - Set random seeds for reproducibility
   - Define paths and constants
   - Configure plotting styles
   
3. **Load Data** (Code)
   - Load required datasets
   - Verify data integrity
   
4. **Main Analysis/Experiments** (Mixed Code + Markdown)
   - Step-by-step implementation
   - Inline comments and explanations
   - Visualizations after each major step
   - Results interpretation
   
5. **Save Outputs** (Code)
   - Save models, datasets, or results
   - Export figures
   
6. **Summary and Next Steps** (Markdown)
   - Key findings from this notebook
   - What to do next
   - Links to subsequent notebooks

### Reusable Functions Approach

**Option 1: Inline Functions (Recommended for transparency)**
- Define helper functions directly in each notebook
- Copy-paste common functions between notebooks
- Advantage: Everything visible, fully transparent, no imports needed

**Option 2: Minimal Utils Module (Optional)**
- Create `utils/helper_functions.py` for truly universal functions
- Keep it minimal - only plotting styles, path management, data loading
- Import at top of notebooks: `from utils.helper_functions import *`

**Option 3: Notebook-to-Notebook Functions (Advanced)**
- Use `%run notebook.ipynb` to import from another notebook
- Useful for loading data preprocessing steps
- Example: `%run 02_feature_engineering.ipynb` to reuse feature functions

---

## Implementation Workflow (Notebook-by-Notebook)

### Phase 1: Setup and Data Exploration (Week 1)

#### Notebook 01: Data Exploration and EDA
**File**: `01_data_exploration_and_eda.ipynb`

**Objectives**:
- Load and validate all raw data files
- Perform comprehensive exploratory data analysis
- Understand data characteristics and quality issues
- Compare volatility across assets

**Key Cells/Sections** (~25-30 cells):
1. **Setup**: Import libraries (pandas, numpy, matplotlib, seaborn)
2. **Load Data**: Read all 4 CSV files
3. **Data Validation**: Check for missing values, duplicates, date ranges
4. **Basic Statistics**: Describe each dataset (mean, std, min, max)
5. **Price Visualizations**: Plot close prices for all assets
6. **Returns Analysis**: Calculate and plot returns distributions
7. **Volatility Metrics**: Calculate realized volatility, compare across assets
8. **Stationarity Tests**: ADF test, KPSS test for each asset
9. **Autocorrelation Analysis**: ACF and PACF plots
10. **Correlation Matrix**: Cross-asset correlations
11. **Trading Hours Analysis**: Volume patterns by hour of day
12. **Summary**: Key findings and data quality conclusions

**Outputs**:
- Price time series plots (saved to `results/figures/eda/`)
- Volatility comparison charts
- Returns distribution histograms
- Correlation heatmap
- Data quality summary table

---

### Phase 2: Feature Engineering (Week 2)

#### Notebook 02: Feature Engineering
**File**: `02_feature_engineering.ipynb`

**Objectives**:
- Create technical indicators for all assets
- Generate derived features (returns, ratios)
- Add time-based features
- Save processed datasets

**Key Cells/Sections** (~30-40 cells):
1. **Setup**: Import libraries, load raw data
2. **Basic Features**: Calculate ratios (close/open, high/low, typical price)
3. **Technical Indicators**:
   - Momentum: RSI, MACD, ROC
   - Trend: SMA, EMA, Bollinger Bands, ADX
   - Volatility: ATR, Bollinger Width
   - Volume: OBV, Volume SMA, VWAP
4. **Derived Features**: Returns, cumulative returns, lagged features
5. **Time Features**: Hour of day (cyclical encoding), day of week
6. **Volatility Regime**: Calculate and categorize volatility regimes
7. **Handle NaN Values**: Drop initial rows with NaN from rolling calculations
8. **Feature Validation**: Check distributions, detect outliers
9. **Save Features**: Export to `data/processed/*.pkl`
10. **Feature Summary**: Document all created features

**Outputs**:
- Processed datasets with all features (4 files)
- Feature distribution plots
- Feature correlation matrix
- Feature documentation

**Time Estimate**: 2-3 days

---

#### Notebook 03: Data Preprocessing and Sequence Creation
**File**: `03_data_preprocessing_and_sequences.ipynb`

**Objectives**:
- Create target labels for all time horizons
- Scale/normalize features
- Generate sequences for deep learning models
- Split data into train/validation/test sets

**Key Cells/Sections** (~25-30 cells):
1. **Setup**: Import libraries, load processed features
2. **Target Creation**: Generate labels for 24h, 168h, 720h horizons
3. **Class Balance Analysis**: Check UP vs DOWN distribution
4. **Data Splitting**: Temporal split (80% train, 20% test)
5. **Feature Scaling**: Implement rolling window normalization
6. **Sequence Generation Functions**: Define sequence creation logic
7. **Create Sequences**: Generate 3D arrays for each asset x horizon
8. **Validate Sequences**: Check shapes, no data leakage
9. **Save Sequences**: Export to `data/sequences/*.npz`
10. **Summary**: Report dataset sizes, class distributions

**Outputs**:
- Sequence datasets (12 files: 4 assets x 3 horizons)
- Scalers saved for inference
- Data split summary table

**Time Estimate**: 1-2 days

---

### Phase 3: Baseline Models (Week 3)

#### Notebook 04: Baseline Models
**File**: `04_baseline_models.ipynb`

**Objectives**:
- Implement non-deep-learning baseline models
- Establish performance benchmarks
- Validate that problem is predictable above random

**Key Cells/Sections** (~35-45 cells):
1. **Setup**: Import libraries, load sequences
2. **Random Classifier**: Implement and evaluate
3. **Persistence Model**: "Tomorrow = Today" baseline
4. **Moving Average Crossover**: Traditional technical analysis
5. **Logistic Regression**: Linear model on engineered features
6. **Random Forest**: Tree-based ensemble
7. **Evaluation Functions**: Calculate all metrics (accuracy, F1, AUC, etc.)
8. **Results Comparison**: Compare all baselines across assets and horizons
9. **Visualization**: Bar charts, confusion matrices
10. **Statistical Testing**: Significance tests between models
11. **Save Results**: Export metrics to `results/metrics/baseline_results.csv`
12. **Conclusions**: Which baselines work best? Is problem predictable?

**Outputs**:
- Baseline performance metrics table
- Comparison visualizations
- Saved baseline models (for reference)

**Time Estimate**: 2-3 days

---

### Phase 4: Deep Learning Models (Weeks 4-5)

#### Notebook 05: LSTM Models
**File**: `05_lstm_models.ipynb`

**Objectives**:
- Build and train LSTM architectures
- Experiment with different configurations
- Evaluate across all assets and time horizons

**Key Cells/Sections** (~50-70 cells):
1. **Setup**: Import TensorFlow/Keras, set random seeds
2. **Load Data**: Load sequence data for first asset (AAPL)
3. **LSTM Architecture Definition**: Build model function
4. **Model Compilation**: Optimizer, loss, metrics
5. **Callbacks Setup**: Early stopping, learning rate reduction, checkpointing
6. **Train AAPL 24h**: Full training loop with visualizations
7. **Evaluate AAPL 24h**: Calculate metrics, plot results
8. **Train AAPL 168h**: Repeat for second horizon
9. **Train AAPL 720h**: Repeat for third horizon
10. **Repeat for AMZN, NVDA, BTC**: All assets, all horizons
11. **Hyperparameter Experiments**: Try different layer sizes, dropout rates
12. **Results Summary**: Aggregate metrics, compare across experiments
13. **Save Models**: Export best models to `models/`
14. **Training Curves**: Plot loss/accuracy for all experiments
15. **Conclusions**: Which configurations work best?

**Outputs**:
- Trained LSTM models (12+ models)
- Training history visualizations
- Performance metrics table
- Confusion matrices and ROC curves

**Time Estimate**: 4-5 days

---

#### Notebook 06: GRU Models
**File**: `06_gru_models.ipynb`

**Structure**: Similar to Notebook 05, but with GRU architecture
- Implement GRU models
- Compare training speed vs LSTM
- Evaluate performance differences

**Time Estimate**: 3-4 days

---

#### Notebook 07: CNN Models
**File**: `07_cnn_models.ipynb`

**Structure**: Similar to Notebook 05, but with 1D CNN architecture
- Implement Conv1D layers
- Experiment with kernel sizes and filter numbers
- Compare performance especially on shorter horizons

**Time Estimate**: 3-4 days

---

#### Notebook 08: Transformer Models
**File**: `08_transformer_models.ipynb`

**Objectives**:
- Implement Transformer with self-attention
- Experiment with attention heads and blocks
- Visualize attention patterns

**Key Sections** (additional to standard structure):
- Multi-head attention implementation
- Positional encoding
- Attention weight visualization
- Compare with RNN models

**Time Estimate**: 4-5 days

---

#### Notebook 09: Hybrid CNN-LSTM Models
**File**: `09_hybrid_cnn_lstm_models.ipynb`

**Structure**: Similar to other model notebooks
- Implement hybrid architecture
- Compare with pure CNN and LSTM
- Analyze speed vs performance trade-off

**Time Estimate**: 3-4 days

---

### Phase 5: Hyperparameter Tuning (Week 6-7)

#### Notebook 10: Hyperparameter Tuning
**File**: `10_hyperparameter_tuning.ipynb`

**Objectives**:
- Systematically search for optimal hyperparameters
- Focus on top 2-3 architectures from previous notebooks

**Key Cells/Sections** (~30-40 cells):
1. **Setup**: Import Keras Tuner or Optuna
2. **Define Search Space**: All hyperparameters to tune
3. **Random Search**: Run 30-50 trials with reduced epochs
4. **Results Analysis**: Identify top 5 configurations
5. **Full Training**: Train top configurations completely
6. **Comparison**: Compare tuned vs default models
7. **Best Model Selection**: Choose optimal configuration
8. **Save Best Models**: Export to `models/`
9. **Hyperparameter Sensitivity**: Visualize impact of each parameter

**Outputs**:
- Tuning results table
- Best hyperparameters documented
- Optimized models saved
- Sensitivity analysis plots

**Time Estimate**: 3-4 days

---

### Phase 6: Cross-Asset Generalization (Week 8-9)

#### Notebook 11: Cross-Asset Experiments
**File**: `11_cross_asset_experiments.ipynb`

**Objectives**:
- Test model generalization across different assets
- Implement all transfer learning scenarios (A, B, C, D)

**Key Cells/Sections** (~60-80 cells):
1. **Setup**: Load best models from previous experiments
2. **Scenario A Recap**: Within-asset results (already completed)
3. **Scenario B: Low to High Volatility**:
   - Train on AAPL, test on NVDA
   - Train on AAPL, test on BTC
   - Train on AMZN, test on NVDA
   - Train on AMZN, test on BTC
   - Train on NVDA, test on BTC
4. **Scenario C: High to Low Volatility**:
   - Train on BTC, test on AAPL, AMZN, NVDA
   - Train on NVDA, test on AAPL, AMZN
5. **Scenario D: Multi-Asset Training**:
   - Combine datasets
   - Train on AAPL+AMZN+NVDA, test on BTC
   - Train on all 4, test on each (hold-out)
6. **Results Analysis**: Compare all scenarios
7. **Generalization Patterns**: Identify what transfers well
8. **Volatility Analysis**: Correlate performance with volatility difference
9. **Statistical Testing**: Test significance of performance differences
10. **Save Results**: Export to `results/metrics/cross_asset_results.csv`

**Outputs**:
- Transfer learning results table
- Heatmaps showing transfer performance
- Analysis of generalization patterns
- Recommendations for model deployment

**Time Estimate**: 5-7 days

---

### Phase 7: Financial Backtesting (Week 10)

#### Notebook 12: Financial Backtesting
**File**: `12_financial_backtesting.ipynb`

**Objectives**:
- Simulate trading strategies based on model predictions
- Calculate financial performance metrics
- Compare models from investor's perspective

**Key Cells/Sections** (~35-50 cells):
1. **Setup**: Import backtesting libraries
2. **Trading Strategy Definition**: 
   - Buy when predict UP
   - Sell/Hold when predict DOWN
3. **Backtesting Functions**: Implement trading simulator
4. **Backtest All Models**: Run simulations on test data
5. **Performance Metrics**:
   - Total returns
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Profit factor
6. **Equity Curves**: Plot portfolio value over time
7. **Trade Analysis**: Analyze individual trades
8. **Risk Analysis**: Calculate VaR, CVaR
9. **Transaction Costs**: Add realistic trading costs
10. **Comparison with Buy-and-Hold**: Benchmark strategy
11. **Statistical Tests**: Test profitability significance
12. **Results Summary**: Financial metrics table

**Outputs**:
- Backtesting results for all models
- Equity curve plots
- Risk-return scatter plots
- Financial metrics comparison table

**Time Estimate**: 3-4 days

---

### Phase 8: Model Interpretation (Week 11)

#### Notebook 13: Model Interpretation
**File**: `13_model_interpretation.ipynb`

**Objectives**:
- Understand what models learn
- Identify important features
- Analyze when and why models fail

**Key Cells/Sections** (~30-45 cells):
1. **Setup**: Import SHAP, analysis libraries
2. **Feature Importance (SHAP)**: Calculate for tree-based baselines
3. **Attention Weight Analysis**: Visualize Transformer attention
4. **Error Analysis**:
   - When do models make mistakes?
   - Are errors clustered in time?
   - Regime-specific errors?
5. **Prediction Confidence**: Analyze prediction probabilities
6. **Learning Curves**: How much data is needed?
7. **Misclassification Patterns**: Common error types
8. **Feature Ablation**: Remove features, measure impact
9. **Temporal Analysis**: Performance over time
10. **Conclusions**: Key insights about model behavior

**Outputs**:
- Feature importance plots
- Attention visualizations
- Error analysis charts
- Insights document

**Time Estimate**: 3-4 days

---

### Phase 9: Final Results and Conclusions (Week 12)

#### Notebook 14: Final Results and Conclusions
**File**: `14_final_results_and_conclusions.ipynb`

**Objectives**:
- Compile all results into comprehensive analysis
- Answer all research questions
- Draw final conclusions
- Provide recommendations

**Key Cells/Sections** (~50-70 cells):
1. **Executive Summary**: High-level findings
2. **Research Question 1: Time Horizons**:
   - Results across all horizons
   - Hypothesis validation
   - Conclusions
3. **Research Question 2: Architecture Comparison**:
   - Model performance ranking
   - Training time vs accuracy trade-offs
   - Best architecture recommendations
4. **Research Question 3: Generalization**:
   - Transfer learning results
   - Volatility regime analysis
   - Multi-asset training conclusions
5. **Financial Performance Summary**:
   - Best models for trading
   - Risk-adjusted returns
   - Practical deployment considerations
6. **Overall Conclusions**:
   - Is stock prediction possible with deep learning?
   - Which approaches work best?
   - Limitations and caveats
7. **Statistical Significance**: Final tests across all comparisons
8. **Future Work**: Recommendations for extensions
9. **Lessons Learned**: Key takeaways
10. **Final Recommendations**: Practical guidance

**Outputs**:
- Comprehensive results tables
- Summary visualizations
- Final report (can be exported as HTML/PDF)
- Project conclusions

**Time Estimate**: 4-5 days

---

### Total Timeline: ~12 weeks

| Phase | Duration | Notebooks | Key Deliverables |
|-------|----------|-----------|------------------|
| 1 | Week 1 | 01 | EDA, data validation |
| 2 | Week 2 | 02-03 | Features, sequences |
| 3 | Week 3 | 04 | Baselines established |
| 4-5 | Weeks 4-5 | 05-09 | All DL models trained |
| 6 | Weeks 6-7 | 10 | Optimized models |
| 7 | Weeks 8-9 | 11 | Generalization analysis |
| 8 | Week 10 | 12 | Financial backtesting |
| 9 | Week 11 | 13 | Model interpretation |
| 10 | Week 12 | 14 | Final conclusions |

---

## Expected Outcomes and Hypotheses

### Research Question 1: Time Horizon Impact

**Hypothesis 1A**: Shorter time horizons (24h) will show higher prediction accuracy than longer horizons (720h) because:
- Less accumulation of random noise over short periods
- Stronger momentum effects at intraday scales
- More consistent patterns in short-term price movements

**Hypothesis 1B**: Model performance will degrade non-linearly as time horizon increases:
- 24h accuracy: 55-60%
- 168h accuracy: 52-56%
- 720h accuracy: 50-53%

**Validation Approach**: Compare accuracy across time horizons for each model on within-asset tests (Scenario A).

### Research Question 2: Model Architecture Comparison

**Hypothesis 2A**: Transformer-based models will outperform LSTM/GRU because:
- Better capture of long-range dependencies through self-attention
- Parallel processing of sequences
- More effective learning of temporal patterns

**Hypothesis 2B**: CNN models will underperform on longer horizons but compete on short horizons:
- CNNs excel at local pattern detection (good for 24h)
- Lack of explicit sequential modeling hurts long-term dependencies

**Hypothesis 2C**: Hybrid CNN-LSTM will balance speed and performance:
- CNN feature extraction + LSTM temporal modeling
- May not be best overall but offers good trade-off

**Expected Ranking (by accuracy)**:
1. Transformer (best on long horizons)
2. Hybrid CNN-LSTM
3. LSTM
4. GRU (close to LSTM, faster training)
5. 1D CNN (best on short horizons only)

**Validation Approach**: Compare models on same dataset (AAPL, 24h horizon) with identical preprocessing and hyperparameters.

### Research Question 3: Cross-Asset Generalization

**Hypothesis 3A**: Models trained on low-volatility assets will fail on high-volatility assets:
- Low-vol training doesn't expose model to extreme price swings
- Different volatility regime = distribution shift
- Expected performance drop: 10-20% accuracy

**Hypothesis 3B**: Models trained on high-volatility assets will partially generalize to low-volatility:
- High-vol training exposes model to diverse scenarios
- Over-fitting to noise may hurt, but pattern recognition may transfer
- Expected performance drop: 5-10% accuracy

**Hypothesis 3C**: Multi-asset training will improve robustness:
- Diverse training data prevents overfitting to single asset dynamics
- Better generalization across volatility regimes
- May slightly underperform single-asset on that specific asset
- But will show best performance on unseen assets

**Hypothesis 3D**: Bitcoin will be hardest to predict:
- Highest volatility and noise
- Different market dynamics (24/7 trading, less regulation)
- Expected within-asset accuracy: 50-53% (barely above random)

**Validation Approach**: Compare transfer learning scenarios (B, C, D) against within-asset baseline (A).

### Success Criteria

**Minimum Success**: 
- At least one model achieves >55% accuracy on 24h horizon for equity stocks
- Clear evidence that deep learning outperforms baseline models (>3% accuracy improvement)
- Reproducible results with proper train/test split

**Good Success**:
- Best model achieves >57% accuracy on 24h horizon
- Transformer or Hybrid model outperforms others by >2%
- Multi-asset training shows better generalization than single-asset
- Positive Sharpe ratio (>0.5) in backtesting

**Excellent Success**:
- Best model achieves >60% accuracy on 24h horizon
- Models maintain >52% accuracy even on 720h horizon
- Clear transfer learning patterns identified and explained
- Financial backtest shows Sharpe ratio >1.0
- Feature importance analysis reveals interpretable patterns

---

## Risk Factors and Mitigation

### Risk 1: Data Leakage
**Problem**: Using future information in features or improper data splitting
**Impact**: Artificially inflated performance, model won't work in production
**Mitigation**: 
- Strict temporal splitting (never use future data)
- Rolling window normalization (only past data)
- Careful feature engineering review
- Forward-chaining cross-validation

### Risk 2: Overfitting
**Problem**: Model memorizes training data, doesn't generalize
**Impact**: Great train accuracy, poor test accuracy
**Mitigation**:
- Dropout layers (0.2-0.3)
- L2 regularization
- Early stopping
- Cross-validation
- Simpler models when complex ones overfit

### Risk 3: Class Imbalance
**Problem**: One class dominates (e.g., 60% UP, 40% DOWN)
**Impact**: Model always predicts majority class
**Mitigation**:
- Class weights in loss function
- SMOTE or undersampling
- Focus on balanced metrics (F1, AUC) not just accuracy
- Threshold tuning

### Risk 4: Non-Stationarity
**Problem**: Financial time series distributions change over time
**Impact**: Model trained on one regime fails on another
**Mitigation**:
- Regular model retraining
- Adaptive features (rolling statistics)
- Multi-regime training data
- Monitoring for distribution drift

### Risk 5: Computational Resources
**Problem**: Training many models with hyperparameter tuning is expensive
**Impact**: Time constraints, GPU memory issues
**Mitigation**:
- Start with smaller models and scale up
- Use reduced epochs for hyperparameter search
- Implement early stopping aggressively
- Focus on best-performing architectures after initial screening
- Use cloud GPUs if local resources insufficient

### Risk 6: Overly Optimistic Results
**Problem**: P-hacking, multiple testing without correction
**Impact**: False conclusions about model effectiveness
**Mitigation**:
- Pre-register hypotheses
- Bonferroni correction for multiple comparisons
- Out-of-sample validation
- Statistical significance testing
- Honest reporting of all experiments, not just successful ones

---

## Tools and Technologies

### Development Environment
- **Jupyter Lab** or **Jupyter Notebook**: Primary development environment
  - Interactive code execution
  - Inline visualizations
  - Markdown documentation
  - Cell-based organization
- **VS Code with Jupyter Extension**: Alternative IDE with excellent notebook support

### Core Deep Learning
- **TensorFlow 2.x / Keras**: Primary deep learning framework (recommended)
  - Sequential and Functional API for model building
  - Extensive documentation and community support
  - Good for financial time series
- **PyTorch**: Alternative deep learning framework
  - More flexible, research-friendly
  - Choose based on preference

### Essential Data Science Libraries
- **NumPy** (>=1.21): Numerical computing, array operations
- **Pandas** (>=1.3): Data manipulation, time series handling
- **Scikit-learn** (>=1.0): Preprocessing, baseline models, metrics
- **SciPy**: Statistical tests, scientific computing

### Financial Analysis Libraries
- **TA-Lib** (>=0.4): Technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
  - Installation: `conda install -c conda-forge ta-lib` or `pip install TA-Lib`
- **pandas_ta**: Pure Python alternative to TA-Lib
- **yfinance**: Stock data download (already used for data collection)

### Visualization Libraries
- **Matplotlib** (>=3.5): Core plotting library
- **Seaborn** (>=0.11): Statistical visualizations, enhanced aesthetics
- **Plotly** (>=5.0): Interactive charts (useful for presentations)
- **TensorBoard**: Training monitoring and visualization

### Model Interpretation
- **SHAP** (>=0.41): Feature importance and model explanation
  - Works with tree-based models and deep learning
- **LIME**: Local interpretable model-agnostic explanations (optional)

### Hyperparameter Tuning
- **Keras Tuner**: Built-in with TensorFlow, easy to use in notebooks
- **Optuna**: More advanced Bayesian optimization
- **scikit-optimize**: Bayesian optimization for scikit-learn

### Notebook-Specific Tools
- **tqdm**: Progress bars for loops (essential for long-running cells)
- **IPython.display**: Enhanced output formatting
- **nbconvert**: Convert notebooks to HTML/PDF for reports
- **jupyter-contrib-nbextensions**: Useful extensions (table of contents, code folding)

### Version Control and Collaboration
- **Git**: Version control for notebooks and code
- **nbdime**: Better diff and merge for notebooks
- **jupytext**: Sync notebooks with Python scripts (optional)

### Optional Enhancements
- **Weights & Biases**: Cloud-based experiment tracking (integrates with notebooks)
- **MLflow**: Experiment tracking and model registry
- **papermill**: Parameterize and execute notebooks programmatically

---

## Jupyter Notebook Best Practices

### Code Organization in Notebooks

#### 1. Cell Structure Guidelines
**Keep cells focused and atomic**:
- One logical operation per cell
- Cells should be runnable independently (when possible)
- Avoid cells longer than ~50 lines - break into smaller cells

**Good Cell Structure**:
```python
# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Load data
df = pd.read_csv('data/AAPL_hourly_yfinance.csv')

# Cell 3: Data validation
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Cell 4: Visualize
plt.figure(figsize=(12, 6))
plt.plot(df['close'])
plt.title('AAPL Close Price')
plt.show()
```

**Bad Cell Structure** (avoid):
```python
# One massive cell doing everything
import pandas as pd
df = pd.read_csv('data/AAPL_hourly_yfinance.csv')
print(df.shape)
# ... 200 more lines ...
# If error occurs, hard to debug!
```

#### 2. Reproducibility Requirements

**Always set random seeds at the start**:
```python
import random
import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# For even more reproducibility
import os
os.environ['PYTHONHASHSEED'] = str(SEED)
```

**Define paths as constants**:
```python
# Paths configuration
DATA_DIR = 'data/'
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/'
FIGURES_DIR = 'results/figures/'

# Use throughout notebook
df = pd.read_csv(f'{DATA_DIR}AAPL_hourly_yfinance.csv')
```

#### 3. Documentation Standards

**Use Markdown cells liberally**:
- Start each notebook with title and overview
- Before each major section, explain what comes next
- After results, interpret and explain findings
- End with summary and next steps

**Example Structure**:
```markdown
# Notebook 05: LSTM Models

## Overview
This notebook implements and evaluates LSTM architectures for stock price 
movement prediction.

## Objectives
1. Build LSTM model architecture
2. Train on AAPL for all time horizons
3. Evaluate performance
4. Compare with baselines

---

## Section 1: Setup and Configuration
```

**Comment complex code**:
```python
# Calculate rolling normalized features
# Using 720-hour (30-day) window to respect temporal ordering
window_size = 720
df['close_normalized'] = (
    df['close'] - df['close'].rolling(window_size).mean()
) / df['close'].rolling(window_size).std()
```

#### 4. Output Management

**Control output verbosity**:
```python
# For training, show progress
model.fit(X_train, y_train, epochs=50, verbose=1)

# For repetitive operations, suppress output
for asset in assets:
    model.fit(X[asset], y[asset], epochs=50, verbose=0)  # Silent
    print(f"Trained on {asset}")  # Manual reporting
```

**Clear outputs before committing**:
- Before saving notebook to Git, clear all outputs: `Cell > All Output > Clear`
- Keep outputs during development for reference
- Exception: Final notebooks can keep outputs for presentation

**Use progress bars for loops**:
```python
from tqdm.notebook import tqdm

for epoch in tqdm(range(100), desc="Training"):
    # Training code
    pass
```

#### 5. Function Definitions in Notebooks

**Define reusable functions in dedicated cells**:
```python
# Cell: Helper Functions

def load_and_preprocess(asset_name):
    """Load and preprocess data for a given asset."""
    df = pd.read_csv(f'{DATA_DIR}{asset_name}_hourly_yfinance.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    return df

def plot_training_history(history, title):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

**When to create external .py file**:
- Function used across 5+ notebooks
- Very complex functions (>100 lines)
- Functions that rarely change
- Keep in `utils/helper_functions.py`

#### 6. Model Training in Notebooks

**Save checkpoints during training**:
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath=f'{MODELS_DIR}lstm_AAPL_24h_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)
```

**Handle long-running cells**:
```python
import time

# Add timing
start_time = time.time()

# Long operation
model.fit(X_train, y_train, epochs=100)

# Report duration
elapsed = time.time() - start_time
print(f"Training completed in {elapsed/60:.2f} minutes")
```

**Restart kernel before final run**:
- Before finalizing experiments: `Kernel > Restart & Run All`
- Ensures notebook runs from clean state
- Catches any hidden state dependencies

#### 7. Visualization Best Practices

**Set global plot styles at start**:
```python
# Configure matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set default figure size
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
```

**Save figures programmatically**:
```python
def save_figure(fig, filename, dpi=300):
    """Save figure to results directory."""
    filepath = f'{FIGURES_DIR}{filename}'
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filepath}")

# Usage
fig, ax = plt.subplots()
# ... plotting code ...
save_figure(fig, 'eda/AAPL_price_series.png')
plt.show()
```

**Use subplots for comparisons**:
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, asset in enumerate(assets):
    ax = axes.flatten()[idx]
    ax.plot(data[asset]['close'])
    ax.set_title(f'{asset} Price')

plt.tight_layout()
plt.show()
```

#### 8. Error Handling and Debugging

**Use try-except for robust operations**:
```python
# When loading models or data that might not exist
try:
    model = tf.keras.models.load_model(f'{MODELS_DIR}lstm_AAPL_24h.h5')
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model not found, training new model...")
    model = build_lstm_model()
```

**Add assertions for data validation**:
```python
# Verify data shapes
assert X_train.shape[1] == sequence_length, "Incorrect sequence length"
assert X_train.shape[2] == n_features, "Incorrect number of features"
print(f"[OK] Data validation passed: {X_train.shape}")
```

**Use IPython magic commands for debugging**:
```python
# Time a cell
%%time
# Cell code here

# Profile a cell
%%prun
# Cell code here

# Debug on error
%pdb on  # Auto-start debugger on exception
```

#### 9. Memory Management

**Delete large objects when done**:
```python
# After processing large dataset
del df_large, X_temp, y_temp

# Force garbage collection
import gc
gc.collect()
```

**Use generators for large datasets**:
```python
def data_generator(file_list, batch_size=32):
    """Generate batches to avoid loading all data at once."""
    while True:
        for file in file_list:
            data = pd.read_csv(file)
            # Process and yield batches
            yield X_batch, y_batch
```

#### 10. Results Storage and Tracking

**Save results to structured formats**:
```python
# Save metrics to CSV for later analysis
results_df = pd.DataFrame({
    'model': ['LSTM', 'GRU', 'CNN'],
    'accuracy': [0.58, 0.56, 0.55],
    'f1_score': [0.57, 0.55, 0.54],
    'training_time': [45, 38, 22]
})

results_df.to_csv(f'{RESULTS_DIR}metrics/model_comparison.csv', index=False)
```

**Track experiments in markdown**:
```markdown
## Experiment Log

### Experiment 1: LSTM Baseline (2024-12-07)
- Architecture: 2 LSTM layers (128, 64 units)
- Dropout: 0.2
- Results: Accuracy=0.58, F1=0.57
- Notes: Good performance on AAPL, struggles on BTC

### Experiment 2: Added Dropout (2024-12-08)
- Changed dropout from 0.2 to 0.3
- Results: Accuracy=0.59, F1=0.58
- Notes: Reduced overfitting
```

### Notebook Workflow Tips

**Run notebooks in order**:
1. Create `00_environment_setup.ipynb` to verify all dependencies
2. Number notebooks sequentially (01, 02, 03...)
3. Each notebook should be runnable independently (load from saved data)

**Share notebooks effectively**:
- Export to HTML: `File > Export Notebook As > HTML`
- Include in README which notebooks to run first
- Document dependencies between notebooks

**Kernel management**:
- Restart kernel regularly to avoid hidden state
- For long experiments, use `nohup` with `nbconvert --execute`
- Monitor memory usage: `!free -h` (Linux) or Activity Monitor

### Common Pitfalls to Avoid

[X] **Don't**: Run cells out of order during development
[OK] **Do**: Regularly "Restart & Run All" to verify order

[X] **Don't**: Import libraries in middle of notebook
[OK] **Do**: All imports at the top (first or second cell)

[X] **Don't**: Hard-code paths like `/home/user/data/`
[OK] **Do**: Use relative paths or variables

[X] **Don't**: Create 500-line cells
[OK] **Do**: Break into logical 20-50 line cells

[X] **Don't**: Forget to save models and results
[OK] **Do**: Save after each major experiment

[X] **Don't**: Leave notebook running overnight without checkpoints
[OK] **Do**: Use ModelCheckpoint callback, save intermediate results

---

## Reporting and Documentation Requirements

### During Development

#### 1. Notebook Documentation
**Each notebook should contain**:
- Title and overview in opening markdown cell
- Clear section headers explaining each part
- Inline comments for complex code
- Interpretation of results after each experiment
- Summary cell at the end with key findings
- Next steps and links to subsequent notebooks

**Keep notebooks clean**:
- Run "Restart & Run All" before final save
- Clear unnecessary outputs to reduce file size
- Save outputs for key visualizations and results
- Add markdown explanations between code cells

#### 2. Version Control
**Git commit strategy**:
- Commit after completing each notebook
- Meaningful commit messages: "Completed LSTM experiments on AAPL"
- Tag important milestones: "v1.0-baseline-complete"
- Consider using `.gitignore` for large model files

**What to commit**:
- [OK] All notebooks (with or without outputs - decide consistently)
- [OK] README.md, requirements.txt
- [OK] Small result CSVs and metrics
- [X] Large model files (>100MB) - use Git LFS or separate storage
- [X] Large datasets - document how to obtain instead

#### 3. Experiment Tracking
**Maintain results spreadsheet**:
```python
# At end of each model training notebook
results = {
    'date': '2024-12-07',
    'notebook': '05_lstm_models',
    'model': 'LSTM',
    'asset': 'AAPL',
    'horizon': '24h',
    'accuracy': 0.58,
    'f1_score': 0.57,
    'auc': 0.62,
    'training_time_min': 45,
    'notes': 'Baseline LSTM with 2 layers'
}

# Append to master results file
import pandas as pd
results_df = pd.read_csv('results/metrics/all_experiments.csv')
results_df = results_df.append(results, ignore_index=True)
results_df.to_csv('results/metrics/all_experiments.csv', index=False)
```

### Final Deliverables

#### 1. Complete Notebook Suite (Primary Deliverable)
**All 14 notebooks fully documented and executable**:
- Notebooks 01-14 with clear outputs
- Each notebook tells a story
- Exported as HTML for easy viewing without Jupyter
- Can be converted to slides using RISE or exported to PDF

**Export notebooks**:
```bash
# Export all notebooks to HTML
jupyter nbconvert --to html notebooks/*.ipynb

# Or create a PDF (requires LaTeX)
jupyter nbconvert --to pdf notebooks/14_final_results_and_conclusions.ipynb
```

#### 2. Executive Summary Notebook (Standalone)
**Create**: `00_executive_summary.ipynb`

**Contents** (~10-15 cells):
- Project overview and objectives
- Key findings summary
- Best model performance table
- Top 5-10 visualizations
- Conclusions and recommendations
- **Purpose**: Quick overview for stakeholders who won't read all 14 notebooks

#### 3. README.md (Project Overview)
**Essential sections**:
```markdown
# Deep Learning Stock Price Prediction Project

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download data (or use provided data in `data/`)
3. Run notebooks in order: 01 -> 02 -> ... -> 14

## Project Structure
- `notebooks/`: All analysis notebooks (01-14)
- `data/`: Raw and processed datasets
- `models/`: Trained model files
- `results/`: Metrics, figures, logs

## Key Results
- Best model: Transformer on AAPL 24h (accuracy: 0.61)
- Sharpe ratio: 0.52 (backtesting)
- [Link to detailed results](notebooks/14_final_results_and_conclusions.html)

## Notebooks Guide
1. **01_data_exploration_and_eda.ipynb**: Data loading and EDA
2. **02_feature_engineering.ipynb**: Create technical indicators
3. **03_data_preprocessing_and_sequences.ipynb**: Prepare sequences
...
14. **14_final_results_and_conclusions.ipynb**: Complete analysis

## Authors
[Your Name/Team]

## License
MIT
```

#### 4. Requirements File
**Create `requirements.txt`**:
```
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0

# Deep learning
tensorflow>=2.8.0
# OR pytorch>=1.10.0

# Financial analysis
TA-Lib>=0.4.0
yfinance>=0.1.70

# Utilities
tqdm>=4.62.0
jupyter>=1.0.0
ipywidgets>=7.6.0

# Optional
shap>=0.41.0
plotly>=5.0.0
```

#### 5. Model Performance Summary Table
**Generated in Notebook 14**:

| Model | Asset | Horizon | Accuracy | F1 | AUC | Sharpe | Training Time |
|-------|-------|---------|----------|----|----|--------|---------------|
| Random | AAPL | 24h | 0.50 | 0.50 | 0.50 | -0.05 | - |
| Logistic Reg | AAPL | 24h | 0.53 | 0.52 | 0.56 | 0.12 | 1 min |
| LSTM | AAPL | 24h | 0.58 | 0.57 | 0.62 | 0.45 | 45 min |
| GRU | AAPL | 24h | 0.57 | 0.56 | 0.61 | 0.42 | 38 min |
| CNN | AAPL | 24h | 0.55 | 0.54 | 0.59 | 0.28 | 22 min |
| Transformer | AAPL | 24h | 0.61 | 0.59 | 0.65 | 0.52 | 90 min |
| Hybrid | AAPL | 24h | 0.59 | 0.58 | 0.63 | 0.48 | 52 min |

Export as CSV: `results/metrics/final_model_comparison.csv`

#### 6. Key Visualizations Collection
**Organize in `results/figures/` and showcase in final notebook**:

**Essential Plots**:
1. EDA:
   - Price time series (all 4 assets)
   - Volatility comparison
   - Returns distribution
   - Correlation heatmap

2. Model Training:
   - Training/validation curves (best models)
   - Accuracy comparison bar chart (all models)
   - Training time comparison

3. Model Evaluation:
   - Confusion matrices (top 3 models)
   - ROC curves (all models overlaid)
   - Precision-recall curves

4. Cross-Asset Analysis:
   - Transfer learning heatmap
   - Generalization performance chart

5. Financial Analysis:
   - Equity curves (top 5 models)
   - Sharpe ratio comparison
   - Drawdown analysis

6. Model Interpretation:
   - Feature importance (SHAP)
   - Attention weights (Transformer)
   - Error distribution over time

#### 7. Final Presentation (Optional)
**Create slides from notebooks**:
- Use RISE extension: turn notebooks into slides
- Or create separate `presentation.ipynb` with key results
- Export to PDF or PowerPoint

**Slide structure** (10-15 slides):
1. Title and objectives
2. Problem statement
3. Data and methodology
4. Baseline results
5. Deep learning results
6. Architecture comparison
7. Cross-asset generalization
8. Financial backtesting
9. Key insights
10. Conclusions and future work

---

## Environment Setup and Installation

### Step 1: Install Python and Jupyter

**Python Version**: 3.8 or higher (3.9 or 3.10 recommended)

**Option A: Using Conda (Recommended)**:
```bash
# Create new environment
conda create -n stock_prediction python=3.9

# Activate environment
conda activate stock_prediction

# Install Jupyter
conda install jupyter jupyterlab

# Install TA-Lib (easier via conda)
conda install -c conda-forge ta-lib
```

**Option B: Using pip and venv**:
```bash
# Create virtual environment
python3 -m venv stock_prediction_env

# Activate environment
# On Linux/Mac:
source stock_prediction_env/bin/activate
# On Windows:
stock_prediction_env\Scripts\activate

# Install Jupyter
pip install jupyter jupyterlab

# Note: TA-Lib requires system dependencies
# See: https://github.com/mrjbq7/ta-lib#installation
```

### Step 2: Install Project Dependencies

**Create `requirements.txt`**:
```txt
# Core Data Science
numpy==1.23.5
pandas==1.5.3
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2

# Deep Learning
tensorflow==2.12.0
# OR for PyTorch users:
# torch==2.0.0
# torchvision==0.15.0

# Financial Analysis
TA-Lib==0.4.26
yfinance==0.2.18
pandas-ta==0.3.14b0

# Machine Learning
scikit-learn==1.2.2

# Visualization
plotly==5.14.1

# Progress Bars
tqdm==4.65.0

# Model Interpretation
shap==0.41.0

# Utilities
ipywidgets==8.0.6

# Optional: Hyperparameter Tuning
keras-tuner==1.3.5
optuna==3.1.1

# Optional: Experiment Tracking
tensorboard==2.12.2
```

**Install all dependencies**:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

**Create notebook `00_environment_setup.ipynb`**:

```python
# Cell 1: Check Python version
import sys
print(f"Python version: {sys.version}")
assert sys.version_info >= (3, 8), "Python 3.8+ required"
print("[OK] Python version OK")

# Cell 2: Check core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print("[OK] Core libraries OK")

# Cell 3: Check deep learning
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    print("[OK] TensorFlow OK")
except ImportError:
    print("[!] TensorFlow not installed")

# Cell 4: Check financial libraries
try:
    import talib
    print(f"TA-Lib: {talib.__version__}")
    print("[OK] TA-Lib OK")
except ImportError:
    print("[!] TA-Lib not installed - some features may not work")

# Cell 5: Check scikit-learn
import sklearn
print(f"scikit-learn: {sklearn.__version__}")
print("[OK] scikit-learn OK")

# Cell 6: Test basic functionality
df = pd.DataFrame({'close': np.random.randn(100)})
fig, ax = plt.subplots()
ax.plot(df['close'])
plt.title("Test Plot")
plt.show()
print("[OK] All tests passed!")
```

### Step 4: Project Structure Setup

**Create directory structure**:
```bash
# From project root
mkdir -p data/processed data/sequences
mkdir -p notebooks
mkdir -p models
mkdir -p results/figures/eda results/figures/training_curves
mkdir -p results/figures/confusion_matrices results/figures/roc_curves
mkdir -p results/figures/backtesting
mkdir -p results/metrics results/logs
```

**Or use Python script**:
```python
# create_structure.py
import os

directories = [
    'data/processed',
    'data/sequences',
    'notebooks',
    'models',
    'results/figures/eda',
    'results/figures/training_curves',
    'results/figures/confusion_matrices',
    'results/figures/roc_curves',
    'results/figures/backtesting',
    'results/metrics',
    'results/logs'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created: {directory}")

print("[OK] Project structure created")
```

### Step 5: Data Preparation

**Place data files in `data/` directory**:
```
data/
+-- AAPL_hourly_yfinance.csv
+-- AMZN_hourly_yfinance.csv
+-- NVDA_hourly_yfinance.csv
+-- BTC_USD_hourly_binance.csv
```

**Verify data files exist**:
```python
# In notebook or script
import os

data_files = [
    'data/AAPL_hourly_yfinance.csv',
    'data/AMZN_hourly_yfinance.csv',
    'data/NVDA_hourly_yfinance.csv',
    'data/BTC_USD_hourly_binance.csv'
]

for file in data_files:
    if os.path.exists(file):
        print(f"[OK] Found: {file}")
    else:
        print(f"✗ Missing: {file}")
```

### Step 6: Launch Jupyter

**Start Jupyter Lab**:
```bash
# From project root directory
jupyter lab

# Or use classic notebook
jupyter notebook
```

**Browser opens automatically** at `http://localhost:8888`

### Step 7: Configure Jupyter (Optional but Recommended)

**Create Jupyter config for better defaults**:
```bash
jupyter lab --generate-config
```

**Edit `~/.jupyter/jupyter_lab_config.py`**:
```python
# Increase max output size (for large outputs)
c.NotebookApp.iopub_data_rate_limit = 10000000

# Auto-save interval (in seconds)
c.FileContentsManager.autosave_interval = 120  # 2 minutes
```

**Install useful Jupyter extensions**:
```bash
# Table of Contents
jupyter labextension install @jupyterlab/toc

# Variable Inspector
pip install lckr-jupyterlab-variableinspector
```

### Troubleshooting Common Issues

#### Issue 1: TA-Lib Installation Fails
**Solution**:
```bash
# On Ubuntu/Debian:
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib

# On Windows:
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# pip install TA_Lib‑0.4.26‑cp39‑cp39‑win_amd64.whl
```

#### Issue 2: TensorFlow GPU Not Detected
**Check CUDA installation**:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# If empty, install CUDA and cuDNN
# See: https://www.tensorflow.org/install/gpu
```

#### Issue 3: Out of Memory Errors
**Solutions**:
- Reduce batch size in training
- Use gradient accumulation
- Clear unused variables: `del large_var; gc.collect()`
- Restart kernel between experiments

#### Issue 4: Kernel Crashes During Training
**Solutions**:
- Monitor system resources (RAM, GPU memory)
- Add memory cleanup between experiments
- Use checkpointing to save progress
- Consider cloud resources (Colab, AWS, etc.)

### Alternative: Google Colab Setup

**For users without local GPU**:

1. **Open Google Colab**: https://colab.research.google.com
2. **Create new notebook**
3. **Enable GPU**: Runtime > Change runtime type > GPU
4. **Install dependencies**:
```python
# In Colab cell
!pip install yfinance TA-Lib pandas-ta shap

# Upload data files
from google.colab import files
uploaded = files.upload()  # Upload CSVs

# Or mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

5. **Limitations**:
- 12-hour session limit
- Need to re-upload data/code each session
- Limited RAM (12-16 GB)

### Development Workflow

**Typical daily workflow**:
1. Activate environment: `conda activate stock_prediction`
2. Navigate to project: `cd /path/to/isf_25_26_deep_learning_project`
3. Start Jupyter: `jupyter lab`
4. Open notebook and work
5. Save frequently (auto-save enabled)
6. Commit to git: `git add notebooks/*.ipynb && git commit -m "Progress update"`
7. Close Jupyter: Ctrl+C in terminal

---

## Future Extensions and Research Directions

### Technical Extensions
1. **Attention Mechanism Analysis**: Visualize what time steps the model focuses on
2. **Ensemble Methods**: Combine multiple models for better predictions
3. **Multi-Task Learning**: Predict both direction and magnitude
4. **Reinforcement Learning**: Train agent to make trading decisions directly
5. **Transfer Learning**: Pre-train on large corpus of stocks, fine-tune on target

### Data Extensions
1. **Alternative Data**: Include sentiment from news, social media
2. **Fundamental Data**: Add earnings, revenue, P/E ratios
3. **Market Regime Features**: VIX, interest rates, macroeconomic indicators
4. **Order Book Data**: For high-frequency prediction (if available)
5. **Cross-Asset Features**: Use correlations between assets explicitly

### Model Extensions
1. **Graph Neural Networks**: Model asset relationships as graph
2. **Temporal Convolutional Networks (TCN)**: Alternative to RNNs
3. **Neural Architecture Search**: Automatically find optimal architecture
4. **Federated Learning**: Train on multiple assets without sharing raw data
5. **Uncertainty Quantification**: Bayesian deep learning for confidence estimates

### Practical Extensions
1. **Real-Time Prediction**: Deploy model for live trading
2. **Portfolio Optimization**: Use predictions for optimal asset allocation
3. **Risk Management**: Incorporate predictions into VaR calculations
4. **Transaction Costs**: Add realistic trading costs to backtesting
5. **Market Impact**: Model how trades affect prices

---

## Ethical Considerations and Limitations

### Limitations
1. **Past Performance != Future Results**: Historical data may not represent future market conditions
2. **Market Efficiency**: If markets are truly efficient, prediction may be impossible
3. **Regime Changes**: Models may fail during black swan events or structural market changes
4. **Look-Ahead Bias**: Despite precautions, subtle data leakage can occur
5. **Survivorship Bias**: Our stocks are selected winners (still trading), not representative
6. **Transaction Costs**: Real trading has costs that reduce profitability
7. **Slippage**: Can't always execute at predicted price
8. **Liquidity**: Large trades may move the market

### Ethical Considerations
1. **Market Manipulation**: Ensure models aren't contributing to market instability
2. **Fairness**: Algorithmic trading may disadvantage retail investors
3. **Transparency**: Black-box models raise concerns in regulated environments
4. **Responsible Use**: This is research, not investment advice
5. **Data Privacy**: If using alternative data, ensure compliance with privacy laws

### Disclaimer
This project is for **educational and research purposes only**. Results should not be interpreted as investment advice. Financial markets are highly complex and risky. Past performance does not guarantee future results. Always consult licensed financial advisors before making investment decisions.

---

## References and Resources

### Key Papers on Financial Machine Learning
1. Bao, W., Yue, J., & Rao, Y. (2017). "A deep learning framework for financial time series using stacked autoencoders and long-short term memory." *PLOS ONE*.
2. Fischer, T., & Krauss, C. (2018). "Deep learning with long short-term memory networks for financial market predictions." *European Journal of Operational Research*.
3. Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). "Financial time series forecasting with deep learning: A systematic literature review: 2005-2019." *Applied Soft Computing*.
4. Lim, B., & Zohren, S. (2021). "Time-series forecasting with deep learning: a survey." *Philosophical Transactions of the Royal Society A*.

### Technical Analysis Resources
- Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.
- Pring, M. J. (2002). *Technical Analysis Explained*. McGraw-Hill.

### Machine Learning for Finance
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt Publishing.

### Online Resources
- Kaggle competitions and notebooks on stock prediction
- ArXiv preprints on financial ML
- QuantConnect and Quantopian (algorithmic trading platforms)
- TensorFlow and PyTorch official tutorials
- Scikit-learn documentation

---

## Contact and Collaboration

**Project Lead**: [Your Name/Team]
**Institution**: [Your Institution]
**Timeline**: 12 weeks (adjust as needed)
**Last Updated**: December 7, 2025

---

## Appendix

### A. Technical Indicator Formulas

**RSI (Relative Strength Index)**:
```
RS = Average Gain / Average Loss (over n periods)
RSI = 100 - (100 / (1 + RS))
```

**MACD (Moving Average Convergence Divergence)**:
```
MACD = EMA(12) - EMA(26)
Signal = EMA(MACD, 9)
Histogram = MACD - Signal
```

**Bollinger Bands**:
```
Middle Band = SMA(close, 20)
Upper Band = Middle Band + (2 x std(close, 20))
Lower Band = Middle Band - (2 x std(close, 20))
```

**ATR (Average True Range)**:
```
True Range = max(high - low, |high - prev_close|, |low - prev_close|)
ATR = SMA(True Range, 14)
```

### B. Data Augmentation Techniques (Optional)

For improving model generalization, consider:

1. **Time Warping**: Slightly stretch or compress time series
2. **Window Slicing**: Extract random subsequences
3. **Magnitude Warping**: Scale magnitude of time series
4. **Jittering**: Add small random noise
5. **Permutation**: Shuffle segments of time series

**Caution**: Use carefully to avoid destroying temporal structure.

### C. Computing Requirements

**Minimum Specifications**:
- CPU: 4 cores, 3.0 GHz
- RAM: 16 GB
- Storage: 50 GB free space
- GPU: Optional but recommended (NVIDIA with CUDA support)

**Recommended Specifications**:
- CPU: 8+ cores, 3.5+ GHz
- RAM: 32 GB
- Storage: 100 GB SSD
- GPU: NVIDIA RTX 3060 or better (12+ GB VRAM)

**Cloud Alternatives**:
- Google Colab Pro (GPU access)
- AWS EC2 with GPU instances
- Google Cloud Platform
- Azure ML

### D. Estimated Training Times

Approximate times for one model on one asset with one horizon:

| Architecture | Training Time (CPU) | Training Time (GPU) |
|--------------|---------------------|---------------------|
| LSTM | 2-4 hours | 20-40 minutes |
| GRU | 1.5-3 hours | 15-30 minutes |
| 1D CNN | 30-60 minutes | 5-10 minutes |
| Transformer | 3-6 hours | 30-60 minutes |
| Hybrid | 2-4 hours | 20-40 minutes |

Total estimated time for all experiments: 100-200 GPU hours or 1000+ CPU hours.

---

## Glossary

- **OHLCV**: Open, High, Low, Close, Volume - standard financial data format
- **Time Horizon**: Length of time into the future for prediction
- **Lookback Window**: Length of historical data used as input
- **Epoch**: One complete pass through the training dataset
- **Batch**: Subset of training data processed together
- **Dropout**: Regularization technique that randomly deactivates neurons
- **Early Stopping**: Stop training when validation performance stops improving
- **Learning Rate**: Step size for gradient descent optimization
- **Overfitting**: Model performs well on training data but poorly on test data
- **Class Imbalance**: Unequal distribution of classes (e.g., 70% UP, 30% DOWN)
- **AUC**: Area Under the Curve (ROC curve) - threshold-independent metric
- **Sharpe Ratio**: Risk-adjusted return metric (return per unit of risk)
- **Drawdown**: Peak-to-trough decline in portfolio value
- **Volatility**: Standard deviation of returns (measure of risk)
- **Regime**: Distinct market condition period (bull, bear, high vol, low vol)

---

## Summary: Jupyter Notebook-First Approach

### Why Notebooks for This Project?

This project is designed to be **entirely notebook-based** for several strategic reasons:

**1. Transparency and Reproducibility**
- Every step is visible: data loading, feature engineering, model training, evaluation
- Easy to re-run experiments and verify results
- Outputs and visualizations are saved with the code
- Perfect for academic/research environments

**2. Iterative Development**
- Quick experimentation and immediate feedback
- Easy to modify hyperparameters and re-train
- Cell-based execution allows testing individual components
- No need to rerun entire scripts for small changes

**3. Documentation and Presentation**
- Markdown cells provide narrative alongside code
- Results are immediately visible for interpretation
- Notebooks can be exported as reports (HTML/PDF)
- Serves as both development environment and final deliverable

**4. Educational Value**
- Clear learning progression from basics to advanced
- Step-by-step explanations
- Visualizations integrated with analysis
- Easy for others to understand and learn from

**5. Flexibility**
- Easy to add experiments without restructuring code
- Can skip or reorder sections as needed
- Natural checkpoint system (each notebook is a milestone)
- Low barrier to entry for beginners

### Project Execution Summary

**14 Notebooks** spanning **12 weeks**:

1. **Notebooks 01-03**: Data preparation (2 weeks)
2. **Notebook 04**: Baselines (1 week)
3. **Notebooks 05-09**: Deep learning models (2 weeks)
4. **Notebook 10**: Hyperparameter tuning (1-2 weeks)
5. **Notebook 11**: Cross-asset experiments (2 weeks)
6. **Notebook 12**: Financial backtesting (1 week)
7. **Notebook 13**: Model interpretation (1 week)
8. **Notebook 14**: Final conclusions (1 week)

**Total Deliverable**: Complete project in 14 executable, documented Jupyter notebooks + supporting files

### Key Differences from Traditional Python Project

| Aspect | Traditional Project | Notebook-Based Project |
|--------|-------------------|----------------------|
| **Code Organization** | Multiple .py modules | 14 sequential notebooks |
| **Execution** | Run scripts from CLI | Execute cells in Jupyter |
| **Functions** | Organized in modules | Defined within notebooks |
| **Documentation** | Separate docs + docstrings | Integrated markdown cells |
| **Results** | Saved to files | Inline + saved to files |
| **Debugging** | Print/logging | Cell outputs + variables |
| **Sharing** | GitHub repo | Notebooks + HTML exports |
| **Testing** | Unit tests in tests/ | Validation within notebooks |

### Best Practices Recap

[OK] **DO**:
- Number notebooks sequentially (01, 02, ...)
- Import all libraries at the top
- Set random seeds for reproducibility
- Save models and results after experiments
- Use markdown to explain what's happening
- Run "Restart & Run All" regularly
- Version control with Git
- Export key notebooks to HTML

[X] **DON'T**:
- Create 500-line cells
- Run cells out of order
- Hard-code absolute paths
- Forget to save intermediate results
- Leave notebooks without documentation
- Commit without testing "Restart & Run All"

### Getting Started Checklist

- [ ] Python 3.8+ installed
- [ ] Jupyter Lab/Notebook installed
- [ ] All dependencies from requirements.txt installed
- [ ] TA-Lib successfully installed and verified
- [ ] TensorFlow/PyTorch installed (GPU optional but recommended)
- [ ] Data files in `data/` directory
- [ ] Project directory structure created
- [ ] Environment setup notebook (00) runs successfully
- [ ] Git repository initialized
- [ ] Ready to start Notebook 01!

---

**END OF SPECIFICATION**

This document serves as a comprehensive guide for implementing the deep learning stock prediction project **entirely in Jupyter notebooks**. All decisions should be justified based on established machine learning and financial engineering principles. The goal is to conduct rigorous, scientific research that provides insights into both model performance and market predictability.

**The notebook-first approach ensures maximum transparency, reproducibility, and educational value** - making this project ideal for academic research, learning, and collaborative development.

---

**Document Version**: 2.0 (Jupyter Notebook Edition)  
**Last Updated**: December 7, 2025  
**Format**: Markdown specification for LLM context and human guidelines 