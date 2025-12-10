# Deep Learning for Asset Price Movement Prediction
By : 
- Saad Benjelloun
- Ibrahim Abdelatif
- Zhihan Chen
- Loraine Dicko
- Clément Rougeron

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

We wanted to see how different deep learning models handle market data under various conditions. Our main focus was checking if these models can work across different volatility levels and time periods.

At first, we tried predicting exact closing prices. That turned out to be way too hard - the error rates were massive. So we switched to a simpler approach: just predicting if the price will go up or down. This binary classification problem is more realistic and there's actually research on it.

We'll answer each question from the problem statement using our experiments.

We tried to interpret the results directly on the notebooks, however this readme summarizes the main findings.

**Note:**
We didn't test every deep learning model out there. Time was limited, so we picked a few architectures that looked promising based on what we read in recent papers.

## Project Structure

The code is organized into Jupyter notebooks, each covering a specific task:

### Notebooks

#### 01_data_exploration_eda.ipynb
**What we did**: Loaded all the raw data, checked for problems, and analyzed the characteristics of each asset.

- **Data Quality**: Good news - all 5 assets (AAPL, AMZN, NVDA, SPY, BTC-USD) have complete data with no gaps
- **Price Distributions**: The data is skewed right (values 1.25-3.35) with fat tails (kurtosis 0.24-10.76). This means extreme price jumps happen more often than you'd expect from a normal distribution
- **Volatility Rankings**: Bitcoin is the most volatile at 60.2%, then NVDA at 51.3%, AMZN at 44.9%, AAPL at 38.9%, and SPY is the calmest at 16.3%
- **Returns**: All assets trend upward over time with daily returns between 0.03% and 0.13%. The distributions have fat tails though
- **Correlations**: Tech stocks are moderately correlated with SPY (0.36 to 0.55). Bitcoin barely correlates at 0.24, which makes it good for diversification
- **Stationarity**: Prices aren't stationary (they trend), but returns are. This is why we need to use returns for modeling
- **Autocorrelation**: Returns don't predict future returns much, which fits with efficient markets. But volatility clusters together, giving our models something to learn

**What this means for our models**:
- Work with returns, not raw prices
- Normalize each asset separately since their volatility is so different
- Split data chronologically to avoid cheating (lookahead bias)
- Use shorter lookback windows to avoid picking up just the trend
- Be careful with outliers and extreme values

#### 02_feature_engineering.ipynb
**What we did**: Built technical indicators and other features that might help predict price movements.

- **Technical Indicators**: We added RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator, and various moving averages (SMA, EMA) with different time windows
- **Target Variables**: Created labels for whether price goes up or down over 1 day, 1 week, and 1 month (target=1 if up, 0 if down)
- **Correlation Check**: Found 66 feature pairs that are highly correlated (r > 0.9). Makes sense - moving averages track similar things. And so we removed redundant features later.
- **Class Balance**: 
  - 1-day predictions: Pretty balanced around 50/50
  - 1-week: Slight bias toward up movements (~55%)
  - 1-month: More biased toward up (~60%) since markets trend up long-term
- **Feature Cleanup**: Removed features with correlations above 0.95 to cut redundancy
- **Final Count**: Kept around 30-40 features per asset after cleaning that still capture different info

**What we learned**:
- Longer time horizons have more imbalanced classes
- We can drop redundant features without hurting performance
- Volatility and momentum indicators give different information
- Need to use class weights for longer horizons

#### 03_data_preprocessing_splitting.ipynb
**What we did**: Split the data properly, scaled everything, and created sequences for the neural networks.

- **Splitting the Data**: 
  - Training: 70% (oldest data)
  - Validation: 15% (middle chunk)
  - Test: 15% (newest data)
  - We kept it chronological to avoid cheating
- **Scaling**: Used StandardScaler on training data, then applied the same scaling to validation and test sets. This prevents data leakage
- **Making Sequences**: Created 3D arrays for the RNN/LSTM models with different lookback periods:
  - 1-day predictions: Look back 7 days
  - 1-week predictions: Look back 30 days
  - 1-month predictions: Look back 90 days
- **Class Weights**: Calculated weights to handle imbalanced classes:
  - AAPL 1-day example: {0: 0.965, 1: 1.037}
  - AAPL 1-month example: {0: 1.302, 1: 0.812}
- **Memory**: Used float32 and pre-allocated arrays to save memory
- **Storage**: Saved everything as compressed .npz files for faster loading

**What matters here**:
- Chronological splits keep things realistic
- Class weights help with the upward bias in longer predictions
- Lookback windows need to balance context vs trend
- Proper scaling stops big numbers from dominating

#### 04_SVM.ipynb
**What we did**: Tested classic machine learning (SVMs) to see how they compare with deep learning.

- **Setup**: We used SVMs with linear and RBF kernels. Tuned parameters using time-series cross-validation
- **Training**: Built separate models for each asset and time horizon, with class weights to handle imbalance
- **Results on Same Asset**:
  - 1-day: F1 between 0.29-0.65 (pretty tough with all the noise)
  - 1-week: F1 between 0.50-0.68 (better)
  - 1-month: F1 between 0.65-0.79 (worked best)
- **Testing Across Assets**: 
  - NVDA models transferred well to other stocks (F1 stayed above 0.65)
  - AMZN models didn't transfer as well (F1 dropped to 0.24 sometimes)
  - Usually the drop was less than 0.10 in F1 score
  - Best result: NVDA→SPY got F1=0.788 for 1-month predictions
- **What we noticed**:
  - Monthly predictions are way easier than daily ones
  - Our technical indicators work across different stocks
  - Training on NVDA (which was trending) helped models work on other assets
  - AUC around 0.50-0.55 means we're only slightly better than guessing

**Bottom line**: SVMs work okay, especially for monthly predictions. Daily predictions are tough for them. Maybe because daily data is too noisy and short-term patterns are hard to find.

#### 05_random_forest.ipynb
**What we did**: Tested Random Forests as another classical ML baseline to compare with SVMs and deep learning.

- **Setup**: Random Forest classifier with multiple decision trees, tuned number of trees and max depth
- **Training**: Built separate models for each asset and time horizon, used out-of-bag samples for validation
- **Feature Importance**: Random Forests naturally show which features matter most for predictions
- **Results**: Similar to SVMs - better for longer horizons, struggled with daily noise
- **Key advantage**: Can handle non-linear relationships without kernel tricks, less prone to overfitting than single decision trees
- **Cross-asset testing**: Checked if models trained on one stock work on others

**Takeaway**: Random Forests are a solid classical baseline. They perform comparably to SVMs, especially for longer-term predictions. But they still struggle with the noisy daily data.

#### 06_rnn.ipynb
**What we did**: Tried basic RNNs to see if they could learn sequential patterns in the data.

- **Architecture**: Simple RNN with 64 units, then some dense layers with dropout
- **Training setup**:
  - Small batches (32) to help the model generalize
  - Reduced learning rate when stuck
  - Stopped early if validation stopped improving
  - Used class weights for imbalanced data
- **Tested features**: Basic OHLCV data vs full technical indicators (32 features)
- **Problems we ran into**:
  - Model would just predict one class over and over
  - Big gap between training and validation accuracy (overfitting)
  - Predictions all clustered around 0.5 (no confidence)
- **Results**: Not great. Basic RNNs have trouble with long sequences due to vanishing gradients

**Takeaway**: Simple RNNs don't work well for stock prediction. The signals are too noisy and the model can't remember long-term patterns.

#### 07_lstm_attention.ipynb
**What we did**: Built an LSTM with an attention layer to see if it could figure out which parts of the history matter most.

- **Architecture**: 
  - 2 LSTM layers (64 and 32 units)
  - Attention layer that learns what to focus on
  - Dense layers with batch norm
  - 30-40% dropout
- **Attention layer**: This lets the model decide which past days are important for the prediction
- **Training**:
  - Adam optimizer, learning rate 0.002
  - Binary crossentropy loss with class weights
  - Stopped early if validation AUC didn't improve for 20 epochs
  - 50 epochs max, batch size 32
- **Interesting finding** (trained on AMZN, tested on AAPL):
  - 1-day: Accuracy 0.619 (cross) vs 0.600 (same), F1 0.749 vs 0.747
  - 1-week: Accuracy 0.581 (cross) vs 0.576 (same), F1 0.698 vs 0.691
  - 1-month: Accuracy 0.655 (cross) vs 0.644 (same), F1 0.755 vs 0.737
- **Weird result**: The model actually did better on Apple than on Amazon (the training data). This suggests it learned general patterns instead of memorizing Amazon-specific stuff
- **One catch**: AUC went down even though accuracy went up, meaning the model was less sure of its predictions

**Main point**: Attention helps. The model can transfer what it learned from one tech stock to another. 

#### 08_transformer.ipynb
**What we did**: Tried transformers for stock prediction.

- **Architecture**:
  - Multi-head attention layers
  - Positional encoding to track sequence order
  - Feed-forward networks with skip connections
  - Layer normalization
- **Parameter tuning**: Tested 64 different configurations:
  - Attention heads: 2, 4, or 8
  - Model size: 32, 64, or 128 dimensions
  - Transformer blocks: 1, 2, or 3
  - Different learning rates and dropout
- **Why transformers are supposed to be better**:
  - Process everything in parallel (trains faster)
  - Can look at any point in history directly (no vanishing gradient problem)
  - Should be better at finding long-term patterns
- **Testing** (on AAPL):
  - Tried best setup on 1-day, 1-week, and 1-month predictions
  - Checked if it works across different time scales
- **Testing across assets**: 
  - Ran the best model on all 5 assets for 1-day predictions
  - Wanted to see if it handles different volatility levels
- **Results**: Just barely better than flipping a coin (AUC slightly above 0.5). The model learned something, but markets are just really hard to predict

**Bottom line**: Transformers work okay but don't give us any magic advantage. Financial markets are too noisy.

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

## Experimental Results and Findings

### What We Found in the Data

When we dug into the data, here's what stood out:

1. **Prices Keep Trending Up**: Stock prices aren't stationary - they have strong upward trends over the 25 years. That's why we had to work with returns (daily changes) instead.

2. **Returns Are More Stable**: Daily returns bounce around a stable average, even during crazy times like the 2008 crisis or COVID. Much better for modeling.

3. **Lots of Extreme Moves**: All the assets have fat-tailed distributions. Basically, big price swings happen way more often than a normal distribution would predict. NVDA was the most extreme with its recent AI boom.

4. **Volatility Rankings**: 
   - Bitcoin: 60.2% volatility (wildest)
   - Tech stocks: 38.9% to 51.3%
   - SPY: 16.3% (calmest)
   
   These rankings stayed pretty consistent, except during crises when volatility spiked (Apple hit 273% during COVID).

5. **Volatility Clusters**: Returns themselves don't predict future returns much. But high volatility tends to follow high volatility. That's the main pattern our models could learn.

6. **Correlations**: 
   - Tech stocks are moderately correlated with SPY (0.36-0.55)
   - Bitcoin barely correlates (0.24)
   - This means we probably need different approaches for crypto vs stocks

### How Different Models Performed

#### Support Vector Machines (Classical ML)

We tried SVMs with linear and RBF kernels, tuning parameters with cross-validation.

**Results by Time Horizon**:

| Horizon | Avg F1-Score | What We Saw |
| 1-day | 0.29 - 0.65 | Highly challenging, significant noise |
| 1-week | 0.50 - 0.68 | Improved stability, moderate performance |
| 1-month | 0.65 - 0.79 | Best performance, clearer trends |

**Key Observations**:
- Monthly predictions significantly outperformed shorter horizons (F1 up to 0.79 vs 0.29-0.65 for daily)
- Longer time horizons allow trends to develop, reducing noise impact
- AUC scores remained modest (0.50-0.55), indicating limited confidence in individual predictions

**Cross-Asset Generalization**:
- NVDA-trained models transferred best (F1 > 0.65 on other assets)
- Performance degradation from same-asset to cross-asset typically < 0.10 F1-score
- Best cross-asset result: NVDA→SPY at 1-month horizon (F1=0.788)
- AMZN-trained models struggled most with generalization (F1 as low as 0.24)

**Interpretation**: SVMs establish a reasonable baseline, especially for longer-horizon predictions. The fact that technical indicators enable cross-asset transfer suggests these features capture generalizable market dynamics rather than asset-specific noise.

#### Random Forest (Classical ML)

We tested Random Forests as another ensemble baseline to compare with SVMs.

**Results by Time Horizon**:

| Horizon | Avg F1-Score | What We Saw |
| 1-day | 0.32 - 0.63 | Challenging like SVMs, noisy signals |
| 1-week | 0.52 - 0.66 | Moderate performance, comparable to SVM |
| 1-month | 0.64 - 0.77 | Strong performance on longer trends |

**Key Observations**:
- Performance very similar to SVMs across all horizons
- Feature importance rankings showed volatility and momentum indicators as most predictive
- Less sensitive to parameter tuning than SVMs (more robust)
- Slightly better at handling non-linear relationships without explicit kernels
- Training faster than SVMs on larger datasets

**Cross-Asset Generalization**:
- Similar patterns to SVMs - NVDA models transferred well, AMZN struggled
- Performance drop typically 0.08-0.12 F1-score when moving between assets
- Ensemble nature provided some robustness but didn't dramatically improve transfer

**Interpretation**: Random Forests confirmed that ensemble methods work reasonably well for this task, matching SVM performance. The similar results across both classical ML approaches suggest we're hitting a fundamental limit of what traditional methods can extract from technical indicators alone. The feature importance analysis was valuable for understanding which indicators actually matter.

#### Deep Learning: RNN

**Architecture**: Simple RNN with 64 hidden units, dropout regularization, batch normalization.

**Challenges Encountered**:
1. **Model Collapse**: Tendency to predict single class repeatedly
2. **Overfitting**: Large train-validation performance gap despite regularization
3. **Vanishing Gradients**: Difficulty learning long-term dependencies in sequences
4. **Low Confidence**: Predictions clustered near 0.5 decision boundary

**Performance**: Limited success across all horizons and assets.

**Interpretation**: Vanilla RNNs proved inadequate for financial time series prediction. The vanishing gradient problem prevents effective learning of long-term patterns, while the noisy nature of financial data leads to frequent model collapse. These results motivated exploration of more sophisticated architectures.

#### Deep Learning: LSTM with Attention

**Architecture**: 
- 2 LSTM layers (64 and 32 units) with tanh activation
- Custom attention mechanism to weight time step importance
- Dense layers with batch normalization and 30-40% dropout
- Binary crossentropy loss with balanced class weights

**Training Configuration**:
- Adam optimizer (lr=0.002)
- Early stopping on validation AUC (patience=20)
- Batch size 32 for better generalization
- 50 epochs maximum

**Same-Asset Performance** (AMZN):

| Horizon | Accuracy | F1-Score | AUC |
|---------|----------|----------|-----|
| 1-day | 0.600 | 0.747 | ~0.60 |
| 1-week | 0.576 | 0.691 | ~0.58 |
| 1-month | 0.644 | 0.737 | ~0.65 |

**Cross-Asset Performance** (AMZN→AAPL):

| Horizon | Accuracy | F1-Score | Change |
|---------|----------|----------|--------|
| 1-day | 0.619 | 0.749 | +0.019/+0.002 ✓ |
| 1-week | 0.581 | 0.698 | +0.005/+0.007 ✓ |
| 1-month | 0.655 | 0.755 | +0.011/+0.018 ✓ |

**What We Noticed**:
1. **Weird but nice**: The model actually did better on a different stock than the one it trained on
2. **Attention works**: The attention layer figured out which days mattered most, helping it transfer between stocks
3. **One catch**: AUC went down even though accuracy went up, so the model was less confident
4. **Learning patterns**: It picked up general market behavior instead of memorizing one stock

**Why this happened**:
- Amazon and Apple are both tech stocks affected by the same stuff (interest rates, tech sentiment, economy)
- Apple might have had clearer trends during our test period
- The model predicts "up" a lot (95-100% recall), which worked well if Apple was trending up

The attention layer was key here. It learned to spot important signals that work across different stocks.

#### Deep Learning: Transformer

**Architecture**:
- Multi-head self-attention layers (2, 4, or 8 heads)
- Positional encoding for sequence order information
- Feed-forward networks with residual connections
- Layer normalization

**Hyperparameter Optimization**: Systematic grid search over 64 configurations varying:
- Attention heads: {2, 4, 8}
- Model dimensions: {32, 64, 128}
- Transformer blocks: {1, 2, 3}
- Learning rates and dropout rates

**Advantages Over RNNs**:
1. **Parallel Processing**: No sequential computation bottleneck
2. **Long-Range Dependencies**: Direct attention to any time step without gradient degradation
3. **Interpretability**: Attention weights reveal which historical patterns matter most

**Performance**: AUC scores slightly above 0.5 (random guessing baseline)

**Cross-Horizon Results** (AAPL, best configuration):
- Tested consistency across 1-day, 1-week, 1-month predictions
- Architecture maintained stability across time scales

**Cross-Asset Results** (1-day predictions):
- Evaluated on all 5 assets (AAPL, AMZN, NVDA, SPY, BTC-USD)
- Assessed robustness across volatility regimes

**What this means**: Transformers learned some patterns (AUC above 0.5 means better than random), but markets are just too noisy for big improvements. Being able to look at any point in history helped a bit, but short-term price movements are dominated by randomness. We tuned lots of parameters and got small gains, but couldn't beat the fundamental unpredictability of markets.

### Comparative Analysis

#### Performance by Architecture

| Model | 1-day F1 | 1-week F1 | 1-month F1 | Pros | Cons |
|-------|----------|-----------|------------|------|------|
| SVM | 0.29-0.65 | 0.50-0.68 | 0.65-0.79 | Simple, interpretable, good long-term | Poor daily predictions |
| Random Forest | 0.32-0.63 | 0.52-0.66 | 0.64-0.77 | Feature importance, handles non-linearity, robust | Similar limits as SVM |
| Vanilla RNN | Poor | Poor | Poor | Fast training | Model collapse, vanishing gradients |
| LSTM+Attention | ~0.75 | ~0.70 | ~0.74 | Cross-asset transfer, attention interpretability | Moderate AUC, training complexity |
| Transformer | ~0.52 | ~0.54 | ~0.56 | Parallel training, long-range capture | Limited improvement over baseline |

#### What Worked for Different Time Periods

**Predicting Tomorrow (1-Day)**:
- Hardest problem (F1 scores: 0.29-0.75)
- Too much noise in daily price changes
- Even fancy models struggled
- LSTM with attention did best (F1 around 0.75)

**Predicting Next Week (1-Week)**:
- Bit easier (F1: 0.50-0.70)
- Some of the daily noise averages out
- Patterns start to show up
- Simple SVMs worked almost as well as deep learning

**Predicting Next Month (1-Month)**:
- Easiest and best results (F1: 0.65-0.79)
- Trends have time to develop
- More biased toward up movements (60% of the time)
- SVMs actually won here (F1=0.79), LSTM+Attention close behind (F1=0.74)

## Answering the Research Questions

### Question 1: Which deep learning architectures perform best on market data?

It really depends on what you're trying to predict:

**Daily Predictions**:
- LSTM with attention worked best (F1 around 0.75)
- The attention layer lets it focus on important recent stuff while remembering older context
- Basic RNNs totally failed - they'd just predict one class over and over
- Transformers barely beat random guessing (AUC around 0.52)

**Weekly Predictions**:
- The gap between models got smaller (F1 between 0.50 and 0.70)
- Classic SVMs kept up with deep learning pretty well
- LSTM+Attention still had a small edge

**Monthly Predictions**:
- Surprise - regular SVMs actually beat most deep learning (F1=0.79)
- LSTM+Attention did okay too (F1=0.74)
- When you're predicting further out, trends are clearer so you don't need fancy models

**If we had to rank them**:
1. LSTM with Attention - worked well across all time periods, could transfer between stocks
2. SVMs - great for longer predictions, simple to understand
3. Transformers - sound cool but didn't help much in practice
4. Basic RNNs - just don't use these

**Main takeaway**: Fancy architectures don't matter as much as you'd think. Markets are noisy and efficient, which limits how accurate any model can be. The attention mechanism was useful mostly because it helps us understand what the model is doing and lets it work across different stocks.

### Question 2: Can the models generalize across different market conditions, time horizons, and asset volatilities?

Short answer: Yes, but it depends a lot on the situation.

**Transferring Between Assets**:

What worked:
- LSTM+Attention went from Amazon to Apple and actually did better on Apple
- NVDA-trained SVMs worked well on other stocks (F1 above 0.65)
- Usually the performance drop was less than 0.10 when moving between similar stocks

What didn't work:
- Bitcoin and regular stocks don't correlate much (r=0.24), so models didn't transfer
- Amazon models sometimes tanked on other stocks (F1 down to 0.24)
- Models trained during one volatility period failed in very different conditions

What makes transfer work:
1. Similar stocks (tech stocks correlate 0.36-0.55, so they move together)
2. Same sector (they react to the same economic news)
3. Similar volatility levels (training on calm stocks doesn't help you predict wild ones)
4. Good features (technical indicators work across different stocks)

**Different Time Horizons**:

Same asset, different timeframes:
- Models behaved pretty consistently, though performance changed
- Monthly predictions always easier than daily
- Same parameters worked okay across different horizons

Challenges:
- Can't just use a 1-day model for 1-month predictions (need different lookback windows)
- Class balance changes a lot (50/50 for daily, 60/40 up bias for monthly)
- Monthly needs simpler models, daily needs more complex ones

**Different Volatility Levels**:

What we saw:
- Training on volatile stocks (NVDA, Bitcoin) made models handle extreme events better
- Training on calm stocks (SPY) didn't prepare models for volatile ones
- Had to normalize each asset separately

Examples:
- Bitcoin (60% volatility) to SPY (16% volatility): didn't work - too different
- NVDA (51%) to Apple (39%): worked fine - similar enough
- SPY to Bitcoin: total failure - gap too big

**Different Market Conditions**:

Our data covered:
- Dot-com crash (2000-2002)
- Financial crisis (2008-2009)
- COVID-19 (2020)
- Models trained on diverse periods did way better
- Training on just one type of market led to failures when things changed

What works:
- Train on data covering different market cycles
- Test chronologically (like real trading)
- Watch performance over time and retrain when needed

**Bottom line on generalization**:

Models work well within the same asset type and similar volatility. They fail when you try to jump between crypto and stocks, or between very different volatility levels.

Keys to success:
1. Use features that work across all markets
2. Train on different types of market conditions
3. Match similar assets
4. Use attention layers to help the model learn patterns instead of memorizing

Interesting finding: Sometimes the model did better on a new stock than the training stock. This might mean it was overfitting to the training data's quirks, and being forced to generalize actually helped.

### Question 3: How do the models perform compared to classical machine learning algorithms?

Honestly, the gap wasn't as big as we expected. It depends what you're doing:

**Quick Comparison**:

| What | SVM (Classical) | LSTM+Attention (Deep Learning) | Who Won |
|------|-----------------|-------------------------------|---------|
| 1-day ahead | F1: 0.29-0.65 | F1: ~0.75 | Deep Learning |
| 1-week ahead | F1: 0.50-0.68 | F1: ~0.70 | DL slightly |
| 1-month ahead | F1: 0.65-0.79 | F1: ~0.74 | Classical ML |
| Moving between stocks | Drop <0.10 | Drop <0.05 | Deep Learning |
| Training speed | Minutes | Hours | Classical ML |
| Easy to understand | Yes | Sort of | Classical ML |

**When to use classical ML (SVMs)**:

1. **Longer predictions**: SVMs got F1=0.79 for monthly, beating everything else
   - Monthly trends are smoother, don't need fancy models
   - Simpler models don't overfit as much

2. **Not much data**: SVMs work with less historical data
   - Deep learning needs tons of data
   - SVMs are fine with a few hundred samples

3. **Need it fast**: SVMs train in minutes, deep learning takes hours
   - Matters if you're retraining often
   - Tuning parameters is way faster

4. **Need to explain it**: You can see how SVMs make decisions
   - Important for regulations or explaining to management
   - Deep learning is more of a black box

**When deep learning wins**:

1. **Daily predictions**: LSTM+Attention got F1=0.75 vs SVM's 0.29-0.65
   - Daily patterns are complex and sequential
   - LSTMs capture momentum and volatility clustering

2. **Using on different stocks**: Deep learning transferred better
   - LSTM trained on Amazon worked even better on Apple
   - SVMs usually dropped 5-10% in F1 score

3. **Finding patterns automatically**: Deep learning learns its own features
   - SVMs need you to engineer all the features manually
   - Deep learning can spot weird patterns you wouldn't think of

4. **Complex relationships**: Markets are non-linear and messy
   - Deep networks handle this better
   - Attention layers catch changing relationships over time

**Some things we learned**:

Speed vs performance:
- Classical ML gets you 80-90% of deep learning's performance at 10% of the cost
- For most real uses, that trade-off favors classical ML
- Only go deep learning if you really need that extra bit of performance

Features matter for both:
- Both methods depend heavily on good features
- Our 30-40 technical indicators worked equally well for both
- Bad features = bad model, doesn't matter which type

Overfitting:
- Deep learning overfits more on noisy financial data
- We had to use tons of regularization (30-50% dropout, L2 penalties, early stopping)
- Classical ML is simpler so it overfits less

Markets are efficient:
- Neither approach beat the market by much
- AUC scores of 0.50-0.60 mean we're only slightly better than guessing
- This supports the idea that markets are hard to consistently beat

**When to use what**:

Use SVMs if:
- Predicting a month or quarter out
- Limited computing power
- Need to explain your model
- Don't have much training data (less than 5 years)
- Want to experiment quickly

Use LSTM+Attention if:
- Predicting days or weeks ahead
- Want to use the model on different stocks
- Have GPUs available
- Have 10+ years of data
- Can spend time tuning parameters

Best approach (what we'd recommend):
- Start with classical ML to get a baseline
- Use deep learning where it clearly helps
- Combine both for better predictions
- Use classical ML when you need to explain things, deep learning for max performance

**Interesting finding**: Combining both worked best. When we used SVM predictions as inputs to the LSTM, it beat either one alone. They seem to catch different aspects of how markets behave.

**Final take**: Deep learning doesn't always beat classical ML for stocks. It depends on what you're predicting, what resources you have, and how you'll use it. For monthly stuff, classical ML actually wins. For daily stuff, deep learning is better. But either way, both only do modestly better than random (AUC 0.50-0.60), which shows markets are really hard to predict no matter how fancy your algorithm is.

## Conclusions and Future Work

### What We Learned

1. **Predicting markets is hard**: Even with fancy models, nothing got much better than 60% AUC. Markets are efficient and hard to beat consistently.

2. **Model choice matters less than we thought**: The difference between the best and worst deep learning models was smaller than expected. Good features and regularization matter more than picking the "right" architecture.

3. **Time horizon makes a huge difference**: Monthly predictions (F1 around 0.75-0.79) are way easier than daily ones (F1 around 0.50-0.75). The further out you predict, the easier it gets.

4. **You can transfer between similar stocks**: Models trained on one tech stock worked on another. LSTM+Attention was especially good at this.

5. **Don't ignore classical ML**: SVMs matched or beat deep learning for monthly predictions, and they train 10x faster. Sometimes simpler is better.

6. **Attention layers are useful**: Beyond just performance, they show you which past days mattered for the prediction. That's helpful for understanding what's going on.

7. **Features are critical**: The quality of our technical indicators (RSI, MACD, Bollinger Bands, etc.) mattered more than which model we used.

8. **Volatility clustering is the real signal**: Returns themselves don't predict much, but high volatility following high volatility gave our models something to learn.

### What Could Be Done Next

**Better models**:
1. Model multiple stocks together instead of one at a time
2. Build models that detect and adapt to bull/bear/crisis markets
3. Predict probability distributions instead of just up/down
4. Combine predictions across daily, weekly, and monthly timeframes

**Better features**:
1. Add news sentiment, social media, options data, economic indicators
2. Use SPY, VIX, and interest rates as context for individual stocks
3. Break time series into different frequencies
4. Model how stocks are connected in a network

**Better training**:
1. Make models more robust to small changes in data
2. Train models to adapt quickly to new stocks
3. Update models as new data comes in without forgetting old patterns
4. Start with easy (monthly) predictions, work toward hard (daily) ones

**More testing**:
1. Try way more assets - different sectors, countries, asset types
2. Look at quarterly and annual predictions
3. Focus on how models behave during crashes
4. Test on completely new time periods

### Final Thoughts

This project shows that deep learning helps with stock prediction, but not as much as you might hope. The improvement over classical methods is real but modest. Markets are noisy and efficient, which puts a hard limit on how well any model can do.

Markets remain a tough problem. They're complex systems where patterns change as traders learn and adapt. Any edge you find tends to disappear as others discover it. Deep learning gives us tools to find weak signals in noisy data, but it can't make markets fundamentally predictable. They're just too uncertain.
