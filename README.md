# Trade selection using machine learning and candlestick patterns

## Goal

The goal of this project is to build, evaluate, and compare machine learning models that identify trading opportunities based on historical candlestick data and signal candles. Models are trained and tested over specific periods, and their performance is measured by metrics such as hit rate, mean return, and Sharpe ratio for the selected trades.

## Example run

```
Training set period: 2024-12-30 to 2025-03-20
Test set period: 2025-03-21 to 2025-04-17

Number of trades in training period: 16297
Number of trades in test period: 6866

Number of (model, threshold) pairs evaluated: 40

Models used:
RandomForest
ExtraTrees
XGBoost
MLP
FusionModel
RandomForest_reg
ExtraTrees_reg
XGBoost_reg
MLP_reg
FusionModel_reg

Summary of the top 5 performing model-selected trading sets with at least 10 trades, ranked by mean return:

Model              threshold  n_trades  n_correct  hit_rate  mean_return  sharpe  p_mean  p_hitrate  p_sharpe  
RandomForest          0.8        15        11        0.73      0.0243      0.48    0.100    0.044     0.028      
XGBoost               0.8        13        7         0.54      0.0239      0.45    0.117    0.450     0.048      
ExtraTrees_reg        0.1        14        7         0.50      0.0139      0.08    0.220    0.550     0.372      
FusionModel           0.65       180       119       0.66      0.0126      0.14    0.013    0.000     0.028      
RandomForest          0.6        460       277       0.60      0.0076      0.10    0.015    0.000     0.009      
baseline              --         1229      574       0.47      -0.0018     -0.02   0.770    0.863     0.760

Global p-value for top 1 mean return (0.0243): 0.87
```

### Metric definitions

- **baseline** – strategy that takes all possible non-overlapping trades
- **threshold** – minimum predicted probability/return to open a trade  
- **n_trades** – number of non-overlaping trades meeting the threshold
- **n_correct** – number of profitable trades  
- **hit_rate** – `n_correct / n_trades`  
- **mean_return** – average return per trade  
- **sharpe** – Sharpe ratio of selected trade returns  
- **p_mean**, **p_hitrate**, **p_sharpe** – p-values from random sampling tests

### Statistical significance (global p-value

The model with the highest mean return has been chosen among 40 possible (model, threshold) pairs (5 shown in the example). A p-value test mimicing this process of taking the maximum mean return of 40 random samples from the test set gives a p-value of 0.87, indicating that the top mean return of 0.0243 could very well have been due to chance.

## Models

Models tested include:

- **Random Forest** (classifier & regressor)
- **ExtraTrees** (classifier & regressor)
- **XGBoost** (classifier & regressor)
- **MLP** (classifier & regressor)
- **FusionModel** - neural network that uses encoders to handle different types of feature families (classifier & regressor)

Models ending with `_reg` corresond to regressors that predict continuous returns, while those without `_reg`
are classifiers predicting probabilities of the binary target (positive vs. negative return).

## Data

The dataset consists ~ 20,000 one-minute candlestick windows, each containing 26 candlesticks. Each window contains a signal candle at candle 21, defined as a candle whose absolute percentage change satisfies

$$\left|\frac{\text{c}_\text{21} - \text{o}_\text{21}}{\text{o}_\text{21}}\right|\geq 0.04.$$

The target return for each window is defined as the percentage change from the opening price of the candle after the signal candle to the closing price of the fifth candle after the signal candle:

$$\text{return}=\frac{\text{c}_\text{26} - \text{o}_\text{22}}{\text{o}_\text{22}}.$$

Windows were extracted from one-minute data downloaded from [massive.com](https://massive.com).

---

**Notebook content structure:**

1. Data preparation  
2. Feature engineering 
3. Variable selection (heuristic)
4. Model building and training 
5. Evaluation
