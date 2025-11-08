# Trade selection using machine learning and candlestick patterns

## Goal

The goal of this project is to build, compare, and evaluate machine learning models that identify trading opportunities based on historical candlestick data and signal candles. Models are trained and tested over specific periods, and their performance is measured by metrics such as hit rate, mean return, and Sharpe ratio for the selected trades.

## Example run

```
Training set period: 2024-12-30 to 2025-03-20
Test set period: 2025-03-21 to 2025-04-17

Number of trades in training period: 16297
Number of trades in test period: 6866

---------------------------------------------------------

Summary of the top 5 model-selected trading sets selected from the test set, each restricted to contain at least 10 trades.
The ranking is based on mean return.

Model              threshold  n_trades  n_correct  hit_rate  mean_return  sharpe  p_mean  p_hitrate  p_sharpe  
RandomForest          0.8        15        11        0.73      0.0243      0.48    0.100    0.044     0.028      
XGBoost               0.8        13        7         0.54      0.0239      0.45    0.117    0.450     0.048      
ExtraTrees_reg        0.1        14        7         0.50      0.0139      0.08    0.220    0.550     0.372      
FusionModel           0.65       180       119       0.66      0.0126      0.14    0.013    0.000     0.028      
RandomForest          0.6        460       277       0.60      0.0076      0.10    0.015    0.000     0.009      
baseline              --         1229      574       0.47      -0.0018     -0.02   0.770    0.863     0.760
```

### Metric definitions

- **baseline** – strategy that takes all non-overlaping trades
- **threshold** – minimum predicted probability/return to open a trade  
- **n_trades** – number of non-overlaping trades meeting the threshold
- **n_correct** – number of profitable trades  
- **hit_rate** – `n_correct / n_trades`  
- **mean_return** – average return per trade  
- **sharpe** – Sharpe ratio of selected trade returns  
- **p_mean**, **p_hitrate**, **p_sharpe** – p-values from random sampling tests 

### Models

Models tested include:

- **Random Forest** (classifier & regressor)
- **ExtraTrees** (classifier & regressor)
- **XGBoost** (classifier & regressor)
- **MLP** (classifier & regressor)
- **FusionModel** - neural network that uses encoders to handle different types of feature families (classifier & regressor)

Models ending with `_reg` corresond to regressors that predict continuous returns, while those without `_reg`
are classifiers predicting probabilities of the binary target (positive vs. negative return).

### Statistical significance

The model with the highest mean return has been chosen among 40 possible (model, threshold) pairs (5 shown in the example). A fair p-value test should mimic this same process, i.e., select the maximum mean return from 40 randomly chosen samples with sizes corresponding to the `n_trades` of the (model, threshold) pairs. The p-value we got from such a test for this particular run is 0.87, indicating that the top mean return of 0.0243 could very well have been due to chance.
