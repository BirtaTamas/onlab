# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `16`
- rows: `215`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.334727 | 0.143648 | 0.442281 | 0.795349 | 0.665273 |
| xgboost | 0.323728 | 0.150780 | 0.440468 | 0.511628 | 0.676272 |

## Closer Per Tick

- lstm: `102`
- xgboost: `113`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
