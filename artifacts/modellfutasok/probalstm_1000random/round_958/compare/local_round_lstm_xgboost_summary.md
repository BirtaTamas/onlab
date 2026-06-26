# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `12`
- rows: `135`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.188089 | 0.078538 | 0.246413 | 0.940741 | 0.188089 |
| xgboost | 0.198724 | 0.078008 | 0.254765 | 0.962963 | 0.198724 |

## Closer Per Tick

- lstm: `96`
- xgboost: `39`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
