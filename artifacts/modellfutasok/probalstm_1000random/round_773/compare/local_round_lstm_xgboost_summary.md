# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `14`
- rows: `185`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.251799 | 0.107045 | 0.354821 | 0.891892 | 0.748201 |
| xgboost | 0.182987 | 0.076687 | 0.244290 | 0.897297 | 0.817013 |

## Closer Per Tick

- lstm: `23`
- xgboost: `162`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
