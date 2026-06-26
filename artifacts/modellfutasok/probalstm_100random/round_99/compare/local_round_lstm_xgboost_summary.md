# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m2-anubis.csv`
- round_num: `9`
- rows: `154`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.011175 | 0.000282 | 0.011320 | 1.000000 | 0.011175 |
| xgboost | 0.026065 | 0.000961 | 0.026559 | 1.000000 | 0.026065 |

## Closer Per Tick

- lstm: `153`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
