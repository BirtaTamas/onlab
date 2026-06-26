# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m1-ancient.csv`
- round_num: `16`
- rows: `163`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.266602 | 0.147850 | 0.411323 | 0.742331 | 0.266602 |
| xgboost | 0.363814 | 0.190192 | 0.553770 | 0.699387 | 0.363814 |

## Closer Per Tick

- lstm: `143`
- xgboost: `20`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
