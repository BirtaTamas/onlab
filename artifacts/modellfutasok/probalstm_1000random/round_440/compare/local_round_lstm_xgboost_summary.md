# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-the-huns-vs-ninja-bo3-8zmdVWrC356tnVH1OFLf2Y/the-huns-vs-ninja-m2-anubis.csv`
- round_num: `8`
- rows: `131`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.335154 | 0.172858 | 0.481634 | 0.511450 | 0.335154 |
| xgboost | 0.336406 | 0.160131 | 0.466521 | 0.541985 | 0.336406 |

## Closer Per Tick

- lstm: `64`
- xgboost: `67`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
