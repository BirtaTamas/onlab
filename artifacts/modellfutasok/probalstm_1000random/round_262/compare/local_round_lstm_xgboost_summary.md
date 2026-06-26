# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-heroic-vs-aurora-bo3-0XrprgXu_t-aBJHUPpJYb4/heroic-vs-aurora-m1-overpass.csv`
- round_num: `7`
- rows: `183`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.218795 | 0.056383 | 0.253579 | 1.000000 | 0.781205 |
| xgboost | 0.216672 | 0.054387 | 0.249804 | 1.000000 | 0.783328 |

## Closer Per Tick

- lstm: `77`
- xgboost: `106`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
