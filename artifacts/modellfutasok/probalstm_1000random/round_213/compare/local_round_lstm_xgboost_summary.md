# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `11`
- rows: `235`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.461196 | 0.345554 | 0.869411 | 0.438298 | 0.461196 |
| xgboost | 0.449554 | 0.315456 | 0.791510 | 0.387234 | 0.449554 |

## Closer Per Tick

- lstm: `120`
- xgboost: `115`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
