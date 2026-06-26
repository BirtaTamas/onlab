# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `21`
- rows: `284`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.302851 | 0.167056 | 0.444118 | 0.619718 | 0.302851 |
| xgboost | 0.291230 | 0.157824 | 0.428516 | 0.732394 | 0.291230 |

## Closer Per Tick

- lstm: `142`
- xgboost: `142`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
