# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `17`
- rows: `191`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.462984 | 0.263309 | 0.741556 | 0.675393 | 0.537016 |
| xgboost | 0.334880 | 0.130642 | 0.431765 | 0.858639 | 0.665120 |

## Closer Per Tick

- lstm: `47`
- xgboost: `144`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
