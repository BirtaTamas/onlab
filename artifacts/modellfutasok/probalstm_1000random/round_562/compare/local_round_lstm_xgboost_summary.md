# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-faze-vs-heroic-dust2-PtQF8ASKD1754yZQHk6148/faze-vs-heroic-dust2.csv`
- round_num: `12`
- rows: `108`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.194802 | 0.068600 | 0.242288 | 1.000000 | 0.805198 |
| xgboost | 0.196744 | 0.074141 | 0.248943 | 1.000000 | 0.803256 |

## Closer Per Tick

- lstm: `43`
- xgboost: `65`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
