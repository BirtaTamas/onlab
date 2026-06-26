# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX/flyquest-vs-fluxo-ancient.csv`
- round_num: `2`
- rows: `121`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.059635 | 0.003719 | 0.061580 | 1.000000 | 0.940365 |
| xgboost | 0.021819 | 0.000498 | 0.022072 | 1.000000 | 0.978181 |

## Closer Per Tick

- lstm: `0`
- xgboost: `121`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
