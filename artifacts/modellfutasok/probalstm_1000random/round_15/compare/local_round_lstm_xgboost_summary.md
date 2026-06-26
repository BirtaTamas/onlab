# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-virtuspro-vs-og-inferno-UyQlNJx_rptvvsTtINI5j3/virtus-pro-vs-og-inferno.csv`
- round_num: `8`
- rows: `263`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.393720 | 0.239768 | 0.610845 | 0.387833 | 0.393720 |
| xgboost | 0.317550 | 0.156844 | 0.440013 | 0.532319 | 0.317550 |

## Closer Per Tick

- lstm: `71`
- xgboost: `192`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
