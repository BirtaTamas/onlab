# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-flyquest-vs-fluxo-ancient-YrTVvYzgDXauKEykMAFJPX/flyquest-vs-fluxo-ancient.csv`
- round_num: `10`
- rows: `170`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.216323 | 0.059054 | 0.254274 | 1.000000 | 0.783677 |
| xgboost | 0.219248 | 0.062222 | 0.259983 | 1.000000 | 0.780752 |

## Closer Per Tick

- lstm: `87`
- xgboost: `83`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
