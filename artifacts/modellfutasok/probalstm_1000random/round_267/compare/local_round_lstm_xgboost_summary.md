# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `9`
- rows: `283`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.252832 | 0.096264 | 0.319209 | 1.000000 | 0.252832 |
| xgboost | 0.343776 | 0.166027 | 0.472260 | 0.484099 | 0.343776 |

## Closer Per Tick

- lstm: `283`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
