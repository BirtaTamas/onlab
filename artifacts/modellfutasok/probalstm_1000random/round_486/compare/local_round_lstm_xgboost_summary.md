# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `18`
- rows: `213`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.485017 | 0.283318 | 0.844216 | 0.521127 | 0.485017 |
| xgboost | 0.561250 | 0.345960 | 1.026893 | 0.366197 | 0.561250 |

## Closer Per Tick

- lstm: `165`
- xgboost: `48`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
