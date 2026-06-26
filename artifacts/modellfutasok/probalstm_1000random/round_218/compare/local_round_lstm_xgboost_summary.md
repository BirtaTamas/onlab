# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-og-vs-falcons-bo3-Q3yO3LacAwamKdCbguw7-l/og-vs-falcons-m1-dust2.csv`
- round_num: `1`
- rows: `150`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.394167 | 0.188160 | 0.542132 | 0.900000 | 0.605833 |
| xgboost | 0.366635 | 0.182165 | 0.509196 | 0.326667 | 0.633365 |

## Closer Per Tick

- lstm: `88`
- xgboost: `62`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
