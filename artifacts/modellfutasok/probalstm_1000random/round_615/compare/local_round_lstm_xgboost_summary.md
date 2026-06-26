# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `11`
- rows: `196`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.428776 | 0.209265 | 0.597052 | 0.627551 | 0.571224 |
| xgboost | 0.331893 | 0.132122 | 0.428324 | 0.806122 | 0.668107 |

## Closer Per Tick

- lstm: `7`
- xgboost: `189`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
