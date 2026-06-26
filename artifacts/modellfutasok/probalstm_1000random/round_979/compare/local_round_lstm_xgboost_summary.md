# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-legacy-ancient-7ivruObh5LTTVaCYe9h-YO/virtus-pro-vs-legacy-ancient.csv`
- round_num: `21`
- rows: `254`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.325586 | 0.177492 | 0.471705 | 0.421260 | 0.325586 |
| xgboost | 0.312932 | 0.160055 | 0.439854 | 0.421260 | 0.312932 |

## Closer Per Tick

- lstm: `122`
- xgboost: `132`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
