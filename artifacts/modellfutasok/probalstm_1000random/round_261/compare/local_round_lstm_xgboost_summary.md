# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `7`
- rows: `224`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.267223 | 0.095506 | 0.338488 | 0.866071 | 0.267223 |
| xgboost | 0.420304 | 0.195233 | 0.568092 | 0.834821 | 0.420304 |

## Closer Per Tick

- lstm: `182`
- xgboost: `42`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
