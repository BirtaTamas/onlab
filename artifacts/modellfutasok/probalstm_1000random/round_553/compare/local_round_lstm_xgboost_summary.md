# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-spirit-vs-the-huns-bo3-TWIJIxJZifB3vPv3OUvjVr/spirit-vs-the-huns-m2-dust2.csv`
- round_num: `11`
- rows: `183`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.135638 | 0.055396 | 0.175730 | 0.945355 | 0.135638 |
| xgboost | 0.191884 | 0.093867 | 0.269811 | 0.939891 | 0.191884 |

## Closer Per Tick

- lstm: `182`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
