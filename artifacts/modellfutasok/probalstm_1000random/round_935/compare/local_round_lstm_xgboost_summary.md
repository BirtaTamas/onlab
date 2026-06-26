# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `14`
- rows: `232`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.057889 | 0.004115 | 0.060066 | 1.000000 | 0.942111 |
| xgboost | 0.038843 | 0.001890 | 0.039825 | 1.000000 | 0.961157 |

## Closer Per Tick

- lstm: `22`
- xgboost: `210`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
