# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `11`
- rows: `216`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.190958 | 0.040249 | 0.214710 | 1.000000 | 0.809042 |
| xgboost | 0.235654 | 0.064873 | 0.276605 | 1.000000 | 0.764346 |

## Closer Per Tick

- lstm: `179`
- xgboost: `37`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
