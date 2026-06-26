# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `15`
- rows: `261`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.295235 | 0.121837 | 0.382978 | 0.946360 | 0.295235 |
| xgboost | 0.353387 | 0.159782 | 0.472215 | 1.000000 | 0.353387 |

## Closer Per Tick

- lstm: `207`
- xgboost: `54`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
