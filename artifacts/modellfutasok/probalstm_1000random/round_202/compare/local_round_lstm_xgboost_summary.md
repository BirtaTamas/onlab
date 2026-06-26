# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `8`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.438034 | 0.237271 | 0.726665 | 0.694444 | 0.561966 |
| xgboost | 0.458225 | 0.252528 | 0.722109 | 0.652778 | 0.541775 |

## Closer Per Tick

- lstm: `103`
- xgboost: `41`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
