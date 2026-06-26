# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `9`
- rows: `228`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.343805 | 0.165710 | 0.473783 | 0.596491 | 0.656195 |
| xgboost | 0.333250 | 0.165316 | 0.463360 | 0.491228 | 0.666750 |

## Closer Per Tick

- lstm: `97`
- xgboost: `131`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
