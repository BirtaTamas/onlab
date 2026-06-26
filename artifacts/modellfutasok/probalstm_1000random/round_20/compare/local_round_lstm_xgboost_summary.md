# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `13`
- rows: `183`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.259936 | 0.091860 | 0.325161 | 1.000000 | 0.740064 |
| xgboost | 0.169018 | 0.057748 | 0.210167 | 1.000000 | 0.830982 |

## Closer Per Tick

- lstm: `0`
- xgboost: `183`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
