# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-virtuspro-vs-vitality-bo3-8Ft8K1evi_LZ8kW_kkrYdB/virtus-pro-vs-vitality-m1-train.csv`
- round_num: `14`
- rows: `257`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.439769 | 0.240902 | 0.641844 | 0.264591 | 0.439769 |
| xgboost | 0.448144 | 0.249610 | 0.663002 | 0.365759 | 0.448144 |

## Closer Per Tick

- lstm: `174`
- xgboost: `83`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
