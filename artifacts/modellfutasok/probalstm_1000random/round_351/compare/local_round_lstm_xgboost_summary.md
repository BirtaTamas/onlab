# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-falcons-vs-g2-bo3-Xf_UKx9fB2btv0Vy_VBVUC/falcons-vs-g2-m3-mirage.csv`
- round_num: `12`
- rows: `130`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.330472 | 0.230385 | 0.608436 | 0.607692 | 0.330472 |
| xgboost | 0.392808 | 0.233820 | 0.643094 | 0.607692 | 0.392808 |

## Closer Per Tick

- lstm: `84`
- xgboost: `46`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
