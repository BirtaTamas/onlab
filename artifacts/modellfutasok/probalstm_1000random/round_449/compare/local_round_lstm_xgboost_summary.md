# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `10`
- rows: `219`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.127579 | 0.041792 | 0.156261 | 0.990868 | 0.127579 |
| xgboost | 0.197009 | 0.080919 | 0.257869 | 0.767123 | 0.197009 |

## Closer Per Tick

- lstm: `219`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
