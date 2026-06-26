# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `12`
- rows: `143`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.091815 | 0.033423 | 0.116111 | 1.000000 | 0.091815 |
| xgboost | 0.150240 | 0.055679 | 0.194612 | 0.874126 | 0.150240 |

## Closer Per Tick

- lstm: `142`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
