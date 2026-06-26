# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m2-dust2.csv`
- round_num: `5`
- rows: `306`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.431176 | 0.244760 | 0.727821 | 0.722222 | 0.431176 |
| xgboost | 0.568577 | 0.383251 | 1.119750 | 0.183007 | 0.568577 |

## Closer Per Tick

- lstm: `304`
- xgboost: `2`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
