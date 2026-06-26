# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m1-inferno.csv`
- round_num: `19`
- rows: `187`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.332726 | 0.123182 | 0.420907 | 0.860963 | 0.667274 |
| xgboost | 0.282929 | 0.092986 | 0.347696 | 0.903743 | 0.717071 |

## Closer Per Tick

- lstm: `19`
- xgboost: `168`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
