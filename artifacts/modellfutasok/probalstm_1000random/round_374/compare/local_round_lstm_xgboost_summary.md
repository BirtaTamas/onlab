# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-falcons-vs-pain-bo3-6mWraId8pA69o5etX6dmBT/falcons-vs-pain-m1-inferno.csv`
- round_num: `12`
- rows: `168`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.304425 | 0.157338 | 0.440784 | 0.607143 | 0.304425 |
| xgboost | 0.293682 | 0.138901 | 0.406358 | 0.607143 | 0.293682 |

## Closer Per Tick

- lstm: `65`
- xgboost: `103`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
