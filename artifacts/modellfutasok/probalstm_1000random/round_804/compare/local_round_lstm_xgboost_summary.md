# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `19`
- rows: `251`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.549220 | 0.325276 | 0.860801 | 0.374502 | 0.450780 |
| xgboost | 0.515199 | 0.291517 | 0.778834 | 0.509960 | 0.484801 |

## Closer Per Tick

- lstm: `77`
- xgboost: `174`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
