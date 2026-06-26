# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `22`
- rows: `248`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.551472 | 0.316862 | 0.860678 | 0.286290 | 0.551472 |
| xgboost | 0.438718 | 0.219835 | 0.635873 | 0.495968 | 0.438718 |

## Closer Per Tick

- lstm: `68`
- xgboost: `180`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
