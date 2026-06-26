# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `9`
- rows: `228`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.405691 | 0.234169 | 0.616005 | 0.364035 | 0.405691 |
| xgboost | 0.412739 | 0.223423 | 0.614347 | 0.372807 | 0.412739 |

## Closer Per Tick

- lstm: `132`
- xgboost: `96`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
