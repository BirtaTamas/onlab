# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `15`
- rows: `126`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.397182 | 0.171155 | 0.522263 | 0.960317 | 0.602818 |
| xgboost | 0.427856 | 0.205665 | 0.586538 | 0.468254 | 0.572144 |

## Closer Per Tick

- lstm: `101`
- xgboost: `25`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
