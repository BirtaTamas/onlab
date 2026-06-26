# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `16`
- rows: `257`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.439837 | 0.233292 | 0.628699 | 0.237354 | 0.439837 |
| xgboost | 0.463606 | 0.258926 | 0.683094 | 0.249027 | 0.463606 |

## Closer Per Tick

- lstm: `192`
- xgboost: `65`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
