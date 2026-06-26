# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m2-dust2.csv`
- round_num: `3`
- rows: `199`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.474682 | 0.238677 | 0.679262 | 0.758794 | 0.474682 |
| xgboost | 0.541419 | 0.304121 | 0.832772 | 0.135678 | 0.541419 |

## Closer Per Tick

- lstm: `173`
- xgboost: `26`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
