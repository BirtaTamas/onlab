# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `5`
- rows: `142`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.244560 | 0.143760 | 0.402471 | 0.852113 | 0.244560 |
| xgboost | 0.292586 | 0.165202 | 0.492027 | 0.880282 | 0.292586 |

## Closer Per Tick

- lstm: `132`
- xgboost: `10`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
