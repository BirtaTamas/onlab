# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `14`
- rows: `215`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.252183 | 0.098175 | 0.323600 | 0.879070 | 0.252183 |
| xgboost | 0.293082 | 0.111824 | 0.372539 | 0.962791 | 0.293082 |

## Closer Per Tick

- lstm: `167`
- xgboost: `48`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
