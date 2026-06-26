# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-spirit-vs-the-mongolz-bo3-Ep_2Z5_t0VWYbCORdH0Tlg/spirit-vs-the-mongolz-m3-mirage.csv`
- round_num: `3`
- rows: `263`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.044583 | 0.013019 | 0.054429 | 0.977186 | 0.044583 |
| xgboost | 0.090740 | 0.021667 | 0.106979 | 0.946768 | 0.090740 |

## Closer Per Tick

- lstm: `259`
- xgboost: `4`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
