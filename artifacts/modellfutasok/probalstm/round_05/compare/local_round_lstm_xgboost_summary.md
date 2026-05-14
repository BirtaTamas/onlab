# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv`
- round_num: `2`
- rows: `250`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.500604 | 0.300669 | 0.800927 | 0.376000 | 0.499396 |
| xgboost | 0.431580 | 0.230041 | 0.627174 | 0.644000 | 0.568420 |

## Closer Per Tick

- lstm: `46`
- xgboost: `204`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
