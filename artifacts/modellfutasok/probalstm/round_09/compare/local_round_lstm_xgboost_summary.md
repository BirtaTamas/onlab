# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\blast_austin_major\blasttv-austin-major-2025-the-mongolz-vs-faze-bo3-HypmoQ2OL2Ts_Mqj1_9ELG\the-mongolz-vs-faze-m2-anubis.csv`
- round_num: `4`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.537320 | 0.357765 | 0.957511 | 0.409722 | 0.462680 |
| xgboost | 0.524128 | 0.329944 | 0.843193 | 0.229167 | 0.475872 |

## Closer Per Tick

- lstm: `62`
- xgboost: `82`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
