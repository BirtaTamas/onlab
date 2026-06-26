# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-tyloo-vs-nrg-anubis-OygKONihup8TZ7k3ClDb0W/tyloo-vs-nrg-anubis.csv`
- round_num: `7`
- rows: `186`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.437994 | 0.261417 | 0.861658 | 0.747312 | 0.562006 |
| xgboost | 0.431431 | 0.256251 | 0.830348 | 0.747312 | 0.568569 |

## Closer Per Tick

- lstm: `78`
- xgboost: `108`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
