# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `10`
- rows: `184`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.196923 | 0.043616 | 0.222844 | 1.000000 | 0.196923 |
| xgboost | 0.298080 | 0.096325 | 0.360331 | 1.000000 | 0.298080 |

## Closer Per Tick

- lstm: `180`
- xgboost: `4`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
