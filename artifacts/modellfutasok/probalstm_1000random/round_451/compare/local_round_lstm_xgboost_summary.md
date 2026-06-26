# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-wildcard-vs-spirit-bo3-VLdaQLy-otUvCLBOl-LFGy/wildcard-vs-spirit-m2-dust2.csv`
- round_num: `9`
- rows: `251`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.203782 | 0.071127 | 0.256970 | 0.940239 | 0.203782 |
| xgboost | 0.337909 | 0.148133 | 0.460638 | 0.788845 | 0.337909 |

## Closer Per Tick

- lstm: `251`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
