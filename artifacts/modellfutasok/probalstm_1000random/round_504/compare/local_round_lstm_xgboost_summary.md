# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m3-ancient.csv`
- round_num: `9`
- rows: `215`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.359145 | 0.178397 | 0.511571 | 0.744186 | 0.359145 |
| xgboost | 0.315565 | 0.136003 | 0.422450 | 0.781395 | 0.315565 |

## Closer Per Tick

- lstm: `61`
- xgboost: `154`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
