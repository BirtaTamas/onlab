# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `12`
- rows: `189`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.418438 | 0.202321 | 0.579912 | 0.444444 | 0.581562 |
| xgboost | 0.341005 | 0.140120 | 0.443484 | 0.968254 | 0.658995 |

## Closer Per Tick

- lstm: `2`
- xgboost: `187`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
