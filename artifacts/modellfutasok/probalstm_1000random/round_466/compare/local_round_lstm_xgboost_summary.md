# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `21`
- rows: `183`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.253012 | 0.073408 | 0.301148 | 0.978142 | 0.746988 |
| xgboost | 0.205759 | 0.066243 | 0.253285 | 0.961749 | 0.794241 |

## Closer Per Tick

- lstm: `46`
- xgboost: `137`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
