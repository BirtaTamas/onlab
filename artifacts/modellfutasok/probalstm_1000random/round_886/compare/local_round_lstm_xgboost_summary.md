# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-faze-bo3-q02I_n27c_oaVV09Kplodn/mouz-vs-faze-m2-mirage.csv`
- round_num: `13`
- rows: `206`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.555815 | 0.323131 | 0.879970 | 0.330097 | 0.444185 |
| xgboost | 0.485014 | 0.251200 | 0.694453 | 0.529126 | 0.514986 |

## Closer Per Tick

- lstm: `35`
- xgboost: `171`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
