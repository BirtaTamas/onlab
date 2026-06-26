# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `11`
- rows: `185`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.655333 | 0.466466 | 1.170222 | 0.102703 | 0.344667 |
| xgboost | 0.457235 | 0.227288 | 0.633221 | 0.459459 | 0.542765 |

## Closer Per Tick

- lstm: `0`
- xgboost: `185`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
