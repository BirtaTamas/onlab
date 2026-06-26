# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-falcons-vs-vitality-bo3-948Z-JwufPJ8ROXkhPE5QF/falcons-vs-vitality-m2-nuke.csv`
- round_num: `15`
- rows: `262`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.355682 | 0.198279 | 0.522046 | 0.385496 | 0.355682 |
| xgboost | 0.328386 | 0.168760 | 0.462213 | 0.400763 | 0.328386 |

## Closer Per Tick

- lstm: `88`
- xgboost: `174`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
