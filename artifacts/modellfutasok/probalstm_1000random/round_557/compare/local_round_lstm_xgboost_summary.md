# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-falcons-vs-vitality-bo3-948Z-JwufPJ8ROXkhPE5QF/falcons-vs-vitality-m2-nuke.csv`
- round_num: `4`
- rows: `250`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.337009 | 0.202827 | 0.546182 | 0.624000 | 0.337009 |
| xgboost | 0.437429 | 0.296896 | 0.794335 | 0.452000 | 0.437429 |

## Closer Per Tick

- lstm: `247`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
