# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `3`
- rows: `135`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.056278 | 0.003880 | 0.058324 | 1.000000 | 0.943722 |
| xgboost | 0.018852 | 0.000449 | 0.019081 | 1.000000 | 0.981148 |

## Closer Per Tick

- lstm: `0`
- xgboost: `135`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
