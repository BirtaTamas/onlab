# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `32`
- rows: `139`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.304094 | 0.142463 | 0.413696 | 0.791367 | 0.695906 |
| xgboost | 0.259311 | 0.111165 | 0.339636 | 1.000000 | 0.740689 |

## Closer Per Tick

- lstm: `1`
- xgboost: `138`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
