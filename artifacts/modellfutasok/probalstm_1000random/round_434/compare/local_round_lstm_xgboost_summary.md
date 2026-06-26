# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `27`
- rows: `220`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.137249 | 0.053000 | 0.176024 | 1.000000 | 0.862751 |
| xgboost | 0.108188 | 0.042133 | 0.137751 | 1.000000 | 0.891812 |

## Closer Per Tick

- lstm: `0`
- xgboost: `220`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
