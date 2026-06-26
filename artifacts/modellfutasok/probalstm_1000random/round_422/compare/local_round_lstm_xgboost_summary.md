# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `23`
- rows: `175`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.109121 | 0.022268 | 0.122724 | 1.000000 | 0.109121 |
| xgboost | 0.086237 | 0.009653 | 0.091467 | 1.000000 | 0.086237 |

## Closer Per Tick

- lstm: `120`
- xgboost: `55`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
