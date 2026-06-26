# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-spirit-vs-vitality-bo3-KtWhzrlsNkWCCS0U9BIlr3/spirit-vs-vitality-m2-nuke.csv`
- round_num: `4`
- rows: `128`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.332632 | 0.125841 | 0.420733 | 0.976562 | 0.667368 |
| xgboost | 0.329610 | 0.129813 | 0.423962 | 0.976562 | 0.670390 |

## Closer Per Tick

- lstm: `68`
- xgboost: `60`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
