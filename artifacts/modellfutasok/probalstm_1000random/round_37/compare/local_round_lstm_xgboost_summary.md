# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `19`
- rows: `213`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.434877 | 0.198898 | 0.582843 | 0.901408 | 0.565123 |
| xgboost | 0.444164 | 0.211729 | 0.604482 | 0.924883 | 0.555836 |

## Closer Per Tick

- lstm: `133`
- xgboost: `80`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
