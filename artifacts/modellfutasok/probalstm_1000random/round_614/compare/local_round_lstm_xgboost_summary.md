# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-vitality-bo3-3MYCYJWfx_8le7ueost7BH/furia-vs-vitality-m1-nuke.csv`
- round_num: `1`
- rows: `166`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.343093 | 0.172434 | 0.478539 | 0.373494 | 0.343093 |
| xgboost | 0.404462 | 0.222314 | 0.593761 | 0.355422 | 0.404462 |

## Closer Per Tick

- lstm: `166`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
