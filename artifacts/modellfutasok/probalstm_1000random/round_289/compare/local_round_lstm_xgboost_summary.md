# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `3`
- rows: `305`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.093621 | 0.016579 | 0.103177 | 1.000000 | 0.093621 |
| xgboost | 0.117637 | 0.023560 | 0.131452 | 1.000000 | 0.117637 |

## Closer Per Tick

- lstm: `256`
- xgboost: `49`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
