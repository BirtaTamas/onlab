# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `2`
- rows: `134`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.113030 | 0.034265 | 0.135592 | 1.000000 | 0.113030 |
| xgboost | 0.132448 | 0.040359 | 0.159260 | 1.000000 | 0.132448 |

## Closer Per Tick

- lstm: `126`
- xgboost: `8`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
