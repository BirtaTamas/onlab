# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-astralis-vs-natus-vincere-bo3-4-6Sb81TUo41h9OxcK0xKz/astralis-vs-natus-vincere-m3-nuke.csv`
- round_num: `5`
- rows: `146`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.117674 | 0.025043 | 0.132556 | 1.000000 | 0.117674 |
| xgboost | 0.196342 | 0.069383 | 0.242738 | 1.000000 | 0.196342 |

## Closer Per Tick

- lstm: `124`
- xgboost: `22`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
