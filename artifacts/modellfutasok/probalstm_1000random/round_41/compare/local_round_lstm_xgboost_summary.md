# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `8`
- rows: `128`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.169514 | 0.048375 | 0.201846 | 1.000000 | 0.830486 |
| xgboost | 0.192547 | 0.069253 | 0.242573 | 0.984375 | 0.807453 |

## Closer Per Tick

- lstm: `75`
- xgboost: `53`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
