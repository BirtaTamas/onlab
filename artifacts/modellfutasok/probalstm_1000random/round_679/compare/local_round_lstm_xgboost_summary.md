# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-liquid-vs-furia-bo3-oYHD2J45okzf6eapD2F9CM/liquid-vs-furia-m1-mirage.csv`
- round_num: `13`
- rows: `136`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.259468 | 0.103673 | 0.336570 | 0.808824 | 0.259468 |
| xgboost | 0.417137 | 0.199996 | 0.577480 | 0.669118 | 0.417137 |

## Closer Per Tick

- lstm: `133`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
