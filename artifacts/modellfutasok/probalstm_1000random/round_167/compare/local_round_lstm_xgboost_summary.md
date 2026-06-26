# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-g2-vs-betboom-bo3-pCfbtiY01aL_JW2Hy1pnZ6/g2-vs-betboom-m1-anubis.csv`
- round_num: `8`
- rows: `221`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.341114 | 0.166155 | 0.479937 | 0.542986 | 0.341114 |
| xgboost | 0.381680 | 0.172188 | 0.514550 | 0.592760 | 0.381680 |

## Closer Per Tick

- lstm: `127`
- xgboost: `94`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
