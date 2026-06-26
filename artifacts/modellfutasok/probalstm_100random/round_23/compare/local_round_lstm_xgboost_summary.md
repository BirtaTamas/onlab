# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-eternal-fire-vs-spirit-bo5-7H36TpK_LYGHtCXpF3Cgdr/eternal-fire-vs-spirit-m3-dust2.csv`
- round_num: `6`
- rows: `217`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.288659 | 0.097167 | 0.354579 | 0.953917 | 0.711341 |
| xgboost | 0.233334 | 0.063009 | 0.272310 | 1.000000 | 0.766666 |

## Closer Per Tick

- lstm: `69`
- xgboost: `148`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
