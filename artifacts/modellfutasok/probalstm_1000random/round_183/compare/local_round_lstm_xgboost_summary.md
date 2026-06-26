# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `20`
- rows: `227`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.123186 | 0.021370 | 0.135525 | 1.000000 | 0.876814 |
| xgboost | 0.091172 | 0.014698 | 0.099551 | 1.000000 | 0.908828 |

## Closer Per Tick

- lstm: `2`
- xgboost: `225`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
