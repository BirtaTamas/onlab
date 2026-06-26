# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `18`
- rows: `196`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.773135 | 0.609157 | 1.555303 | 0.020408 | 0.773135 |
| xgboost | 0.833216 | 0.709328 | 1.959209 | 0.020408 | 0.833216 |

## Closer Per Tick

- lstm: `164`
- xgboost: `32`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
