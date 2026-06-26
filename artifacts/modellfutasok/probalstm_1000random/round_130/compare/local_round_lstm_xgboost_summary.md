# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `9`
- rows: `156`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.227259 | 0.081797 | 0.283455 | 1.000000 | 0.772741 |
| xgboost | 0.238088 | 0.103654 | 0.313810 | 0.948718 | 0.761912 |

## Closer Per Tick

- lstm: `69`
- xgboost: `87`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
