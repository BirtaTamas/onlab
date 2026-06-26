# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `14`
- rows: `262`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.131474 | 0.029401 | 0.149027 | 1.000000 | 0.131474 |
| xgboost | 0.187322 | 0.054883 | 0.222020 | 1.000000 | 0.187322 |

## Closer Per Tick

- lstm: `261`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
