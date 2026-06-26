# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `10`
- rows: `118`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.624598 | 0.418576 | 1.111112 | 0.194915 | 0.624598 |
| xgboost | 0.656810 | 0.453044 | 1.168211 | 0.076271 | 0.656810 |

## Closer Per Tick

- lstm: `53`
- xgboost: `65`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
