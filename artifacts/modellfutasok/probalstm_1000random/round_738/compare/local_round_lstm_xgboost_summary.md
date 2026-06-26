# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `4`
- rows: `175`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.228374 | 0.088856 | 0.293183 | 0.977143 | 0.228374 |
| xgboost | 0.270049 | 0.107522 | 0.347926 | 0.994286 | 0.270049 |

## Closer Per Tick

- lstm: `144`
- xgboost: `31`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
