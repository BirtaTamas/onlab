# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-jijiehao-vs-lynn-vision-bo3-vHZRr1xxhgwfg-A38MzOQQ/jijiehao-vs-lynn-vision-m2-dust2.csv`
- round_num: `8`
- rows: `132`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.110462 | 0.023090 | 0.124149 | 1.000000 | 0.110462 |
| xgboost | 0.147172 | 0.036901 | 0.169604 | 1.000000 | 0.147172 |

## Closer Per Tick

- lstm: `102`
- xgboost: `30`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
