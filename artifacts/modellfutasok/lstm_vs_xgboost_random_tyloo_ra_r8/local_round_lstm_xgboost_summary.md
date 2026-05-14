# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full\asian_champions_league\hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62\tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `8`
- rows: `206`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.450701 | 0.207650 | 0.606050 | 0.946602 | 0.549299 |
| xgboost | 0.481065 | 0.237870 | 0.667378 | 0.805825 | 0.518935 |

## Closer Per Tick

- lstm: `202`
- xgboost: `4`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
