# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `9`
- rows: `119`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.444343 | 0.217230 | 0.616013 | 0.512605 | 0.555657 |
| xgboost | 0.516709 | 0.304493 | 0.801616 | 0.470588 | 0.483291 |

## Closer Per Tick

- lstm: `97`
- xgboost: `22`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
