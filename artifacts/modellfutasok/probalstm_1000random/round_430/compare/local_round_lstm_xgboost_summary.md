# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `1`
- rows: `150`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.522570 | 0.294495 | 0.792684 | 0.373333 | 0.477430 |
| xgboost | 0.432166 | 0.205096 | 0.599515 | 0.846667 | 0.567834 |

## Closer Per Tick

- lstm: `9`
- xgboost: `141`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
