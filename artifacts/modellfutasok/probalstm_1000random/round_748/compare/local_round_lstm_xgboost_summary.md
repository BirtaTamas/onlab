# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `7`
- rows: `198`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.319408 | 0.108437 | 0.390962 | 1.000000 | 0.680592 |
| xgboost | 0.321874 | 0.110041 | 0.394775 | 1.000000 | 0.678126 |

## Closer Per Tick

- lstm: `103`
- xgboost: `95`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
