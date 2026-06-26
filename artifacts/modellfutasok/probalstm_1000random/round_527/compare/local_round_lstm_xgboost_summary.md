# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `13`
- rows: `234`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.391934 | 0.195309 | 0.552929 | 0.512821 | 0.391934 |
| xgboost | 0.502900 | 0.298758 | 0.806913 | 0.358974 | 0.502900 |

## Closer Per Tick

- lstm: `230`
- xgboost: `4`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
