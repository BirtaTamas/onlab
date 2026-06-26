# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `9`
- rows: `100`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.434950 | 0.221094 | 0.665971 | 0.770000 | 0.565050 |
| xgboost | 0.497682 | 0.287851 | 0.786141 | 0.730000 | 0.502318 |

## Closer Per Tick

- lstm: `76`
- xgboost: `24`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
