# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-tyloo-bo3-tXRL8tbpb2Kb27VbUgWfK1/lynn-vision-vs-tyloo-m2-ancient.csv`
- round_num: `11`
- rows: `173`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.286793 | 0.090191 | 0.345476 | 1.000000 | 0.713207 |
| xgboost | 0.340194 | 0.129897 | 0.430866 | 1.000000 | 0.659806 |

## Closer Per Tick

- lstm: `128`
- xgboost: `45`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
