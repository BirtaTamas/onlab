# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `10`
- rows: `154`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.484574 | 0.259466 | 0.697505 | 0.305195 | 0.484574 |
| xgboost | 0.643874 | 0.450565 | 1.129237 | 0.155844 | 0.643874 |

## Closer Per Tick

- lstm: `154`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
