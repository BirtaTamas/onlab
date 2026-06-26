# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `10`
- rows: `255`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.441554 | 0.229239 | 0.640599 | 0.650980 | 0.558446 |
| xgboost | 0.531402 | 0.335501 | 0.899167 | 0.639216 | 0.468598 |

## Closer Per Tick

- lstm: `232`
- xgboost: `23`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
