# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `17`
- rows: `267`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 267 | 1.000 | 0.595779 | 0.638798 | -0.043019 | 36 | 231 | 0.483146 | 0.868914 |
| active/recent utility | 267 | 1.000 | 0.595779 | 0.638798 | -0.043019 | 36 | 231 | 0.483146 | 0.868914 |
| strong utility action | 215 | 0.805 | 0.630278 | 0.670466 | -0.040188 | 36 | 179 | 0.590698 | 0.934884 |
| utility damage | 20 | 0.075 | 0.704841 | 0.706512 | -0.001671 | 10 | 10 | 0.550000 | 1.000000 |
| active smoke/inferno | 212 | 0.794 | 0.625358 | 0.665939 | -0.040582 | 36 | 176 | 0.584906 | 0.933962 |
| recent utility last 5s | 6 | 0.022 | 0.976354 | 0.990404 | -0.014051 | 0 | 6 | 1.000000 | 1.000000 |
| flash effect present | 267 | 1.000 | 0.595779 | 0.638798 | -0.043019 | 36 | 231 | 0.483146 | 0.868914 |

## Active Smoke/Inferno Intervals

- `9.0s` - `52.5s`, rows `88`
- `57.5s` - `66.5s`, rows `19`
- `79.5s` - `131.5s`, rows `105`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `118.5`, LSTM `0.6172`, XGBoost `0.8719`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `120.0`, LSTM `0.6405`, XGBoost `0.8621`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `119.5`, LSTM `0.6689`, XGBoost `0.8663`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `118.0`, LSTM `0.6819`, XGBoost `0.8719`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `119.0`, LSTM `0.6918`, XGBoost `0.8663`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.3889`, XGBoost `0.5192`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.3957`, XGBoost `0.5220`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.3970`, XGBoost `0.5230`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.6225`, XGBoost `0.7485`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.3961`, XGBoost `0.5211`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
