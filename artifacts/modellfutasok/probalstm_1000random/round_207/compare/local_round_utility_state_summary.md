# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-b8-bo3--nzkpOWiS4qFgkFOwM8Hun/legacy-vs-b8-m2-ancient.csv`
- round_num: `19`
- rows: `117`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 117 | 1.000 | 0.543944 | 0.520043 | 0.023901 | 50 | 67 | 0.418803 | 0.418803 |
| active/recent utility | 117 | 1.000 | 0.543944 | 0.520043 | 0.023901 | 50 | 67 | 0.418803 | 0.418803 |
| strong utility action | 97 | 0.829 | 0.509560 | 0.494607 | 0.014953 | 47 | 50 | 0.474227 | 0.463918 |
| utility damage | 26 | 0.222 | 0.572481 | 0.646589 | -0.074108 | 23 | 3 | 0.192308 | 0.115385 |
| active smoke/inferno | 86 | 0.735 | 0.519276 | 0.492571 | 0.026705 | 36 | 50 | 0.406977 | 0.523256 |
| recent utility last 5s | 11 | 0.094 | 0.433595 | 0.510524 | -0.076928 | 11 | 0 | 1.000000 | 0.000000 |
| flash effect present | 117 | 1.000 | 0.543944 | 0.520043 | 0.023901 | 50 | 67 | 0.418803 | 0.418803 |

## Active Smoke/Inferno Intervals

- `7.0s` - `49.5s`, rows `86`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.0`, LSTM `0.5271`, XGBoost `0.2793`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.5056`, XGBoost `0.2761`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.4989`, XGBoost `0.2793`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.4945`, XGBoost `0.2761`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6523`, XGBoost `0.4498`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `26.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.6456`, XGBoost `0.4498`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `26.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6303`, XGBoost `0.4463`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6258`, XGBoost `0.4463`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `26.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.5345`, XGBoost `0.7133`, closer `lstm`, smoke `4`, inferno `5`, utility_damage `11.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.6282`, XGBoost `0.4498`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `26.0`, recent_utility `0`
