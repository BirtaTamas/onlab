# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-pain-vs-housebets-bo3-SOezkQe1hszxnf1QDg0VUC/pain-vs-housebets-m1-dust2.csv`
- round_num: `2`
- rows: `210`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 210 | 1.000 | 0.308945 | 0.419904 | -0.110959 | 14 | 196 | 0.161905 | 0.161905 |
| active/recent utility | 210 | 1.000 | 0.308945 | 0.419904 | -0.110959 | 14 | 196 | 0.161905 | 0.161905 |
| strong utility action | 171 | 0.814 | 0.231344 | 0.345982 | -0.114638 | 10 | 161 | 0.058480 | 0.046784 |
| utility damage | 38 | 0.181 | 0.264794 | 0.364796 | -0.100002 | 0 | 38 | 0.000000 | 0.000000 |
| active smoke/inferno | 159 | 0.757 | 0.211417 | 0.337317 | -0.125900 | 0 | 159 | 0.000000 | 0.000000 |
| recent utility last 5s | 10 | 0.048 | 0.542519 | 0.478816 | 0.063703 | 10 | 0 | 1.000000 | 0.800000 |
| flash effect present | 210 | 1.000 | 0.308945 | 0.419904 | -0.110959 | 14 | 196 | 0.161905 | 0.161905 |

## Active Smoke/Inferno Intervals

- `9.5s` - `43.5s`, rows `69`
- `46.5s` - `91.0s`, rows `90`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.0551`, XGBoost `0.3148`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.0628`, XGBoost `0.3178`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0566`, XGBoost `0.3113`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0571`, XGBoost `0.3110`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.0688`, XGBoost `0.3178`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0596`, XGBoost `0.3043`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1469`, XGBoost `0.3873`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0669`, XGBoost `0.3061`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.0780`, XGBoost `0.3166`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0737`, XGBoost `0.3045`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
