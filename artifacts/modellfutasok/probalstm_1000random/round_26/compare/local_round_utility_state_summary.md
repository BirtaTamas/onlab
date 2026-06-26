# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `6`
- rows: `244`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 244 | 1.000 | 0.313388 | 0.449932 | -0.136543 | 0 | 244 | 0.098361 | 0.188525 |
| active/recent utility | 244 | 1.000 | 0.313388 | 0.449932 | -0.136543 | 0 | 244 | 0.098361 | 0.188525 |
| strong utility action | 183 | 0.750 | 0.244822 | 0.389964 | -0.145142 | 0 | 183 | 0.000000 | 0.049180 |
| utility damage | 10 | 0.041 | 0.288403 | 0.438313 | -0.149909 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 183 | 0.750 | 0.244822 | 0.389964 | -0.145142 | 0 | 183 | 0.000000 | 0.049180 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 244 | 1.000 | 0.313388 | 0.449932 | -0.136543 | 0 | 244 | 0.098361 | 0.188525 |

## Active Smoke/Inferno Intervals

- `5.5s` - `36.0s`, rows `62`
- `40.0s` - `93.0s`, rows `107`
- `100.0s` - `106.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `106.5`, LSTM `0.3393`, XGBoost `0.7385`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.1735`, XGBoost `0.5620`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.0395`, XGBoost `0.3348`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.0600`, XGBoost `0.3443`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `106.0`, LSTM `0.4612`, XGBoost `0.7385`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.1153`, XGBoost `0.3883`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.0746`, XGBoost `0.3440`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.1658`, XGBoost `0.4245`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.2584`, XGBoost `0.5142`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1798`, XGBoost `0.4327`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
