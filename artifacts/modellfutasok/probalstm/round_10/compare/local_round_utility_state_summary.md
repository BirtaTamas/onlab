# Local Round Utility State Analysis

- csv_path: `processed_full\esl_pro_league_season_22\esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9\vitality-vs-hotu-m2-dust2.csv`
- round_num: `3`
- rows: `126`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 126 | 1.000 | 0.144102 | 0.128175 | 0.015927 | 77 | 49 | 0.857143 | 1.000000 |
| active/recent utility | 126 | 1.000 | 0.144102 | 0.128175 | 0.015927 | 77 | 49 | 0.857143 | 1.000000 |
| strong utility action | 88 | 0.698 | 0.101155 | 0.094162 | 0.006994 | 57 | 31 | 0.943182 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 88 | 0.698 | 0.101155 | 0.094162 | 0.006994 | 57 | 31 | 0.943182 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 126 | 1.000 | 0.144102 | 0.128175 | 0.015927 | 77 | 49 | 0.857143 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `52.5s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.5`, LSTM `0.5222`, XGBoost `0.4126`, closer `xgboost`, smoke `6`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.5130`, XGBoost `0.4126`, closer `xgboost`, smoke `6`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.5033`, XGBoost `0.4126`, closer `xgboost`, smoke `6`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.1425`, XGBoost `0.0565`, closer `xgboost`, smoke `11`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.1411`, XGBoost `0.0565`, closer `xgboost`, smoke `11`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4950`, XGBoost `0.4126`, closer `xgboost`, smoke `4`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.1412`, XGBoost `0.0592`, closer `xgboost`, smoke `11`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1344`, XGBoost `0.0592`, closer `xgboost`, smoke `11`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1328`, XGBoost `0.0592`, closer `xgboost`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1315`, XGBoost `0.0592`, closer `xgboost`, smoke `7`, inferno `3`, utility_damage `0.0`, recent_utility `0`
