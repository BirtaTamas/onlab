# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `11`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.671270 | 0.696640 | -0.025370 | 21 | 209 | 0.886957 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.671270 | 0.696640 | -0.025370 | 21 | 209 | 0.886957 | 1.000000 |
| strong utility action | 178 | 0.774 | 0.642781 | 0.673559 | -0.030778 | 7 | 171 | 0.910112 | 1.000000 |
| utility damage | 21 | 0.091 | 0.514571 | 0.562860 | -0.048289 | 0 | 21 | 1.000000 | 1.000000 |
| active smoke/inferno | 178 | 0.774 | 0.642781 | 0.673559 | -0.030778 | 7 | 171 | 0.910112 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.671270 | 0.696640 | -0.025370 | 21 | 209 | 0.886957 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `51.0s`, rows `89`
- `56.5s` - `100.5s`, rows `89`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.0`, LSTM `0.6725`, XGBoost `0.7489`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5012`, XGBoost `0.5707`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5063`, XGBoost `0.5753`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5080`, XGBoost `0.5731`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5057`, XGBoost `0.5707`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5120`, XGBoost `0.5753`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5011`, XGBoost `0.5640`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `2.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5013`, XGBoost `0.5640`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5105`, XGBoost `0.5731`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5014`, XGBoost `0.5640`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `2.0`, recent_utility `0`
