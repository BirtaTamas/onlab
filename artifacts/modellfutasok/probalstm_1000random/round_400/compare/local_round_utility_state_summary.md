# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-gamerlegion-vs-inner-circle-bo3-TOF4f6Uhtdi7Vqylk0QEY6/gamerlegion-vs-inner-circle-m1-ancient.csv`
- round_num: `11`
- rows: `163`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 163 | 1.000 | 0.304112 | 0.408212 | -0.104101 | 13 | 150 | 0.116564 | 0.196319 |
| active/recent utility | 163 | 1.000 | 0.304112 | 0.408212 | -0.104101 | 13 | 150 | 0.116564 | 0.196319 |
| strong utility action | 141 | 0.865 | 0.311862 | 0.403635 | -0.091773 | 13 | 128 | 0.134752 | 0.170213 |
| utility damage | 21 | 0.129 | 0.481873 | 0.615579 | -0.133705 | 0 | 21 | 0.285714 | 0.714286 |
| active smoke/inferno | 135 | 0.828 | 0.302661 | 0.396006 | -0.093345 | 13 | 122 | 0.133333 | 0.133333 |
| recent utility last 5s | 10 | 0.061 | 0.467754 | 0.516109 | -0.048355 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 163 | 1.000 | 0.304112 | 0.408212 | -0.104101 | 13 | 150 | 0.116564 | 0.196319 |

## Active Smoke/Inferno Intervals

- `6.5s` - `68.0s`, rows `124`
- `75.5s` - `80.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.1941`, XGBoost `0.5381`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.0861`, XGBoost `0.3965`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.2344`, XGBoost `0.5381`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.0951`, XGBoost `0.3965`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2881`, XGBoost `0.5820`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.1108`, XGBoost `0.4021`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.2523`, XGBoost `0.5381`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `11.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.1140`, XGBoost `0.3965`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.1240`, XGBoost `0.4021`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.1296`, XGBoost `0.4021`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
