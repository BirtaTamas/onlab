# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `18`
- rows: `146`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.244890 | 0.316440 | -0.071551 | 10 | 136 | 0.157534 | 0.164384 |
| active/recent utility | 146 | 1.000 | 0.244890 | 0.316440 | -0.071551 | 10 | 136 | 0.157534 | 0.164384 |
| strong utility action | 114 | 0.781 | 0.205161 | 0.245833 | -0.040672 | 10 | 104 | 0.070175 | 0.070175 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 114 | 0.781 | 0.205161 | 0.245833 | -0.040672 | 10 | 104 | 0.070175 | 0.070175 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 146 | 1.000 | 0.244890 | 0.316440 | -0.071551 | 10 | 136 | 0.157534 | 0.164384 |

## Active Smoke/Inferno Intervals

- `2.5s` - `37.0s`, rows `70`
- `40.0s` - `61.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `59.5`, LSTM `0.5206`, XGBoost `0.6692`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.5237`, XGBoost `0.6696`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0732`, XGBoost `0.2130`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5439`, XGBoost `0.6801`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.2153`, XGBoost `0.3468`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5372`, XGBoost `0.6675`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5541`, XGBoost `0.6814`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.2906`, XGBoost `0.3858`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6943`, XGBoost `0.6084`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0534`, XGBoost `0.1374`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
