# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `7`
- rows: `150`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 150 | 1.000 | 0.194587 | 0.312889 | -0.118302 | 150 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 150 | 1.000 | 0.194587 | 0.312889 | -0.118302 | 150 | 0 | 1.000000 | 1.000000 |
| strong utility action | 137 | 0.913 | 0.196316 | 0.306237 | -0.109921 | 137 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.067 | 0.281270 | 0.373247 | -0.091977 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 137 | 0.913 | 0.196316 | 0.306237 | -0.109921 | 137 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 150 | 1.000 | 0.194587 | 0.312889 | -0.118302 | 150 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `74.5s`, rows `137`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.0691`, XGBoost `0.3725`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.0756`, XGBoost `0.3725`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `21.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.0859`, XGBoost `0.3790`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `20.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0939`, XGBoost `0.3790`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0996`, XGBoost `0.3790`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `20.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1213`, XGBoost `0.3835`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `15.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.2290`, XGBoost `0.4835`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `35.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1537`, XGBoost `0.3835`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `15.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.1897`, XGBoost `0.3948`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1804`, XGBoost `0.3835`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `15.0`, recent_utility `0`
