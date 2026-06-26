# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `4`
- rows: `165`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.107196 | 0.132524 | -0.025328 | 153 | 12 | 1.000000 | 1.000000 |
| active/recent utility | 165 | 1.000 | 0.107196 | 0.132524 | -0.025328 | 153 | 12 | 1.000000 | 1.000000 |
| strong utility action | 114 | 0.691 | 0.150008 | 0.180565 | -0.030557 | 103 | 11 | 1.000000 | 1.000000 |
| utility damage | 26 | 0.158 | 0.177162 | 0.209633 | -0.032471 | 26 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 94 | 0.570 | 0.157063 | 0.182336 | -0.025273 | 83 | 11 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.121 | 0.274111 | 0.323022 | -0.048911 | 16 | 4 | 1.000000 | 1.000000 |
| flash effect present | 165 | 1.000 | 0.107196 | 0.132524 | -0.025328 | 153 | 12 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `52.5s`, rows `94`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `4.0`, LSTM `0.1559`, XGBoost `0.3422`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.1527`, XGBoost `0.3312`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.1556`, XGBoost `0.3312`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.1608`, XGBoost `0.3309`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.1671`, XGBoost `0.3306`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.1723`, XGBoost `0.3309`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.5`, LSTM `0.1802`, XGBoost `0.3313`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.1947`, XGBoost `0.3422`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.0`, LSTM `0.1951`, XGBoost `0.3313`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.2080`, XGBoost `0.3345`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
