# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m2-dust2.csv`
- round_num: `6`
- rows: `146`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 146 | 1.000 | 0.075938 | 0.149469 | -0.073531 | 146 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 146 | 1.000 | 0.075938 | 0.149469 | -0.073531 | 146 | 0 | 1.000000 | 1.000000 |
| strong utility action | 100 | 0.685 | 0.096475 | 0.176427 | -0.079952 | 100 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 95 | 0.651 | 0.096541 | 0.175680 | -0.079139 | 95 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.137 | 0.084795 | 0.186235 | -0.101441 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 146 | 1.000 | 0.075938 | 0.149469 | -0.073531 | 146 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `36.0s`, rows `67`
- `49.0s` - `55.5s`, rows `14`
- `65.5s` - `72.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `7.5`, LSTM `0.0411`, XGBoost `0.1838`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.0398`, XGBoost `0.1825`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0426`, XGBoost `0.1842`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.0518`, XGBoost `0.1888`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0489`, XGBoost `0.1842`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.0646`, XGBoost `0.1888`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.0582`, XGBoost `0.1819`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.0789`, XGBoost `0.1903`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0941`, XGBoost `0.2049`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0722`, XGBoost `0.1819`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `1`
