# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-legacy-bo3-o9hWAn-ugamRsSw8ngEfHF/astralis-vs-legacy-m3-ancient.csv`
- round_num: `11`
- rows: `275`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 275 | 1.000 | 0.094163 | 0.146975 | -0.052813 | 275 | 0 | 0.996364 | 0.905455 |
| active/recent utility | 275 | 1.000 | 0.094163 | 0.146975 | -0.052813 | 275 | 0 | 0.996364 | 0.905455 |
| strong utility action | 203 | 0.738 | 0.116289 | 0.172733 | -0.056444 | 203 | 0 | 0.995074 | 0.881773 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 194 | 0.705 | 0.099838 | 0.157040 | -0.057202 | 194 | 0 | 1.000000 | 0.922680 |
| recent utility last 5s | 10 | 0.036 | 0.469411 | 0.510613 | -0.041202 | 10 | 0 | 0.900000 | 0.000000 |
| flash effect present | 275 | 1.000 | 0.094163 | 0.146975 | -0.052813 | 275 | 0 | 0.996364 | 0.905455 |

## Active Smoke/Inferno Intervals

- `5.5s` - `53.5s`, rows `97`
- `63.0s` - `111.0s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `53.5`, LSTM `0.0597`, XGBoost `0.1904`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.0499`, XGBoost `0.1793`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.0545`, XGBoost `0.1835`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.0453`, XGBoost `0.1735`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.0581`, XGBoost `0.1851`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0546`, XGBoost `0.1813`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.0592`, XGBoost `0.1855`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0530`, XGBoost `0.1785`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.0589`, XGBoost `0.1841`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.0619`, XGBoost `0.1863`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
