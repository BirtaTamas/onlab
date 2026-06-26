# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mibr-bo3-vjmAHfXA4PQfROTmirSCCF/vitality-vs-mibr-m2-inferno.csv`
- round_num: `10`
- rows: `263`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 263 | 1.000 | 0.150350 | 0.194662 | -0.044312 | 226 | 37 | 0.942966 | 1.000000 |
| active/recent utility | 263 | 1.000 | 0.150350 | 0.194662 | -0.044312 | 226 | 37 | 0.942966 | 1.000000 |
| strong utility action | 223 | 0.848 | 0.143843 | 0.177389 | -0.033546 | 191 | 32 | 0.932735 | 1.000000 |
| utility damage | 39 | 0.148 | 0.396912 | 0.373191 | 0.023721 | 13 | 26 | 0.615385 | 1.000000 |
| active smoke/inferno | 213 | 0.810 | 0.141225 | 0.172267 | -0.031042 | 181 | 32 | 0.929577 | 1.000000 |
| recent utility last 5s | 10 | 0.038 | 0.199609 | 0.286484 | -0.086876 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 263 | 1.000 | 0.150350 | 0.194662 | -0.044312 | 226 | 37 | 0.942966 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `37.0s`, rows `61`
- `42.0s` - `48.5s`, rows `14`
- `62.5s` - `131.0s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.5820`, XGBoost `0.4145`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `104.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1185`, XGBoost `0.2837`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.1250`, XGBoost `0.2837`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.1253`, XGBoost `0.2837`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.1268`, XGBoost `0.2832`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1279`, XGBoost `0.2837`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1284`, XGBoost `0.2837`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.1339`, XGBoost `0.2837`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.1353`, XGBoost `0.2832`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1486`, XGBoost `0.2869`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
