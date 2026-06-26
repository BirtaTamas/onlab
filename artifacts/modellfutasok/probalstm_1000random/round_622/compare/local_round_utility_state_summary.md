# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `9`
- rows: `202`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 202 | 1.000 | 0.631690 | 0.611021 | 0.020669 | 155 | 47 | 1.000000 | 0.985149 |
| active/recent utility | 202 | 1.000 | 0.631690 | 0.611021 | 0.020669 | 155 | 47 | 1.000000 | 0.985149 |
| strong utility action | 167 | 0.827 | 0.617777 | 0.595673 | 0.022104 | 128 | 39 | 1.000000 | 0.982036 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 167 | 0.827 | 0.617777 | 0.595673 | 0.022104 | 128 | 39 | 1.000000 | 0.982036 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 202 | 1.000 | 0.631690 | 0.611021 | 0.020669 | 155 | 47 | 1.000000 | 0.985149 |

## Active Smoke/Inferno Intervals

- `8.5s` - `87.0s`, rows `158`
- `96.5s` - `100.5s`, rows `9`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.0`, LSTM `0.6608`, XGBoost `0.4954`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.7451`, XGBoost `0.8749`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.7471`, XGBoost `0.8749`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6068`, XGBoost `0.4935`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.8411`, XGBoost `0.7340`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5954`, XGBoost `0.4935`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5910`, XGBoost `0.5038`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.7525`, XGBoost `0.6659`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.7905`, XGBoost `0.8754`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `52.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.7484`, XGBoost `0.6659`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
