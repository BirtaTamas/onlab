# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-aurora-bo3-0icw3xvkvOZhHsCT2PEavZ/furia-vs-aurora-m1-inferno.csv`
- round_num: `8`
- rows: `253`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 253 | 1.000 | 0.564240 | 0.464310 | 0.099929 | 236 | 17 | 0.667984 | 0.557312 |
| active/recent utility | 253 | 1.000 | 0.564240 | 0.464310 | 0.099929 | 236 | 17 | 0.667984 | 0.557312 |
| strong utility action | 159 | 0.628 | 0.554253 | 0.481234 | 0.073019 | 143 | 16 | 0.698113 | 0.698113 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 152 | 0.601 | 0.548591 | 0.477372 | 0.071219 | 136 | 16 | 0.684211 | 0.684211 |
| recent utility last 5s | 20 | 0.079 | 0.623015 | 0.544146 | 0.078869 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 253 | 1.000 | 0.564240 | 0.464310 | 0.099929 | 236 | 17 | 0.667984 | 0.557312 |

## Active Smoke/Inferno Intervals

- `10.5s` - `63.5s`, rows `107`
- `78.5s` - `100.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `62.5`, LSTM `0.4971`, XGBoost `0.3045`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.4770`, XGBoost `0.3045`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.4964`, XGBoost `0.3268`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.4669`, XGBoost `0.3045`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.4841`, XGBoost `0.3261`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.4838`, XGBoost `0.3261`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.4698`, XGBoost `0.3131`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.4816`, XGBoost `0.3256`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.4767`, XGBoost `0.3229`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `85.0`, LSTM `0.4723`, XGBoost `0.3187`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
