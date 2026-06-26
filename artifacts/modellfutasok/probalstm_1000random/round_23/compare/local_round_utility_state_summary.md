# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-pain-bo3-zcuZjSa9VUSMkJoK5k8I3c/gamerlegion-vs-pain-m3-mirage.csv`
- round_num: `5`
- rows: `208`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 208 | 1.000 | 0.127470 | 0.150383 | -0.022914 | 184 | 24 | 0.975962 | 0.817308 |
| active/recent utility | 208 | 1.000 | 0.127470 | 0.150383 | -0.022914 | 184 | 24 | 0.975962 | 0.817308 |
| strong utility action | 109 | 0.524 | 0.188295 | 0.214767 | -0.026473 | 85 | 24 | 0.954128 | 0.761468 |
| utility damage | 10 | 0.048 | 0.256370 | 0.239943 | 0.016427 | 4 | 6 | 1.000000 | 1.000000 |
| active smoke/inferno | 109 | 0.524 | 0.188295 | 0.214767 | -0.026473 | 85 | 24 | 0.954128 | 0.761468 |
| recent utility last 5s | 10 | 0.048 | 0.117988 | 0.203502 | -0.085514 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 208 | 1.000 | 0.127470 | 0.150383 | -0.022914 | 184 | 24 | 0.975962 | 0.817308 |

## Active Smoke/Inferno Intervals

- `6.0s` - `60.0s`, rows `109`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.0`, LSTM `0.4483`, XGBoost `0.5647`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.1176`, XGBoost `0.2280`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `44.5`, LSTM `0.1179`, XGBoost `0.2280`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `20.5`, LSTM `0.1948`, XGBoost `0.3043`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `26.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.1081`, XGBoost `0.2154`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `10.5`, LSTM `0.4473`, XGBoost `0.5516`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.1174`, XGBoost `0.2180`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `42.5`, LSTM `0.1211`, XGBoost `0.2183`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `1`
- seconds `41.5`, LSTM `0.1235`, XGBoost `0.2174`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `19.5`, LSTM `0.1885`, XGBoost `0.2794`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `26.0`, recent_utility `0`
