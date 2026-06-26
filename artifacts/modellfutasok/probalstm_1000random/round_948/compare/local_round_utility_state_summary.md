# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-faze-bo3-1414ljxN3FRmXv6-03KYFL/spirit-vs-faze-m2-mirage.csv`
- round_num: `14`
- rows: `200`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.319496 | 0.417987 | -0.098491 | 9 | 191 | 0.150000 | 0.260000 |
| active/recent utility | 200 | 1.000 | 0.319496 | 0.417987 | -0.098491 | 9 | 191 | 0.150000 | 0.260000 |
| strong utility action | 151 | 0.755 | 0.338811 | 0.431699 | -0.092889 | 9 | 142 | 0.178808 | 0.298013 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 151 | 0.755 | 0.338811 | 0.431699 | -0.092889 | 9 | 142 | 0.178808 | 0.298013 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 200 | 1.000 | 0.319496 | 0.417987 | -0.098491 | 9 | 191 | 0.150000 | 0.260000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `44.0s`, rows `76`
- `59.0s` - `96.0s`, rows `75`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `94.5`, LSTM `0.6747`, XGBoost `0.9185`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.6914`, XGBoost `0.9185`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.0804`, XGBoost `0.3054`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.0834`, XGBoost `0.3054`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.0862`, XGBoost `0.3062`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0860`, XGBoost `0.3054`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.0898`, XGBoost `0.3062`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.7078`, XGBoost `0.9196`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.0976`, XGBoost `0.3054`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.0982`, XGBoost `0.3054`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
