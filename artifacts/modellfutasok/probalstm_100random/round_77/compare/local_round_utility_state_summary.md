# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-gentle-mates-vs-aurora-bo3-gDH2lDrlT5ROvKI-0e6nmI/gentle-mates-vs-aurora-m1-nuke.csv`
- round_num: `15`
- rows: `203`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 203 | 1.000 | 0.217220 | 0.231566 | -0.014346 | 132 | 71 | 0.852217 | 1.000000 |
| active/recent utility | 203 | 1.000 | 0.217220 | 0.231566 | -0.014346 | 132 | 71 | 0.852217 | 1.000000 |
| strong utility action | 131 | 0.645 | 0.278308 | 0.275888 | 0.002419 | 65 | 66 | 0.786260 | 1.000000 |
| utility damage | 10 | 0.049 | 0.490777 | 0.483765 | 0.007012 | 5 | 5 | 0.600000 | 1.000000 |
| active smoke/inferno | 131 | 0.645 | 0.278308 | 0.275888 | 0.002419 | 65 | 66 | 0.786260 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 203 | 1.000 | 0.217220 | 0.231566 | -0.014346 | 132 | 71 | 0.852217 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `34.5s`, rows `51`
- `36.5s` - `65.0s`, rows `58`
- `68.5s` - `79.0s`, rows `22`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `10.0`, LSTM `0.2918`, XGBoost `0.4701`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.3007`, XGBoost `0.4699`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.3225`, XGBoost `0.4719`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.3541`, XGBoost `0.4711`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.3577`, XGBoost `0.4727`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3201`, XGBoost `0.2233`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.3797`, XGBoost `0.4718`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.3811`, XGBoost `0.4714`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.3111`, XGBoost `0.2233`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.3899`, XGBoost `0.4717`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
