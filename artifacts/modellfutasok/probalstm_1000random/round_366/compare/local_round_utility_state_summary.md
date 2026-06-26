# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `13`
- rows: `207`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 207 | 1.000 | 0.378815 | 0.437846 | -0.059030 | 189 | 18 | 0.743961 | 0.502415 |
| active/recent utility | 207 | 1.000 | 0.378815 | 0.437846 | -0.059030 | 189 | 18 | 0.743961 | 0.502415 |
| strong utility action | 71 | 0.343 | 0.445427 | 0.488926 | -0.043499 | 57 | 14 | 0.704225 | 0.690141 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 71 | 0.343 | 0.445427 | 0.488926 | -0.043499 | 57 | 14 | 0.704225 | 0.690141 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 207 | 1.000 | 0.378815 | 0.437846 | -0.059030 | 189 | 18 | 0.743961 | 0.502415 |

## Active Smoke/Inferno Intervals

- `14.5s` - `21.0s`, rows `14`
- `23.0s` - `51.0s`, rows `57`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.5`, LSTM `0.5464`, XGBoost `0.7233`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.2532`, XGBoost `0.4156`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.5620`, XGBoost `0.7233`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1963`, XGBoost `0.3565`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2631`, XGBoost `0.4177`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5157`, XGBoost `0.6428`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2336`, XGBoost `0.3592`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.3851`, XGBoost `0.5087`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5157`, XGBoost `0.6316`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.5157`, XGBoost `0.4006`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
