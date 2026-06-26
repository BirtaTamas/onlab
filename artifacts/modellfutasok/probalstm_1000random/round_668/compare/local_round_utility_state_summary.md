# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `22`
- rows: `184`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.625278 | 0.636343 | -0.011065 | 84 | 100 | 0.902174 | 0.907609 |
| active/recent utility | 184 | 1.000 | 0.625278 | 0.636343 | -0.011065 | 84 | 100 | 0.902174 | 0.907609 |
| strong utility action | 100 | 0.543 | 0.643988 | 0.643210 | 0.000778 | 50 | 50 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 90 | 0.489 | 0.637794 | 0.649348 | -0.011555 | 40 | 50 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.109 | 0.673031 | 0.651855 | 0.021176 | 10 | 10 | 1.000000 | 1.000000 |
| flash effect present | 184 | 1.000 | 0.625278 | 0.636343 | -0.011065 | 84 | 100 | 0.902174 | 0.907609 |

## Active Smoke/Inferno Intervals

- `8.5s` - `52.5s`, rows `89`
- `91.5s` - `91.5s`, rows `1`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `3.5`, LSTM `0.7131`, XGBoost `0.5822`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.0`, LSTM `0.7119`, XGBoost `0.5813`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.0`, LSTM `0.6993`, XGBoost `0.5829`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `5.5`, LSTM `0.7078`, XGBoost `0.5920`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.5`, LSTM `0.7059`, XGBoost `0.5909`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `5.0`, LSTM `0.7048`, XGBoost `0.5909`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `6.0`, LSTM `0.7068`, XGBoost `0.5959`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.5`, LSTM `0.6927`, XGBoost `0.5831`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `49.5`, LSTM `0.6623`, XGBoost `0.7678`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `50.0`, LSTM `0.6666`, XGBoost `0.7678`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
