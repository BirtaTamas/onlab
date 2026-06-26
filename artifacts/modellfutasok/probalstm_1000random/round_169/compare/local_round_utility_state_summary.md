# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-eternal-fire-vs-falcons-bo3-Bm3FkXiO5h_cvpKxUnOmaW/eternal-fire-vs-falcons-m1-inferno.csv`
- round_num: `8`
- rows: `251`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 251 | 1.000 | 0.064890 | 0.155413 | -0.090523 | 251 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 251 | 1.000 | 0.064890 | 0.155413 | -0.090523 | 251 | 0 | 1.000000 | 1.000000 |
| strong utility action | 200 | 0.797 | 0.060903 | 0.155114 | -0.094211 | 200 | 0 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.040 | 0.068996 | 0.215403 | -0.146408 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 200 | 0.797 | 0.060903 | 0.155114 | -0.094211 | 200 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 251 | 1.000 | 0.064890 | 0.155413 | -0.090523 | 251 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `32.5s`, rows `46`
- `34.0s` - `78.5s`, rows `90`
- `79.5s` - `111.0s`, rows `64`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.0`, LSTM `0.2165`, XGBoost `0.3896`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `43.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.0431`, XGBoost `0.2100`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.0466`, XGBoost `0.2064`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.0545`, XGBoost `0.2140`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0475`, XGBoost `0.2064`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.0539`, XGBoost `0.2124`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1881`, XGBoost `0.3465`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.0567`, XGBoost `0.2142`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0493`, XGBoost `0.2064`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.0575`, XGBoost `0.2142`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
