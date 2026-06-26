# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `16`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.841164 | 0.847781 | -0.006617 | 56 | 114 | 1.000000 | 1.000000 |
| active/recent utility | 170 | 1.000 | 0.841164 | 0.847781 | -0.006617 | 56 | 114 | 1.000000 | 1.000000 |
| strong utility action | 138 | 0.812 | 0.843963 | 0.855720 | -0.011756 | 34 | 104 | 1.000000 | 1.000000 |
| utility damage | 22 | 0.129 | 0.888091 | 0.922160 | -0.034069 | 0 | 22 | 1.000000 | 1.000000 |
| active smoke/inferno | 138 | 0.812 | 0.843963 | 0.855720 | -0.011756 | 34 | 104 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 170 | 1.000 | 0.841164 | 0.847781 | -0.006617 | 56 | 114 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `67.5s`, rows `124`
- `75.5s` - `82.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.5`, LSTM `0.6837`, XGBoost `0.5541`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6805`, XGBoost `0.5591`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.6778`, XGBoost `0.5600`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.6689`, XGBoost `0.5628`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6700`, XGBoost `0.5672`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.6563`, XGBoost `0.5630`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.8223`, XGBoost `0.9146`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.8226`, XGBoost `0.9138`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6580`, XGBoost `0.5672`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.6474`, XGBoost `0.5632`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
