# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-astralis-bo3-AOc9ksnKaf2n3lWssI4XgX/falcons-vs-astralis-m2-mirage.csv`
- round_num: `10`
- rows: `205`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.531545 | 0.592788 | -0.061243 | 23 | 182 | 0.673171 | 0.692683 |
| active/recent utility | 205 | 1.000 | 0.531545 | 0.592788 | -0.061243 | 23 | 182 | 0.673171 | 0.692683 |
| strong utility action | 136 | 0.663 | 0.519660 | 0.570050 | -0.050390 | 23 | 113 | 0.654412 | 0.654412 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 136 | 0.663 | 0.519660 | 0.570050 | -0.050390 | 23 | 113 | 0.654412 | 0.654412 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 205 | 1.000 | 0.531545 | 0.592788 | -0.061243 | 23 | 182 | 0.673171 | 0.692683 |

## Active Smoke/Inferno Intervals

- `6.5s` - `74.0s`, rows `136`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.5`, LSTM `0.2193`, XGBoost `0.3842`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.2203`, XGBoost `0.3842`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.2273`, XGBoost `0.3876`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.2310`, XGBoost `0.3873`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.3229`, XGBoost `0.4654`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.3483`, XGBoost `0.4844`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3318`, XGBoost `0.4654`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.2510`, XGBoost `0.3842`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5141`, XGBoost `0.3847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.2621`, XGBoost `0.3876`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
