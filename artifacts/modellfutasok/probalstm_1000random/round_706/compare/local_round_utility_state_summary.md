# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-3dmax-bo3-Oe166BQltZjvHlE8qlepgF/furia-vs-3dmax-m1-nuke.csv`
- round_num: `9`
- rows: `212`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 212 | 1.000 | 0.878909 | 0.898876 | -0.019967 | 36 | 176 | 1.000000 | 1.000000 |
| active/recent utility | 212 | 1.000 | 0.878909 | 0.898876 | -0.019967 | 36 | 176 | 1.000000 | 1.000000 |
| strong utility action | 133 | 0.627 | 0.857261 | 0.882149 | -0.024888 | 23 | 110 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.047 | 0.597225 | 0.612679 | -0.015454 | 4 | 6 | 1.000000 | 1.000000 |
| active smoke/inferno | 132 | 0.623 | 0.859052 | 0.884072 | -0.025019 | 23 | 109 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.047 | 0.589473 | 0.610612 | -0.021139 | 3 | 7 | 1.000000 | 1.000000 |
| flash effect present | 212 | 1.000 | 0.878909 | 0.898876 | -0.019967 | 36 | 176 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `37.0s`, rows `60`
- `70.0s` - `105.5s`, rows `72`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `105.5`, LSTM `0.8627`, XGBoost `0.9379`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `105.0`, LSTM `0.8690`, XGBoost `0.9379`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.8696`, XGBoost `0.9383`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.9288`, XGBoost `0.9962`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.9297`, XGBoost `0.9963`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `104.5`, LSTM `0.8738`, XGBoost `0.9391`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.9316`, XGBoost `0.9961`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.8755`, XGBoost `0.9391`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.9332`, XGBoost `0.9963`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.9343`, XGBoost `0.9963`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
