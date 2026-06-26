# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-heroic-vs-3dmax-bo3-OVT4ch_FfOW2E26liKqT_k/heroic-vs-3dmax-m2-inferno.csv`
- round_num: `3`
- rows: `269`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 269 | 1.000 | 0.850047 | 0.886079 | -0.036032 | 39 | 230 | 0.959108 | 0.921933 |
| active/recent utility | 269 | 1.000 | 0.850047 | 0.886079 | -0.036032 | 39 | 230 | 0.959108 | 0.921933 |
| strong utility action | 224 | 0.833 | 0.860992 | 0.895317 | -0.034325 | 22 | 202 | 0.955357 | 0.937500 |
| utility damage | 32 | 0.119 | 0.712352 | 0.766490 | -0.054138 | 5 | 27 | 1.000000 | 1.000000 |
| active smoke/inferno | 209 | 0.777 | 0.884564 | 0.918779 | -0.034215 | 19 | 190 | 1.000000 | 0.985646 |
| recent utility last 5s | 11 | 0.041 | 0.464583 | 0.469392 | -0.004809 | 3 | 8 | 0.090909 | 0.000000 |
| flash effect present | 269 | 1.000 | 0.850047 | 0.886079 | -0.036032 | 39 | 230 | 0.959108 | 0.921933 |

## Active Smoke/Inferno Intervals

- `9.0s` - `49.0s`, rows `81`
- `55.5s` - `97.0s`, rows `84`
- `112.5s` - `134.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `24.0`, LSTM `0.6001`, XGBoost `0.7618`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6006`, XGBoost `0.7605`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6053`, XGBoost `0.7609`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5934`, XGBoost `0.7459`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5945`, XGBoost `0.7456`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6003`, XGBoost `0.7474`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.8121`, XGBoost `0.9466`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `111.5`, LSTM `0.7161`, XGBoost `0.8495`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `46.0`, recent_utility `0`
- seconds `110.5`, LSTM `0.6995`, XGBoost `0.8299`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `39.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.7329`, XGBoost `0.8508`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `46.0`, recent_utility `0`
