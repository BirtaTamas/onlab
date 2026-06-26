# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-eternal-fire-vs-flyquest-bo3-bOv4otMGdpLsO1VdhzI_AV/eternal-fire-vs-flyquest-m1-inferno.csv`
- round_num: `6`
- rows: `174`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 174 | 1.000 | 0.759213 | 0.762294 | -0.003081 | 54 | 120 | 1.000000 | 1.000000 |
| active/recent utility | 174 | 1.000 | 0.759213 | 0.762294 | -0.003081 | 54 | 120 | 1.000000 | 1.000000 |
| strong utility action | 153 | 0.879 | 0.747434 | 0.753158 | -0.005724 | 43 | 110 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.115 | 0.813608 | 0.790448 | 0.023160 | 15 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 142 | 0.816 | 0.745824 | 0.755790 | -0.009967 | 33 | 109 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.115 | 0.758505 | 0.720888 | 0.037617 | 19 | 1 | 1.000000 | 1.000000 |
| flash effect present | 174 | 1.000 | 0.759213 | 0.762294 | -0.003081 | 54 | 120 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.5s` - `82.0s`, rows `142`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `5.5`, LSTM `0.7980`, XGBoost `0.7159`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `72.0`, LSTM `0.8244`, XGBoost `0.7460`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `106.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.7942`, XGBoost `0.7159`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `72.5`, LSTM `0.8225`, XGBoost `0.7484`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `106.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.8173`, XGBoost `0.7489`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `106.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.8136`, XGBoost `0.7453`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `106.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.8674`, XGBoost `0.8032`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `56.0`, recent_utility `0`
- seconds `1.0`, LSTM `0.7795`, XGBoost `0.7181`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `1.5`, LSTM `0.7810`, XGBoost `0.7202`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.5`, LSTM `0.7733`, XGBoost `0.7159`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
