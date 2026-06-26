# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-flyquest-vs-lynn-vision-bo3-tBzyC_GrP1HzVZ3u3bXk3k/flyquest-vs-lynn-vision-m2-anubis.csv`
- round_num: `8`
- rows: `226`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 226 | 1.000 | 0.525151 | 0.513276 | 0.011874 | 155 | 71 | 0.747788 | 0.707965 |
| active/recent utility | 226 | 1.000 | 0.525151 | 0.513276 | 0.011874 | 155 | 71 | 0.747788 | 0.707965 |
| strong utility action | 204 | 0.903 | 0.500951 | 0.488796 | 0.012155 | 144 | 60 | 0.725490 | 0.676471 |
| utility damage | 27 | 0.119 | 0.399118 | 0.255395 | 0.143723 | 27 | 0 | 0.259259 | 0.111111 |
| active smoke/inferno | 194 | 0.858 | 0.500114 | 0.486013 | 0.014101 | 139 | 55 | 0.731959 | 0.659794 |
| recent utility last 5s | 10 | 0.044 | 0.517194 | 0.542781 | -0.025587 | 5 | 5 | 0.600000 | 1.000000 |
| flash effect present | 226 | 1.000 | 0.525151 | 0.513276 | 0.011874 | 155 | 71 | 0.747788 | 0.707965 |

## Active Smoke/Inferno Intervals

- `9.0s` - `83.5s`, rows `150`
- `86.0s` - `107.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `78.0`, LSTM `0.3787`, XGBoost `0.1097`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `62.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.3520`, XGBoost `0.1097`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `62.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.4205`, XGBoost `0.6522`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.5675`, XGBoost `0.3391`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `32.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.3338`, XGBoost `0.1097`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `62.0`, recent_utility `0`
- seconds `104.0`, LSTM `0.3909`, XGBoost `0.6120`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.4357`, XGBoost `0.6556`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.3953`, XGBoost `0.6120`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.5410`, XGBoost `0.3379`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `32.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.5380`, XGBoost `0.3391`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `32.0`, recent_utility `0`
