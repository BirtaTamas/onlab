# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `15`
- rows: `170`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 170 | 1.000 | 0.732364 | 0.645295 | 0.087069 | 160 | 10 | 1.000000 | 1.000000 |
| active/recent utility | 170 | 1.000 | 0.732364 | 0.645295 | 0.087069 | 160 | 10 | 1.000000 | 1.000000 |
| strong utility action | 153 | 0.900 | 0.727999 | 0.635496 | 0.092503 | 143 | 10 | 1.000000 | 1.000000 |
| utility damage | 34 | 0.200 | 0.720493 | 0.600235 | 0.120258 | 34 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 138 | 0.812 | 0.735326 | 0.639190 | 0.096136 | 128 | 10 | 1.000000 | 1.000000 |
| recent utility last 5s | 15 | 0.088 | 0.660591 | 0.601513 | 0.059078 | 15 | 0 | 1.000000 | 1.000000 |
| flash effect present | 170 | 1.000 | 0.732364 | 0.645295 | 0.087069 | 160 | 10 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `78.0s`, rows `138`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.7201`, XGBoost `0.5207`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.7845`, XGBoost `0.5983`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.7828`, XGBoost `0.5983`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.7812`, XGBoost `0.5983`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7769`, XGBoost `0.6008`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7764`, XGBoost `0.6017`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7748`, XGBoost `0.6008`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6822`, XGBoost `0.5085`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.6816`, XGBoost `0.5085`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7721`, XGBoost `0.6008`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
