# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m2-nuke.csv`
- round_num: `7`
- rows: `172`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 172 | 1.000 | 0.793584 | 0.795017 | -0.001433 | 61 | 111 | 1.000000 | 1.000000 |
| active/recent utility | 172 | 1.000 | 0.793584 | 0.795017 | -0.001433 | 61 | 111 | 1.000000 | 1.000000 |
| strong utility action | 113 | 0.657 | 0.750945 | 0.751307 | -0.000362 | 48 | 65 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.058 | 0.951330 | 0.990833 | -0.039503 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 113 | 0.657 | 0.750945 | 0.751307 | -0.000362 | 48 | 65 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 172 | 1.000 | 0.793584 | 0.795017 | -0.001433 | 61 | 111 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `64.0s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.8511`, XGBoost `0.9254`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6122`, XGBoost `0.5385`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.6119`, XGBoost `0.5385`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6106`, XGBoost `0.5385`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6097`, XGBoost `0.5385`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6089`, XGBoost `0.5385`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.8484`, XGBoost `0.9181`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.6068`, XGBoost `0.5381`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.8528`, XGBoost `0.9182`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6985`, XGBoost `0.7634`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
