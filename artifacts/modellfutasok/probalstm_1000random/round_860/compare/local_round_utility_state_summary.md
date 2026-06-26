# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `3`
- rows: `287`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 287 | 1.000 | 0.017390 | 0.023785 | -0.006395 | 206 | 81 | 1.000000 | 1.000000 |
| active/recent utility | 287 | 1.000 | 0.017390 | 0.023785 | -0.006395 | 206 | 81 | 1.000000 | 1.000000 |
| strong utility action | 184 | 0.641 | 0.023422 | 0.031802 | -0.008380 | 126 | 58 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 170 | 0.592 | 0.016701 | 0.028440 | -0.011739 | 122 | 48 | 1.000000 | 1.000000 |
| recent utility last 5s | 14 | 0.049 | 0.105037 | 0.072628 | 0.032409 | 4 | 10 | 1.000000 | 1.000000 |
| flash effect present | 287 | 1.000 | 0.017390 | 0.023785 | -0.006395 | 206 | 81 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `52.0s`, rows `85`
- `59.0s` - `80.5s`, rows `44`
- `123.0s` - `143.0s`, rows `41`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `6.5`, LSTM `0.1890`, XGBoost `0.0768`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `6.0`, LSTM `0.1526`, XGBoost `0.0737`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `5.5`, LSTM `0.1483`, XGBoost `0.0737`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `5.0`, LSTM `0.1434`, XGBoost `0.0737`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `7.0`, LSTM `0.1342`, XGBoost `0.0780`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.1278`, XGBoost `0.0737`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `4`
- seconds `10.5`, LSTM `0.0328`, XGBoost `0.0842`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0338`, XGBoost `0.0844`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0247`, XGBoost `0.0735`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0150`, XGBoost `0.0632`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `45.0`, recent_utility `0`
