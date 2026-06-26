# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-liquid-bo3-9v1WXdmzbeO2q7iD5Nu_mP/mouz-vs-liquid-m2-nuke.csv`
- round_num: `11`
- rows: `224`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.635777 | 0.711478 | -0.075701 | 0 | 224 | 0.491071 | 1.000000 |
| active/recent utility | 224 | 1.000 | 0.635777 | 0.711478 | -0.075701 | 0 | 224 | 0.491071 | 1.000000 |
| strong utility action | 207 | 0.924 | 0.647584 | 0.722446 | -0.074861 | 0 | 207 | 0.526570 | 1.000000 |
| utility damage | 35 | 0.156 | 0.917475 | 0.957343 | -0.039867 | 0 | 35 | 1.000000 | 1.000000 |
| active smoke/inferno | 199 | 0.888 | 0.634959 | 0.711883 | -0.076924 | 0 | 199 | 0.507538 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 224 | 1.000 | 0.635777 | 0.711478 | -0.075701 | 0 | 224 | 0.491071 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `93.0s`, rows `171`
- `94.0s` - `107.5s`, rows `28`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.3126`, XGBoost `0.5618`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.3143`, XGBoost `0.5621`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.3201`, XGBoost `0.5618`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3205`, XGBoost `0.5564`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.3208`, XGBoost `0.5564`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.3254`, XGBoost `0.5564`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.3377`, XGBoost `0.5557`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.3403`, XGBoost `0.5564`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.3227`, XGBoost `0.5373`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.3431`, XGBoost `0.5564`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
