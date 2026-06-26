# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `12`
- rows: `160`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 160 | 1.000 | 0.405778 | 0.406801 | -0.001023 | 116 | 44 | 0.362500 | 0.406250 |
| active/recent utility | 160 | 1.000 | 0.405778 | 0.406801 | -0.001023 | 116 | 44 | 0.362500 | 0.406250 |
| strong utility action | 140 | 0.875 | 0.397452 | 0.398755 | -0.001302 | 106 | 34 | 0.378571 | 0.428571 |
| utility damage | 21 | 0.131 | 0.393824 | 0.377411 | 0.016413 | 9 | 12 | 0.523810 | 0.523810 |
| active smoke/inferno | 140 | 0.875 | 0.397452 | 0.398755 | -0.001302 | 106 | 34 | 0.378571 | 0.428571 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 160 | 1.000 | 0.405778 | 0.406801 | -0.001023 | 116 | 44 | 0.362500 | 0.406250 |

## Active Smoke/Inferno Intervals

- `7.5s` - `72.5s`, rows `131`
- `75.5s` - `79.5s`, rows `9`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.2739`, XGBoost `0.1109`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.2501`, XGBoost `0.1109`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.2443`, XGBoost `0.1114`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.2204`, XGBoost `0.1116`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.2182`, XGBoost `0.1114`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `17.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5392`, XGBoost `0.4425`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5381`, XGBoost `0.4430`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5375`, XGBoost `0.4519`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5355`, XGBoost `0.4519`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5353`, XGBoost `0.4522`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `2.0`, recent_utility `0`
