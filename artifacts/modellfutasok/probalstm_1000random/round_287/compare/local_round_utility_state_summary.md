# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `13`
- rows: `108`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 108 | 1.000 | 0.683057 | 0.720893 | -0.037836 | 18 | 90 | 1.000000 | 1.000000 |
| active/recent utility | 108 | 1.000 | 0.683057 | 0.720893 | -0.037836 | 18 | 90 | 1.000000 | 1.000000 |
| strong utility action | 48 | 0.444 | 0.707508 | 0.768542 | -0.061033 | 5 | 43 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.102 | 0.583664 | 0.674072 | -0.090408 | 0 | 11 | 1.000000 | 1.000000 |
| active smoke/inferno | 48 | 0.444 | 0.707508 | 0.768542 | -0.061033 | 5 | 43 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 108 | 1.000 | 0.683057 | 0.720893 | -0.037836 | 18 | 90 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `20.0s` - `43.5s`, rows `48`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.6133`, XGBoost `0.7841`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.6269`, XGBoost `0.7836`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6579`, XGBoost `0.8001`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6419`, XGBoost `0.7838`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.6759`, XGBoost `0.8106`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6482`, XGBoost `0.7697`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.6939`, XGBoost `0.8106`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `55.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5702`, XGBoost `0.6721`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `54.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5087`, XGBoost `0.6034`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5012`, XGBoost `0.5916`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `55.0`, recent_utility `0`
