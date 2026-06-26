# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-falcons-vs-natus-vincere-bo3-z3OpWwYDPa33wwfDY8_B1Q/falcons-vs-natus-vincere-m1-nuke.csv`
- round_num: `10`
- rows: `176`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.683198 | 0.696439 | -0.013241 | 79 | 97 | 1.000000 | 1.000000 |
| active/recent utility | 176 | 1.000 | 0.683198 | 0.696439 | -0.013241 | 79 | 97 | 1.000000 | 1.000000 |
| strong utility action | 109 | 0.619 | 0.655606 | 0.681876 | -0.026270 | 46 | 63 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 109 | 0.619 | 0.655606 | 0.681876 | -0.026270 | 46 | 63 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 176 | 1.000 | 0.683198 | 0.696439 | -0.013241 | 79 | 97 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `54.5s`, rows `95`
- `72.0s` - `78.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.5967`, XGBoost `0.7594`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.6036`, XGBoost `0.7594`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.6663`, XGBoost `0.8146`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6550`, XGBoost `0.7988`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.6182`, XGBoost `0.7585`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.6651`, XGBoost `0.7988`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6760`, XGBoost `0.7988`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6783`, XGBoost `0.7988`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6787`, XGBoost `0.7988`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6388`, XGBoost `0.7585`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
