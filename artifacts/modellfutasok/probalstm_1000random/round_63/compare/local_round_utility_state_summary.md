# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `7`
- rows: `127`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 127 | 1.000 | 0.862133 | 0.902695 | -0.040562 | 0 | 127 | 1.000000 | 1.000000 |
| active/recent utility | 127 | 1.000 | 0.862133 | 0.902695 | -0.040562 | 0 | 127 | 1.000000 | 1.000000 |
| strong utility action | 112 | 0.882 | 0.857472 | 0.900144 | -0.042672 | 0 | 112 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.157 | 0.890242 | 0.933815 | -0.043573 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 102 | 0.803 | 0.869874 | 0.912935 | -0.043060 | 0 | 102 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.079 | 0.730967 | 0.769681 | -0.038714 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 127 | 1.000 | 0.862133 | 0.902695 | -0.040562 | 0 | 127 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `58.0s`, rows `102`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `20.5`, LSTM `0.6455`, XGBoost `0.7496`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.6458`, XGBoost `0.7496`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6494`, XGBoost `0.7496`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.6529`, XGBoost `0.7481`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `2.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.6542`, XGBoost `0.7491`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6528`, XGBoost `0.7476`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6548`, XGBoost `0.7496`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.6549`, XGBoost `0.7496`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6548`, XGBoost `0.7476`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.6554`, XGBoost `0.7481`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `2.0`, recent_utility `0`
