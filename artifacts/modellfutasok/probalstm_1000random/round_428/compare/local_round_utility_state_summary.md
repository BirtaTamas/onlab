# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-gamerlegion-bo3-8K-MOEPC1meC7FXyBc8fA2/astralis-vs-gamerlegion-m1-nuke.csv`
- round_num: `12`
- rows: `197`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 197 | 1.000 | 0.796105 | 0.825851 | -0.029746 | 47 | 150 | 1.000000 | 1.000000 |
| active/recent utility | 197 | 1.000 | 0.796105 | 0.825851 | -0.029746 | 47 | 150 | 1.000000 | 1.000000 |
| strong utility action | 162 | 0.822 | 0.784363 | 0.828239 | -0.043876 | 26 | 136 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 162 | 0.822 | 0.784363 | 0.828239 | -0.043876 | 26 | 136 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 197 | 1.000 | 0.796105 | 0.825851 | -0.029746 | 47 | 150 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `59.5s`, rows `104`
- `64.5s` - `71.0s`, rows `14`
- `73.5s` - `95.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.7355`, XGBoost `0.8908`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6926`, XGBoost `0.8187`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.7655`, XGBoost `0.8913`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.7650`, XGBoost `0.8896`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.6809`, XGBoost `0.7991`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7029`, XGBoost `0.8187`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.6859`, XGBoost `0.8016`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.6883`, XGBoost `0.8016`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.6893`, XGBoost `0.8022`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.7022`, XGBoost `0.8132`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
