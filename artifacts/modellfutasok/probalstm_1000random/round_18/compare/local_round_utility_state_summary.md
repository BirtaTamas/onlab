# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-3dmax-bo3-Oe166BQltZjvHlE8qlepgF/furia-vs-3dmax-m1-nuke.csv`
- round_num: `6`
- rows: `168`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 168 | 1.000 | 0.633111 | 0.705941 | -0.072831 | 10 | 158 | 1.000000 | 1.000000 |
| active/recent utility | 168 | 1.000 | 0.633111 | 0.705941 | -0.072831 | 10 | 158 | 1.000000 | 1.000000 |
| strong utility action | 153 | 0.911 | 0.640783 | 0.720103 | -0.079321 | 4 | 149 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 153 | 0.911 | 0.640783 | 0.720103 | -0.079321 | 4 | 149 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 168 | 1.000 | 0.633111 | 0.705941 | -0.072831 | 10 | 158 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `83.5s`, rows `153`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.0`, LSTM `0.5341`, XGBoost `0.7964`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.5445`, XGBoost `0.7964`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.5436`, XGBoost `0.7835`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.5567`, XGBoost `0.7964`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.5575`, XGBoost `0.7964`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.5609`, XGBoost `0.7964`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.5560`, XGBoost `0.7835`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.5859`, XGBoost `0.7971`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.6065`, XGBoost `0.7971`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.6161`, XGBoost `0.7971`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
