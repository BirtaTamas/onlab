# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `5`
- rows: `178`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 178 | 1.000 | 0.795212 | 0.865294 | -0.070082 | 0 | 178 | 1.000000 | 1.000000 |
| active/recent utility | 178 | 1.000 | 0.795212 | 0.865294 | -0.070082 | 0 | 178 | 1.000000 | 1.000000 |
| strong utility action | 123 | 0.691 | 0.783357 | 0.856308 | -0.072951 | 0 | 123 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.112 | 0.832706 | 0.888242 | -0.055536 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 112 | 0.629 | 0.788906 | 0.860295 | -0.071390 | 0 | 112 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.112 | 0.749668 | 0.823925 | -0.074257 | 0 | 20 | 1.000000 | 1.000000 |
| flash effect present | 178 | 1.000 | 0.795212 | 0.865294 | -0.070082 | 0 | 178 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `19.0s` - `47.0s`, rows `57`
- `52.5s` - `79.5s`, rows `55`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `62.0`, LSTM `0.6736`, XGBoost `0.8334`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.6743`, XGBoost `0.8334`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.6839`, XGBoost `0.8305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.6843`, XGBoost `0.8305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.6846`, XGBoost `0.8305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.6887`, XGBoost `0.8318`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.6874`, XGBoost `0.8305`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.6926`, XGBoost `0.8318`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6963`, XGBoost `0.8345`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.6927`, XGBoost `0.8305`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
