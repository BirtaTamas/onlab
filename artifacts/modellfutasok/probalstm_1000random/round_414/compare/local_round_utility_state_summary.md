# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-b8-vs-hotu-bo3-tmCfOETKzYqjV6vSvNp3-F/b8-vs-hotu-m3-ancient.csv`
- round_num: `7`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.831638 | 0.853293 | -0.021655 | 29 | 129 | 1.000000 | 1.000000 |
| active/recent utility | 158 | 1.000 | 0.831638 | 0.853293 | -0.021655 | 29 | 129 | 1.000000 | 1.000000 |
| strong utility action | 132 | 0.835 | 0.821608 | 0.848478 | -0.026871 | 17 | 115 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.063 | 0.681296 | 0.708953 | -0.027658 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 131 | 0.829 | 0.822239 | 0.849291 | -0.027052 | 17 | 114 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.063 | 0.742496 | 0.735184 | 0.007312 | 6 | 4 | 1.000000 | 1.000000 |
| flash effect present | 158 | 1.000 | 0.831638 | 0.853293 | -0.021655 | 29 | 129 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `71.5s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.0`, LSTM `0.9012`, XGBoost `0.9750`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.9061`, XGBoost `0.9749`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.9086`, XGBoost `0.9749`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.6409`, XGBoost `0.7016`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.6439`, XGBoost `0.7038`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.9168`, XGBoost `0.9747`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.9171`, XGBoost `0.9749`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.6474`, XGBoost `0.7038`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.9188`, XGBoost `0.9747`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.6484`, XGBoost `0.7038`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
