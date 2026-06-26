# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m3-ancient.csv`
- round_num: `1`
- rows: `102`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 102 | 1.000 | 0.499003 | 0.593315 | -0.094312 | 4 | 98 | 0.313725 | 0.931373 |
| active/recent utility | 102 | 1.000 | 0.499003 | 0.593315 | -0.094312 | 4 | 98 | 0.313725 | 0.931373 |
| strong utility action | 11 | 0.108 | 0.577535 | 0.643286 | -0.065752 | 2 | 9 | 0.818182 | 0.727273 |
| utility damage | 11 | 0.108 | 0.577535 | 0.643286 | -0.065752 | 2 | 9 | 0.818182 | 0.727273 |
| active smoke/inferno | 3 | 0.029 | 0.638011 | 0.696264 | -0.058253 | 0 | 3 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 102 | 1.000 | 0.499003 | 0.593315 | -0.094312 | 4 | 98 | 0.313725 | 0.931373 |

## Active Smoke/Inferno Intervals

- `38.0s` - `39.0s`, rows `3`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.5371`, XGBoost `0.7159`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `204.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5678`, XGBoost `0.7159`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `204.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.4010`, XGBoost `0.4919`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `204.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6195`, XGBoost `0.6907`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `204.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.6318`, XGBoost `0.6949`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `204.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.6349`, XGBoost `0.6956`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `204.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.4419`, XGBoost `0.4933`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `204.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.6473`, XGBoost `0.6983`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `204.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6672`, XGBoost `0.6917`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `204.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5036`, XGBoost `0.4898`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `204.0`, recent_utility `0`
