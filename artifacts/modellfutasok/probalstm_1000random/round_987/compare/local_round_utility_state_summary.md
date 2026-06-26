# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `6`
- rows: `195`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 195 | 1.000 | 0.685200 | 0.638418 | 0.046782 | 172 | 23 | 1.000000 | 0.800000 |
| active/recent utility | 195 | 1.000 | 0.685200 | 0.638418 | 0.046782 | 172 | 23 | 1.000000 | 0.800000 |
| strong utility action | 126 | 0.646 | 0.649962 | 0.603461 | 0.046501 | 115 | 11 | 1.000000 | 0.746032 |
| utility damage | 36 | 0.185 | 0.696868 | 0.648473 | 0.048395 | 34 | 2 | 1.000000 | 0.722222 |
| active smoke/inferno | 116 | 0.595 | 0.655530 | 0.612056 | 0.043473 | 105 | 11 | 1.000000 | 0.758621 |
| recent utility last 5s | 10 | 0.051 | 0.585380 | 0.503755 | 0.081624 | 10 | 0 | 1.000000 | 0.600000 |
| flash effect present | 195 | 1.000 | 0.685200 | 0.638418 | 0.046782 | 172 | 23 | 1.000000 | 0.800000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `67.5s`, rows `116`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `12.5`, LSTM `0.6212`, XGBoost `0.4971`, closer `lstm`, smoke `1`, inferno `4`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6047`, XGBoost `0.4808`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.6207`, XGBoost `0.4971`, closer `lstm`, smoke `1`, inferno `4`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6200`, XGBoost `0.4971`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `19.0`, LSTM `0.6043`, XGBoost `0.4829`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6168`, XGBoost `0.4971`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5955`, XGBoost `0.4808`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `16.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.6107`, XGBoost `0.4971`, closer `lstm`, smoke `1`, inferno `4`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6076`, XGBoost `0.4978`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6054`, XGBoost `0.4978`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
