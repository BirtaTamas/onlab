# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-nrg-vs-fluxo-bo3-aFv0UX6WO0txoeY8N630nT/nrg-vs-fluxo-m1-nuke.csv`
- round_num: `5`
- rows: `217`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.553737 | 0.611665 | -0.057929 | 70 | 147 | 0.700461 | 0.792627 |
| active/recent utility | 217 | 1.000 | 0.553737 | 0.611665 | -0.057929 | 70 | 147 | 0.700461 | 0.792627 |
| strong utility action | 140 | 0.645 | 0.529577 | 0.569259 | -0.039681 | 65 | 75 | 0.735714 | 0.857143 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 130 | 0.599 | 0.542313 | 0.594870 | -0.052556 | 55 | 75 | 0.792308 | 0.923077 |
| recent utility last 5s | 10 | 0.046 | 0.364008 | 0.236315 | 0.127694 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 217 | 1.000 | 0.553737 | 0.611665 | -0.057929 | 70 | 147 | 0.700461 | 0.792627 |

## Active Smoke/Inferno Intervals

- `8.0s` - `50.5s`, rows `86`
- `66.0s` - `87.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `81.5`, LSTM `0.5136`, XGBoost `0.8524`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.5`, LSTM `0.5139`, XGBoost `0.8517`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.5161`, XGBoost `0.8524`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.5165`, XGBoost `0.8517`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `82.0`, LSTM `0.5180`, XGBoost `0.8524`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.5381`, XGBoost `0.8723`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `84.0`, LSTM `0.5210`, XGBoost `0.8521`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.5409`, XGBoost `0.8697`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.5582`, XGBoost `0.8713`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.5674`, XGBoost `0.8673`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
