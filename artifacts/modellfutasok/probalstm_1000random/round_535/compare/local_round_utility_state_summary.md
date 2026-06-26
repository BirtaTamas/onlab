# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `20`
- rows: `130`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 130 | 1.000 | 0.684114 | 0.594528 | 0.089586 | 126 | 4 | 1.000000 | 0.915385 |
| active/recent utility | 130 | 1.000 | 0.684114 | 0.594528 | 0.089586 | 126 | 4 | 1.000000 | 0.915385 |
| strong utility action | 96 | 0.738 | 0.705533 | 0.616724 | 0.088809 | 92 | 4 | 1.000000 | 0.885417 |
| utility damage | 29 | 0.223 | 0.719484 | 0.636817 | 0.082667 | 27 | 2 | 1.000000 | 0.793103 |
| active smoke/inferno | 86 | 0.662 | 0.715305 | 0.626918 | 0.088387 | 82 | 4 | 1.000000 | 0.872093 |
| recent utility last 5s | 10 | 0.077 | 0.621492 | 0.529054 | 0.092438 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 130 | 1.000 | 0.684114 | 0.594528 | 0.089586 | 126 | 4 | 1.000000 | 0.915385 |

## Active Smoke/Inferno Intervals

- `10.0s` - `38.5s`, rows `58`
- `51.0s` - `64.5s`, rows `28`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.0`, LSTM `0.5492`, XGBoost `0.2934`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5292`, XGBoost `0.2816`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5271`, XGBoost `0.2834`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `14.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.5319`, XGBoost `0.2934`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5284`, XGBoost `0.2934`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.5277`, XGBoost `0.2934`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5248`, XGBoost `0.2934`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5099`, XGBoost `0.2834`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5057`, XGBoost `0.2834`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.7120`, XGBoost `0.5587`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
