# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-betboom-vs-nemiga-train-khA7BVyAiKBjWcyTrFzube/betboom-vs-nemiga-train.csv`
- round_num: `5`
- rows: `123`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 123 | 1.000 | 0.632424 | 0.613537 | 0.018887 | 72 | 51 | 0.902439 | 1.000000 |
| active/recent utility | 123 | 1.000 | 0.632424 | 0.613537 | 0.018887 | 72 | 51 | 0.902439 | 1.000000 |
| strong utility action | 104 | 0.846 | 0.651327 | 0.629157 | 0.022170 | 67 | 37 | 0.942308 | 1.000000 |
| utility damage | 20 | 0.163 | 0.649540 | 0.648082 | 0.001457 | 7 | 13 | 0.800000 | 1.000000 |
| active smoke/inferno | 104 | 0.846 | 0.651327 | 0.629157 | 0.022170 | 67 | 37 | 0.942308 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 123 | 1.000 | 0.632424 | 0.613537 | 0.018887 | 72 | 51 | 0.902439 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `42.0s`, rows `70`
- `44.5s` - `61.0s`, rows `34`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `47.0`, LSTM `0.6689`, XGBoost `0.5380`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6644`, XGBoost `0.5367`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6641`, XGBoost `0.5383`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.6617`, XGBoost `0.5367`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.6587`, XGBoost `0.5386`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6335`, XGBoost `0.5282`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6587`, XGBoost `0.5641`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `34.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.6525`, XGBoost `0.5636`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `34.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6216`, XGBoost `0.5344`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.6320`, XGBoost `0.5588`, closer `lstm`, smoke `1`, inferno `3`, utility_damage `34.0`, recent_utility `0`
