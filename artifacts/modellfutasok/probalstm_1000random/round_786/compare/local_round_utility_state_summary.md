# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `21`
- rows: `221`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 221 | 1.000 | 0.170954 | 0.267482 | -0.096528 | 211 | 10 | 0.959276 | 0.936652 |
| active/recent utility | 221 | 1.000 | 0.170954 | 0.267482 | -0.096528 | 211 | 10 | 0.959276 | 0.936652 |
| strong utility action | 201 | 0.910 | 0.185781 | 0.289810 | -0.104029 | 191 | 10 | 0.955224 | 0.930348 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 186 | 0.842 | 0.178598 | 0.288694 | -0.110096 | 179 | 7 | 0.951613 | 0.924731 |
| recent utility last 5s | 15 | 0.068 | 0.274851 | 0.303652 | -0.028800 | 12 | 3 | 1.000000 | 1.000000 |
| flash effect present | 221 | 1.000 | 0.170954 | 0.267482 | -0.096528 | 211 | 10 | 0.959276 | 0.936652 |

## Active Smoke/Inferno Intervals

- `8.5s` - `101.0s`, rows `186`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `69.0`, LSTM `0.0310`, XGBoost `0.3248`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.0329`, XGBoost `0.3248`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0334`, XGBoost `0.3215`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0377`, XGBoost `0.3256`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.0349`, XGBoost `0.3210`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.0344`, XGBoost `0.3196`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.0504`, XGBoost `0.3256`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0467`, XGBoost `0.3215`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.0639`, XGBoost `0.3242`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.0624`, XGBoost `0.3196`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
