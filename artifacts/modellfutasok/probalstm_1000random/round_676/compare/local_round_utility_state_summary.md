# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m2-nuke.csv`
- round_num: `10`
- rows: `131`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 131 | 1.000 | 0.738539 | 0.787743 | -0.049204 | 1 | 130 | 0.816794 | 0.992366 |
| active/recent utility | 131 | 1.000 | 0.738539 | 0.787743 | -0.049204 | 1 | 130 | 0.816794 | 0.992366 |
| strong utility action | 102 | 0.779 | 0.754022 | 0.805054 | -0.051032 | 1 | 101 | 0.921569 | 0.990196 |
| utility damage | 11 | 0.084 | 0.547815 | 0.587023 | -0.039208 | 1 | 10 | 0.727273 | 0.909091 |
| active smoke/inferno | 102 | 0.779 | 0.754022 | 0.805054 | -0.051032 | 1 | 101 | 0.921569 | 0.990196 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 131 | 1.000 | 0.738539 | 0.787743 | -0.049204 | 1 | 130 | 0.816794 | 0.992366 |

## Active Smoke/Inferno Intervals

- `8.0s` - `55.0s`, rows `95`
- `62.0s` - `65.0s`, rows `7`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `31.5`, LSTM `0.7266`, XGBoost `0.8676`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.7316`, XGBoost `0.8670`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7337`, XGBoost `0.8676`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.7381`, XGBoost `0.8660`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.7454`, XGBoost `0.8660`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.7751`, XGBoost `0.8687`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.7759`, XGBoost `0.8660`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5731`, XGBoost `0.6535`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.5732`, XGBoost `0.6535`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.4233`, XGBoost `0.5034`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
