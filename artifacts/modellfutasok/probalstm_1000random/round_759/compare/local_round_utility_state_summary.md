# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `4`
- rows: `144`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.698038 | 0.701519 | -0.003481 | 44 | 100 | 0.902778 | 0.875000 |
| active/recent utility | 144 | 1.000 | 0.698038 | 0.701519 | -0.003481 | 44 | 100 | 0.902778 | 0.875000 |
| strong utility action | 125 | 0.868 | 0.692137 | 0.694794 | -0.002658 | 39 | 86 | 0.888000 | 0.856000 |
| utility damage | 24 | 0.167 | 0.640855 | 0.592973 | 0.047883 | 12 | 12 | 0.958333 | 0.791667 |
| active smoke/inferno | 125 | 0.868 | 0.692137 | 0.694794 | -0.002658 | 39 | 86 | 0.888000 | 0.856000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.698038 | 0.701519 | -0.003481 | 44 | 100 | 0.902778 | 0.875000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `33.0s`, rows `51`
- `34.0s` - `70.5s`, rows `74`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.4707`, XGBoost `0.2501`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.6375`, XGBoost `0.4336`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `41.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.4468`, XGBoost `0.2501`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.6952`, XGBoost `0.5011`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `41.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.7831`, XGBoost `0.6116`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `41.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.5805`, XGBoost `0.4274`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `41.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4101`, XGBoost `0.2649`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.4828`, XGBoost `0.3400`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.5559`, XGBoost `0.4194`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `41.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.7478`, XGBoost `0.6116`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `41.0`, recent_utility `0`
