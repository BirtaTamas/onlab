# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `14`
- rows: `149`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 149 | 1.000 | 0.772965 | 0.829987 | -0.057023 | 4 | 145 | 1.000000 | 1.000000 |
| active/recent utility | 149 | 1.000 | 0.772965 | 0.829987 | -0.057023 | 4 | 145 | 1.000000 | 1.000000 |
| strong utility action | 119 | 0.799 | 0.747336 | 0.806498 | -0.059162 | 4 | 115 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 110 | 0.738 | 0.758525 | 0.818371 | -0.059846 | 4 | 106 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.067 | 0.607085 | 0.661397 | -0.054312 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 149 | 1.000 | 0.772965 | 0.829987 | -0.057023 | 4 | 145 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `62.0s`, rows `110`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.7156`, XGBoost `0.8985`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.7687`, XGBoost `0.9302`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.7788`, XGBoost `0.9280`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.7913`, XGBoost `0.9289`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.7931`, XGBoost `0.9285`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.7957`, XGBoost `0.9302`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.7955`, XGBoost `0.9289`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.7966`, XGBoost `0.9299`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.7979`, XGBoost `0.9309`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.7961`, XGBoost `0.9289`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
