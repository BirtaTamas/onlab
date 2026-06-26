# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-heroic-vs-flyquest-bo3-ErQHzvBcWPHiA-H04IjPMf/heroic-vs-flyquest-m2-anubis.csv`
- round_num: `15`
- rows: `164`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 164 | 1.000 | 0.045590 | 0.110098 | -0.064508 | 158 | 6 | 1.000000 | 1.000000 |
| active/recent utility | 164 | 1.000 | 0.045590 | 0.110098 | -0.064508 | 158 | 6 | 1.000000 | 1.000000 |
| strong utility action | 132 | 0.805 | 0.043115 | 0.110520 | -0.067405 | 126 | 6 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.061 | 0.104467 | 0.258001 | -0.153533 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.744 | 0.041892 | 0.097655 | -0.055763 | 116 | 6 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.061 | 0.058034 | 0.267463 | -0.209429 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 164 | 1.000 | 0.045590 | 0.110098 | -0.064508 | 158 | 6 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `29.5s`, rows `44`
- `34.0s` - `72.5s`, rows `78`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `2.5`, LSTM `0.0496`, XGBoost `0.2735`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.0520`, XGBoost `0.2735`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.0532`, XGBoost `0.2735`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.0`, LSTM `0.0555`, XGBoost `0.2735`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.0469`, XGBoost `0.2620`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.0496`, XGBoost `0.2620`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.0548`, XGBoost `0.2620`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.0680`, XGBoost `0.2735`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.0692`, XGBoost `0.2620`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.0`, LSTM `0.0815`, XGBoost `0.2594`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
