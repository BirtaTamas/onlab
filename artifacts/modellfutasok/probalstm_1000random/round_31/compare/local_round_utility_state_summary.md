# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-vitality-vs-g2-bo5-ieXHvClzA7f_aJ_85fPFqK/vitality-vs-g2-m1-dust2.csv`
- round_num: `16`
- rows: `184`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.470287 | 0.509994 | -0.039708 | 133 | 51 | 0.396739 | 0.375000 |
| active/recent utility | 184 | 1.000 | 0.470287 | 0.509994 | -0.039708 | 133 | 51 | 0.396739 | 0.375000 |
| strong utility action | 143 | 0.777 | 0.504291 | 0.530213 | -0.025922 | 99 | 44 | 0.335664 | 0.321678 |
| utility damage | 10 | 0.054 | 0.435973 | 0.519945 | -0.083972 | 10 | 0 | 0.400000 | 0.400000 |
| active smoke/inferno | 132 | 0.717 | 0.489130 | 0.524639 | -0.035510 | 99 | 33 | 0.363636 | 0.348485 |
| recent utility last 5s | 21 | 0.114 | 0.635944 | 0.584434 | 0.051510 | 3 | 18 | 0.000000 | 0.000000 |
| flash effect present | 184 | 1.000 | 0.470287 | 0.509994 | -0.039708 | 133 | 51 | 0.396739 | 0.375000 |

## Active Smoke/Inferno Intervals

- `9.0s` - `36.0s`, rows `55`
- `38.5s` - `69.5s`, rows `63`
- `85.0s` - `91.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `60.0`, LSTM `0.2596`, XGBoost `0.4163`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.2516`, XGBoost `0.4021`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.2736`, XGBoost `0.4021`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.2801`, XGBoost `0.4021`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.4957`, XGBoost `0.6171`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `43.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.4976`, XGBoost `0.6178`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `43.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.2868`, XGBoost `0.4021`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `2.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.7160`, XGBoost `0.6014`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `3`
- seconds `69.5`, LSTM `0.0927`, XGBoost `0.2059`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5028`, XGBoost `0.6105`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `45.0`, recent_utility `0`
