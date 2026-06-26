# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m1-train.csv`
- round_num: `14`
- rows: `215`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 215 | 1.000 | 0.252183 | 0.293082 | -0.040899 | 167 | 48 | 0.879070 | 0.962791 |
| active/recent utility | 215 | 1.000 | 0.252183 | 0.293082 | -0.040899 | 167 | 48 | 0.879070 | 0.962791 |
| strong utility action | 137 | 0.637 | 0.232886 | 0.279300 | -0.046414 | 105 | 32 | 0.810219 | 0.941606 |
| utility damage | 10 | 0.047 | 0.497907 | 0.446861 | 0.051046 | 1 | 9 | 0.400000 | 1.000000 |
| active smoke/inferno | 129 | 0.600 | 0.226268 | 0.272961 | -0.046693 | 97 | 32 | 0.798450 | 0.937984 |
| recent utility last 5s | 10 | 0.047 | 0.349377 | 0.385089 | -0.035712 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 215 | 1.000 | 0.252183 | 0.293082 | -0.040899 | 167 | 48 | 0.879070 | 0.962791 |

## Active Smoke/Inferno Intervals

- `6.0s` - `48.0s`, rows `85`
- `83.5s` - `105.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.0`, LSTM `0.1262`, XGBoost `0.3847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1652`, XGBoost `0.4179`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1392`, XGBoost `0.3847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1437`, XGBoost `0.3847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.2071`, XGBoost `0.4151`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1767`, XGBoost `0.3847`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.0778`, XGBoost `0.2656`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0795`, XGBoost `0.2656`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0817`, XGBoost `0.2656`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0868`, XGBoost `0.2656`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
