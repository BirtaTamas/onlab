# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m3-overpass.csv`
- round_num: `3`
- rows: `156`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 156 | 1.000 | 0.011600 | 0.040637 | -0.029037 | 156 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 156 | 1.000 | 0.011600 | 0.040637 | -0.029037 | 156 | 0 | 1.000000 | 1.000000 |
| strong utility action | 83 | 0.532 | 0.014903 | 0.053006 | -0.038104 | 83 | 0 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.071 | 0.004887 | 0.033226 | -0.028339 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 83 | 0.532 | 0.014903 | 0.053006 | -0.038104 | 83 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 156 | 1.000 | 0.011600 | 0.040637 | -0.029037 | 156 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `50.5s`, rows `83`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.5`, LSTM `0.0118`, XGBoost `0.0822`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `30.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.0136`, XGBoost `0.0726`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0169`, XGBoost `0.0721`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.0166`, XGBoost `0.0709`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.0179`, XGBoost `0.0721`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.0161`, XGBoost `0.0702`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0185`, XGBoost `0.0720`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0156`, XGBoost `0.0691`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0163`, XGBoost `0.0695`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.0164`, XGBoost `0.0694`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
