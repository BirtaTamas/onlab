# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-spirit-vs-heroic-bo3-NSK1XoxyhXK-sIe0204dzp/spirit-vs-heroic-m3-mirage.csv`
- round_num: `17`
- rows: `160`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 160 | 1.000 | 0.377631 | 0.519940 | -0.142309 | 160 | 0 | 1.000000 | 0.081250 |
| active/recent utility | 160 | 1.000 | 0.377631 | 0.519940 | -0.142309 | 160 | 0 | 1.000000 | 0.081250 |
| strong utility action | 156 | 0.975 | 0.375827 | 0.519367 | -0.143540 | 156 | 0 | 1.000000 | 0.083333 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 148 | 0.925 | 0.373831 | 0.517749 | -0.143918 | 148 | 0 | 1.000000 | 0.087838 |
| recent utility last 5s | 20 | 0.125 | 0.418718 | 0.558258 | -0.139540 | 20 | 0 | 1.000000 | 0.000000 |
| flash effect present | 160 | 1.000 | 0.377631 | 0.519940 | -0.142309 | 160 | 0 | 1.000000 | 0.081250 |

## Active Smoke/Inferno Intervals

- `6.0s` - `79.5s`, rows `148`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.0`, LSTM `0.2471`, XGBoost `0.5740`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `1`
- seconds `14.5`, LSTM `0.2504`, XGBoost `0.5740`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.2676`, XGBoost `0.5740`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.2721`, XGBoost `0.5047`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.3132`, XGBoost `0.5429`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.1719`, XGBoost `0.4005`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.3409`, XGBoost `0.5672`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3251`, XGBoost `0.5414`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.3467`, XGBoost `0.5625`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3245`, XGBoost `0.5402`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
