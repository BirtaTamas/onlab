# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-imperial-vs-liquid-bo3-eiIGPV5tjvJFQ73hC8D8JI/imperial-vs-liquid-m3-anubis.csv`
- round_num: `14`
- rows: `188`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 188 | 1.000 | 0.225330 | 0.223090 | 0.002239 | 134 | 54 | 0.579787 | 0.765957 |
| active/recent utility | 188 | 1.000 | 0.225330 | 0.223090 | 0.002239 | 134 | 54 | 0.579787 | 0.765957 |
| strong utility action | 44 | 0.234 | 0.050054 | 0.057272 | -0.007218 | 40 | 4 | 0.954545 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.234 | 0.050054 | 0.057272 | -0.007218 | 40 | 4 | 0.954545 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 188 | 1.000 | 0.225330 | 0.223090 | 0.002239 | 134 | 54 | 0.579787 | 0.765957 |

## Active Smoke/Inferno Intervals

- `39.0s` - `60.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.5`, LSTM `0.2337`, XGBoost `0.1483`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0994`, XGBoost `0.1472`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.0336`, XGBoost `0.0664`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `64.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.0343`, XGBoost `0.0667`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `64.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.0345`, XGBoost `0.0651`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `64.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5122`, XGBoost `0.4839`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5021`, XGBoost `0.4741`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0433`, XGBoost `0.0649`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `64.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0016`, XGBoost `0.0232`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0018`, XGBoost `0.0233`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
