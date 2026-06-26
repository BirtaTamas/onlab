# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-g2-bo3-3aFk7fRwd7iUE0VJycUPHK/spirit-vs-g2-m3-ancient.csv`
- round_num: `5`
- rows: `218`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 218 | 1.000 | 0.063916 | 0.119789 | -0.055873 | 218 | 0 | 1.000000 | 1.000000 |
| active/recent utility | 218 | 1.000 | 0.063916 | 0.119789 | -0.055873 | 218 | 0 | 1.000000 | 1.000000 |
| strong utility action | 163 | 0.748 | 0.083138 | 0.154561 | -0.071423 | 163 | 0 | 1.000000 | 1.000000 |
| utility damage | 13 | 0.060 | 0.127608 | 0.315789 | -0.188181 | 13 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 151 | 0.693 | 0.080667 | 0.141012 | -0.060345 | 151 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 12 | 0.055 | 0.114233 | 0.325056 | -0.210824 | 12 | 0 | 1.000000 | 1.000000 |
| flash effect present | 218 | 1.000 | 0.063916 | 0.119789 | -0.055873 | 218 | 0 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `34.0s`, rows `56`
- `35.0s` - `82.0s`, rows `95`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `2.5`, LSTM `0.0983`, XGBoost `0.3401`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `3.0`, LSTM `0.0950`, XGBoost `0.3358`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `9.0`, LSTM `0.0839`, XGBoost `0.3207`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.0998`, XGBoost `0.3358`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.0`, LSTM `0.1051`, XGBoost `0.3409`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.0`, LSTM `0.1082`, XGBoost `0.3336`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `9.5`, LSTM `0.0976`, XGBoost `0.3225`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0987`, XGBoost `0.3207`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `1.5`, LSTM `0.0927`, XGBoost `0.3138`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `10.0`, LSTM `0.1086`, XGBoost `0.3212`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `7.0`, recent_utility `0`
