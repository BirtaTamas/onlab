# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-gamerlegion-vs-the-mongolz-bo3-zdjI5BKx0DIgDYoNAnfKpI/gamerlegion-vs-the-mongolz-m2-mirage.csv`
- round_num: `15`
- rows: `199`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.870526 | 0.871660 | -0.001133 | 97 | 102 | 1.000000 | 1.000000 |
| active/recent utility | 199 | 1.000 | 0.870526 | 0.871660 | -0.001133 | 97 | 102 | 1.000000 | 1.000000 |
| strong utility action | 166 | 0.834 | 0.877986 | 0.880096 | -0.002110 | 80 | 86 | 1.000000 | 1.000000 |
| utility damage | 40 | 0.201 | 0.884111 | 0.875886 | 0.008224 | 25 | 15 | 1.000000 | 1.000000 |
| active smoke/inferno | 166 | 0.834 | 0.877986 | 0.880096 | -0.002110 | 80 | 86 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 199 | 1.000 | 0.870526 | 0.871660 | -0.001133 | 97 | 102 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.5s` - `18.0s`, rows `22`
- `20.5s` - `27.0s`, rows `14`
- `33.5s` - `98.0s`, rows `130`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.8704`, XGBoost `0.7954`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.8683`, XGBoost `0.7954`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.8631`, XGBoost `0.7950`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.8059`, XGBoost `0.8690`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.8761`, XGBoost `0.9219`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `19.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.8640`, XGBoost `0.8204`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `12.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.9235`, XGBoost `0.9663`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.8510`, XGBoost `0.8936`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.8502`, XGBoost `0.8927`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.8516`, XGBoost `0.8936`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
