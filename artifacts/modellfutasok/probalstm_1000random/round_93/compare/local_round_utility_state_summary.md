# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_lisbon/blast-open-lisbon-2025-the-mongolz-vs-g2-bo3-_aqP5h00uQDg161T2kCLGM/the-mongolz-vs-g2-m2-dust2.csv`
- round_num: `10`
- rows: `186`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 186 | 1.000 | 0.666159 | 0.604989 | 0.061170 | 170 | 16 | 1.000000 | 1.000000 |
| active/recent utility | 186 | 1.000 | 0.666159 | 0.604989 | 0.061170 | 170 | 16 | 1.000000 | 1.000000 |
| strong utility action | 175 | 0.941 | 0.666934 | 0.609992 | 0.056942 | 159 | 16 | 1.000000 | 1.000000 |
| utility damage | 40 | 0.215 | 0.661742 | 0.586112 | 0.075630 | 40 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 165 | 0.887 | 0.668122 | 0.615173 | 0.052950 | 149 | 16 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 186 | 1.000 | 0.666159 | 0.604989 | 0.061170 | 170 | 16 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `92.5s`, rows `165`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `3.5`, LSTM `0.6623`, XGBoost `0.5240`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `4.0`, LSTM `0.6563`, XGBoost `0.5250`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.6575`, XGBoost `0.5265`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.6542`, XGBoost `0.5260`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.7119`, XGBoost `0.5846`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `51.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.7120`, XGBoost `0.5868`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.6465`, XGBoost `0.5229`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `5.5`, LSTM `0.6479`, XGBoost `0.5260`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.7046`, XGBoost `0.5846`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `51.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.7042`, XGBoost `0.5846`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
