# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `14`
- rows: `288`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 288 | 1.000 | 0.150659 | 0.239327 | -0.088668 | 260 | 28 | 1.000000 | 1.000000 |
| active/recent utility | 288 | 1.000 | 0.150659 | 0.239327 | -0.088668 | 260 | 28 | 1.000000 | 1.000000 |
| strong utility action | 224 | 0.778 | 0.167537 | 0.265888 | -0.098350 | 215 | 9 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 214 | 0.743 | 0.156931 | 0.262155 | -0.105225 | 214 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.035 | 0.394521 | 0.345754 | 0.048767 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 288 | 1.000 | 0.150659 | 0.239327 | -0.088668 | 260 | 28 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `33.0s`, rows `45`
- `39.0s` - `82.5s`, rows `88`
- `84.0s` - `124.0s`, rows `81`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.1477`, XGBoost `0.3657`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.1578`, XGBoost `0.3657`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1600`, XGBoost `0.3654`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.1614`, XGBoost `0.3657`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1640`, XGBoost `0.3654`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.1661`, XGBoost `0.3657`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.1671`, XGBoost `0.3658`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1703`, XGBoost `0.3659`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.1721`, XGBoost `0.3657`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.1723`, XGBoost `0.3657`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
