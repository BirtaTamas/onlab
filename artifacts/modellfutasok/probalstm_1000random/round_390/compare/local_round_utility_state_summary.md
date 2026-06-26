# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `3`
- rows: `190`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 190 | 1.000 | 0.012848 | 0.039526 | -0.026678 | 157 | 33 | 1.000000 | 1.000000 |
| active/recent utility | 190 | 1.000 | 0.012848 | 0.039526 | -0.026678 | 157 | 33 | 1.000000 | 1.000000 |
| strong utility action | 111 | 0.584 | 0.010982 | 0.033063 | -0.022081 | 78 | 33 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 100 | 0.526 | 0.010032 | 0.028938 | -0.018907 | 67 | 33 | 1.000000 | 1.000000 |
| recent utility last 5s | 11 | 0.058 | 0.019621 | 0.070557 | -0.050936 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 190 | 1.000 | 0.012848 | 0.039526 | -0.026678 | 157 | 33 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `17.5s`, rows `16`
- `29.0s` - `70.5s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.0`, LSTM `0.0436`, XGBoost `0.1473`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.0460`, XGBoost `0.1471`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.0493`, XGBoost `0.1500`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.0321`, XGBoost `0.1242`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.0290`, XGBoost `0.1207`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.0300`, XGBoost `0.1179`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0347`, XGBoost `0.1175`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.0281`, XGBoost `0.1107`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0349`, XGBoost `0.1174`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.0292`, XGBoost `0.1109`, closer `lstm`, smoke `0`, inferno `3`, utility_damage `0.0`, recent_utility `0`
