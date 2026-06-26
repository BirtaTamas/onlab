# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-faze-vs-aurora-bo3-ZssSxRC3p7Nn5A_BOLQ-lD/faze-vs-aurora-m2-mirage.csv`
- round_num: `8`
- rows: `144`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 144 | 1.000 | 0.561966 | 0.541775 | 0.020191 | 103 | 41 | 0.694444 | 0.652778 |
| active/recent utility | 144 | 1.000 | 0.561966 | 0.541775 | 0.020191 | 103 | 41 | 0.694444 | 0.652778 |
| strong utility action | 131 | 0.910 | 0.551000 | 0.538932 | 0.012068 | 90 | 41 | 0.664122 | 0.618321 |
| utility damage | 44 | 0.306 | 0.575794 | 0.500326 | 0.075468 | 42 | 2 | 0.727273 | 0.522727 |
| active smoke/inferno | 131 | 0.910 | 0.551000 | 0.538932 | 0.012068 | 90 | 41 | 0.664122 | 0.618321 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 144 | 1.000 | 0.561966 | 0.541775 | 0.020191 | 103 | 41 | 0.694444 | 0.652778 |

## Active Smoke/Inferno Intervals

- `6.5s` - `71.5s`, rows `131`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `66.0`, LSTM `0.4376`, XGBoost `0.7883`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.4392`, XGBoost `0.7858`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.4803`, XGBoost `0.7887`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.0`, LSTM `0.1532`, XGBoost `0.4360`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.1669`, XGBoost `0.4360`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.1339`, XGBoost `0.3964`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1268`, XGBoost `0.3864`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.1115`, XGBoost `0.3645`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.1323`, XGBoost `0.3415`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5221`, XGBoost `0.3519`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `14.0`, recent_utility `0`
