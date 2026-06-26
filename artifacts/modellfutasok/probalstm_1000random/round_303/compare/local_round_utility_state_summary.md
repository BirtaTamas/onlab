# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-legacy-vs-mouz-bo3-D4mE8XcULbH9iT3IhMhdJY/legacy-vs-mouz-m1-ancient.csv`
- round_num: `6`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.862022 | 0.887794 | -0.025772 | 27 | 203 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.862022 | 0.887794 | -0.025772 | 27 | 203 | 1.000000 | 1.000000 |
| strong utility action | 168 | 0.730 | 0.859136 | 0.885370 | -0.026234 | 20 | 148 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.043 | 0.549144 | 0.550445 | -0.001301 | 4 | 6 | 1.000000 | 1.000000 |
| active smoke/inferno | 168 | 0.730 | 0.859136 | 0.885370 | -0.026234 | 20 | 148 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.862022 | 0.887794 | -0.025772 | 27 | 203 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `46.0s`, rows `80`
- `62.0s` - `83.5s`, rows `44`
- `89.5s` - `111.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.0`, LSTM `0.5985`, XGBoost `0.7148`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.6000`, XGBoost `0.7153`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6017`, XGBoost `0.7152`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.6207`, XGBoost `0.7131`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6259`, XGBoost `0.7084`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6349`, XGBoost `0.7135`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.6336`, XGBoost `0.7114`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6344`, XGBoost `0.7079`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.8361`, XGBoost `0.8970`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6597`, XGBoost `0.7135`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
