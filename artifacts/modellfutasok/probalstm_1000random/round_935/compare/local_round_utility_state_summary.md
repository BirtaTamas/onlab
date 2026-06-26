# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-vitality-bo3-ZpOL0o26IrRvvgFRbFxVou/lynn-vision-vs-vitality-m1-dust2.csv`
- round_num: `14`
- rows: `232`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 232 | 1.000 | 0.942111 | 0.961157 | -0.019046 | 22 | 210 | 1.000000 | 1.000000 |
| active/recent utility | 232 | 1.000 | 0.942111 | 0.961157 | -0.019046 | 22 | 210 | 1.000000 | 1.000000 |
| strong utility action | 89 | 0.384 | 0.942686 | 0.966835 | -0.024149 | 0 | 89 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 89 | 0.384 | 0.942686 | 0.966835 | -0.024149 | 0 | 89 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 232 | 1.000 | 0.942111 | 0.961157 | -0.019046 | 22 | 210 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `31.5s`, rows `45`
- `32.5s` - `54.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `42.5`, LSTM `0.9001`, XGBoost `0.9660`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.8941`, XGBoost `0.9588`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.9071`, XGBoost `0.9660`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.9047`, XGBoost `0.9588`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.9056`, XGBoost `0.9589`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.9134`, XGBoost `0.9660`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.9136`, XGBoost `0.9650`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.9077`, XGBoost `0.9589`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.9151`, XGBoost `0.9660`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.9156`, XGBoost `0.9660`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
