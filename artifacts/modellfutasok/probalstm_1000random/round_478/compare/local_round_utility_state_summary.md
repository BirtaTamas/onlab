# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `4`
- rows: `207`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 207 | 1.000 | 0.488198 | 0.469015 | 0.019183 | 96 | 111 | 0.376812 | 0.439614 |
| active/recent utility | 207 | 1.000 | 0.488198 | 0.469015 | 0.019183 | 96 | 111 | 0.376812 | 0.439614 |
| strong utility action | 112 | 0.541 | 0.715508 | 0.691356 | 0.024152 | 25 | 87 | 0.098214 | 0.142857 |
| utility damage | 34 | 0.164 | 0.787596 | 0.760325 | 0.027272 | 4 | 30 | 0.000000 | 0.000000 |
| active smoke/inferno | 109 | 0.527 | 0.714780 | 0.690049 | 0.024732 | 24 | 85 | 0.100917 | 0.146789 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 207 | 1.000 | 0.488198 | 0.469015 | 0.019183 | 96 | 111 | 0.376812 | 0.439614 |

## Active Smoke/Inferno Intervals

- `11.5s` - `60.0s`, rows `98`
- `65.5s` - `70.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `65.5`, LSTM `0.3825`, XGBoost `0.1817`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6250`, XGBoost `0.4693`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.8257`, XGBoost `0.6786`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.7964`, XGBoost `0.6771`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `55.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.8029`, XGBoost `0.7008`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.1355`, XGBoost `0.0416`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.7178`, XGBoost `0.6286`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1297`, XGBoost `0.0419`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.7842`, XGBoost `0.8696`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.7104`, XGBoost `0.6336`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
