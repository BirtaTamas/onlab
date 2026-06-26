# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `10`
- rows: `199`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 199 | 1.000 | 0.679195 | 0.685508 | -0.006313 | 96 | 103 | 1.000000 | 1.000000 |
| active/recent utility | 199 | 1.000 | 0.679195 | 0.685508 | -0.006313 | 96 | 103 | 1.000000 | 1.000000 |
| strong utility action | 187 | 0.940 | 0.685905 | 0.695878 | -0.009974 | 84 | 103 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 177 | 0.889 | 0.693770 | 0.705412 | -0.011642 | 74 | 103 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.101 | 0.708323 | 0.699168 | 0.009155 | 14 | 6 | 1.000000 | 1.000000 |
| flash effect present | 199 | 1.000 | 0.679195 | 0.685508 | -0.006313 | 96 | 103 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `11.0s` - `99.0s`, rows `177`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `98.0`, LSTM `0.6915`, XGBoost `0.8945`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.7021`, XGBoost `0.8945`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.7304`, XGBoost `0.8945`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.7497`, XGBoost `0.8945`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.7544`, XGBoost `0.8945`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.6458`, XGBoost `0.5237`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.7995`, XGBoost `0.9117`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.6359`, XGBoost `0.5240`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6356`, XGBoost `0.5253`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.8052`, XGBoost `0.9117`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
