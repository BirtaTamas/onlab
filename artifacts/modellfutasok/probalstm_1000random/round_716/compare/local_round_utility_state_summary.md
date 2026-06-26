# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-gamerlegion-vs-complexity-bo3-A8nOd44IyEYHGVOxrkExMv/gamerlegion-vs-complexity-m1-inferno.csv`
- round_num: `2`
- rows: `222`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 222 | 1.000 | 0.508784 | 0.567185 | -0.058401 | 77 | 145 | 0.707207 | 0.765766 |
| active/recent utility | 222 | 1.000 | 0.508784 | 0.567185 | -0.058401 | 77 | 145 | 0.707207 | 0.765766 |
| strong utility action | 190 | 0.856 | 0.464226 | 0.529976 | -0.065749 | 67 | 123 | 0.657895 | 0.726316 |
| utility damage | 31 | 0.140 | 0.447070 | 0.478766 | -0.031696 | 14 | 17 | 0.645161 | 0.645161 |
| active smoke/inferno | 180 | 0.811 | 0.456825 | 0.527564 | -0.070738 | 57 | 123 | 0.638889 | 0.711111 |
| recent utility last 5s | 10 | 0.045 | 0.597449 | 0.573392 | 0.024057 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 222 | 1.000 | 0.508784 | 0.567185 | -0.058401 | 77 | 145 | 0.707207 | 0.765766 |

## Active Smoke/Inferno Intervals

- `11.0s` - `100.5s`, rows `180`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.0`, LSTM `0.2422`, XGBoost `0.5831`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.2704`, XGBoost `0.5877`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.2983`, XGBoost `0.5831`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.1954`, XGBoost `0.4559`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.1976`, XGBoost `0.4547`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.1982`, XGBoost `0.4551`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.2965`, XGBoost `0.5532`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.2017`, XGBoost `0.4547`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `88.5`, LSTM `0.1796`, XGBoost `0.4323`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.2130`, XGBoost `0.4547`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
