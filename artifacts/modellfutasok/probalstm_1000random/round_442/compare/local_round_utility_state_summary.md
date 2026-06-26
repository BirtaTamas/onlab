# Local Round Utility State Analysis

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `2`
- rows: `162`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 162 | 1.000 | 0.080770 | 0.075522 | 0.005249 | 131 | 31 | 1.000000 | 1.000000 |
| active/recent utility | 162 | 1.000 | 0.080770 | 0.075522 | 0.005249 | 131 | 31 | 1.000000 | 1.000000 |
| strong utility action | 102 | 0.630 | 0.088967 | 0.085352 | 0.003615 | 83 | 19 | 1.000000 | 1.000000 |
| utility damage | 24 | 0.148 | 0.018854 | 0.023432 | -0.004578 | 23 | 1 | 1.000000 | 1.000000 |
| active smoke/inferno | 95 | 0.586 | 0.071619 | 0.070520 | 0.001099 | 81 | 14 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.062 | 0.283946 | 0.283475 | 0.000471 | 5 | 5 | 1.000000 | 1.000000 |
| flash effect present | 162 | 1.000 | 0.080770 | 0.075522 | 0.005249 | 131 | 31 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `56.5s`, rows `95`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.4540`, XGBoost `0.2649`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.4391`, XGBoost `0.2652`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.4211`, XGBoost `0.2635`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.3835`, XGBoost `0.2641`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.1776`, XGBoost `0.2872`, closer `lstm`, smoke `0`, inferno `2`, utility_damage `2.0`, recent_utility `1`
- seconds `11.0`, LSTM `0.1485`, XGBoost `0.2552`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `22.0`, recent_utility `0`
- seconds `6.0`, LSTM `0.3852`, XGBoost `0.2877`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `12.0`, LSTM `0.1650`, XGBoost `0.2583`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `22.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.1641`, XGBoost `0.2556`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `22.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.3697`, XGBoost `0.2877`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
