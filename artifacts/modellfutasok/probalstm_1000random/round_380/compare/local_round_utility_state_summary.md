# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `6`
- rows: `305`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 305 | 1.000 | 0.366531 | 0.358643 | 0.007889 | 149 | 156 | 0.511475 | 0.557377 |
| active/recent utility | 305 | 1.000 | 0.366531 | 0.358643 | 0.007889 | 149 | 156 | 0.511475 | 0.557377 |
| strong utility action | 227 | 0.744 | 0.443319 | 0.432216 | 0.011104 | 86 | 141 | 0.427313 | 0.502203 |
| utility damage | 10 | 0.033 | 0.562198 | 0.527520 | 0.034678 | 0 | 10 | 0.000000 | 0.000000 |
| active smoke/inferno | 217 | 0.711 | 0.463651 | 0.452046 | 0.011605 | 85 | 132 | 0.400922 | 0.479263 |
| recent utility last 5s | 10 | 0.033 | 0.002128 | 0.001892 | 0.000236 | 1 | 9 | 1.000000 | 1.000000 |
| flash effect present | 305 | 1.000 | 0.366531 | 0.358643 | 0.007889 | 149 | 156 | 0.511475 | 0.557377 |

## Active Smoke/Inferno Intervals

- `9.5s` - `57.5s`, rows `97`
- `59.5s` - `119.0s`, rows `120`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `108.0`, LSTM `0.5084`, XGBoost `0.2734`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.5`, LSTM `0.5214`, XGBoost `0.3799`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `107.0`, LSTM `0.5144`, XGBoost `0.3768`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.3656`, XGBoost `0.4963`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.3701`, XGBoost `0.4978`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.5251`, XGBoost `0.3980`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.3724`, XGBoost `0.4990`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.3730`, XGBoost `0.4963`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.3802`, XGBoost `0.5016`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.3823`, XGBoost `0.4999`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
