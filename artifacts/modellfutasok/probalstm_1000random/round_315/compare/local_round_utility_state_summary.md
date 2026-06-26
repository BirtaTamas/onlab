# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `8`
- rows: `184`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 184 | 1.000 | 0.043429 | 0.046854 | -0.003424 | 146 | 38 | 1.000000 | 1.000000 |
| active/recent utility | 184 | 1.000 | 0.043429 | 0.046854 | -0.003424 | 146 | 38 | 1.000000 | 1.000000 |
| strong utility action | 74 | 0.402 | 0.102433 | 0.106119 | -0.003686 | 36 | 38 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.054 | 0.012927 | 0.012958 | -0.000031 | 5 | 5 | 1.000000 | 1.000000 |
| active smoke/inferno | 57 | 0.310 | 0.105372 | 0.092559 | 0.012813 | 19 | 38 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.109 | 0.100408 | 0.151814 | -0.051406 | 19 | 1 | 1.000000 | 1.000000 |
| flash effect present | 184 | 1.000 | 0.043429 | 0.046854 | -0.003424 | 146 | 38 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `10.0s` - `38.0s`, rows `57`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `1.0`, LSTM `0.0685`, XGBoost `0.1519`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `1.5`, LSTM `0.0700`, XGBoost `0.1501`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.0`, LSTM `0.0748`, XGBoost `0.1501`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `0.5`, LSTM `0.0782`, XGBoost `0.1527`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `2.5`, LSTM `0.0780`, XGBoost `0.1501`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.5`, LSTM `0.0791`, XGBoost `0.1501`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `3.0`, LSTM `0.0809`, XGBoost `0.1501`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.0809`, XGBoost `0.1501`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `24.0`, LSTM `0.2073`, XGBoost `0.1387`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.2107`, XGBoost `0.1440`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `12.0`, recent_utility `0`
