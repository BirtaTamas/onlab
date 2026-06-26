# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-heroic-bo3-ReZhZ3UThZvWjRyUeuYiIR/falcons-vs-heroic-m3-dust2.csv`
- round_num: `15`
- rows: `149`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 149 | 1.000 | 0.687975 | 0.725655 | -0.037680 | 28 | 121 | 1.000000 | 1.000000 |
| active/recent utility | 149 | 1.000 | 0.687975 | 0.725655 | -0.037680 | 28 | 121 | 1.000000 | 1.000000 |
| strong utility action | 146 | 0.980 | 0.687418 | 0.725731 | -0.038313 | 26 | 120 | 1.000000 | 1.000000 |
| utility damage | 41 | 0.275 | 0.711354 | 0.765090 | -0.053736 | 16 | 25 | 1.000000 | 1.000000 |
| active smoke/inferno | 136 | 0.913 | 0.687848 | 0.728379 | -0.040531 | 23 | 113 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.067 | 0.587619 | 0.592036 | -0.004416 | 5 | 5 | 1.000000 | 1.000000 |
| flash effect present | 149 | 1.000 | 0.687975 | 0.725655 | -0.037680 | 28 | 121 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `4.0s` - `43.0s`, rows `79`
- `45.5s` - `72.5s`, rows `55`
- `73.5s` - `74.0s`, rows `2`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `64.0`, LSTM `0.6476`, XGBoost `0.8906`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.6481`, XGBoost `0.8906`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.6626`, XGBoost `0.8906`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.6677`, XGBoost `0.8906`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.7038`, XGBoost `0.9064`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.7039`, XGBoost `0.9056`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.6949`, XGBoost `0.8814`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.7204`, XGBoost `0.9064`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.7097`, XGBoost `0.8895`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.7111`, XGBoost `0.8906`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
