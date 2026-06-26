# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-faze-vs-aurora-bo3-OcxcOl9bFIHQQ2588nwUWG/faze-vs-aurora-m2-mirage.csv`
- round_num: `5`
- rows: `192`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 192 | 1.000 | 0.800315 | 0.826836 | -0.026521 | 48 | 144 | 1.000000 | 1.000000 |
| active/recent utility | 192 | 1.000 | 0.800315 | 0.826836 | -0.026521 | 48 | 144 | 1.000000 | 1.000000 |
| strong utility action | 134 | 0.698 | 0.789535 | 0.812727 | -0.023193 | 42 | 92 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.104 | 0.727184 | 0.763332 | -0.036148 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 134 | 0.698 | 0.789535 | 0.812727 | -0.023193 | 42 | 92 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 192 | 1.000 | 0.800315 | 0.826836 | -0.026521 | 48 | 144 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `74.5s`, rows `134`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `67.0`, LSTM `0.7285`, XGBoost `0.8408`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.7314`, XGBoost `0.8408`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.7370`, XGBoost `0.8408`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.7462`, XGBoost `0.8408`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.8362`, XGBoost `0.9225`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.8425`, XGBoost `0.9225`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.8660`, XGBoost `0.9428`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.8483`, XGBoost `0.9225`, closer `xgboost`, smoke `4`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.7044`, XGBoost `0.7770`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `37.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6955`, XGBoost `0.7678`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
