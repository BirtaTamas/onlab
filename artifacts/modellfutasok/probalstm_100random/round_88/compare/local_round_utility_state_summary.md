# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-aurora-vs-heroic-bo3-Z9VnvF_JkEDX6y_HyMsFXx/aurora-vs-heroic-m3-mirage.csv`
- round_num: `10`
- rows: `158`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 158 | 1.000 | 0.953779 | 0.981432 | -0.027653 | 0 | 158 | 1.000000 | 1.000000 |
| active/recent utility | 158 | 1.000 | 0.953779 | 0.981432 | -0.027653 | 0 | 158 | 1.000000 | 1.000000 |
| strong utility action | 103 | 0.652 | 0.953045 | 0.981705 | -0.028661 | 0 | 103 | 1.000000 | 1.000000 |
| utility damage | 31 | 0.196 | 0.948786 | 0.980390 | -0.031603 | 0 | 31 | 1.000000 | 1.000000 |
| active smoke/inferno | 103 | 0.652 | 0.953045 | 0.981705 | -0.028661 | 0 | 103 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 158 | 1.000 | 0.953779 | 0.981432 | -0.027653 | 0 | 158 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `35.0s`, rows `54`
- `44.5s` - `68.5s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.5`, LSTM `0.9360`, XGBoost `0.9824`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.9367`, XGBoost `0.9820`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.9369`, XGBoost `0.9821`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.9396`, XGBoost `0.9820`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.9405`, XGBoost `0.9824`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.9383`, XGBoost `0.9800`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.9412`, XGBoost `0.9824`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.9400`, XGBoost `0.9813`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.9399`, XGBoost `0.9807`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.9417`, XGBoost `0.9824`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
