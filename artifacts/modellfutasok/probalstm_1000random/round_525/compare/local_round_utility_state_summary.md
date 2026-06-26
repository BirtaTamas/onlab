# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `6`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.617008 | 0.656487 | -0.039480 | 33 | 132 | 0.630303 | 0.890909 |
| active/recent utility | 165 | 1.000 | 0.617008 | 0.656487 | -0.039480 | 33 | 132 | 0.630303 | 0.890909 |
| strong utility action | 104 | 0.630 | 0.504669 | 0.555399 | -0.050730 | 27 | 77 | 0.548077 | 0.826923 |
| utility damage | 10 | 0.061 | 0.631138 | 0.739485 | -0.108347 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 104 | 0.630 | 0.504669 | 0.555399 | -0.050730 | 27 | 77 | 0.548077 | 0.826923 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 165 | 1.000 | 0.617008 | 0.656487 | -0.039480 | 33 | 132 | 0.630303 | 0.890909 |

## Active Smoke/Inferno Intervals

- `7.0s` - `58.5s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `29.5`, LSTM `0.5558`, XGBoost `0.7790`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5873`, XGBoost `0.7836`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.3685`, XGBoost `0.5559`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.3752`, XGBoost `0.5559`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.4058`, XGBoost `0.5860`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `40.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.5808`, XGBoost `0.7334`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `31.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.3849`, XGBoost `0.5341`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.3870`, XGBoost `0.5355`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `32.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.3855`, XGBoost `0.5325`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.5789`, XGBoost `0.7248`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `22.0`, recent_utility `0`
