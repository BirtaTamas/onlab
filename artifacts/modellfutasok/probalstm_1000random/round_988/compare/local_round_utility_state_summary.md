# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `10`
- rows: `132`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 132 | 1.000 | 0.602835 | 0.631871 | -0.029036 | 17 | 115 | 0.477273 | 0.909091 |
| active/recent utility | 132 | 1.000 | 0.602835 | 0.631871 | -0.029036 | 17 | 115 | 0.477273 | 0.909091 |
| strong utility action | 114 | 0.864 | 0.621464 | 0.651667 | -0.030203 | 16 | 98 | 0.543860 | 0.956140 |
| utility damage | 20 | 0.152 | 0.692240 | 0.737744 | -0.045504 | 2 | 18 | 0.950000 | 1.000000 |
| active smoke/inferno | 114 | 0.864 | 0.621464 | 0.651667 | -0.030203 | 16 | 98 | 0.543860 | 0.956140 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 132 | 1.000 | 0.602835 | 0.631871 | -0.029036 | 17 | 115 | 0.477273 | 0.909091 |

## Active Smoke/Inferno Intervals

- `8.5s` - `38.5s`, rows `61`
- `39.5s` - `65.5s`, rows `53`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `38.0`, LSTM `0.6306`, XGBoost `0.7471`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.6359`, XGBoost `0.7471`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.6475`, XGBoost `0.7523`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.6429`, XGBoost `0.7471`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6478`, XGBoost `0.7517`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.6535`, XGBoost `0.7524`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `5.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.4263`, XGBoost `0.5231`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4443`, XGBoost `0.5066`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.5254`, XGBoost `0.5867`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.4468`, XGBoost `0.5066`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
