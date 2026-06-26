# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-g2-bo3-_Q5oHsnKyowoqEbJh5o3f8/furia-vs-g2-m2-overpass.csv`
- round_num: `11`
- rows: `200`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.643797 | 0.764277 | -0.120480 | 2 | 198 | 0.690000 | 0.985000 |
| active/recent utility | 200 | 1.000 | 0.643797 | 0.764277 | -0.120480 | 2 | 198 | 0.690000 | 0.985000 |
| strong utility action | 185 | 0.925 | 0.631686 | 0.756413 | -0.124727 | 2 | 183 | 0.675676 | 0.983784 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 170 | 0.850 | 0.654437 | 0.777948 | -0.123511 | 2 | 168 | 0.735294 | 1.000000 |
| recent utility last 5s | 17 | 0.085 | 0.372095 | 0.513393 | -0.141298 | 0 | 17 | 0.000000 | 0.823529 |
| flash effect present | 200 | 1.000 | 0.643797 | 0.764277 | -0.120480 | 2 | 198 | 0.690000 | 0.985000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `86.0s`, rows `156`
- `90.5s` - `97.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `11.0`, LSTM `0.2412`, XGBoost `0.5165`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.2622`, XGBoost `0.5141`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.6571`, XGBoost `0.8761`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6572`, XGBoost `0.8761`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.6596`, XGBoost `0.8761`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6627`, XGBoost `0.8761`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2960`, XGBoost `0.5081`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `49.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.6662`, XGBoost `0.8766`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.6662`, XGBoost `0.8761`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6690`, XGBoost `0.8764`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
