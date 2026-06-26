# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-mouz-vs-vitality-bo3-pYxpz34IEN-t8y4DgB-MSD/mouz-vs-vitality-m3-train.csv`
- round_num: `11`
- rows: `209`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 209 | 1.000 | 0.620208 | 0.641066 | -0.020858 | 39 | 170 | 0.933014 | 0.990431 |
| active/recent utility | 209 | 1.000 | 0.620208 | 0.641066 | -0.020858 | 39 | 170 | 0.933014 | 0.990431 |
| strong utility action | 178 | 0.852 | 0.601073 | 0.622608 | -0.021534 | 30 | 148 | 0.921348 | 0.988764 |
| utility damage | 10 | 0.048 | 0.597183 | 0.596095 | 0.001088 | 6 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 178 | 0.852 | 0.601073 | 0.622608 | -0.021534 | 30 | 148 | 0.921348 | 0.988764 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 209 | 1.000 | 0.620208 | 0.641066 | -0.020858 | 39 | 170 | 0.933014 | 0.990431 |

## Active Smoke/Inferno Intervals

- `8.5s` - `74.0s`, rows `132`
- `75.5s` - `98.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `87.0`, LSTM `0.5851`, XGBoost `0.6898`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.6032`, XGBoost `0.6898`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `10.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.4545`, XGBoost `0.5394`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.4783`, XGBoost `0.5407`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.5744`, XGBoost `0.6366`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.5555`, XGBoost `0.6176`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.5602`, XGBoost `0.6218`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.5404`, XGBoost `0.6016`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5607`, XGBoost `0.6210`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.5`, LSTM `0.4835`, XGBoost `0.5416`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
