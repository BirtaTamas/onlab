# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `2`
- rows: `202`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 202 | 1.000 | 0.832793 | 0.897166 | -0.064374 | 0 | 202 | 0.910891 | 1.000000 |
| active/recent utility | 202 | 1.000 | 0.832793 | 0.897166 | -0.064374 | 0 | 202 | 0.910891 | 1.000000 |
| strong utility action | 80 | 0.396 | 0.904621 | 0.935561 | -0.030940 | 0 | 80 | 0.937500 | 1.000000 |
| utility damage | 10 | 0.050 | 0.942960 | 0.982410 | -0.039450 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 80 | 0.396 | 0.904621 | 0.935561 | -0.030940 | 0 | 80 | 0.937500 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 202 | 1.000 | 0.832793 | 0.897166 | -0.064374 | 0 | 202 | 0.910891 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `49.0s`, rows `80`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.3853`, XGBoost `0.5288`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.3883`, XGBoost `0.5292`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4171`, XGBoost `0.5326`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.4276`, XGBoost `0.5326`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.4424`, XGBoost `0.5326`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.6648`, XGBoost `0.7317`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.6740`, XGBoost `0.7344`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.9042`, XGBoost `0.9626`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.9283`, XGBoost `0.9792`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.9337`, XGBoost `0.9831`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `8.0`, recent_utility `0`
