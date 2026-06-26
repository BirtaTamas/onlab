# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `15`
- rows: `173`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.576381 | 0.647734 | -0.071353 | 3 | 170 | 0.520231 | 0.803468 |
| active/recent utility | 173 | 1.000 | 0.576381 | 0.647734 | -0.071353 | 3 | 170 | 0.520231 | 0.803468 |
| strong utility action | 168 | 0.971 | 0.579336 | 0.652592 | -0.073256 | 3 | 165 | 0.535714 | 0.827381 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 160 | 0.925 | 0.584982 | 0.660984 | -0.076002 | 3 | 157 | 0.562500 | 0.868750 |
| recent utility last 5s | 20 | 0.116 | 0.418612 | 0.494355 | -0.075742 | 0 | 20 | 0.000000 | 0.450000 |
| flash effect present | 173 | 1.000 | 0.576381 | 0.647734 | -0.071353 | 3 | 170 | 0.520231 | 0.803468 |

## Active Smoke/Inferno Intervals

- `6.5s` - `86.0s`, rows `160`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.3270`, XGBoost `0.5055`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `18.5`, LSTM `0.3438`, XGBoost `0.5055`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `19.5`, LSTM `0.3598`, XGBoost `0.5042`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `61.5`, LSTM `0.5878`, XGBoost `0.7307`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.5956`, XGBoost `0.7307`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.5969`, XGBoost `0.7317`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3747`, XGBoost `0.5091`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `59.5`, LSTM `0.5983`, XGBoost `0.7318`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.6051`, XGBoost `0.7381`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.5996`, XGBoost `0.7311`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
