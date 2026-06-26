# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `12`
- rows: `122`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 122 | 1.000 | 0.826612 | 0.868923 | -0.042311 | 10 | 112 | 1.000000 | 1.000000 |
| active/recent utility | 122 | 1.000 | 0.826612 | 0.868923 | -0.042311 | 10 | 112 | 1.000000 | 1.000000 |
| strong utility action | 108 | 0.885 | 0.863755 | 0.910174 | -0.046419 | 4 | 104 | 1.000000 | 1.000000 |
| utility damage | 23 | 0.189 | 0.870068 | 0.933978 | -0.063910 | 0 | 23 | 1.000000 | 1.000000 |
| active smoke/inferno | 108 | 0.885 | 0.863755 | 0.910174 | -0.046419 | 4 | 104 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 122 | 1.000 | 0.826612 | 0.868923 | -0.042311 | 10 | 112 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `60.5s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `17.0`, LSTM `0.8301`, XGBoost `0.9208`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `50.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.8511`, XGBoost `0.9270`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `50.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.8357`, XGBoost `0.9108`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.8506`, XGBoost `0.9257`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `50.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.8494`, XGBoost `0.9215`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `50.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.8505`, XGBoost `0.9216`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `50.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.8525`, XGBoost `0.9226`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `50.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.9176`, XGBoost `0.9871`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.9192`, XGBoost `0.9871`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.9197`, XGBoost `0.9872`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `1.0`, recent_utility `0`
