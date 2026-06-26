# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-faze-vs-bcgame-bo3-daIGc_M_y7Qq42fF9AoQhi/faze-vs-bcgame-m2-anubis.csv`
- round_num: `8`
- rows: `138`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 138 | 1.000 | 0.619703 | 0.622086 | -0.002383 | 56 | 82 | 0.710145 | 0.543478 |
| active/recent utility | 138 | 1.000 | 0.619703 | 0.622086 | -0.002383 | 56 | 82 | 0.710145 | 0.543478 |
| strong utility action | 135 | 0.978 | 0.618741 | 0.621423 | -0.002682 | 55 | 80 | 0.703704 | 0.540741 |
| utility damage | 17 | 0.123 | 0.452959 | 0.417186 | 0.035773 | 11 | 6 | 0.470588 | 0.117647 |
| active smoke/inferno | 124 | 0.899 | 0.626805 | 0.632461 | -0.005655 | 47 | 77 | 0.693548 | 0.564516 |
| recent utility last 5s | 21 | 0.152 | 0.594858 | 0.577096 | 0.017761 | 12 | 9 | 0.904762 | 0.571429 |
| flash effect present | 138 | 1.000 | 0.619703 | 0.622086 | -0.002383 | 56 | 82 | 0.710145 | 0.543478 |

## Active Smoke/Inferno Intervals

- `6.5s` - `68.0s`, rows `124`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.5`, LSTM `0.1886`, XGBoost `0.3661`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.2052`, XGBoost `0.3657`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.5209`, XGBoost `0.3699`, closer `lstm`, smoke `2`, inferno `4`, utility_damage `48.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.1749`, XGBoost `0.3226`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.2296`, XGBoost `0.3678`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.2589`, XGBoost `0.3905`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.2703`, XGBoost `0.3877`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.2751`, XGBoost `0.3908`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2578`, XGBoost `0.3659`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.7521`, XGBoost `0.6443`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
