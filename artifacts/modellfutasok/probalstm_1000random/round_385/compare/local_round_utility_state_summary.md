# Local Round Utility State Analysis

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `10`
- rows: `128`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 128 | 1.000 | 0.275201 | 0.335166 | -0.059964 | 30 | 98 | 0.140625 | 0.125000 |
| active/recent utility | 128 | 1.000 | 0.275201 | 0.335166 | -0.059964 | 30 | 98 | 0.140625 | 0.125000 |
| strong utility action | 61 | 0.477 | 0.346157 | 0.362827 | -0.016669 | 20 | 41 | 0.196721 | 0.196721 |
| utility damage | 11 | 0.086 | 0.326119 | 0.391200 | -0.065081 | 0 | 11 | 0.000000 | 0.000000 |
| active smoke/inferno | 61 | 0.477 | 0.346157 | 0.362827 | -0.016669 | 20 | 41 | 0.196721 | 0.196721 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 128 | 1.000 | 0.275201 | 0.335166 | -0.059964 | 30 | 98 | 0.140625 | 0.125000 |

## Active Smoke/Inferno Intervals

- `10.5s` - `33.5s`, rows `47`
- `57.0s` - `63.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.4407`, XGBoost `0.2631`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.4431`, XGBoost `0.2699`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.2935`, XGBoost `0.3950`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `18.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.3101`, XGBoost `0.3950`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `18.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.3823`, XGBoost `0.3025`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.8932`, XGBoost `0.9727`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.3116`, XGBoost `0.3901`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `18.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.3155`, XGBoost `0.3936`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.3778`, XGBoost `0.3025`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `3.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.3152`, XGBoost `0.3901`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
