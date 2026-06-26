# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-the-mongolz-vs-natus-vincere-bo3-C0GZxMhpGHBr28LeyjgICZ/the-mongolz-vs-natus-vincere-m1-mirage.csv`
- round_num: `9`
- rows: `221`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 221 | 1.000 | 0.147797 | 0.233127 | -0.085330 | 218 | 3 | 0.972851 | 0.954751 |
| active/recent utility | 221 | 1.000 | 0.147797 | 0.233127 | -0.085330 | 218 | 3 | 0.972851 | 0.954751 |
| strong utility action | 179 | 0.810 | 0.123268 | 0.202767 | -0.079500 | 176 | 3 | 0.966480 | 0.944134 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 166 | 0.751 | 0.118267 | 0.192731 | -0.074465 | 163 | 3 | 0.963855 | 0.939759 |
| recent utility last 5s | 13 | 0.059 | 0.187127 | 0.330918 | -0.143792 | 13 | 0 | 1.000000 | 1.000000 |
| flash effect present | 221 | 1.000 | 0.147797 | 0.233127 | -0.085330 | 218 | 3 | 0.972851 | 0.954751 |

## Active Smoke/Inferno Intervals

- `7.5s` - `29.0s`, rows `44`
- `46.5s` - `107.0s`, rows `122`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.0177`, XGBoost `0.3688`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.0181`, XGBoost `0.3688`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.0191`, XGBoost `0.3688`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.0205`, XGBoost `0.3674`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.0240`, XGBoost `0.3698`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.0279`, XGBoost `0.3707`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.0275`, XGBoost `0.3696`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.0293`, XGBoost `0.3701`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.0503`, XGBoost `0.3726`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.0164`, XGBoost `0.2841`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
