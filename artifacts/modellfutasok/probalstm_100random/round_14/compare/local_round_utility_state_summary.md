# Local Round Utility State Analysis

- csv_path: `processed_full/iem_katowice/iem-katowice-2025-vitality-vs-faze-bo3-hDX5yjYYbla4cw8aPwAYi3/vitality-vs-faze-m1-nuke.csv`
- round_num: `8`
- rows: `102`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 102 | 1.000 | 0.208544 | 0.289213 | -0.080669 | 90 | 12 | 0.784314 | 0.784314 |
| active/recent utility | 102 | 1.000 | 0.208544 | 0.289213 | -0.080669 | 90 | 12 | 0.784314 | 0.784314 |
| strong utility action | 82 | 0.804 | 0.237512 | 0.317558 | -0.080047 | 70 | 12 | 0.731707 | 0.731707 |
| utility damage | 11 | 0.108 | 0.453410 | 0.544320 | -0.090910 | 10 | 1 | 0.454545 | 0.454545 |
| active smoke/inferno | 82 | 0.804 | 0.237512 | 0.317558 | -0.080047 | 70 | 12 | 0.731707 | 0.731707 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 102 | 1.000 | 0.208544 | 0.289213 | -0.080669 | 90 | 12 | 0.784314 | 0.784314 |

## Active Smoke/Inferno Intervals

- `10.0s` - `50.5s`, rows `82`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `19.0`, LSTM `0.5198`, XGBoost `0.7924`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `19.5`, LSTM `0.5225`, XGBoost `0.7777`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.5249`, XGBoost `0.7767`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.5288`, XGBoost `0.7636`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5183`, XGBoost `0.7369`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5269`, XGBoost `0.7369`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.5300`, XGBoost `0.7374`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.5203`, XGBoost `0.7050`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.5243`, XGBoost `0.7050`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.5251`, XGBoost `0.7050`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
