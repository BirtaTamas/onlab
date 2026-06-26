# Local Round Utility State Analysis

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-3dmax-vs-faze-bo3-oPmZd_fSjJ93cG46OkNLxM/3dmax-vs-faze-m3-dust2.csv`
- round_num: `10`
- rows: `205`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.259571 | 0.316679 | -0.057108 | 194 | 11 | 0.697561 | 0.687805 |
| active/recent utility | 205 | 1.000 | 0.259571 | 0.316679 | -0.057108 | 194 | 11 | 0.697561 | 0.687805 |
| strong utility action | 172 | 0.839 | 0.266624 | 0.324986 | -0.058362 | 161 | 11 | 0.697674 | 0.686047 |
| utility damage | 13 | 0.063 | 0.693271 | 0.772797 | -0.079527 | 13 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 172 | 0.839 | 0.266624 | 0.324986 | -0.058362 | 161 | 11 | 0.697674 | 0.686047 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 205 | 1.000 | 0.259571 | 0.316679 | -0.057108 | 194 | 11 | 0.697561 | 0.687805 |

## Active Smoke/Inferno Intervals

- `5.0s` - `90.5s`, rows `172`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.0`, LSTM `0.5195`, XGBoost `0.7525`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5852`, XGBoost `0.7727`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5985`, XGBoost `0.7831`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5903`, XGBoost `0.7738`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5762`, XGBoost `0.7580`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5782`, XGBoost `0.7586`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5857`, XGBoost `0.7589`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.6467`, XGBoost `0.8031`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5329`, XGBoost `0.3793`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.4203`, XGBoost `0.5683`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
