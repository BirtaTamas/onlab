# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `18`
- rows: `147`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 147 | 1.000 | 0.642380 | 0.615911 | 0.026469 | 117 | 30 | 1.000000 | 1.000000 |
| active/recent utility | 147 | 1.000 | 0.642380 | 0.615911 | 0.026469 | 117 | 30 | 1.000000 | 1.000000 |
| strong utility action | 144 | 0.980 | 0.642915 | 0.616689 | 0.026227 | 114 | 30 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.136 | 0.655433 | 0.628355 | 0.027078 | 13 | 7 | 1.000000 | 1.000000 |
| active smoke/inferno | 134 | 0.912 | 0.642512 | 0.620009 | 0.022503 | 104 | 30 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.136 | 0.620487 | 0.578558 | 0.041929 | 16 | 4 | 1.000000 | 1.000000 |
| flash effect present | 147 | 1.000 | 0.642380 | 0.615911 | 0.026469 | 117 | 30 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `73.0s`, rows `134`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `2.0`, LSTM `0.6525`, XGBoost `0.5724`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `5.5`, LSTM `0.6516`, XGBoost `0.5717`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `2.5`, LSTM `0.6515`, XGBoost `0.5724`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `10.0`, LSTM `0.6516`, XGBoost `0.5726`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.6497`, XGBoost `0.5724`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `13.0`, LSTM `0.6556`, XGBoost `0.5797`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.6477`, XGBoost `0.5717`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `4.0`, LSTM `0.6481`, XGBoost `0.5724`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `16.5`, LSTM `0.6579`, XGBoost `0.5829`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `11.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.6471`, XGBoost `0.5724`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
