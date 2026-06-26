# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `16`
- rows: `188`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 188 | 1.000 | 0.699020 | 0.733669 | -0.034649 | 41 | 147 | 1.000000 | 0.893617 |
| active/recent utility | 188 | 1.000 | 0.699020 | 0.733669 | -0.034649 | 41 | 147 | 1.000000 | 0.893617 |
| strong utility action | 174 | 0.926 | 0.712402 | 0.752744 | -0.040342 | 27 | 147 | 1.000000 | 0.959770 |
| utility damage | 20 | 0.106 | 0.676389 | 0.736624 | -0.060235 | 0 | 20 | 1.000000 | 1.000000 |
| active smoke/inferno | 174 | 0.926 | 0.712402 | 0.752744 | -0.040342 | 27 | 147 | 1.000000 | 0.959770 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 188 | 1.000 | 0.699020 | 0.733669 | -0.034649 | 41 | 147 | 1.000000 | 0.893617 |

## Active Smoke/Inferno Intervals

- `7.0s` - `93.5s`, rows `174`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.0`, LSTM `0.6174`, XGBoost `0.7364`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.6266`, XGBoost `0.5079`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.6257`, XGBoost `0.5090`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.6261`, XGBoost `0.7405`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.6221`, XGBoost `0.7364`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.6211`, XGBoost `0.5083`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.6264`, XGBoost `0.7364`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.6350`, XGBoost `0.7440`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `12.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.6324`, XGBoost `0.7402`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.6335`, XGBoost `0.7405`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
