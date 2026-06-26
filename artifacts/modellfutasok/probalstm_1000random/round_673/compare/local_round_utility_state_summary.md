# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major/blasttv-austin-major-2025-vitality-vs-the-mongolz-bo3-vhm_UWcBfYfYcOeLh9JIDA/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `8`
- rows: `101`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 101 | 1.000 | 0.428379 | 0.627271 | -0.198891 | 94 | 7 | 0.514851 | 0.257426 |
| active/recent utility | 101 | 1.000 | 0.428379 | 0.627271 | -0.198891 | 94 | 7 | 0.514851 | 0.257426 |
| strong utility action | 88 | 0.871 | 0.417913 | 0.645856 | -0.227944 | 88 | 0 | 0.534091 | 0.238636 |
| utility damage | 11 | 0.109 | 0.658320 | 0.791725 | -0.133404 | 11 | 0 | 0.090909 | 0.000000 |
| active smoke/inferno | 88 | 0.871 | 0.417913 | 0.645856 | -0.227944 | 88 | 0 | 0.534091 | 0.238636 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 101 | 1.000 | 0.428379 | 0.627271 | -0.198891 | 94 | 7 | 0.514851 | 0.257426 |

## Active Smoke/Inferno Intervals

- `6.5s` - `50.0s`, rows `88`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `50.0`, LSTM `0.1357`, XGBoost `0.5977`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.1983`, XGBoost `0.6427`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2197`, XGBoost `0.6596`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0478`, XGBoost `0.4648`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0551`, XGBoost `0.4681`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.4702`, XGBoost `0.8685`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0492`, XGBoost `0.4320`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2822`, XGBoost `0.6596`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.0566`, XGBoost `0.4339`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0575`, XGBoost `0.4334`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
