# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-og-vs-tyloo-ancient-6bJQWEKo0L9rHQMGqH72Vs/og-vs-tyloo-ancient.csv`
- round_num: `14`
- rows: `246`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 246 | 1.000 | 0.630664 | 0.661218 | -0.030554 | 70 | 176 | 0.971545 | 0.975610 |
| active/recent utility | 246 | 1.000 | 0.630664 | 0.661218 | -0.030554 | 70 | 176 | 0.971545 | 0.975610 |
| strong utility action | 191 | 0.776 | 0.604842 | 0.625030 | -0.020188 | 59 | 132 | 0.984293 | 0.968586 |
| utility damage | 21 | 0.085 | 0.564187 | 0.584220 | -0.020032 | 11 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 191 | 0.776 | 0.604842 | 0.625030 | -0.020188 | 59 | 132 | 0.984293 | 0.968586 |
| recent utility last 5s | 10 | 0.041 | 0.542180 | 0.599914 | -0.057734 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 246 | 1.000 | 0.630664 | 0.661218 | -0.030554 | 70 | 176 | 0.971545 | 0.975610 |

## Active Smoke/Inferno Intervals

- `7.0s` - `57.0s`, rows `101`
- `58.0s` - `102.5s`, rows `90`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `91.0`, LSTM `0.6001`, XGBoost `0.4039`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.5992`, XGBoost `0.4039`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.5968`, XGBoost `0.4039`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.0`, LSTM `0.5950`, XGBoost `0.4039`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.0`, LSTM `0.5885`, XGBoost `0.4039`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.5883`, XGBoost `0.4039`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.4608`, XGBoost `0.5991`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.7493`, XGBoost `0.8815`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.4695`, XGBoost `0.5991`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.4888`, XGBoost `0.5997`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
