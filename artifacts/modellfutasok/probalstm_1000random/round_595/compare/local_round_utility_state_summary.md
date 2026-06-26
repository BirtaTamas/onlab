# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-legacy-vs-vitality-bo3-43WNFDazpfbmBN3Sj5hWmP/vitality-vs-legacy-m2-dust2.csv`
- round_num: `18`
- rows: `196`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.773135 | 0.833216 | -0.060082 | 164 | 32 | 0.020408 | 0.020408 |
| active/recent utility | 196 | 1.000 | 0.773135 | 0.833216 | -0.060082 | 164 | 32 | 0.020408 | 0.020408 |
| strong utility action | 188 | 0.959 | 0.773289 | 0.834893 | -0.061605 | 156 | 32 | 0.021277 | 0.021277 |
| utility damage | 31 | 0.158 | 0.692222 | 0.732512 | -0.040289 | 28 | 3 | 0.129032 | 0.129032 |
| active smoke/inferno | 177 | 0.903 | 0.779274 | 0.840277 | -0.061003 | 145 | 32 | 0.016949 | 0.016949 |
| recent utility last 5s | 10 | 0.051 | 0.720941 | 0.793290 | -0.072350 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 196 | 1.000 | 0.773135 | 0.833216 | -0.060082 | 164 | 32 | 0.020408 | 0.020408 |

## Active Smoke/Inferno Intervals

- `9.0s` - `97.0s`, rows `177`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `82.0`, LSTM `0.6797`, XGBoost `0.9190`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `82.5`, LSTM `0.6983`, XGBoost `0.9195`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `81.5`, LSTM `0.7012`, XGBoost `0.9190`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `83.0`, LSTM `0.7163`, XGBoost `0.9207`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.7260`, XGBoost `0.9253`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.7374`, XGBoost `0.9348`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.7256`, XGBoost `0.9190`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.7035`, XGBoost `0.8964`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.7375`, XGBoost `0.9291`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.7492`, XGBoost `0.9398`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
