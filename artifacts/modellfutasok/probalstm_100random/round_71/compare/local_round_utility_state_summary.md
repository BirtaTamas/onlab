# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-mibr-vs-legacy-nuke-uERfHmzId5aHOSWUmDGvHY/mibr-vs-legacy-nuke.csv`
- round_num: `2`
- rows: `243`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 243 | 1.000 | 0.568952 | 0.641742 | -0.072790 | 9 | 234 | 0.876543 | 0.934156 |
| active/recent utility | 243 | 1.000 | 0.568952 | 0.641742 | -0.072790 | 9 | 234 | 0.876543 | 0.934156 |
| strong utility action | 139 | 0.572 | 0.510862 | 0.596148 | -0.085286 | 3 | 136 | 0.812950 | 0.884892 |
| utility damage | 13 | 0.053 | 0.626427 | 0.645580 | -0.019153 | 0 | 13 | 1.000000 | 1.000000 |
| active smoke/inferno | 129 | 0.531 | 0.506944 | 0.591106 | -0.084161 | 3 | 126 | 0.798450 | 0.875969 |
| recent utility last 5s | 10 | 0.041 | 0.561397 | 0.661193 | -0.099796 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 243 | 1.000 | 0.568952 | 0.641742 | -0.072790 | 9 | 234 | 0.876543 | 0.934156 |

## Active Smoke/Inferno Intervals

- `8.0s` - `50.0s`, rows `85`
- `87.5s` - `109.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `98.0`, LSTM `0.2102`, XGBoost `0.5837`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.1690`, XGBoost `0.4932`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.1669`, XGBoost `0.4830`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.1782`, XGBoost `0.4860`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.0`, LSTM `0.3269`, XGBoost `0.6322`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.2746`, XGBoost `0.5790`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.1861`, XGBoost `0.4863`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `99.0`, LSTM `0.1491`, XGBoost `0.4493`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.1934`, XGBoost `0.4911`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.1688`, XGBoost `0.4631`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
