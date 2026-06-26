# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `15`
- rows: `162`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 162 | 1.000 | 0.139852 | 0.177675 | -0.037823 | 159 | 3 | 0.876543 | 0.753086 |
| active/recent utility | 162 | 1.000 | 0.139852 | 0.177675 | -0.037823 | 159 | 3 | 0.876543 | 0.753086 |
| strong utility action | 107 | 0.660 | 0.202009 | 0.255387 | -0.053379 | 104 | 3 | 0.822430 | 0.644860 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 97 | 0.599 | 0.169913 | 0.226246 | -0.056333 | 94 | 3 | 0.907216 | 0.711340 |
| recent utility last 5s | 10 | 0.062 | 0.513332 | 0.538053 | -0.024721 | 10 | 0 | 0.000000 | 0.000000 |
| flash effect present | 162 | 1.000 | 0.139852 | 0.177675 | -0.037823 | 159 | 3 | 0.876543 | 0.753086 |

## Active Smoke/Inferno Intervals

- `6.0s` - `54.0s`, rows `97`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.0`, LSTM `0.0873`, XGBoost `0.2980`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.0997`, XGBoost `0.3011`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.1003`, XGBoost `0.2852`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `30.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.0821`, XGBoost `0.2571`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `46.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.1206`, XGBoost `0.2946`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1224`, XGBoost `0.2933`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `14.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.1300`, XGBoost `0.3002`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1409`, XGBoost `0.2959`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `6.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.1552`, XGBoost `0.3036`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.1621`, XGBoost `0.3046`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `1.0`, recent_utility `0`
