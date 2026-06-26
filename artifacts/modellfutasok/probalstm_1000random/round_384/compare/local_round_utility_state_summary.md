# Local Round Utility State Analysis

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-the-mongolz-vs-heroic-bo3-lz59_87ZRvJjbdTai7Ev35/heroic-vs-3dmax-m3-ancient.csv`
- round_num: `7`
- rows: `141`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 141 | 1.000 | 0.330327 | 0.359102 | -0.028775 | 123 | 18 | 0.581560 | 0.411348 |
| active/recent utility | 141 | 1.000 | 0.330327 | 0.359102 | -0.028775 | 123 | 18 | 0.581560 | 0.411348 |
| strong utility action | 130 | 0.922 | 0.307380 | 0.341032 | -0.033651 | 123 | 7 | 0.630769 | 0.446154 |
| utility damage | 10 | 0.071 | 0.445159 | 0.459672 | -0.014513 | 8 | 2 | 0.900000 | 0.200000 |
| active smoke/inferno | 130 | 0.922 | 0.307380 | 0.341032 | -0.033651 | 123 | 7 | 0.630769 | 0.446154 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 141 | 1.000 | 0.330327 | 0.359102 | -0.028775 | 123 | 18 | 0.581560 | 0.411348 |

## Active Smoke/Inferno Intervals

- `5.5s` - `70.0s`, rows `130`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `40.0`, LSTM `0.2687`, XGBoost `0.3850`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.2843`, XGBoost `0.3898`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.0253`, XGBoost `0.1215`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.3015`, XGBoost `0.3894`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.4826`, XGBoost `0.5688`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.0863`, XGBoost `0.1668`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.4944`, XGBoost `0.5653`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.4953`, XGBoost `0.5656`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.4834`, XGBoost `0.5527`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.4970`, XGBoost `0.5656`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
