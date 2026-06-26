# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-vitality-vs-falcons-bo3-8ZTMZQ0BkOa0azICXTbCYv/vitality-vs-falcons-m2-train.csv`
- round_num: `16`
- rows: `142`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 142 | 1.000 | 0.851258 | 0.859821 | -0.008563 | 48 | 94 | 1.000000 | 1.000000 |
| active/recent utility | 142 | 1.000 | 0.851258 | 0.859821 | -0.008563 | 48 | 94 | 1.000000 | 1.000000 |
| strong utility action | 103 | 0.725 | 0.835544 | 0.852042 | -0.016497 | 33 | 70 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 103 | 0.725 | 0.835544 | 0.852042 | -0.016497 | 33 | 70 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 142 | 1.000 | 0.851258 | 0.859821 | -0.008563 | 48 | 94 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `57.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.8090`, XGBoost `0.9282`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.7898`, XGBoost `0.6863`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.7886`, XGBoost `0.6899`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.8298`, XGBoost `0.9282`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.8309`, XGBoost `0.9291`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.8311`, XGBoost `0.9289`, closer `xgboost`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.8319`, XGBoost `0.9291`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.8317`, XGBoost `0.9282`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7812`, XGBoost `0.6873`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.8362`, XGBoost `0.9290`, closer `xgboost`, smoke `3`, inferno `3`, utility_damage `0.0`, recent_utility `0`
