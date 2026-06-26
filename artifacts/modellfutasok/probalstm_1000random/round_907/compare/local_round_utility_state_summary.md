# Local Round Utility State Analysis

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-b8-vs-flyquest-bo3-ROTxQXIIApwC88KHLMMjQT/b8-vs-flyquest-m3-inferno.csv`
- round_num: `18`
- rows: `264`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 264 | 1.000 | 0.266754 | 0.281386 | -0.014632 | 185 | 79 | 0.799242 | 0.958333 |
| active/recent utility | 264 | 1.000 | 0.266754 | 0.281386 | -0.014632 | 185 | 79 | 0.799242 | 0.958333 |
| strong utility action | 205 | 0.777 | 0.319955 | 0.315815 | 0.004140 | 127 | 78 | 0.741463 | 0.946341 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 205 | 0.777 | 0.319955 | 0.315815 | 0.004140 | 127 | 78 | 0.741463 | 0.946341 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 264 | 1.000 | 0.266754 | 0.281386 | -0.014632 | 185 | 79 | 0.799242 | 0.958333 |

## Active Smoke/Inferno Intervals

- `9.5s` - `90.0s`, rows `162`
- `110.5s` - `131.5s`, rows `43`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `77.0`, LSTM `0.4813`, XGBoost `0.2937`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.5139`, XGBoost `0.3265`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.5137`, XGBoost `0.3265`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `79.0`, LSTM `0.4818`, XGBoost `0.2963`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.4675`, XGBoost `0.2963`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.5146`, XGBoost `0.3454`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.4590`, XGBoost `0.2963`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.3466`, XGBoost `0.5079`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.5060`, XGBoost `0.3454`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.4519`, XGBoost `0.2931`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
