# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-aurora-vs-falcons-bo3-5oHSxtVT-5F3Op7ZcgBMjW/aurora-vs-falcons-m2-train.csv`
- round_num: `22`
- rows: `254`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 254 | 1.000 | 0.454368 | 0.573702 | -0.119334 | 4 | 250 | 0.444882 | 0.444882 |
| active/recent utility | 254 | 1.000 | 0.454368 | 0.573702 | -0.119334 | 4 | 250 | 0.444882 | 0.444882 |
| strong utility action | 119 | 0.469 | 0.450757 | 0.567236 | -0.116479 | 4 | 115 | 0.537815 | 0.537815 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 119 | 0.469 | 0.450757 | 0.567236 | -0.116479 | 4 | 115 | 0.537815 | 0.537815 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 254 | 1.000 | 0.454368 | 0.573702 | -0.119334 | 4 | 250 | 0.444882 | 0.444882 |

## Active Smoke/Inferno Intervals

- `8.0s` - `44.5s`, rows `74`
- `93.5s` - `115.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `114.0`, LSTM `0.5302`, XGBoost `0.8082`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `114.5`, LSTM `0.5398`, XGBoost `0.8058`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `115.0`, LSTM `0.5408`, XGBoost `0.7997`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.5508`, XGBoost `0.8082`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `115.5`, LSTM `0.5426`, XGBoost `0.7997`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1329`, XGBoost `0.3697`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.1480`, XGBoost `0.3697`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1525`, XGBoost `0.3697`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.1530`, XGBoost `0.3697`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.6411`, XGBoost `0.8567`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
