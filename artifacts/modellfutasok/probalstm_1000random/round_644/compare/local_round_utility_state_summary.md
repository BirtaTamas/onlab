# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-passion-ua-vs-spirit-bo3-WimU0hRkNcqhh3KAjCozBx/passion-ua-vs-spirit-m2-mirage.csv`
- round_num: `3`
- rows: `135`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 135 | 1.000 | 0.366225 | 0.523705 | -0.157480 | 32 | 103 | 0.496296 | 0.681481 |
| active/recent utility | 135 | 1.000 | 0.366225 | 0.523705 | -0.157480 | 32 | 103 | 0.496296 | 0.681481 |
| strong utility action | 92 | 0.681 | 0.429250 | 0.520557 | -0.091308 | 32 | 60 | 0.586957 | 0.728261 |
| utility damage | 20 | 0.148 | 0.574860 | 0.552854 | 0.022006 | 10 | 10 | 0.900000 | 0.900000 |
| active smoke/inferno | 92 | 0.681 | 0.429250 | 0.520557 | -0.091308 | 32 | 60 | 0.586957 | 0.728261 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 135 | 1.000 | 0.366225 | 0.523705 | -0.157480 | 32 | 103 | 0.496296 | 0.681481 |

## Active Smoke/Inferno Intervals

- `6.5s` - `52.0s`, rows `92`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.5`, LSTM `0.2909`, XGBoost `0.6449`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.2886`, XGBoost `0.6423`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3447`, XGBoost `0.6866`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.0351`, XGBoost `0.3750`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.0423`, XGBoost `0.3775`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.0431`, XGBoost `0.3775`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.3084`, XGBoost `0.6322`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.0430`, XGBoost `0.3653`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.0452`, XGBoost `0.3649`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.0437`, XGBoost `0.3563`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
