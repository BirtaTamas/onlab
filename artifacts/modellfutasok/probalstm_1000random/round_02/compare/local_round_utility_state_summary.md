# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `4`
- rows: `194`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 194 | 1.000 | 0.324482 | 0.433952 | -0.109470 | 6 | 188 | 0.231959 | 0.247423 |
| active/recent utility | 194 | 1.000 | 0.324482 | 0.433952 | -0.109470 | 6 | 188 | 0.231959 | 0.247423 |
| strong utility action | 151 | 0.778 | 0.304294 | 0.408679 | -0.104385 | 6 | 145 | 0.198675 | 0.218543 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 151 | 0.778 | 0.304294 | 0.408679 | -0.104385 | 6 | 145 | 0.198675 | 0.218543 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 194 | 1.000 | 0.324482 | 0.433952 | -0.109470 | 6 | 188 | 0.231959 | 0.247423 |

## Active Smoke/Inferno Intervals

- `11.0s` - `39.5s`, rows `58`
- `43.0s` - `89.0s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `68.5`, LSTM `0.0834`, XGBoost `0.3025`, closer `xgboost`, smoke `1`, inferno `4`, utility_damage `0.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.0893`, XGBoost `0.3037`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `67.5`, LSTM `0.0979`, XGBoost `0.3030`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `70.5`, LSTM `0.0914`, XGBoost `0.2932`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.0998`, XGBoost `0.3010`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.1009`, XGBoost `0.3020`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0963`, XGBoost `0.2946`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `70.0`, LSTM `0.0970`, XGBoost `0.2947`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.1083`, XGBoost `0.3023`, closer `xgboost`, smoke `1`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.1104`, XGBoost `0.3020`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
