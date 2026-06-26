# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-flyquest-vs-mibr-bo3-qPrK-wzQgATa8KQ5HjYeOS/flyquest-vs-mibr-m1-nuke.csv`
- round_num: `5`
- rows: `152`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 152 | 1.000 | 0.190625 | 0.198920 | -0.008295 | 128 | 24 | 0.710526 | 0.710526 |
| active/recent utility | 152 | 1.000 | 0.190625 | 0.198920 | -0.008295 | 128 | 24 | 0.710526 | 0.710526 |
| strong utility action | 128 | 0.842 | 0.131639 | 0.144034 | -0.012394 | 118 | 10 | 0.812500 | 0.812500 |
| utility damage | 34 | 0.224 | 0.312847 | 0.329958 | -0.017111 | 25 | 9 | 0.529412 | 0.529412 |
| active smoke/inferno | 128 | 0.842 | 0.131639 | 0.144034 | -0.012394 | 118 | 10 | 0.812500 | 0.812500 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 152 | 1.000 | 0.190625 | 0.198920 | -0.008295 | 128 | 24 | 0.710526 | 0.710526 |

## Active Smoke/Inferno Intervals

- `10.0s` - `73.5s`, rows `128`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `22.0`, LSTM `0.3815`, XGBoost `0.2588`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6367`, XGBoost `0.7476`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6215`, XGBoost `0.5507`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.5421`, XGBoost `0.6070`, closer `lstm`, smoke `2`, inferno `4`, utility_damage `10.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.5541`, XGBoost `0.6053`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.5593`, XGBoost `0.6069`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `24.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.2044`, XGBoost `0.2514`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0110`, XGBoost `0.0576`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.5602`, XGBoost `0.6053`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `16.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.5563`, XGBoost `0.6008`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
