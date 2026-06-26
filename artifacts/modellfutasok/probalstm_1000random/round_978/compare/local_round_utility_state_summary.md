# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-heroic-vs-aurora-bo3-QigxwcikBDdlIOkrYDpY7y/heroic-vs-aurora-m2-dust2.csv`
- round_num: `21`
- rows: `140`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 140 | 1.000 | 0.129420 | 0.122474 | 0.006946 | 35 | 105 | 0.850000 | 0.850000 |
| active/recent utility | 140 | 1.000 | 0.129420 | 0.122474 | 0.006946 | 35 | 105 | 0.850000 | 0.850000 |
| strong utility action | 113 | 0.807 | 0.131513 | 0.129556 | 0.001957 | 35 | 78 | 0.849558 | 0.849558 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 103 | 0.736 | 0.088926 | 0.092737 | -0.003811 | 35 | 68 | 0.932039 | 0.932039 |
| recent utility last 5s | 22 | 0.157 | 0.318037 | 0.281117 | 0.036921 | 0 | 22 | 0.454545 | 0.454545 |
| flash effect present | 140 | 1.000 | 0.129420 | 0.122474 | 0.006946 | 35 | 105 | 0.850000 | 0.850000 |

## Active Smoke/Inferno Intervals

- `7.0s` - `58.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `8.0`, LSTM `0.6304`, XGBoost `0.5040`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.6281`, XGBoost `0.5053`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.5849`, XGBoost `0.5040`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.5829`, XGBoost `0.5040`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.5819`, XGBoost `0.5042`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `21.0`, LSTM `0.0642`, XGBoost `0.1408`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.0482`, XGBoost `0.1220`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.5766`, XGBoost `0.5075`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `2`
- seconds `22.0`, LSTM `0.0333`, XGBoost `0.1022`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.0239`, XGBoost `0.0923`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
