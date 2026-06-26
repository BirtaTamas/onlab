# Local Round Utility State Analysis

- csv_path: `processed_full/esports_world_cup/esports-world-cup-2025-gamerlegion-vs-the-mongolz-bo3-bupFip4WbObttNLCPYz_Zo/gamerlegion-vs-the-mongolz-m2-inferno.csv`
- round_num: `15`
- rows: `204`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 204 | 1.000 | 0.546999 | 0.549133 | -0.002134 | 89 | 115 | 0.514706 | 0.416667 |
| active/recent utility | 204 | 1.000 | 0.546999 | 0.549133 | -0.002134 | 89 | 115 | 0.514706 | 0.416667 |
| strong utility action | 188 | 0.922 | 0.538958 | 0.542489 | -0.003531 | 86 | 102 | 0.542553 | 0.452128 |
| utility damage | 37 | 0.181 | 0.635747 | 0.578905 | 0.056841 | 7 | 30 | 0.297297 | 0.297297 |
| active smoke/inferno | 188 | 0.922 | 0.538958 | 0.542489 | -0.003531 | 86 | 102 | 0.542553 | 0.452128 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 204 | 1.000 | 0.546999 | 0.549133 | -0.002134 | 89 | 115 | 0.514706 | 0.416667 |

## Active Smoke/Inferno Intervals

- `6.5s` - `42.5s`, rows `73`
- `44.5s` - `101.5s`, rows `115`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `94.0`, LSTM `0.4179`, XGBoost `0.7472`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.5`, LSTM `0.4751`, XGBoost `0.7597`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.4605`, XGBoost `0.7399`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.4750`, XGBoost `0.7517`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.4774`, XGBoost `0.7399`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `96.0`, LSTM `0.4967`, XGBoost `0.7404`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.1708`, XGBoost `0.3961`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.5169`, XGBoost `0.7399`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.5618`, XGBoost `0.7708`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.3190`, XGBoost `0.5224`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
