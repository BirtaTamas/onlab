# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-saw-bo3-PeKJ4V-uBfKnBCIB8ocl58/natus-vincere-vs-saw-m1-inferno.csv`
- round_num: `8`
- rows: `173`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 173 | 1.000 | 0.634051 | 0.611348 | 0.022703 | 121 | 52 | 0.780347 | 0.658960 |
| active/recent utility | 173 | 1.000 | 0.634051 | 0.611348 | 0.022703 | 121 | 52 | 0.780347 | 0.658960 |
| strong utility action | 147 | 0.850 | 0.619391 | 0.594994 | 0.024397 | 104 | 43 | 0.741497 | 0.598639 |
| utility damage | 10 | 0.058 | 0.500804 | 0.458840 | 0.041964 | 10 | 0 | 0.700000 | 0.000000 |
| active smoke/inferno | 137 | 0.792 | 0.626181 | 0.601438 | 0.024743 | 94 | 43 | 0.722628 | 0.569343 |
| recent utility last 5s | 10 | 0.058 | 0.526368 | 0.506718 | 0.019650 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 173 | 1.000 | 0.634051 | 0.611348 | 0.022703 | 121 | 52 | 0.780347 | 0.658960 |

## Active Smoke/Inferno Intervals

- `11.0s` - `73.5s`, rows `126`
- `75.0s` - `80.0s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `60.5`, LSTM `0.4980`, XGBoost `0.3391`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.4868`, XGBoost `0.3391`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.4908`, XGBoost `0.3491`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.7215`, XGBoost `0.5848`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.7111`, XGBoost `0.5798`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.4916`, XGBoost `0.3652`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.0`, LSTM `0.6165`, XGBoost `0.7402`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.4863`, XGBoost `0.3646`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.5058`, XGBoost `0.3882`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.5048`, XGBoost `0.3882`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
