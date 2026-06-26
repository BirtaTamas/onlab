# Local Round Utility State Analysis

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `15`
- rows: `227`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 227 | 1.000 | 0.178337 | 0.188202 | -0.009865 | 177 | 50 | 0.991189 | 0.991189 |
| active/recent utility | 227 | 1.000 | 0.178337 | 0.188202 | -0.009865 | 177 | 50 | 0.991189 | 0.991189 |
| strong utility action | 176 | 0.775 | 0.198406 | 0.203814 | -0.005409 | 130 | 46 | 0.988636 | 0.988636 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 176 | 0.775 | 0.198406 | 0.203814 | -0.005409 | 130 | 46 | 0.988636 | 0.988636 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 227 | 1.000 | 0.178337 | 0.188202 | -0.009865 | 177 | 50 | 0.991189 | 0.991189 |

## Active Smoke/Inferno Intervals

- `12.0s` - `109.5s`, rows `176`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `76.5`, LSTM `0.4812`, XGBoost `0.3478`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.4762`, XGBoost `0.3474`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.4759`, XGBoost `0.3478`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.4726`, XGBoost `0.3474`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.4726`, XGBoost `0.3474`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `75.5`, LSTM `0.4704`, XGBoost `0.3474`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.4601`, XGBoost `0.3474`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.4404`, XGBoost `0.3474`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.2535`, XGBoost `0.1657`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `78.0`, LSTM `0.2407`, XGBoost `0.1657`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
