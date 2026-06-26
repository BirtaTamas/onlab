# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `3`
- rows: `251`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 251 | 1.000 | 0.479266 | 0.531308 | -0.052042 | 109 | 142 | 0.577689 | 0.840637 |
| active/recent utility | 251 | 1.000 | 0.479266 | 0.531308 | -0.052042 | 109 | 142 | 0.577689 | 0.840637 |
| strong utility action | 188 | 0.749 | 0.499699 | 0.533891 | -0.034192 | 89 | 99 | 0.590426 | 0.867021 |
| utility damage | 20 | 0.080 | 0.624414 | 0.571228 | 0.053186 | 20 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 188 | 0.749 | 0.499699 | 0.533891 | -0.034192 | 89 | 99 | 0.590426 | 0.867021 |
| recent utility last 5s | 10 | 0.040 | 0.469486 | 0.543558 | -0.074073 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 251 | 1.000 | 0.479266 | 0.531308 | -0.052042 | 109 | 142 | 0.577689 | 0.840637 |

## Active Smoke/Inferno Intervals

- `6.5s` - `49.5s`, rows `87`
- `59.5s` - `109.5s`, rows `101`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `94.5`, LSTM `0.2195`, XGBoost `0.5667`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.2701`, XGBoost `0.5569`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `103.5`, LSTM `0.1894`, XGBoost `0.4725`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.1915`, XGBoost `0.4725`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.1961`, XGBoost `0.4732`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.1980`, XGBoost `0.4732`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.3010`, XGBoost `0.5667`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.2082`, XGBoost `0.4732`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.0`, LSTM `0.3047`, XGBoost `0.5667`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.3134`, XGBoost `0.5658`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
