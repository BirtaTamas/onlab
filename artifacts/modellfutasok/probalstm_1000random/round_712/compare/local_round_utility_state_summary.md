# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-the-mongolz-bo3-7VmOOQFfF_Xgx4vOG4cYIY/vitality-vs-the-mongolz-m3-inferno.csv`
- round_num: `4`
- rows: `205`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 205 | 1.000 | 0.606674 | 0.646335 | -0.039661 | 40 | 165 | 0.775610 | 0.897561 |
| active/recent utility | 205 | 1.000 | 0.606674 | 0.646335 | -0.039661 | 40 | 165 | 0.775610 | 0.897561 |
| strong utility action | 186 | 0.907 | 0.607593 | 0.646725 | -0.039132 | 40 | 146 | 0.822581 | 0.887097 |
| utility damage | 10 | 0.049 | 0.513337 | 0.515670 | -0.002334 | 4 | 6 | 0.800000 | 1.000000 |
| active smoke/inferno | 186 | 0.907 | 0.607593 | 0.646725 | -0.039132 | 40 | 146 | 0.822581 | 0.887097 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 205 | 1.000 | 0.606674 | 0.646335 | -0.039661 | 40 | 165 | 0.775610 | 0.897561 |

## Active Smoke/Inferno Intervals

- `7.0s` - `99.5s`, rows `186`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.0`, LSTM `0.1332`, XGBoost `0.3044`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.1422`, XGBoost `0.3044`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `28.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.1435`, XGBoost `0.2957`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `28.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.2167`, XGBoost `0.3591`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.1685`, XGBoost `0.3009`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `28.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.3913`, XGBoost `0.5219`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `28.0`, recent_utility `0`
- seconds `26.0`, LSTM `0.2316`, XGBoost `0.3591`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.6084`, XGBoost `0.7340`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.6105`, XGBoost `0.7342`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.2375`, XGBoost `0.3591`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
