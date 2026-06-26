# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m2-dust2.csv`
- round_num: `10`
- rows: `297`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 297 | 1.000 | 0.237233 | 0.344281 | -0.107048 | 297 | 0 | 1.000000 | 0.794613 |
| active/recent utility | 297 | 1.000 | 0.237233 | 0.344281 | -0.107048 | 297 | 0 | 1.000000 | 0.794613 |
| strong utility action | 224 | 0.754 | 0.283500 | 0.413231 | -0.129731 | 224 | 0 | 1.000000 | 0.763393 |
| utility damage | 42 | 0.141 | 0.211497 | 0.383482 | -0.171985 | 42 | 0 | 1.000000 | 0.833333 |
| active smoke/inferno | 224 | 0.754 | 0.283500 | 0.413231 | -0.129731 | 224 | 0 | 1.000000 | 0.763393 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 297 | 1.000 | 0.237233 | 0.344281 | -0.107048 | 297 | 0 | 1.000000 | 0.794613 |

## Active Smoke/Inferno Intervals

- `4.0s` - `86.5s`, rows `166`
- `91.5s` - `113.0s`, rows `44`
- `114.0s` - `120.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.5`, LSTM `0.0673`, XGBoost `0.3521`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `103.0`, LSTM `0.0499`, XGBoost `0.3162`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.0549`, XGBoost `0.3097`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.0623`, XGBoost `0.3097`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `99.5`, LSTM `0.0547`, XGBoost `0.3007`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `28.0`, recent_utility `0`
- seconds `102.0`, LSTM `0.0640`, XGBoost `0.3097`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `100.5`, LSTM `0.0556`, XGBoost `0.3007`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.0653`, XGBoost `0.3080`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `100.0`, LSTM `0.0615`, XGBoost `0.3007`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `19.0`, recent_utility `0`
- seconds `97.5`, LSTM `0.0816`, XGBoost `0.3115`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `34.0`, recent_utility `0`
