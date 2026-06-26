# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-faze-vs-inner-circle-bo3-runM3q2zOKSAHTeRui0Q2h/faze-vs-inner-circle-m2-nuke.csv`
- round_num: `18`
- rows: `228`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 228 | 1.000 | 0.520681 | 0.387592 | 0.133089 | 0 | 228 | 0.368421 | 0.688596 |
| active/recent utility | 228 | 1.000 | 0.520681 | 0.387592 | 0.133089 | 0 | 228 | 0.368421 | 0.688596 |
| strong utility action | 125 | 0.548 | 0.517604 | 0.393852 | 0.123752 | 0 | 125 | 0.496000 | 0.696000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 125 | 0.548 | 0.517604 | 0.393852 | 0.123752 | 0 | 125 | 0.496000 | 0.696000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 228 | 1.000 | 0.520681 | 0.387592 | 0.133089 | 0 | 228 | 0.368421 | 0.688596 |

## Active Smoke/Inferno Intervals

- `8.0s` - `47.0s`, rows `79`
- `68.5s` - `91.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `75.5`, LSTM `0.4982`, XGBoost `0.2189`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.4980`, XGBoost `0.2209`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.5`, LSTM `0.4840`, XGBoost `0.2209`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.4606`, XGBoost `0.2189`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.4796`, XGBoost `0.2387`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.5`, LSTM `0.4357`, XGBoost `0.2189`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `76.0`, LSTM `0.4504`, XGBoost `0.2387`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `74.0`, LSTM `0.4292`, XGBoost `0.2209`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.4442`, XGBoost `0.2397`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.4329`, XGBoost `0.2397`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
