# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-spirit-vs-saw-bo3-_1uD70D_aUzOV8qHt5kBr9/spirit-vs-saw-m1-dust2.csv`
- round_num: `12`
- rows: `154`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 154 | 1.000 | 0.376668 | 0.402287 | -0.025619 | 127 | 27 | 0.928571 | 0.707792 |
| active/recent utility | 154 | 1.000 | 0.376668 | 0.402287 | -0.025619 | 127 | 27 | 0.928571 | 0.707792 |
| strong utility action | 141 | 0.916 | 0.376114 | 0.400986 | -0.024871 | 117 | 24 | 0.943262 | 0.695035 |
| utility damage | 20 | 0.130 | 0.260477 | 0.308205 | -0.047727 | 20 | 0 | 1.000000 | 0.500000 |
| active smoke/inferno | 131 | 0.851 | 0.368554 | 0.394187 | -0.025633 | 108 | 23 | 0.938931 | 0.671756 |
| recent utility last 5s | 10 | 0.065 | 0.475158 | 0.490048 | -0.014891 | 9 | 1 | 1.000000 | 1.000000 |
| flash effect present | 154 | 1.000 | 0.376668 | 0.402287 | -0.025619 | 127 | 27 | 0.928571 | 0.707792 |

## Active Smoke/Inferno Intervals

- `9.5s` - `36.0s`, rows `54`
- `37.5s` - `75.5s`, rows `77`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.5207`, XGBoost `0.2832`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.5142`, XGBoost `0.2829`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.5092`, XGBoost `0.2829`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.0`, LSTM `0.5123`, XGBoost `0.2866`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.5040`, XGBoost `0.2829`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.5065`, XGBoost `0.2870`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.5020`, XGBoost `0.2870`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5012`, XGBoost `0.2866`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.4977`, XGBoost `0.2866`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.4959`, XGBoost `0.2866`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
