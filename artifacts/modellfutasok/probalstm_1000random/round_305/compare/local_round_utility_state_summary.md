# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-spirit-vs-inner-circle-bo3-YbhHiIk4CcU9clhSbtidF_/spirit-vs-inner-circle-m1-ancient.csv`
- round_num: `12`
- rows: `116`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 116 | 1.000 | 0.254353 | 0.289807 | -0.035453 | 77 | 39 | 0.715517 | 0.681034 |
| active/recent utility | 116 | 1.000 | 0.254353 | 0.289807 | -0.035453 | 77 | 39 | 0.715517 | 0.681034 |
| strong utility action | 103 | 0.888 | 0.201513 | 0.250873 | -0.049360 | 77 | 26 | 0.805825 | 0.766990 |
| utility damage | 11 | 0.095 | 0.561894 | 0.519952 | 0.041942 | 0 | 11 | 0.363636 | 0.363636 |
| active smoke/inferno | 103 | 0.888 | 0.201513 | 0.250873 | -0.049360 | 77 | 26 | 0.805825 | 0.766990 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 116 | 1.000 | 0.254353 | 0.289807 | -0.035453 | 77 | 39 | 0.715517 | 0.681034 |

## Active Smoke/Inferno Intervals

- `6.5s` - `57.5s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.0`, LSTM `0.0333`, XGBoost `0.2205`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.0396`, XGBoost `0.2213`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0535`, XGBoost `0.2206`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.3361`, XGBoost `0.5032`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.3341`, XGBoost `0.5006`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.3372`, XGBoost `0.5006`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.1335`, XGBoost `0.2942`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.3420`, XGBoost `0.5024`, closer `lstm`, smoke `5`, inferno `2`, utility_damage `32.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0499`, XGBoost `0.2063`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.0`, LSTM `0.1807`, XGBoost `0.3345`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
