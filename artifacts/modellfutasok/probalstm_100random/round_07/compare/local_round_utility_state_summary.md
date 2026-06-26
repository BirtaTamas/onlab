# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-natus-vincere-vs-tyloo-bo3-u9zlDGjnIy0eSohnO5P-Xx/natus-vincere-vs-tyloo-m2-mirage.csv`
- round_num: `17`
- rows: `217`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 217 | 1.000 | 0.410235 | 0.384582 | 0.025653 | 82 | 135 | 0.479263 | 0.889401 |
| active/recent utility | 217 | 1.000 | 0.410235 | 0.384582 | 0.025653 | 82 | 135 | 0.479263 | 0.889401 |
| strong utility action | 128 | 0.590 | 0.417152 | 0.369005 | 0.048147 | 49 | 79 | 0.437500 | 0.960938 |
| utility damage | 41 | 0.189 | 0.368296 | 0.357062 | 0.011234 | 25 | 16 | 0.682927 | 0.975610 |
| active smoke/inferno | 128 | 0.590 | 0.417152 | 0.369005 | 0.048147 | 49 | 79 | 0.437500 | 0.960938 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 217 | 1.000 | 0.410235 | 0.384582 | 0.025653 | 82 | 135 | 0.479263 | 0.889401 |

## Active Smoke/Inferno Intervals

- `6.5s` - `47.5s`, rows `83`
- `73.5s` - `95.5s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `95.5`, LSTM `0.0597`, XGBoost `0.3514`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.0`, LSTM `0.0649`, XGBoost `0.3490`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.0764`, XGBoost `0.3514`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.0`, LSTM `0.0810`, XGBoost `0.3514`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `93.5`, LSTM `0.0927`, XGBoost `0.3514`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.6517`, XGBoost `0.4015`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6418`, XGBoost `0.3976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `20.5`, LSTM `0.6384`, XGBoost `0.3976`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6379`, XGBoost `0.4015`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6356`, XGBoost `0.4015`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `20.0`, recent_utility `0`
