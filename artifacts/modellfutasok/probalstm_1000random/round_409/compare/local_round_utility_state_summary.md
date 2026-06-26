# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `2`
- rows: `224`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 224 | 1.000 | 0.639148 | 0.740060 | -0.100913 | 0 | 224 | 0.977679 | 1.000000 |
| active/recent utility | 224 | 1.000 | 0.639148 | 0.740060 | -0.100913 | 0 | 224 | 0.977679 | 1.000000 |
| strong utility action | 156 | 0.696 | 0.676969 | 0.770622 | -0.093653 | 0 | 156 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 146 | 0.652 | 0.687690 | 0.777848 | -0.090158 | 0 | 146 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.045 | 0.520434 | 0.665125 | -0.144691 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 224 | 1.000 | 0.639148 | 0.740060 | -0.100913 | 0 | 224 | 0.977679 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `35.0s`, rows `55`
- `65.5s` - `110.5s`, rows `91`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `103.0`, LSTM `0.6760`, XGBoost `0.8444`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.5`, LSTM `0.6894`, XGBoost `0.8496`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5079`, XGBoost `0.6647`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5097`, XGBoost `0.6651`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `102.0`, LSTM `0.6947`, XGBoost `0.8496`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.5117`, XGBoost `0.6651`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `45.5`, LSTM `0.5161`, XGBoost `0.6651`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `81.0`, LSTM `0.7190`, XGBoost `0.8678`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `102.5`, LSTM `0.7018`, XGBoost `0.8496`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `101.0`, LSTM `0.7023`, XGBoost `0.8496`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
