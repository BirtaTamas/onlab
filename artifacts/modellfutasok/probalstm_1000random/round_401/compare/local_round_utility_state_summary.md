# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `1`
- rows: `169`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 169 | 1.000 | 0.700538 | 0.758027 | -0.057489 | 26 | 143 | 0.698225 | 0.615385 |
| active/recent utility | 99 | 0.586 | 0.864088 | 0.945841 | -0.081753 | 0 | 99 | 1.000000 | 1.000000 |
| strong utility action | 56 | 0.331 | 0.893942 | 0.962157 | -0.068215 | 0 | 56 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 56 | 0.331 | 0.893942 | 0.962157 | -0.068215 | 0 | 56 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 98 | 0.580 | 0.864901 | 0.946091 | -0.081190 | 0 | 98 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `35.0s` - `62.5s`, rows `56`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.0`, LSTM `0.7844`, XGBoost `0.9213`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7291`, XGBoost `0.8551`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `25.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.8409`, XGBoost `0.9662`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.8427`, XGBoost `0.9680`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `47.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.7176`, XGBoost `0.8365`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `40.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.8484`, XGBoost `0.9658`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.7440`, XGBoost `0.8555`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `10.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.8562`, XGBoost `0.9676`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.8552`, XGBoost `0.9658`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.8572`, XGBoost `0.9676`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `48.0`, recent_utility `0`
