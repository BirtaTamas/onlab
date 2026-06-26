# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-falcons-bo5-L7CZVGSHd1AqjKPyYU04lA/furia-vs-falcons-m1-inferno.csv`
- round_num: `5`
- rows: `246`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 246 | 1.000 | 0.657645 | 0.687711 | -0.030066 | 31 | 215 | 1.000000 | 1.000000 |
| active/recent utility | 246 | 1.000 | 0.657645 | 0.687711 | -0.030066 | 31 | 215 | 1.000000 | 1.000000 |
| strong utility action | 207 | 0.841 | 0.630006 | 0.662077 | -0.032071 | 30 | 177 | 1.000000 | 1.000000 |
| utility damage | 21 | 0.085 | 0.672275 | 0.712420 | -0.040146 | 2 | 19 | 1.000000 | 1.000000 |
| active smoke/inferno | 207 | 0.841 | 0.630006 | 0.662077 | -0.032071 | 30 | 177 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.041 | 0.566757 | 0.652010 | -0.085253 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 246 | 1.000 | 0.657645 | 0.687711 | -0.030066 | 31 | 215 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `42.5s`, rows `67`
- `43.5s` - `113.0s`, rows `140`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `63.0`, LSTM `0.5618`, XGBoost `0.6572`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `74.0`, LSTM `0.5568`, XGBoost `0.6518`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.5530`, XGBoost `0.6468`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `74.5`, LSTM `0.5590`, XGBoost `0.6518`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.5651`, XGBoost `0.6572`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `63.5`, LSTM `0.5661`, XGBoost `0.6572`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `66.5`, LSTM `0.5563`, XGBoost `0.6468`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `112.5`, LSTM `0.7972`, XGBoost `0.8871`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `92.5`, LSTM `0.7813`, XGBoost `0.8706`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `75.0`, LSTM `0.5630`, XGBoost `0.6506`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
