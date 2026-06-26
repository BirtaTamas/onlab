# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `5`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.739382 | 0.820848 | -0.081466 | 0 | 230 | 0.626087 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.739382 | 0.820848 | -0.081466 | 0 | 230 | 0.626087 | 1.000000 |
| strong utility action | 119 | 0.517 | 0.621730 | 0.725657 | -0.103927 | 0 | 119 | 0.411765 | 1.000000 |
| utility damage | 13 | 0.057 | 0.442223 | 0.566742 | -0.124520 | 0 | 13 | 0.000000 | 1.000000 |
| active smoke/inferno | 119 | 0.517 | 0.621730 | 0.725657 | -0.103927 | 0 | 119 | 0.411765 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.739382 | 0.820848 | -0.081466 | 0 | 230 | 0.626087 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `60.0s`, rows `105`
- `68.5s` - `75.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `41.5`, LSTM `0.2609`, XGBoost `0.6060`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3075`, XGBoost `0.6310`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.3132`, XGBoost `0.6061`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.2978`, XGBoost `0.5772`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.2997`, XGBoost `0.5760`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.3362`, XGBoost `0.6061`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.3638`, XGBoost `0.6321`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.3577`, XGBoost `0.5907`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.3634`, XGBoost `0.5907`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.4149`, XGBoost `0.6321`, closer `xgboost`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
