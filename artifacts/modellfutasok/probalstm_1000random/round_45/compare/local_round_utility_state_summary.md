# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `7`
- rows: `180`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 180 | 1.000 | 0.244663 | 0.368705 | -0.124042 | 178 | 2 | 0.850000 | 0.594444 |
| active/recent utility | 180 | 1.000 | 0.244663 | 0.368705 | -0.124042 | 178 | 2 | 0.850000 | 0.594444 |
| strong utility action | 101 | 0.561 | 0.320298 | 0.432904 | -0.112606 | 101 | 0 | 0.732673 | 0.435644 |
| utility damage | 10 | 0.056 | 0.517925 | 0.603435 | -0.085511 | 10 | 0 | 0.000000 | 0.000000 |
| active smoke/inferno | 101 | 0.561 | 0.320298 | 0.432904 | -0.112606 | 101 | 0 | 0.732673 | 0.435644 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 180 | 1.000 | 0.244663 | 0.368705 | -0.124042 | 178 | 2 | 0.850000 | 0.594444 |

## Active Smoke/Inferno Intervals

- `8.0s` - `52.5s`, rows `90`
- `79.5s` - `84.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.0`, LSTM `0.1171`, XGBoost `0.3243`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.0`, LSTM `0.0754`, XGBoost `0.2811`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1290`, XGBoost `0.3302`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.1182`, XGBoost `0.3172`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1588`, XGBoost `0.3538`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.1232`, XGBoost `0.3172`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.1277`, XGBoost `0.3172`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1291`, XGBoost `0.3167`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1430`, XGBoost `0.3302`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.1304`, XGBoost `0.3172`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
