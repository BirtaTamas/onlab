# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `6`
- rows: `200`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.188815 | 0.224498 | -0.035683 | 199 | 1 | 0.805000 | 0.680000 |
| active/recent utility | 200 | 1.000 | 0.188815 | 0.224498 | -0.035683 | 199 | 1 | 0.805000 | 0.680000 |
| strong utility action | 129 | 0.645 | 0.229260 | 0.280861 | -0.051601 | 128 | 1 | 0.813953 | 0.627907 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 129 | 0.645 | 0.229260 | 0.280861 | -0.051601 | 128 | 1 | 0.813953 | 0.627907 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 200 | 1.000 | 0.188815 | 0.224498 | -0.035683 | 199 | 1 | 0.805000 | 0.680000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `72.0s`, rows `129`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `43.0`, LSTM `0.2790`, XGBoost `0.4764`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `30.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.1060`, XGBoost `0.3026`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.0683`, XGBoost `0.2503`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.1285`, XGBoost `0.3019`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.0826`, XGBoost `0.2503`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `30.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.1323`, XGBoost `0.2994`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.0876`, XGBoost `0.2513`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `30.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.0833`, XGBoost `0.2459`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `48.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.1388`, XGBoost `0.2997`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `22.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.1440`, XGBoost `0.3024`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `22.0`, recent_utility `0`
