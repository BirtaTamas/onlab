# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-the-mongolz-vs-natus-vincere-bo3-PG4ywdeF4kSxWHc10zCBZ3/the-mongolz-vs-natus-vincere-m1-nuke.csv`
- round_num: `1`
- rows: `193`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 193 | 1.000 | 0.786403 | 0.826923 | -0.040520 | 16 | 177 | 0.917098 | 0.917098 |
| active/recent utility | 193 | 1.000 | 0.786403 | 0.826923 | -0.040520 | 16 | 177 | 0.917098 | 0.917098 |
| strong utility action | 64 | 0.332 | 0.650674 | 0.686184 | -0.035510 | 16 | 48 | 0.750000 | 0.750000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 64 | 0.332 | 0.650674 | 0.686184 | -0.035510 | 16 | 48 | 0.750000 | 0.750000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 193 | 1.000 | 0.786403 | 0.826923 | -0.040520 | 16 | 177 | 0.917098 | 0.917098 |

## Active Smoke/Inferno Intervals

- `13.0s` - `44.5s`, rows `64`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `28.0`, LSTM `0.5272`, XGBoost `0.6899`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5404`, XGBoost `0.6899`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.5426`, XGBoost `0.6902`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5422`, XGBoost `0.6888`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.5529`, XGBoost `0.6954`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.5495`, XGBoost `0.6921`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.7536`, XGBoost `0.8816`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.5585`, XGBoost `0.6857`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `20.0`, LSTM `0.4460`, XGBoost `0.3288`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `23.5`, LSTM `0.4424`, XGBoost `0.3300`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
