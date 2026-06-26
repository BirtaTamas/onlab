# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `22`
- rows: `200`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 200 | 1.000 | 0.706946 | 0.653037 | 0.053909 | 159 | 41 | 1.000000 | 1.000000 |
| active/recent utility | 200 | 1.000 | 0.706946 | 0.653037 | 0.053909 | 159 | 41 | 1.000000 | 1.000000 |
| strong utility action | 180 | 0.900 | 0.707810 | 0.657711 | 0.050099 | 142 | 38 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 180 | 0.900 | 0.707810 | 0.657711 | 0.050099 | 142 | 38 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 200 | 1.000 | 0.706946 | 0.653037 | 0.053909 | 159 | 41 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `98.0s`, rows `180`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `35.5`, LSTM `0.7331`, XGBoost `0.5965`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.7175`, XGBoost `0.5840`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.7254`, XGBoost `0.6015`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `66.5`, LSTM `0.6628`, XGBoost `0.5441`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7194`, XGBoost `0.6018`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.7082`, XGBoost `0.5914`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.7133`, XGBoost `0.5972`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `67.0`, LSTM `0.6623`, XGBoost `0.5464`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.6966`, XGBoost `0.5811`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.7124`, XGBoost `0.5972`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
