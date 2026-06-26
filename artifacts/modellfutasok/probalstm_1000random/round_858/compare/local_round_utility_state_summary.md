# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `5`
- rows: `276`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 276 | 1.000 | 0.281378 | 0.331353 | -0.049975 | 250 | 26 | 0.652174 | 0.594203 |
| active/recent utility | 276 | 1.000 | 0.281378 | 0.331353 | -0.049975 | 250 | 26 | 0.652174 | 0.594203 |
| strong utility action | 121 | 0.438 | 0.503951 | 0.542647 | -0.038696 | 106 | 15 | 0.347107 | 0.214876 |
| utility damage | 20 | 0.072 | 0.528605 | 0.582735 | -0.054129 | 19 | 1 | 0.400000 | 0.000000 |
| active smoke/inferno | 121 | 0.438 | 0.503951 | 0.542647 | -0.038696 | 106 | 15 | 0.347107 | 0.214876 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 276 | 1.000 | 0.281378 | 0.331353 | -0.049975 | 250 | 26 | 0.652174 | 0.594203 |

## Active Smoke/Inferno Intervals

- `8.5s` - `68.5s`, rows `121`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.5`, LSTM `0.4447`, XGBoost `0.5961`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.4334`, XGBoost `0.5672`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.4379`, XGBoost `0.5672`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.4683`, XGBoost `0.5968`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.4715`, XGBoost `0.5961`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.4430`, XGBoost `0.5651`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `12.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.4775`, XGBoost `0.5961`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.4974`, XGBoost `0.6101`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.6158`, XGBoost `0.7269`, closer `lstm`, smoke `4`, inferno `3`, utility_damage `65.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5725`, XGBoost `0.6803`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `11.0`, recent_utility `0`
