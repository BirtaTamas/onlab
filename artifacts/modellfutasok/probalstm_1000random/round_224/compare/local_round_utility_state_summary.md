# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m1-dust2.csv`
- round_num: `16`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.360130 | 0.401009 | -0.040879 | 119 | 34 | 0.849673 | 0.745098 |
| active/recent utility | 153 | 1.000 | 0.360130 | 0.401009 | -0.040879 | 119 | 34 | 0.849673 | 0.745098 |
| strong utility action | 99 | 0.647 | 0.380240 | 0.437438 | -0.057198 | 85 | 14 | 0.878788 | 0.646465 |
| utility damage | 20 | 0.131 | 0.319984 | 0.370172 | -0.050188 | 16 | 4 | 1.000000 | 1.000000 |
| active smoke/inferno | 99 | 0.647 | 0.380240 | 0.437438 | -0.057198 | 85 | 14 | 0.878788 | 0.646465 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.360130 | 0.401009 | -0.040879 | 119 | 34 | 0.849673 | 0.745098 |

## Active Smoke/Inferno Intervals

- `3.5s` - `52.5s`, rows `99`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `34.0`, LSTM `0.3338`, XGBoost `0.5308`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.3642`, XGBoost `0.5308`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.3893`, XGBoost `0.5287`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.5`, LSTM `0.1524`, XGBoost `0.2890`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `12.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.3949`, XGBoost `0.5287`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.1528`, XGBoost `0.2839`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `12.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.4004`, XGBoost `0.5266`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.4057`, XGBoost `0.5255`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.1806`, XGBoost `0.3003`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.1723`, XGBoost `0.2845`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `12.0`, recent_utility `0`
