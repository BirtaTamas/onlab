# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `20`
- rows: `179`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 179 | 1.000 | 0.380932 | 0.562042 | -0.181110 | 175 | 4 | 0.614525 | 0.234637 |
| active/recent utility | 179 | 1.000 | 0.380932 | 0.562042 | -0.181110 | 175 | 4 | 0.614525 | 0.234637 |
| strong utility action | 150 | 0.838 | 0.428158 | 0.596181 | -0.168023 | 147 | 3 | 0.546667 | 0.120000 |
| utility damage | 10 | 0.056 | 0.478778 | 0.624607 | -0.145829 | 10 | 0 | 0.500000 | 0.000000 |
| active smoke/inferno | 140 | 0.782 | 0.427731 | 0.600981 | -0.173250 | 137 | 3 | 0.514286 | 0.128571 |
| recent utility last 5s | 10 | 0.056 | 0.434137 | 0.528975 | -0.094839 | 10 | 0 | 1.000000 | 0.000000 |
| flash effect present | 179 | 1.000 | 0.380932 | 0.562042 | -0.181110 | 175 | 4 | 0.614525 | 0.234637 |

## Active Smoke/Inferno Intervals

- `7.0s` - `43.0s`, rows `73`
- `44.0s` - `77.0s`, rows `67`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `58.0`, LSTM `0.1478`, XGBoost `0.5882`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.5`, LSTM `0.1662`, XGBoost `0.5882`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.1315`, XGBoost `0.5358`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `57.0`, LSTM `0.1851`, XGBoost `0.5882`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.2031`, XGBoost `0.5882`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.2055`, XGBoost `0.5882`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.2347`, XGBoost `0.6109`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.2488`, XGBoost `0.6109`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.1156`, XGBoost `0.4732`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.1203`, XGBoost `0.4732`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
