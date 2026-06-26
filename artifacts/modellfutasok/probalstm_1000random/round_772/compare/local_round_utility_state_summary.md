# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `10`
- rows: `151`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.662636 | 0.639055 | 0.023581 | 99 | 52 | 0.754967 | 0.629139 |
| active/recent utility | 151 | 1.000 | 0.662636 | 0.639055 | 0.023581 | 99 | 52 | 0.754967 | 0.629139 |
| strong utility action | 117 | 0.775 | 0.628377 | 0.594052 | 0.034325 | 84 | 33 | 0.683761 | 0.521368 |
| utility damage | 27 | 0.179 | 0.748468 | 0.723801 | 0.024667 | 17 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 117 | 0.775 | 0.628377 | 0.594052 | 0.034325 | 84 | 33 | 0.683761 | 0.521368 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 151 | 1.000 | 0.662636 | 0.639055 | 0.023581 | 99 | 52 | 0.754967 | 0.629139 |

## Active Smoke/Inferno Intervals

- `7.5s` - `65.5s`, rows `117`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.4798`, XGBoost `0.3404`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.4731`, XGBoost `0.3350`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.4701`, XGBoost `0.3368`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5582`, XGBoost `0.4256`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5056`, XGBoost `0.3736`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4635`, XGBoost `0.3368`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.0`, LSTM `0.7880`, XGBoost `0.9130`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.4742`, XGBoost `0.3501`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.4720`, XGBoost `0.3491`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.4757`, XGBoost `0.3531`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
