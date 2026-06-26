# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-ence-bo3-avzZyhQ46OR9GyYE_6NeM7/astralis-vs-ence-m3-overpass.csv`
- round_num: `12`
- rows: `180`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 180 | 1.000 | 0.376383 | 0.348892 | 0.027491 | 71 | 109 | 0.400000 | 0.400000 |
| active/recent utility | 180 | 1.000 | 0.376383 | 0.348892 | 0.027491 | 71 | 109 | 0.400000 | 0.400000 |
| strong utility action | 148 | 0.822 | 0.381234 | 0.356853 | 0.024381 | 56 | 92 | 0.385135 | 0.385135 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 148 | 0.822 | 0.381234 | 0.356853 | 0.024381 | 56 | 92 | 0.385135 | 0.385135 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 180 | 1.000 | 0.376383 | 0.348892 | 0.027491 | 71 | 109 | 0.400000 | 0.400000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `59.5s`, rows `103`
- `63.0s` - `85.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.0`, LSTM `0.3835`, XGBoost `0.1759`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.6053`, XGBoost `0.5102`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6160`, XGBoost `0.5229`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6237`, XGBoost `0.5346`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.6216`, XGBoost `0.5326`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5944`, XGBoost `0.5102`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6025`, XGBoost `0.5196`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.5915`, XGBoost `0.5102`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.5903`, XGBoost `0.5102`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5916`, XGBoost `0.5129`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
