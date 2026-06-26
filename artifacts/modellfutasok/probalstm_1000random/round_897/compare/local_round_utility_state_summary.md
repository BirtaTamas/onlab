# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-rare-atom-vs-astralis-bo3-2mbRF781jI0kkV-FX6ZCr7/rare-atom-vs-astralis-m1-ancient.csv`
- round_num: `2`
- rows: `244`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 244 | 1.000 | 0.863106 | 0.906484 | -0.043379 | 34 | 210 | 1.000000 | 1.000000 |
| active/recent utility | 244 | 1.000 | 0.863106 | 0.906484 | -0.043379 | 34 | 210 | 1.000000 | 1.000000 |
| strong utility action | 131 | 0.537 | 0.870118 | 0.900430 | -0.030312 | 34 | 97 | 1.000000 | 1.000000 |
| utility damage | 20 | 0.082 | 0.867181 | 0.882867 | -0.015685 | 12 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 131 | 0.537 | 0.870118 | 0.900430 | -0.030312 | 34 | 97 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 244 | 1.000 | 0.863106 | 0.906484 | -0.043379 | 34 | 210 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `48.5s`, rows `85`
- `65.5s` - `88.0s`, rows `46`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.7526`, XGBoost `0.8595`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `77.0`, LSTM `0.8260`, XGBoost `0.9166`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `76.5`, LSTM `0.8296`, XGBoost `0.9166`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.7666`, XGBoost `0.8530`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `10.0`, LSTM `0.7686`, XGBoost `0.8531`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.7754`, XGBoost `0.8586`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.7756`, XGBoost `0.8584`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `78.5`, LSTM `0.8339`, XGBoost `0.9164`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.7763`, XGBoost `0.8573`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `77.5`, LSTM `0.8365`, XGBoost `0.9164`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
