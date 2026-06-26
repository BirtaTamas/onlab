# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `8`
- rows: `214`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 214 | 1.000 | 0.257354 | 0.477925 | -0.220571 | 175 | 39 | 0.822430 | 0.621495 |
| active/recent utility | 214 | 1.000 | 0.257354 | 0.477925 | -0.220571 | 175 | 39 | 0.822430 | 0.621495 |
| strong utility action | 141 | 0.659 | 0.299875 | 0.507796 | -0.207921 | 113 | 28 | 0.758865 | 0.574468 |
| utility damage | 13 | 0.061 | 0.445221 | 0.387483 | 0.057737 | 1 | 12 | 0.846154 | 1.000000 |
| active smoke/inferno | 141 | 0.659 | 0.299875 | 0.507796 | -0.207921 | 113 | 28 | 0.758865 | 0.574468 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 214 | 1.000 | 0.257354 | 0.477925 | -0.220571 | 175 | 39 | 0.822430 | 0.621495 |

## Active Smoke/Inferno Intervals

- `10.0s` - `44.0s`, rows `69`
- `50.0s` - `78.5s`, rows `58`
- `97.0s` - `103.5s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `56.0`, LSTM `0.1141`, XGBoost `0.6773`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.1115`, XGBoost `0.6720`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.1175`, XGBoost `0.6693`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.1488`, XGBoost `0.6773`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1174`, XGBoost `0.6376`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.5`, LSTM `0.1153`, XGBoost `0.6239`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.1276`, XGBoost `0.6213`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.1398`, XGBoost `0.6220`, closer `lstm`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.1394`, XGBoost `0.6198`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `52.0`, LSTM `0.1485`, XGBoost `0.6207`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
