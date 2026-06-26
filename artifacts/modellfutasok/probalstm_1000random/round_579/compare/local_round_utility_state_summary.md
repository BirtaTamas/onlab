# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-astralis-vs-fluxo-bo3-sWQe-jgKNP3vaioXQrjxgB/astralis-vs-fluxo-m3-nuke.csv`
- round_num: `13`
- rows: `157`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.727801 | 0.771266 | -0.043464 | 20 | 137 | 0.891720 | 0.904459 |
| active/recent utility | 157 | 1.000 | 0.727801 | 0.771266 | -0.043464 | 20 | 137 | 0.891720 | 0.904459 |
| strong utility action | 44 | 0.280 | 0.555442 | 0.643097 | -0.087655 | 11 | 33 | 0.659091 | 0.659091 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.280 | 0.555442 | 0.643097 | -0.087655 | 11 | 33 | 0.659091 | 0.659091 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 157 | 1.000 | 0.727801 | 0.771266 | -0.043464 | 20 | 137 | 0.891720 | 0.904459 |

## Active Smoke/Inferno Intervals

- `21.0s` - `42.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `26.5`, LSTM `0.5297`, XGBoost `0.7271`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.7564`, XGBoost `0.9517`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.5260`, XGBoost `0.7198`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.5428`, XGBoost `0.7309`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.5357`, XGBoost `0.7205`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5420`, XGBoost `0.7205`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.5193`, XGBoost `0.6943`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.5203`, XGBoost `0.6943`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5209`, XGBoost `0.6943`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5252`, XGBoost `0.6943`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
