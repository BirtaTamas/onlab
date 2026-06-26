# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `6`
- rows: `162`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 162 | 1.000 | 0.661172 | 0.726262 | -0.065090 | 16 | 146 | 1.000000 | 1.000000 |
| active/recent utility | 162 | 1.000 | 0.661172 | 0.726262 | -0.065090 | 16 | 146 | 1.000000 | 1.000000 |
| strong utility action | 101 | 0.623 | 0.663728 | 0.740098 | -0.076371 | 0 | 101 | 1.000000 | 1.000000 |
| utility damage | 31 | 0.191 | 0.609138 | 0.673583 | -0.064444 | 0 | 31 | 1.000000 | 1.000000 |
| active smoke/inferno | 101 | 0.623 | 0.663728 | 0.740098 | -0.076371 | 0 | 101 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 162 | 1.000 | 0.661172 | 0.726262 | -0.065090 | 16 | 146 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `45.0s`, rows `75`
- `65.0s` - `71.5s`, rows `14`
- `75.0s` - `80.5s`, rows `12`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.0`, LSTM `0.6271`, XGBoost `0.7899`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6273`, XGBoost `0.7899`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.6335`, XGBoost `0.7896`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.6365`, XGBoost `0.7897`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.6394`, XGBoost `0.7899`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.6414`, XGBoost `0.7899`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6413`, XGBoost `0.7882`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6470`, XGBoost `0.7878`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.6518`, XGBoost `0.7882`, closer `xgboost`, smoke `7`, inferno `0`, utility_damage `5.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.6377`, XGBoost `0.7736`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
