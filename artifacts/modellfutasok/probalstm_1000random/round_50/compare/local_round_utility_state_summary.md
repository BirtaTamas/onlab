# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-legacy-vs-nrg-bo3-_uO_eo-VIGwp_pYoaUs9Le/legacy-vs-nrg-m3-dust2.csv`
- round_num: `4`
- rows: `103`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 103 | 1.000 | 0.008541 | 0.018474 | -0.009933 | 100 | 3 | 1.000000 | 1.000000 |
| active/recent utility | 103 | 1.000 | 0.008541 | 0.018474 | -0.009933 | 100 | 3 | 1.000000 | 1.000000 |
| strong utility action | 59 | 0.573 | 0.011013 | 0.024273 | -0.013261 | 57 | 2 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.107 | 0.014956 | 0.033559 | -0.018603 | 11 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 59 | 0.573 | 0.011013 | 0.024273 | -0.013261 | 57 | 2 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 103 | 1.000 | 0.008541 | 0.018474 | -0.009933 | 100 | 3 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `3.0s` - `25.0s`, rows `45`
- `33.5s` - `40.0s`, rows `14`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `9.5`, LSTM `0.0389`, XGBoost `0.1014`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `58.0`, recent_utility `0`
- seconds `3.0`, LSTM `0.0190`, XGBoost `0.0553`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `7.5`, LSTM `0.0216`, XGBoost `0.0568`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `7.0`, LSTM `0.0226`, XGBoost `0.0568`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.0232`, XGBoost `0.0568`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `4.5`, LSTM `0.0230`, XGBoost `0.0545`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `3.5`, LSTM `0.0235`, XGBoost `0.0543`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.0264`, XGBoost `0.0568`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.0267`, XGBoost `0.0568`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `5.0`, LSTM `0.0244`, XGBoost `0.0545`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
