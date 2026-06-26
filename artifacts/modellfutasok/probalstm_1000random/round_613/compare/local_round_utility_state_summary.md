# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `3`
- rows: `127`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 127 | 1.000 | 0.620038 | 0.643922 | -0.023885 | 51 | 76 | 0.834646 | 0.732283 |
| active/recent utility | 127 | 1.000 | 0.620038 | 0.643922 | -0.023885 | 51 | 76 | 0.834646 | 0.732283 |
| strong utility action | 108 | 0.850 | 0.612989 | 0.643971 | -0.030981 | 38 | 70 | 0.805556 | 0.805556 |
| utility damage | 20 | 0.157 | 0.531426 | 0.514362 | 0.017064 | 18 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 108 | 0.850 | 0.612989 | 0.643971 | -0.030981 | 38 | 70 | 0.805556 | 0.805556 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 127 | 1.000 | 0.620038 | 0.643922 | -0.023885 | 51 | 76 | 0.834646 | 0.732283 |

## Active Smoke/Inferno Intervals

- `6.5s` - `60.0s`, rows `108`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `51.5`, LSTM `0.6638`, XGBoost `0.7994`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.6120`, XGBoost `0.7466`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.6135`, XGBoost `0.7466`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.6116`, XGBoost `0.7427`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.6174`, XGBoost `0.7476`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.6176`, XGBoost `0.7466`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.6265`, XGBoost `0.7476`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.6257`, XGBoost `0.7466`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.6282`, XGBoost `0.7476`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.6308`, XGBoost `0.7476`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
