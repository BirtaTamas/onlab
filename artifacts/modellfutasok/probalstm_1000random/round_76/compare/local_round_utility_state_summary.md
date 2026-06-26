# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `9`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.546374 | 0.599723 | -0.053349 | 38 | 192 | 0.782609 | 0.882609 |
| active/recent utility | 230 | 1.000 | 0.546374 | 0.599723 | -0.053349 | 38 | 192 | 0.782609 | 0.882609 |
| strong utility action | 142 | 0.617 | 0.560771 | 0.623634 | -0.062863 | 10 | 132 | 0.866197 | 1.000000 |
| utility damage | 10 | 0.043 | 0.591863 | 0.732229 | -0.140366 | 0 | 10 | 1.000000 | 1.000000 |
| active smoke/inferno | 132 | 0.574 | 0.567721 | 0.629680 | -0.061960 | 10 | 122 | 0.924242 | 1.000000 |
| recent utility last 5s | 10 | 0.043 | 0.469039 | 0.543831 | -0.074792 | 0 | 10 | 0.100000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.546374 | 0.599723 | -0.053349 | 38 | 192 | 0.782609 | 0.882609 |

## Active Smoke/Inferno Intervals

- `7.5s` - `66.5s`, rows `119`
- `108.5s` - `114.5s`, rows `13`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `111.5`, LSTM `0.6575`, XGBoost `0.8807`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `114.5`, LSTM `0.4952`, XGBoost `0.7067`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `114.0`, LSTM `0.4959`, XGBoost `0.7067`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5359`, XGBoost `0.7441`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `110.0`, LSTM `0.6781`, XGBoost `0.8859`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `111.0`, LSTM `0.6843`, XGBoost `0.8864`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.6082`, XGBoost `0.8043`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `113.5`, LSTM `0.5071`, XGBoost `0.6970`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `19.0`, recent_utility `0`
- seconds `109.0`, LSTM `0.7083`, XGBoost `0.8873`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `1.0`, recent_utility `0`
- seconds `112.0`, LSTM `0.7078`, XGBoost `0.8850`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `19.0`, recent_utility `0`
