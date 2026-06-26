# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m2-mirage.csv`
- round_num: `14`
- rows: `128`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 128 | 1.000 | 0.607011 | 0.723323 | -0.116313 | 6 | 122 | 1.000000 | 1.000000 |
| active/recent utility | 128 | 1.000 | 0.607011 | 0.723323 | -0.116313 | 6 | 122 | 1.000000 | 1.000000 |
| strong utility action | 104 | 0.812 | 0.606625 | 0.728812 | -0.122187 | 6 | 98 | 1.000000 | 1.000000 |
| utility damage | 11 | 0.086 | 0.683214 | 0.714393 | -0.031178 | 3 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 104 | 0.812 | 0.606625 | 0.728812 | -0.122187 | 6 | 98 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 128 | 1.000 | 0.607011 | 0.723323 | -0.116313 | 6 | 122 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `59.5s`, rows `104`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.5409`, XGBoost `0.7960`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.5560`, XGBoost `0.8077`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.5576`, XGBoost `0.8091`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.5620`, XGBoost `0.8091`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.5583`, XGBoost `0.8046`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.5445`, XGBoost `0.7901`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5685`, XGBoost `0.8114`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5675`, XGBoost `0.8091`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5699`, XGBoost `0.8112`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5669`, XGBoost `0.8068`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
