# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-furia-vs-fluxo-bo3-cy88FeSpEinhT8XDRxQGHo/furia-vs-fluxo-m2-mirage.csv`
- round_num: `1`
- rows: `151`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 151 | 1.000 | 0.519630 | 0.615344 | -0.095714 | 1 | 150 | 0.642384 | 0.761589 |
| active/recent utility | 151 | 1.000 | 0.519630 | 0.615344 | -0.095714 | 1 | 150 | 0.642384 | 0.761589 |
| strong utility action | 49 | 0.325 | 0.532445 | 0.665276 | -0.132831 | 0 | 49 | 0.571429 | 0.795918 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 49 | 0.325 | 0.532445 | 0.665276 | -0.132831 | 0 | 49 | 0.571429 | 0.795918 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 151 | 1.000 | 0.519630 | 0.615344 | -0.095714 | 1 | 150 | 0.642384 | 0.761589 |

## Active Smoke/Inferno Intervals

- `26.0s` - `50.0s`, rows `49`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `45.0`, LSTM `0.3182`, XGBoost `0.5390`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.7176`, XGBoost `0.9215`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.3521`, XGBoost `0.5495`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.4505`, XGBoost `0.6477`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.7328`, XGBoost `0.9215`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1280`, XGBoost `0.3151`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3653`, XGBoost `0.5514`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.7363`, XGBoost `0.9215`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1261`, XGBoost `0.3091`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.7395`, XGBoost `0.9215`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
