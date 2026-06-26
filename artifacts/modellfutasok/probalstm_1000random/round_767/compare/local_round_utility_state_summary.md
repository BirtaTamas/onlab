# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-mouz-bo3-Ko5VJMvyF1OsCx2TbVU9pb/vitality-vs-mouz-m1-inferno.csv`
- round_num: `13`
- rows: `115`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 115 | 1.000 | 0.518003 | 0.552327 | -0.034324 | 25 | 90 | 0.330435 | 0.765217 |
| active/recent utility | 115 | 1.000 | 0.518003 | 0.552327 | -0.034324 | 25 | 90 | 0.330435 | 0.765217 |
| strong utility action | 25 | 0.217 | 0.601557 | 0.652520 | -0.050963 | 5 | 20 | 0.440000 | 0.760000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 15 | 0.130 | 0.682511 | 0.725754 | -0.043243 | 5 | 10 | 0.733333 | 0.600000 |
| recent utility last 5s | 10 | 0.087 | 0.480126 | 0.542669 | -0.062543 | 0 | 10 | 0.000000 | 1.000000 |
| flash effect present | 115 | 1.000 | 0.518003 | 0.552327 | -0.034324 | 25 | 90 | 0.330435 | 0.765217 |

## Active Smoke/Inferno Intervals

- `50.0s` - `57.0s`, rows `15`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `54.5`, LSTM `0.8135`, XGBoost `0.9568`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.8172`, XGBoost `0.9564`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6413`, XGBoost `0.7651`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.8877`, XGBoost `0.9853`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.8939`, XGBoost `0.9854`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.9031`, XGBoost `0.9858`, closer `xgboost`, smoke `0`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.4721`, XGBoost `0.5427`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `15.5`, LSTM `0.4745`, XGBoost `0.5427`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `15.0`, LSTM `0.4751`, XGBoost `0.5427`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `14.0`, LSTM `0.4763`, XGBoost `0.5427`, closer `xgboost`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
