# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-gamerlegion-bo3-HfAhqHTEhpe_HlObeToa76/vitality-vs-gamerlegion-m1-overpass.csv`
- round_num: `1`
- rows: `106`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 106 | 1.000 | 0.453183 | 0.637239 | -0.184056 | 0 | 106 | 0.311321 | 0.858491 |
| active/recent utility | 106 | 1.000 | 0.453183 | 0.637239 | -0.184056 | 0 | 106 | 0.311321 | 0.858491 |
| strong utility action | 44 | 0.415 | 0.482877 | 0.760133 | -0.277256 | 0 | 44 | 0.431818 | 0.954545 |
| utility damage | 10 | 0.094 | 0.300102 | 0.659791 | -0.359688 | 0 | 10 | 0.000000 | 1.000000 |
| active smoke/inferno | 44 | 0.415 | 0.482877 | 0.760133 | -0.277256 | 0 | 44 | 0.431818 | 0.954545 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 106 | 1.000 | 0.453183 | 0.637239 | -0.184056 | 0 | 106 | 0.311321 | 0.858491 |

## Active Smoke/Inferno Intervals

- `27.0s` - `48.5s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.5`, LSTM `0.1516`, XGBoost `0.6534`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.2115`, XGBoost `0.7086`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `32.0`, recent_utility `0`
- seconds `30.0`, LSTM `0.1621`, XGBoost `0.6339`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.3778`, XGBoost `0.8338`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.1686`, XGBoost `0.6156`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.1700`, XGBoost `0.6164`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.1709`, XGBoost `0.6164`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `34.0`, LSTM `0.2625`, XGBoost `0.6936`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `32.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.2226`, XGBoost `0.6180`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.2549`, XGBoost `0.6402`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
