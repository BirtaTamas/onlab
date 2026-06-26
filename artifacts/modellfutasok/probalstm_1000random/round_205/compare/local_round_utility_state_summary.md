# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-inner-circle-vs-furia-bo3-bgGti4JPo_3k74mZn1hWMp/inner-circle-vs-furia-m1-mirage.csv`
- round_num: `5`
- rows: `165`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 165 | 1.000 | 0.598444 | 0.608713 | -0.010269 | 57 | 108 | 0.860606 | 0.860606 |
| active/recent utility | 165 | 1.000 | 0.598444 | 0.608713 | -0.010269 | 57 | 108 | 0.860606 | 0.860606 |
| strong utility action | 152 | 0.921 | 0.597960 | 0.610017 | -0.012056 | 46 | 106 | 0.848684 | 0.848684 |
| utility damage | 20 | 0.121 | 0.680684 | 0.624190 | 0.056493 | 20 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 152 | 0.921 | 0.597960 | 0.610017 | -0.012056 | 46 | 106 | 0.848684 | 0.848684 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 165 | 1.000 | 0.598444 | 0.608713 | -0.010269 | 57 | 108 | 0.860606 | 0.860606 |

## Active Smoke/Inferno Intervals

- `6.5s` - `82.0s`, rows `152`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `23.5`, LSTM `0.7071`, XGBoost `0.6090`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `10.0`, recent_utility `0`
- seconds `60.5`, LSTM `0.5257`, XGBoost `0.6182`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `60.0`, LSTM `0.5295`, XGBoost `0.6182`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `73.0`, LSTM `0.5908`, XGBoost `0.6730`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.2524`, XGBoost `0.3312`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.2529`, XGBoost `0.3312`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `58.5`, LSTM `0.5300`, XGBoost `0.6074`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `59.5`, LSTM `0.5437`, XGBoost `0.6182`, closer `xgboost`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.6933`, XGBoost `0.6217`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `14.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.5063`, XGBoost `0.5772`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
