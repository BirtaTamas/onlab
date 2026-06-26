# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-astralis-bo3-EH-le-_LuObR5nGefXVoZY/aurora-vs-astralis-m3-overpass.csv`
- round_num: `11`
- rows: `190`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 190 | 1.000 | 0.533457 | 0.514191 | 0.019266 | 71 | 119 | 0.184211 | 0.221053 |
| active/recent utility | 190 | 1.000 | 0.533457 | 0.514191 | 0.019266 | 71 | 119 | 0.184211 | 0.221053 |
| strong utility action | 162 | 0.853 | 0.531948 | 0.526089 | 0.005859 | 64 | 98 | 0.172840 | 0.216049 |
| utility damage | 19 | 0.100 | 0.613189 | 0.518543 | 0.094645 | 0 | 19 | 0.000000 | 0.157895 |
| active smoke/inferno | 162 | 0.853 | 0.531948 | 0.526089 | 0.005859 | 64 | 98 | 0.172840 | 0.216049 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 190 | 1.000 | 0.533457 | 0.514191 | 0.019266 | 71 | 119 | 0.184211 | 0.221053 |

## Active Smoke/Inferno Intervals

- `9.0s` - `54.5s`, rows `92`
- `56.5s` - `91.0s`, rows `70`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.6885`, XGBoost `0.4949`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.6868`, XGBoost `0.4960`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.6842`, XGBoost `0.4949`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.6757`, XGBoost `0.5005`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.6678`, XGBoost `0.4992`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.6613`, XGBoost `0.4992`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.6331`, XGBoost `0.5024`, closer `xgboost`, smoke `1`, inferno `2`, utility_damage `4.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.6270`, XGBoost `0.5005`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `80.5`, LSTM `0.0319`, XGBoost `0.1535`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `81.0`, LSTM `0.0330`, XGBoost `0.1537`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
