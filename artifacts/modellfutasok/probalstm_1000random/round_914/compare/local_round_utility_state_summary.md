# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-falcons-bo3-yayytstbo8IxTFlUpfbUPR/mouz-vs-falcons-m1-train.csv`
- round_num: `4`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.539541 | 0.621627 | -0.082086 | 6 | 224 | 0.347826 | 0.991304 |
| active/recent utility | 230 | 1.000 | 0.539541 | 0.621627 | -0.082086 | 6 | 224 | 0.347826 | 0.991304 |
| strong utility action | 208 | 0.904 | 0.542712 | 0.622537 | -0.079825 | 6 | 202 | 0.370192 | 0.990385 |
| utility damage | 24 | 0.104 | 0.454311 | 0.569855 | -0.115544 | 0 | 24 | 0.083333 | 1.000000 |
| active smoke/inferno | 208 | 0.904 | 0.542712 | 0.622537 | -0.079825 | 6 | 202 | 0.370192 | 0.990385 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 230 | 1.000 | 0.539541 | 0.621627 | -0.082086 | 6 | 224 | 0.347826 | 0.991304 |

## Active Smoke/Inferno Intervals

- `8.5s` - `55.5s`, rows `95`
- `57.0s` - `113.0s`, rows `113`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `36.5`, LSTM `0.3786`, XGBoost `0.5614`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.3916`, XGBoost `0.5710`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.3954`, XGBoost `0.5710`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.3985`, XGBoost `0.5710`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `36.0`, LSTM `0.4014`, XGBoost `0.5710`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `33.5`, LSTM `0.4571`, XGBoost `0.6244`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.4171`, XGBoost `0.5794`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.0`, LSTM `0.4250`, XGBoost `0.5845`, closer `xgboost`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.4292`, XGBoost `0.5846`, closer `xgboost`, smoke `0`, inferno `3`, utility_damage `1.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.2030`, XGBoost `0.3561`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
