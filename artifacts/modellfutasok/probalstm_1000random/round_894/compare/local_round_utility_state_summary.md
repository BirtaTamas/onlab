# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `15`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.845098 | 0.869352 | -0.024254 | 37 | 193 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.845098 | 0.869352 | -0.024254 | 37 | 193 | 1.000000 | 1.000000 |
| strong utility action | 105 | 0.457 | 0.730344 | 0.762556 | -0.032213 | 32 | 73 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.043 | 0.766273 | 0.751497 | 0.014776 | 4 | 6 | 1.000000 | 1.000000 |
| active smoke/inferno | 93 | 0.404 | 0.741665 | 0.785287 | -0.043622 | 20 | 73 | 1.000000 | 1.000000 |
| recent utility last 5s | 18 | 0.078 | 0.624099 | 0.573204 | 0.050896 | 18 | 0 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.845098 | 0.869352 | -0.024254 | 37 | 193 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `54.5s`, rows `93`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `30.0`, LSTM `0.7576`, XGBoost `0.9139`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.7568`, XGBoost `0.9114`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `30.5`, LSTM `0.7663`, XGBoost `0.9138`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.7681`, XGBoost `0.9114`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.7745`, XGBoost `0.9142`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.7777`, XGBoost `0.9142`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.7768`, XGBoost `0.9114`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.7876`, XGBoost `0.9112`, closer `xgboost`, smoke `6`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.7916`, XGBoost `0.9142`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.5`, LSTM `0.6769`, XGBoost `0.5607`, closer `lstm`, smoke `5`, inferno `0`, utility_damage `14.0`, recent_utility `0`
