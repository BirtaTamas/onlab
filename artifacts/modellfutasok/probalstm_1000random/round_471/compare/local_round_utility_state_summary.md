# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-mouz-vs-vitality-bo5-RwgqrXEuhDJTxQHhSIn72X/mouz-vs-vitality-m2-nuke.csv`
- round_num: `5`
- rows: `143`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 143 | 1.000 | 0.824512 | 0.882684 | -0.058171 | 3 | 140 | 1.000000 | 1.000000 |
| active/recent utility | 143 | 1.000 | 0.824512 | 0.882684 | -0.058171 | 3 | 140 | 1.000000 | 1.000000 |
| strong utility action | 103 | 0.720 | 0.800978 | 0.871269 | -0.070292 | 0 | 103 | 1.000000 | 1.000000 |
| utility damage | 21 | 0.147 | 0.665553 | 0.768005 | -0.102452 | 0 | 21 | 1.000000 | 1.000000 |
| active smoke/inferno | 99 | 0.692 | 0.804308 | 0.875262 | -0.070954 | 0 | 99 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.070 | 0.705413 | 0.771675 | -0.066262 | 0 | 10 | 1.000000 | 1.000000 |
| flash effect present | 143 | 1.000 | 0.824512 | 0.882684 | -0.058171 | 3 | 140 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `51.5s`, rows `88`
- `56.5s` - `61.5s`, rows `11`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `25.0`, LSTM `0.6171`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `25.5`, LSTM `0.6174`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.6196`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `26.5`, LSTM `0.6214`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `18.0`, LSTM `0.6269`, XGBoost `0.7715`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.6269`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `27.5`, LSTM `0.6271`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.6298`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `22.5`, LSTM `0.6303`, XGBoost `0.7698`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.5`, LSTM `0.6299`, XGBoost `0.7692`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `1.0`, recent_utility `0`
