# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-vitality-vs-3dmax-bo3-SFueR4Yd1u5-bIhh5XKwOq/vitality-vs-3dmax-m2-dust2.csv`
- round_num: `1`
- rows: `152`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 152 | 1.000 | 0.661731 | 0.739877 | -0.078147 | 38 | 114 | 0.953947 | 0.776316 |
| active/recent utility | 152 | 1.000 | 0.661731 | 0.739877 | -0.078147 | 38 | 114 | 0.953947 | 0.776316 |
| strong utility action | 135 | 0.888 | 0.666151 | 0.750366 | -0.084215 | 31 | 104 | 0.948148 | 0.762963 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 125 | 0.822 | 0.679403 | 0.770962 | -0.091559 | 23 | 102 | 0.976000 | 0.808000 |
| recent utility last 5s | 10 | 0.066 | 0.500502 | 0.492916 | 0.007586 | 8 | 2 | 0.600000 | 0.200000 |
| flash effect present | 152 | 1.000 | 0.661731 | 0.739877 | -0.078147 | 38 | 114 | 0.953947 | 0.776316 |

## Active Smoke/Inferno Intervals

- `10.5s` - `72.5s`, rows `125`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `37.5`, LSTM `0.5346`, XGBoost `0.7323`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `29.0`, LSTM `0.5302`, XGBoost `0.7260`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5382`, XGBoost `0.7323`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5419`, XGBoost `0.7349`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5406`, XGBoost `0.7323`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.5347`, XGBoost `0.7260`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5415`, XGBoost `0.7323`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.5432`, XGBoost `0.7328`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5462`, XGBoost `0.7349`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5459`, XGBoost `0.7323`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
