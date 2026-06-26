# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-lynn-vision-vs-housebets-bo3-GrWDn9AJOxYQcZMXkSI-Tw/lynn-vision-vs-housebets-m2-dust2.csv`
- round_num: `13`
- rows: `157`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 157 | 1.000 | 0.476671 | 0.482914 | -0.006243 | 89 | 68 | 0.452229 | 0.464968 |
| active/recent utility | 157 | 1.000 | 0.476671 | 0.482914 | -0.006243 | 89 | 68 | 0.452229 | 0.464968 |
| strong utility action | 44 | 0.280 | 0.494411 | 0.513476 | -0.019065 | 24 | 20 | 0.318182 | 0.090909 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 44 | 0.280 | 0.494411 | 0.513476 | -0.019065 | 24 | 20 | 0.318182 | 0.090909 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 157 | 1.000 | 0.476671 | 0.482914 | -0.006243 | 89 | 68 | 0.452229 | 0.464968 |

## Active Smoke/Inferno Intervals

- `27.5s` - `49.0s`, rows `44`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.0`, LSTM `0.5020`, XGBoost `0.6289`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.5`, LSTM `0.3653`, XGBoost `0.2852`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.4726`, XGBoost `0.5451`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.7241`, XGBoost `0.7915`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `48.0`, LSTM `0.2198`, XGBoost `0.2868`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.5134`, XGBoost `0.5793`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.4742`, XGBoost `0.5398`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.4612`, XGBoost `0.5265`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.4745`, XGBoost `0.5362`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.4792`, XGBoost `0.5409`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
