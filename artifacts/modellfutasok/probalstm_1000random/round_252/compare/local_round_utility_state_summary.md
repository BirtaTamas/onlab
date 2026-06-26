# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m2-mirage.csv`
- round_num: `7`
- rows: `219`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 219 | 1.000 | 0.196499 | 0.212771 | -0.016272 | 172 | 47 | 0.853881 | 1.000000 |
| active/recent utility | 219 | 1.000 | 0.196499 | 0.212771 | -0.016272 | 172 | 47 | 0.853881 | 1.000000 |
| strong utility action | 122 | 0.557 | 0.304060 | 0.323740 | -0.019679 | 77 | 45 | 0.737705 | 1.000000 |
| utility damage | 12 | 0.055 | 0.185240 | 0.217236 | -0.031996 | 10 | 2 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.557 | 0.304060 | 0.323740 | -0.019679 | 77 | 45 | 0.737705 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 219 | 1.000 | 0.196499 | 0.212771 | -0.016272 | 172 | 47 | 0.853881 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `67.0s`, rows `122`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `48.0`, LSTM `0.2000`, XGBoost `0.3518`, closer `lstm`, smoke `4`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.0569`, XGBoost `0.2024`, closer `lstm`, smoke `5`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `48.5`, LSTM `0.2218`, XGBoost `0.3518`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.2205`, XGBoost `0.3402`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `29.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.0054`, XGBoost `0.1141`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.0`, LSTM `0.0055`, XGBoost `0.1141`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.5`, LSTM `0.0057`, XGBoost `0.1141`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.5`, LSTM `0.0058`, XGBoost `0.1133`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `63.5`, LSTM `0.0052`, XGBoost `0.1122`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `61.0`, LSTM `0.0069`, XGBoost `0.1133`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
