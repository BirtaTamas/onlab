# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-pain-vs-housebets-bo3-SOezkQe1hszxnf1QDg0VUC/pain-vs-housebets-m1-dust2.csv`
- round_num: `12`
- rows: `189`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.346654 | 0.281547 | 0.065107 | 64 | 125 | 0.656085 | 0.878307 |
| active/recent utility | 189 | 1.000 | 0.346654 | 0.281547 | 0.065107 | 64 | 125 | 0.656085 | 0.878307 |
| strong utility action | 162 | 0.857 | 0.338702 | 0.271378 | 0.067324 | 52 | 110 | 0.691358 | 0.950617 |
| utility damage | 10 | 0.053 | 0.050741 | 0.102910 | -0.052169 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 162 | 0.857 | 0.338702 | 0.271378 | 0.067324 | 52 | 110 | 0.691358 | 0.950617 |
| recent utility last 5s | 10 | 0.053 | 0.003433 | 0.017543 | -0.014110 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 189 | 1.000 | 0.346654 | 0.281547 | 0.065107 | 64 | 125 | 0.656085 | 0.878307 |

## Active Smoke/Inferno Intervals

- `7.5s` - `88.0s`, rows `162`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `49.5`, LSTM `0.4766`, XGBoost `0.2657`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.4469`, XGBoost `0.2686`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5179`, XGBoost `0.3541`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5157`, XGBoost `0.3541`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5157`, XGBoost `0.3541`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.5137`, XGBoost `0.3531`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.4182`, XGBoost `0.2578`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5259`, XGBoost `0.3670`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.5`, LSTM `0.5119`, XGBoost `0.3537`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.5119`, XGBoost `0.3537`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
