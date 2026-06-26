# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22_stage_1/esl-pro-league-season-22-stage-1-g2-vs-fluxo-bo3-IhqycqXYyOA3DyfY0xuGyX/g2-vs-fluxo-m2-inferno.csv`
- round_num: `1`
- rows: `124`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 124 | 1.000 | 0.392812 | 0.365062 | 0.027750 | 27 | 97 | 0.370968 | 0.959677 |
| active/recent utility | 124 | 1.000 | 0.392812 | 0.365062 | 0.027750 | 27 | 97 | 0.370968 | 0.959677 |
| strong utility action | 89 | 0.718 | 0.388113 | 0.355074 | 0.033039 | 20 | 69 | 0.359551 | 0.943820 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 89 | 0.718 | 0.388113 | 0.355074 | 0.033039 | 20 | 69 | 0.359551 | 0.943820 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 124 | 1.000 | 0.392812 | 0.365062 | 0.027750 | 27 | 97 | 0.370968 | 0.959677 |

## Active Smoke/Inferno Intervals

- `9.0s` - `30.5s`, rows `44`
- `36.0s` - `58.0s`, rows `45`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `46.5`, LSTM `0.4889`, XGBoost `0.3647`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.5132`, XGBoost `0.4020`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.5288`, XGBoost `0.4381`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.5263`, XGBoost `0.4365`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.5262`, XGBoost `0.4365`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.5`, LSTM `0.5262`, XGBoost `0.4387`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `43.0`, LSTM `0.5252`, XGBoost `0.4387`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.5253`, XGBoost `0.4390`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.5`, LSTM `0.5218`, XGBoost `0.4390`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.5189`, XGBoost `0.4364`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
