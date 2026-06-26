# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-vitality-vs-hotu-bo3-g2oB3RySVGugmKq6yJcHo9/vitality-vs-hotu-m2-dust2.csv`
- round_num: `16`
- rows: `117`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 117 | 1.000 | 0.788546 | 0.815061 | -0.026515 | 39 | 78 | 0.957265 | 1.000000 |
| active/recent utility | 117 | 1.000 | 0.788546 | 0.815061 | -0.026515 | 39 | 78 | 0.957265 | 1.000000 |
| strong utility action | 80 | 0.684 | 0.769551 | 0.805541 | -0.035990 | 26 | 54 | 0.937500 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 65 | 0.556 | 0.772590 | 0.799572 | -0.026983 | 26 | 39 | 0.923077 | 1.000000 |
| recent utility last 5s | 15 | 0.128 | 0.756382 | 0.831404 | -0.075022 | 0 | 15 | 1.000000 | 1.000000 |
| flash effect present | 117 | 1.000 | 0.788546 | 0.815061 | -0.026515 | 39 | 78 | 0.957265 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `13.5s`, rows `11`
- `15.5s` - `42.0s`, rows `54`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `39.5`, LSTM `0.4707`, XGBoost `0.7423`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `39.0`, LSTM `0.4811`, XGBoost `0.7441`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3114`, XGBoost `0.5617`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.4935`, XGBoost `0.7385`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.5`, LSTM `0.5109`, XGBoost `0.7402`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.5529`, XGBoost `0.7397`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `40.5`, LSTM `0.5627`, XGBoost `0.7385`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `41.0`, LSTM `0.5872`, XGBoost `0.7485`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.4158`, XGBoost `0.5617`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `42.0`, LSTM `0.6070`, XGBoost `0.7470`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
