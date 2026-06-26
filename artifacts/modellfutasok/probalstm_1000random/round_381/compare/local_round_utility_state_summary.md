# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21/esl-pro-league-season-21-gamerlegion-vs-tyloo-bo3-0g9mXt3FIxC8XzjXNUjRL7/gamerlegion-vs-tyloo-m1-ancient-p3.csv`
- round_num: `6`
- rows: `164`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 164 | 1.000 | 0.450586 | 0.593737 | -0.143151 | 10 | 154 | 0.640244 | 0.774390 |
| active/recent utility | 164 | 1.000 | 0.450586 | 0.593737 | -0.143151 | 10 | 154 | 0.640244 | 0.774390 |
| strong utility action | 103 | 0.628 | 0.527068 | 0.631371 | -0.104303 | 10 | 93 | 0.854369 | 0.854369 |
| utility damage | 10 | 0.061 | 0.448653 | 0.558144 | -0.109491 | 1 | 9 | 0.300000 | 0.300000 |
| active smoke/inferno | 103 | 0.628 | 0.527068 | 0.631371 | -0.104303 | 10 | 93 | 0.854369 | 0.854369 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 164 | 1.000 | 0.450586 | 0.593737 | -0.143151 | 10 | 154 | 0.640244 | 0.774390 |

## Active Smoke/Inferno Intervals

- `6.0s` - `57.0s`, rows `103`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `57.0`, LSTM `0.1362`, XGBoost `0.4163`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.5`, LSTM `0.1567`, XGBoost `0.4163`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `56.0`, LSTM `0.1630`, XGBoost `0.4163`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.5`, LSTM `0.1724`, XGBoost `0.4163`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.1753`, XGBoost `0.4163`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `55.0`, LSTM `0.1820`, XGBoost `0.4163`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `31.0`, LSTM `0.2781`, XGBoost `0.4976`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `7.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.5267`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `32.0`, LSTM `0.5284`, XGBoost `0.7387`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `31.5`, LSTM `0.3114`, XGBoost `0.4976`, closer `xgboost`, smoke `4`, inferno `1`, utility_damage `7.0`, recent_utility `0`
