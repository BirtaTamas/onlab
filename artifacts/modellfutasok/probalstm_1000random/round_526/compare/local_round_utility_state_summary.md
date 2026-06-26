# Local Round Utility State Analysis

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-chinggis-warriors-vs-fluxo-bo3-q_dqfGh9bi4kDnaRAX0wjf/chinggis-warriors-vs-fluxo-m2-mirage.csv`
- round_num: `17`
- rows: `230`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.617728 | 0.695664 | -0.077936 | 19 | 211 | 0.604348 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.617728 | 0.695664 | -0.077936 | 19 | 211 | 0.604348 | 1.000000 |
| strong utility action | 138 | 0.600 | 0.498144 | 0.593581 | -0.095437 | 18 | 120 | 0.405797 | 1.000000 |
| utility damage | 10 | 0.043 | 0.557059 | 0.567751 | -0.010692 | 2 | 8 | 1.000000 | 1.000000 |
| active smoke/inferno | 122 | 0.530 | 0.486922 | 0.585453 | -0.098530 | 15 | 107 | 0.385246 | 1.000000 |
| recent utility last 5s | 20 | 0.087 | 0.561506 | 0.639252 | -0.077746 | 3 | 17 | 0.500000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.617728 | 0.695664 | -0.077936 | 19 | 211 | 0.604348 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `67.0s`, rows `122`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.5`, LSTM `0.3641`, XGBoost `0.6028`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.0`, LSTM `0.3686`, XGBoost `0.6028`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.3561`, XGBoost `0.5883`, closer `xgboost`, smoke `5`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `40.0`, LSTM `0.3751`, XGBoost `0.5951`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.3838`, XGBoost `0.6028`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.3371`, XGBoost `0.5532`, closer `xgboost`, smoke `6`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.3871`, XGBoost `0.6028`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.5`, LSTM `0.3562`, XGBoost `0.5710`, closer `xgboost`, smoke `6`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `37.5`, LSTM `0.3841`, XGBoost `0.5915`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `38.0`, LSTM `0.3890`, XGBoost `0.5931`, closer `xgboost`, smoke `5`, inferno `1`, utility_damage `0.0`, recent_utility `0`
