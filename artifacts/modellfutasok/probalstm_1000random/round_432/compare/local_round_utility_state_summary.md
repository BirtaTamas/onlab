# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-housebets-bo3-NgyLHfqCvYO4WZnaqhUlfi/heroic-vs-housebets-m2-mirage.csv`
- round_num: `2`
- rows: `116`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 116 | 1.000 | 0.765567 | 0.797400 | -0.031832 | 0 | 116 | 1.000000 | 1.000000 |
| active/recent utility | 116 | 1.000 | 0.765567 | 0.797400 | -0.031832 | 0 | 116 | 1.000000 | 1.000000 |
| strong utility action | 84 | 0.724 | 0.745522 | 0.779988 | -0.034466 | 0 | 84 | 1.000000 | 1.000000 |
| utility damage | 12 | 0.103 | 0.860343 | 0.907007 | -0.046665 | 0 | 12 | 1.000000 | 1.000000 |
| active smoke/inferno | 84 | 0.724 | 0.745522 | 0.779988 | -0.034466 | 0 | 84 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 116 | 1.000 | 0.765567 | 0.797400 | -0.031832 | 0 | 116 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.0s` - `47.5s`, rows `84`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `27.5`, LSTM `0.6170`, XGBoost `0.7637`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `18.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.6755`, XGBoost `0.7735`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `24.0`, recent_utility `0`
- seconds `27.0`, LSTM `0.6581`, XGBoost `0.7505`, closer `xgboost`, smoke `7`, inferno `1`, utility_damage `9.0`, recent_utility `0`
- seconds `25.0`, LSTM `0.5296`, XGBoost `0.6187`, closer `xgboost`, smoke `5`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.5`, LSTM `0.5352`, XGBoost `0.6177`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `24.0`, LSTM `0.5557`, XGBoost `0.6177`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.5458`, XGBoost `0.6074`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.5540`, XGBoost `0.6098`, closer `xgboost`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `6.5`, LSTM `0.5449`, XGBoost `0.5990`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `8.5`, LSTM `0.5449`, XGBoost `0.5970`, closer `xgboost`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
