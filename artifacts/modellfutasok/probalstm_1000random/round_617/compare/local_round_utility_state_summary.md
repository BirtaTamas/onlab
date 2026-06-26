# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-nemiga-bo3-HBPh0RFmxqP1tE9QMaq3nA/heroic-vs-nemiga-m2-mirage.csv`
- round_num: `9`
- rows: `176`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 176 | 1.000 | 0.115984 | 0.145105 | -0.029121 | 165 | 11 | 1.000000 | 1.000000 |
| active/recent utility | 176 | 1.000 | 0.115984 | 0.145105 | -0.029121 | 165 | 11 | 1.000000 | 1.000000 |
| strong utility action | 152 | 0.864 | 0.130144 | 0.162190 | -0.032046 | 141 | 11 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.057 | 0.099236 | 0.201411 | -0.102175 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 134 | 0.761 | 0.116800 | 0.148346 | -0.031546 | 123 | 11 | 1.000000 | 1.000000 |
| recent utility last 5s | 20 | 0.114 | 0.232785 | 0.273279 | -0.040494 | 20 | 0 | 1.000000 | 1.000000 |
| flash effect present | 176 | 1.000 | 0.115984 | 0.145105 | -0.029121 | 165 | 11 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.0s` - `74.5s`, rows `134`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `33.5`, LSTM `0.0662`, XGBoost `0.2103`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `63.0`, recent_utility `0`
- seconds `28.0`, LSTM `0.1787`, XGBoost `0.3165`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `28.5`, LSTM `0.2115`, XGBoost `0.3447`, closer `lstm`, smoke `4`, inferno `2`, utility_damage `1.0`, recent_utility `0`
- seconds `36.5`, LSTM `0.0720`, XGBoost `0.2048`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `34.5`, LSTM `0.0682`, XGBoost `0.1975`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `35.0`, LSTM `0.0701`, XGBoost `0.1975`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `33.0`, LSTM `0.0829`, XGBoost `0.2103`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `63.0`, recent_utility `0`
- seconds `37.0`, LSTM `0.0656`, XGBoost `0.1922`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `8.0`, recent_utility `0`
- seconds `35.5`, LSTM `0.0755`, XGBoost `0.2010`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `32.5`, LSTM `0.0863`, XGBoost `0.2103`, closer `lstm`, smoke `3`, inferno `3`, utility_damage `63.0`, recent_utility `0`
