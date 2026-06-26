# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `3`
- rows: `230`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 230 | 1.000 | 0.009848 | 0.026218 | -0.016370 | 227 | 3 | 1.000000 | 1.000000 |
| active/recent utility | 230 | 1.000 | 0.009848 | 0.026218 | -0.016370 | 227 | 3 | 1.000000 | 1.000000 |
| strong utility action | 125 | 0.543 | 0.009411 | 0.027268 | -0.017857 | 125 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 115 | 0.500 | 0.008600 | 0.024583 | -0.015983 | 115 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 10 | 0.043 | 0.018739 | 0.058144 | -0.039405 | 10 | 0 | 1.000000 | 1.000000 |
| flash effect present | 230 | 1.000 | 0.009848 | 0.026218 | -0.016370 | 227 | 3 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `9.5s` - `33.0s`, rows `48`
- `60.0s` - `66.5s`, rows `14`
- `69.5s` - `95.5s`, rows `53`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `66.5`, LSTM `0.0126`, XGBoost `0.0696`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `8.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.0121`, XGBoost `0.0663`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `21.0`, recent_utility `0`
- seconds `65.5`, LSTM `0.0136`, XGBoost `0.0664`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `65.0`, LSTM `0.0138`, XGBoost `0.0664`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.0111`, XGBoost `0.0625`, closer `lstm`, smoke `1`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `64.5`, LSTM `0.0150`, XGBoost `0.0664`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `34.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.0114`, XGBoost `0.0627`, closer `lstm`, smoke `2`, inferno `3`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.0115`, XGBoost `0.0625`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `10.5`, LSTM `0.0127`, XGBoost `0.0625`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `9.5`, LSTM `0.0133`, XGBoost `0.0616`, closer `lstm`, smoke `0`, inferno `1`, utility_damage `0.0`, recent_utility `0`
