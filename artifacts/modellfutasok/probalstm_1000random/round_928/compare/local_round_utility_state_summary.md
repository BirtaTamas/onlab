# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_21_stage_1/esl-pro-league-season-21-stage-1-heroic-vs-saw-bo3-tIR5RlOpBrnlpEe6MBVyNd/heroic-vs-saw-m2-train.csv`
- round_num: `17`
- rows: `201`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 201 | 1.000 | 0.365321 | 0.447519 | -0.082198 | 191 | 10 | 0.955224 | 0.507463 |
| active/recent utility | 201 | 1.000 | 0.365321 | 0.447519 | -0.082198 | 191 | 10 | 0.955224 | 0.507463 |
| strong utility action | 86 | 0.428 | 0.363425 | 0.448298 | -0.084872 | 76 | 10 | 0.941860 | 0.825581 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 77 | 0.383 | 0.382753 | 0.456716 | -0.073963 | 67 | 10 | 0.935065 | 0.805195 |
| recent utility last 5s | 11 | 0.055 | 0.201273 | 0.376571 | -0.175298 | 11 | 0 | 1.000000 | 1.000000 |
| flash effect present | 201 | 1.000 | 0.365321 | 0.447519 | -0.082198 | 191 | 10 | 0.955224 | 0.507463 |

## Active Smoke/Inferno Intervals

- `8.5s` - `46.5s`, rows `77`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `5.0`, LSTM `0.1711`, XGBoost `0.3648`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `5.5`, LSTM `0.1905`, XGBoost `0.3823`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.0`, LSTM `0.1908`, XGBoost `0.3823`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.0`, LSTM `0.1755`, XGBoost `0.3648`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `4.5`, LSTM `0.1774`, XGBoost `0.3648`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `8.5`, LSTM `0.2074`, XGBoost `0.3813`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `6.5`, LSTM `0.2114`, XGBoost `0.3823`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `19.5`, LSTM `0.0948`, XGBoost `0.2641`, closer `lstm`, smoke `6`, inferno `1`, utility_damage `37.0`, recent_utility `0`
- seconds `8.0`, LSTM `0.2210`, XGBoost `0.3817`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
- seconds `7.0`, LSTM `0.2223`, XGBoost `0.3817`, closer `lstm`, smoke `0`, inferno `0`, utility_damage `0.0`, recent_utility `1`
