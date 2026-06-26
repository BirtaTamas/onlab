# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-aurora-vs-faze-bo3-ZgdBOa3Yi0KCkwa_Ap1ef3/aurora-vs-faze-m2-train.csv`
- round_num: `6`
- rows: `236`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 236 | 1.000 | 0.261034 | 0.329789 | -0.068755 | 236 | 0 | 0.872881 | 0.491525 |
| active/recent utility | 236 | 1.000 | 0.261034 | 0.329789 | -0.068755 | 236 | 0 | 0.872881 | 0.491525 |
| strong utility action | 176 | 0.746 | 0.308652 | 0.385444 | -0.076792 | 176 | 0 | 0.829545 | 0.409091 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 176 | 0.746 | 0.308652 | 0.385444 | -0.076792 | 176 | 0 | 0.829545 | 0.409091 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 236 | 1.000 | 0.261034 | 0.329789 | -0.068755 | 236 | 0 | 0.872881 | 0.491525 |

## Active Smoke/Inferno Intervals

- `8.0s` - `95.5s`, rows `176`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `73.0`, LSTM `0.0868`, XGBoost `0.3641`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.5`, LSTM `0.0516`, XGBoost `0.2985`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `71.0`, LSTM `0.0582`, XGBoost `0.2957`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.5`, LSTM `0.0505`, XGBoost `0.2876`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `72.0`, LSTM `0.0535`, XGBoost `0.2876`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `62.0`, LSTM `0.1173`, XGBoost `0.3493`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `68.0`, LSTM `0.0443`, XGBoost `0.2747`, closer `lstm`, smoke `2`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `66.0`, LSTM `0.1376`, XGBoost `0.3676`, closer `lstm`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `69.5`, LSTM `0.0691`, XGBoost `0.2970`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.0501`, XGBoost `0.2778`, closer `lstm`, smoke `3`, inferno `2`, utility_damage `0.0`, recent_utility `0`
