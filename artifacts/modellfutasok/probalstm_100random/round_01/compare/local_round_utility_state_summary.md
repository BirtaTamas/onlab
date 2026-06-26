# Local Round Utility State Analysis

- csv_path: `processed_full/esl_pro_league_season_22/esl-pro-league-season-22-furia-vs-vitality-bo3-ZNzuF_vw0WBzn8QEbGrbgj/furia-vs-vitality-m1-overpass.csv`
- round_num: `20`
- rows: `153`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 153 | 1.000 | 0.604246 | 0.604924 | -0.000678 | 92 | 61 | 0.183007 | 0.183007 |
| active/recent utility | 153 | 1.000 | 0.604246 | 0.604924 | -0.000678 | 92 | 61 | 0.183007 | 0.183007 |
| strong utility action | 126 | 0.824 | 0.625387 | 0.623812 | 0.001575 | 79 | 47 | 0.150794 | 0.150794 |
| utility damage | 20 | 0.131 | 0.711505 | 0.649612 | 0.061893 | 6 | 14 | 0.000000 | 0.000000 |
| active smoke/inferno | 126 | 0.824 | 0.625387 | 0.623812 | 0.001575 | 79 | 47 | 0.150794 | 0.150794 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 153 | 1.000 | 0.604246 | 0.604924 | -0.000678 | 92 | 61 | 0.183007 | 0.183007 |

## Active Smoke/Inferno Intervals

- `9.0s` - `71.5s`, rows `126`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `52.0`, LSTM `0.7384`, XGBoost `0.5619`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `51.5`, LSTM `0.7428`, XGBoost `0.5693`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `52.5`, LSTM `0.7252`, XGBoost `0.5619`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `53.0`, LSTM `0.7161`, XGBoost `0.5619`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `53.5`, LSTM `0.7080`, XGBoost `0.5619`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.0`, LSTM `0.7036`, XGBoost `0.5619`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `54.5`, LSTM `0.6819`, XGBoost `0.5622`, closer `xgboost`, smoke `1`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `51.0`, LSTM `0.7047`, XGBoost `0.5903`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `4.0`, recent_utility `0`
- seconds `69.0`, LSTM `0.1649`, XGBoost `0.2789`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `68.5`, LSTM `0.1689`, XGBoost `0.2789`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
