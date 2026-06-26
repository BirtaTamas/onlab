# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-the-mongolz-vs-aurora-bo3-BL3Rcvf733R8cy7TtLY5HE/the-mongolz-vs-aurora-m1-dust2.csv`
- round_num: `18`
- rows: `252`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 252 | 1.000 | 0.685254 | 0.691275 | -0.006021 | 110 | 142 | 0.932540 | 0.932540 |
| active/recent utility | 252 | 1.000 | 0.685254 | 0.691275 | -0.006021 | 110 | 142 | 0.932540 | 0.932540 |
| strong utility action | 176 | 0.698 | 0.628403 | 0.624685 | 0.003718 | 104 | 72 | 0.903409 | 0.903409 |
| utility damage | 10 | 0.040 | 0.664085 | 0.636997 | 0.027087 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 176 | 0.698 | 0.628403 | 0.624685 | 0.003718 | 104 | 72 | 0.903409 | 0.903409 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 252 | 1.000 | 0.685254 | 0.691275 | -0.006021 | 110 | 142 | 0.932540 | 0.932540 |

## Active Smoke/Inferno Intervals

- `11.0s` - `98.5s`, rows `176`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `98.0`, LSTM `0.7622`, XGBoost `0.9399`, closer `xgboost`, smoke `2`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `98.5`, LSTM `0.7990`, XGBoost `0.9399`, closer `xgboost`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.5`, LSTM `0.6087`, XGBoost `0.7469`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `87.0`, LSTM `0.6073`, XGBoost `0.7427`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `95.5`, LSTM `0.2865`, XGBoost `0.1546`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.5`, LSTM `0.6188`, XGBoost `0.7364`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `86.0`, LSTM `0.6271`, XGBoost `0.7429`, closer `xgboost`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.0`, LSTM `0.3978`, XGBoost `0.2876`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `94.5`, LSTM `0.2752`, XGBoost `0.1697`, closer `lstm`, smoke `3`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `91.5`, LSTM `0.3948`, XGBoost `0.2894`, closer `lstm`, smoke `4`, inferno `0`, utility_damage `0.0`, recent_utility `0`
