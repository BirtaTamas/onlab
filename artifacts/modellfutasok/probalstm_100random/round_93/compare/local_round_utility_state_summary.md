# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1_finals/blast-bounty-2025-season-1-finals-spirit-vs-heroic-bo3-8PNegF4uXnTykkGvqzloIi/spirit-vs-heroic-m2-nuke.csv`
- round_num: `3`
- rows: `177`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 177 | 1.000 | 0.174615 | 0.271490 | -0.096875 | 170 | 7 | 1.000000 | 1.000000 |
| active/recent utility | 102 | 0.576 | 0.098191 | 0.178534 | -0.080343 | 95 | 7 | 1.000000 | 1.000000 |
| strong utility action | 85 | 0.480 | 0.112728 | 0.189739 | -0.077012 | 78 | 7 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 85 | 0.480 | 0.112728 | 0.189739 | -0.077012 | 78 | 7 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 88 | 0.497 | 0.089067 | 0.145218 | -0.056151 | 81 | 7 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `8.5s` - `50.5s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `44.0`, LSTM `0.0808`, XGBoost `0.3691`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.0`, LSTM `0.1470`, XGBoost `0.3971`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `44.5`, LSTM `0.1255`, XGBoost `0.3691`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `47.5`, LSTM `0.1562`, XGBoost `0.3971`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `46.0`, LSTM `0.1429`, XGBoost `0.3801`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `45.5`, LSTM `0.1432`, XGBoost `0.3801`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `49.0`, LSTM `0.1644`, XGBoost `0.3971`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `49.5`, LSTM `0.1689`, XGBoost `0.3971`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
- seconds `45.0`, LSTM `0.1515`, XGBoost `0.3788`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `50.0`, LSTM `0.1730`, XGBoost `0.3971`, closer `lstm`, smoke `1`, inferno `0`, utility_damage `0.0`, recent_utility `0`
