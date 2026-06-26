# Local Round Utility State Analysis

- csv_path: `processed_full/blast_bounty_season_1/blast-bounty-2025-season-1-astralis-vs-wildcard-bo3-qSXX__H_dx2QMbEuGWf0Qb/astralis-vs-wildcard-m2-mirage.csv`
- round_num: `15`
- rows: `196`
- true ct_win: `1`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 196 | 1.000 | 0.795570 | 0.744283 | 0.051288 | 162 | 34 | 1.000000 | 1.000000 |
| active/recent utility | 196 | 1.000 | 0.795570 | 0.744283 | 0.051288 | 162 | 34 | 1.000000 | 1.000000 |
| strong utility action | 158 | 0.806 | 0.795701 | 0.753591 | 0.042110 | 124 | 34 | 1.000000 | 1.000000 |
| utility damage | 10 | 0.051 | 0.789305 | 0.653467 | 0.135839 | 10 | 0 | 1.000000 | 1.000000 |
| active smoke/inferno | 158 | 0.806 | 0.795701 | 0.753591 | 0.042110 | 124 | 34 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 196 | 1.000 | 0.795570 | 0.744283 | 0.051288 | 162 | 34 | 1.000000 | 1.000000 |

## Active Smoke/Inferno Intervals

- `6.5s` - `59.5s`, rows `107`
- `61.0s` - `66.0s`, rows `11`
- `78.0s` - `97.5s`, rows `40`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `15.0`, LSTM `0.8132`, XGBoost `0.6525`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `97.0`, recent_utility `0`
- seconds `14.5`, LSTM `0.8120`, XGBoost `0.6525`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `97.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.8143`, XGBoost `0.6621`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `97.0`, recent_utility `0`
- seconds `16.5`, LSTM `0.7956`, XGBoost `0.6452`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `97.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.8046`, XGBoost `0.6575`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `97.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.8026`, XGBoost `0.6575`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `97.0`, recent_utility `0`
- seconds `16.0`, LSTM `0.7874`, XGBoost `0.6459`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `97.0`, recent_utility `0`
- seconds `22.0`, LSTM `0.7871`, XGBoost `0.6493`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `21.5`, LSTM `0.7772`, XGBoost `0.6493`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `17.0`, LSTM `0.7721`, XGBoost `0.6452`, closer `lstm`, smoke `3`, inferno `1`, utility_damage `97.0`, recent_utility `0`
