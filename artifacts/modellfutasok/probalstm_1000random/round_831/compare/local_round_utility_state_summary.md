# Local Round Utility State Analysis

- csv_path: `processed_full/blast_rivals_season_2/blast-rivals-2025-season-2-furia-vs-pain-bo3-BGpRMXEt8xpbRAS7KbpPH6/furia-vs-pain-m2-overpass.csv`
- round_num: `14`
- rows: `189`
- true ct_win: `0`

## Cohort Summary

| cohort | rows | row_rate | lstm_mean_prob | xgboost_mean_prob | lstm-xgb | lstm_closer | xgb_closer | lstm_acc@0.5 | xgb_acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all ticks | 189 | 1.000 | 0.130547 | 0.222536 | -0.091989 | 189 | 0 | 1.000000 | 0.873016 |
| active/recent utility | 189 | 1.000 | 0.130547 | 0.222536 | -0.091989 | 189 | 0 | 1.000000 | 0.873016 |
| strong utility action | 85 | 0.450 | 0.081878 | 0.146083 | -0.064205 | 85 | 0 | 1.000000 | 1.000000 |
| utility damage | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| active smoke/inferno | 85 | 0.450 | 0.081878 | 0.146083 | -0.064205 | 85 | 0 | 1.000000 | 1.000000 |
| recent utility last 5s | 0 | 0.000 | NA | NA | NA | 0 | 0 | NA | NA |
| flash effect present | 189 | 1.000 | 0.130547 | 0.222536 | -0.091989 | 189 | 0 | 1.000000 | 0.873016 |

## Active Smoke/Inferno Intervals

- `8.0s` - `50.0s`, rows `85`

## Biggest LSTM-XGBoost Differences During Utility States

- seconds `14.5`, LSTM `0.1438`, XGBoost `0.4532`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `14.0`, LSTM `0.1733`, XGBoost `0.4503`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.5`, LSTM `0.1854`, XGBoost `0.4503`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.0`, LSTM `0.2029`, XGBoost `0.4592`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `13.0`, LSTM `0.2021`, XGBoost `0.4441`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.0`, LSTM `0.2308`, XGBoost `0.4526`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `12.5`, LSTM `0.2307`, XGBoost `0.4513`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `15.5`, LSTM `0.2080`, XGBoost `0.4253`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.5`, LSTM `0.2564`, XGBoost `0.4545`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
- seconds `11.0`, LSTM `0.2799`, XGBoost `0.4542`, closer `lstm`, smoke `2`, inferno `1`, utility_damage `0.0`, recent_utility `0`
